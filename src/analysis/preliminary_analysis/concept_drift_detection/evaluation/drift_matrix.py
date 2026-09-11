"""距離×位置行列の構築（日次スライド学習。design.md §4, §6, §8）。

各 seed k（＝別の事前学習エンコーダ）について：
  1. 凍結エンコーダで**全日の集合を 1 回だけ埋め込みキャッシュ**（§6.4）
  2. **評価日ごと・距離ごとに** linear probe を貼り直す（§6.2）
       学習期間 = [x − 窓長 − 刻み×d, x − 1 − 刻み×d]（末尾は前日側。評価日自身は入れない）
       同じ「学習期間の末尾日」の probe は共有（キャッシュ）
  3. その評価日を Δ=1 で予測 → **日次の指標**を算出
  4. 汎用ヘッドも各評価日で評価（距離に非依存。§8.3）
日次の指標を**位置（列）の日数で平均** → seed 横断で中央値＋IQR に集約し、3 種の行列を返す：
  - probe   : 日次 probe の d×p 行列（**d=0 が step1 の fresh 窓と一致**）
  - general : 汎用ヘッドの d×p 行列（距離 d に非依存＝列 p ごとに一定）
  - diff    : probe − 汎用ヘッド（正＝特化が効く／負＝特化が悪化。§8.3）
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from src.analysis.preliminary_analysis.concept_drift_detection.evaluation import metrics
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st

logger = logging.getLogger(__name__)


@dataclass
class MatrixResult:
    """1 指標の d×p 行列。value/iqr は [距離 d(0始まり), 位置 p(0始まり)] の 2 次元。

    bin_count は格子サイズ N（距離の行数 ＝ 位置の列数。design.md §4.2）。
    距離 d は 0 始まりで、**d=0 が step1 の fresh 窓**に対応する。
    """
    metric: str
    bin_count: int
    value: np.ndarray
    iqr: np.ndarray
    per_repeat: dict = field(default_factory=dict)


def _agg(vals: list[float]) -> tuple[float, float]:
    """反復値のリスト → (代表値, IQR)。NaN は除外。空なら (nan, nan)。"""
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(arr)) if constants.REPEAT_AGG == "median" else float(np.mean(arr))
    q75, q25 = np.percentile(arr, [75, 25]) if arr.size > 1 else (arr[0], arr[0])
    return center, float(q75 - q25)


def _empty_matrix(bin_count: int) -> np.ndarray:
    return np.full((bin_count, bin_count), np.nan)


def _counts(sets) -> tuple[int, int]:
    """集合群の (正例数, 負例数)。"""
    total = sum(len(s) for s in sets)
    pos = int(sum(sum(s.labels) for s in sets))
    return pos, total - pos


def _daily_row(day, distance, seed_idx: int, position: int,
               n_pos_e: int, n_neg_e: int, n_pos_t: int, n_neg_t: int,
               missing: bool, reason: str, mvals: dict | None) -> dict:
    """(評価日, 距離, seed) ごとの指標行（保存用。§10）。distance='general' は汎用ヘッド。"""
    row = {"day": day.isoformat(), "distance": distance, "seed": seed_idx, "position": position + 1,
           "n_pos_eval": n_pos_e, "n_neg_eval": n_neg_e,
           "n_pos_train": n_pos_t, "n_neg_train": n_neg_t,
           "missing": int(missing), "missing_reason": reason}
    for col in metrics.metric_columns(constants.K_LIST):
        row[col] = (mvals or {}).get(col, metrics.NAN)
    return row


def build_matrices(day_sets: dict, eval_dates: list, positions: dict, *,
                   encoders, scaler, grid: int, window_days: int, step_days: int,
                   metric_names=None, base_seed=None, device=None,
                   daily_sink=None) -> dict:
    """日次スライド学習で probe / general / diff の行列（指標ごと）を構築する（§4, §6.2）。

    day_sets   : {日付 -> TSet}（学習に遡る範囲＋評価期間を含む。1 日 1 集合）
    eval_dates : 評価日の列（対象サイクルの全日）
    positions  : {日付 -> 位置 p（0..grid-1）}（binning.position_of_days の出力）
    grid       : 格子サイズ N（距離の行数 ＝ 位置の列数）
    window_days: 学習に使う期間の長さ（日）
    step_days  : 距離 1 段ぶんの日数
    encoders   : [(凍結エンコーダ, 汎用ヘッド), ...]（seed ごと。§6.1）
    scaler     : 事前学習データで fit した特徴標準化器
    daily_sink : list を渡すと (評価日, 距離, seed) ごとの指標行を追記（保存用。§10）
    """
    metric_names = metric_names or metrics.metric_columns(constants.K_LIST)
    base_seed = constants.RANDOM_SEED if base_seed is None else base_seed
    device = st.resolve_device() if device is None else device
    thr = constants.CLASSIFY_THRESHOLD

    all_dates = sorted(day_sets)
    # m -> (d,p) -> seed -> [日次値]／m -> p -> seed -> [日次値]（汎用は距離に非依存）
    probe_vals = {m: defaultdict(lambda: defaultdict(list)) for m in metric_names}
    gen_vals = {m: defaultdict(lambda: defaultdict(list)) for m in metric_names}

    for k, (encoder, general_head) in enumerate(encoders):
        seed = base_seed + k
        # 全日の集合を 1 回だけ埋め込み（凍結エンコーダ。使い回す。§6.4）
        embeds = st.embed_sets(encoder, scaler, [day_sets[d] for d in all_dates], device)
        emb_by_date = dict(zip(all_dates, embeds))

        # 学習期間の「末尾日」で probe をキャッシュ（同じ末尾日のセルは再学習しない。§6.2）
        probe_cache: dict = {}

        def get_probe(train_end):
            if train_end not in probe_cache:
                train_start = train_end - timedelta(days=window_days - 1)
                tdates = [d for d in all_dates if train_start <= d <= train_end]
                tsets = [day_sets[d] for d in tdates]
                n_pos_t, n_neg_t = _counts(tsets) if tsets else (0, 0)
                head = None
                if (tsets and n_pos_t >= 1 and n_neg_t >= 1
                        and set_builder.count_records(tsets) >= constants.MIN_TRAIN):
                    head = st.train_probe([emb_by_date[d] for d in tdates], tsets, seed, device)
                probe_cache[train_end] = (head, n_pos_t, n_neg_t)
            return probe_cache[train_end]

        for X in eval_dates:
            eval_set = day_sets.get(X)
            p = positions.get(X)
            if eval_set is None or p is None:
                continue
            yt = np.array(eval_set.labels, dtype=float)
            n_pos_e, n_neg_e = int(yt.sum()), int(len(yt) - yt.sum())
            eval_ok = (n_pos_e >= 1 and n_neg_e >= 1 and len(eval_set) >= constants.MIN_EVAL)
            eval_emb = emb_by_date[X]

            # 汎用ヘッド（§8.3）：その評価日で評価（距離に非依存）
            if eval_ok:
                grows = st.predict(general_head, [eval_emb], [eval_set], device)
                gm = metrics.compute_day_metrics([r[0] for r in grows], [r[1] for r in grows],
                                                 constants.K_LIST, thr)
                for m in metric_names:
                    if not np.isnan(gm[m]):
                        gen_vals[m][p][k].append(gm[m])
                if daily_sink is not None:
                    daily_sink.append(_daily_row(X, "general", k, p, n_pos_e, n_neg_e,
                                                 -1, -1, False, "", gm))
            elif daily_sink is not None:
                daily_sink.append(_daily_row(X, "general", k, p, n_pos_e, n_neg_e,
                                             -1, -1, True, "eval_guard", None))

            # 距離 d ごとに学習期間をずらして probe を貼り直す（§4.1）
            for d in range(grid):
                train_end = X - timedelta(days=1 + step_days * d)
                head, n_pos_t, n_neg_t = get_probe(train_end)
                reason = "" if eval_ok else "eval_guard"
                if not reason and head is None:
                    reason = "train_guard"
                if reason:
                    if daily_sink is not None:
                        daily_sink.append(_daily_row(X, d, k, p, n_pos_e, n_neg_e,
                                                     n_pos_t, n_neg_t, True, reason, None))
                    continue
                rows = st.predict(head, [eval_emb], [eval_set], device)
                mv = metrics.compute_day_metrics([r[0] for r in rows], [r[1] for r in rows],
                                                 constants.K_LIST, thr)
                for m in metric_names:
                    if not np.isnan(mv[m]):
                        probe_vals[m][(d, p)][k].append(mv[m])
                if daily_sink is not None:
                    daily_sink.append(_daily_row(X, d, k, p, n_pos_e, n_neg_e,
                                                 n_pos_t, n_neg_t, False, "", mv))

    # ── 日次値 → 位置（列）で平均 → seed 横断で中央値＋IQR に集約 ──
    probe_res, gen_res, diff_res = {}, {}, {}
    for m in metric_names:
        pv = _empty_matrix(grid); pi = _empty_matrix(grid)
        gv = _empty_matrix(grid); gi = _empty_matrix(grid)
        dv = _empty_matrix(grid); di = _empty_matrix(grid)
        per_repeat_p = {}
        # 汎用ヘッド（位置ごとに seed 内平均 → seed 横断集約。距離方向へブロードキャスト）
        gen_seed_mean: dict = {}
        for p in range(grid):
            per_seed = {kk: float(np.mean(v)) for kk, v in gen_vals[m].get(p, {}).items() if v}
            gen_seed_mean[p] = per_seed
            gc, gq = _agg(list(per_seed.values()))
            for d in range(grid):
                gv[d, p] = gc; gi[d, p] = gq
        # probe と diff（d は 0 始まりでそのまま行 index）
        for d in range(grid):
            for p in range(grid):
                per_seed = {kk: float(np.mean(v))
                            for kk, v in probe_vals[m].get((d, p), {}).items() if v}
                c, q = _agg(list(per_seed.values()))
                pv[d, p] = c; pi[d, p] = q
                per_repeat_p[(d, p)] = list(per_seed.values())
                # diff は seed でペアを取って (probe − 汎用) を計算してから集約
                gp = gen_seed_mean.get(p, {})
                paired = [per_seed[kk] - gp[kk] for kk in per_seed.keys() & gp.keys()]
                dc, dq = _agg(paired)
                dv[d, p] = dc; di[d, p] = dq
        probe_res[m] = MatrixResult(m, grid, pv, pi, per_repeat_p)
        gen_res[m] = MatrixResult(m, grid, gv, gi, {})
        diff_res[m] = MatrixResult(f"{m}_diff", grid, dv, di, {})

    return {"probe": probe_res, "general": gen_res, "diff": diff_res}
