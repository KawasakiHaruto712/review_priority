"""距離×位置行列の構築（日次スライド学習。design.md §4, §6, §8）。

各 seed k（＝別の事前学習エンコーダ）について：
  1. 凍結エンコーダで**全日の集合を 1 回だけ埋め込みキャッシュ**（§6.4）
  2. **評価日ごと・距離ごとに** linear probe を貼り直す（§6.2）
       学習期間 = [x − 窓長 − 刻み×d, x − 1 − 刻み×d]（末尾は前日側。評価日自身は入れない）
       同じ「学習期間の末尾日」の probe は共有（キャッシュ）
  3. その評価日を Δ=1 で予測 → **日次の指標**を算出
  4. 汎用ヘッドも各評価日で評価（距離に非依存。§8.3）
集約の順序は **① 日ごとに seed 中央値 → ② 位置（列）の日数で平均**（§8.1）。逆順ではない理由は
design.md §8.1 参照（その日だけ異常な結果を出したモデルを弾けるのはこの順序だけ）。3 種の行列を返す：
  - probe   : 日次 probe の d×p 行列（**d=0 が窓長の調査の fresh 窓と一致**）
  - general : 汎用ヘッドの d×p 行列（距離 d に非依存＝列 p ごとに一定）
  - diff    : probe − 汎用ヘッド（正＝特化が効く／負＝特化が悪化。§8.3）
              **精度というスカラーの引き算なので seed のペアは取らない**（行列同士の単純な差。§8.3）。
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
    """1 指標の d×p 行列。各配列は [距離 d(0始まり), 位置 p(0始まり)] の 2 次元。

    bin_count は格子サイズ N（距離の行数 ＝ 位置の列数。design.md §4.2）。
    距離 d は 0 始まりで、**d=0 が窓長の調査の fresh 窓**に対応する。

    value   : セル値（日ごとの seed 中央値を、位置の日数で平均したもの。§8.1）
    iqr     : **seed 間 IQR**（日ごとに 5 seed の四分位範囲を取り、日で平均）＝モデル間の不一致
    day_std : **日間 標準偏差**（日ごとの seed 中央値を、位置の日数で取った標準偏差）＝日によるブレ
    n_days  : そのセルで有効だった日数（std が何日から出たか判断するため。§9）
    """
    metric: str
    bin_count: int
    value: np.ndarray
    iqr: np.ndarray
    per_repeat: dict = field(default_factory=dict)
    day_std: np.ndarray | None = None
    n_days: np.ndarray | None = None


def _agg(vals: list[float]) -> tuple[float, float]:
    """**1 評価日**の seed 横断集約：反復値のリスト → (代表値, IQR)。NaN は除外。空なら (nan, nan)。

    `N_REPEATS` は奇数（5）なので、中央値は**実在する 1 個のモデルの値**になる（§6.5）。
    """
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(arr)) if constants.REPEAT_AGG == "median" else float(np.mean(arr))
    q75, q25 = np.percentile(arr, [75, 25]) if arr.size > 1 else (arr[0], arr[0])
    return center, float(q75 - q25)


def _cell_agg(per_day: dict) -> tuple[float, float, float, int]:
    """{評価日 -> {seed -> 日次値}} → (セル値, seed間IQR, 日間std, 有効日数)。§8.1 の ①→②。

    ① 日ごとに seed 中央値（と seed 間 IQR）を取り、② それを位置の日数で平均する。
    日間 std は ① の日次値のばらつきなので、**モデルの入れ替わりではなく日の違いだけ**を測る。
    """
    centers, iqrs = [], []
    for _day in sorted(per_day):
        c, q = _agg(list(per_day[_day].values()))
        if not np.isnan(c):
            centers.append(c)
            iqrs.append(q)
    if not centers:
        nan = float("nan")
        return nan, nan, nan, 0
    arr = np.array(centers, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), float(np.nanmean(iqrs)), std, int(arr.size)


def _seed_day_means(per_day: dict) -> list[float]:
    """{評価日 -> {seed -> 値}} → seed ごとの日平均のリスト（per_repeat 用の参考値）。

    注：**セル値はこれの中央値ではない**（集約は日ごとの seed 中央値が先。§8.1）。
    seed ごとの成績を後から眺めるための記録にすぎない。
    """
    by_seed: dict = defaultdict(list)
    for by in per_day.values():
        for kk, v in by.items():
            by_seed[kk].append(v)
    return [float(np.mean(vs)) for _kk, vs in sorted(by_seed.items()) if vs]


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
                   daily_sink=None, probe_sink=None) -> dict:
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
    probe_sink : 関数を渡すと、probe を新規学習するたびに `probe_sink(seed_idx, 末尾日, head)`
                 を呼ぶ（特徴量の寄与度分析用の保存。§6.7）。probe は版に依存しないので重複は呼び先で弾く。
    """
    metric_names = metric_names or metrics.metric_columns(constants.K_LIST)
    base_seed = constants.RANDOM_SEED if base_seed is None else base_seed
    device = st.resolve_device() if device is None else device
    thr = constants.CLASSIFY_THRESHOLD

    all_dates = sorted(day_sets)
    # m -> (d,p) -> 評価日 -> seed -> 日次値／m -> p -> 評価日 -> seed -> 日次値（汎用は距離に非依存）
    # 日を残すのは、集約が「日ごとに seed 中央値 → 日で平均」の順だから（§8.1）。
    probe_vals = {m: defaultdict(lambda: defaultdict(dict)) for m in metric_names}
    gen_vals = {m: defaultdict(lambda: defaultdict(dict)) for m in metric_names}

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
                    if probe_sink is not None:
                        probe_sink(k, train_end, head)   # 特徴量の寄与度分析用に保存（§6.7）
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
                        gen_vals[m][p][X][k] = gm[m]
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
                        probe_vals[m][(d, p)][X][k] = mv[m]
                if daily_sink is not None:
                    daily_sink.append(_daily_row(X, d, k, p, n_pos_e, n_neg_e,
                                                 n_pos_t, n_neg_t, False, "", mv))

    # ── 日ごとに seed 中央値 → 位置（列）の日数で平均（§8.1） ──
    probe_res, gen_res, diff_res = {}, {}, {}
    for m in metric_names:
        pv = _empty_matrix(grid); pi = _empty_matrix(grid)
        ps = _empty_matrix(grid); pn = _empty_matrix(grid)
        gv = _empty_matrix(grid); gi = _empty_matrix(grid)
        gs = _empty_matrix(grid); gn = _empty_matrix(grid)
        per_repeat_p = {}
        # 汎用ヘッド（位置ごとに集約 → 距離方向へブロードキャスト）。
        # seed 中央値は**汎用ヘッド自身の精度**で取る（probe の中央 seed には合わせない。§8.3）。
        for p in range(grid):
            c, q, sd, nd = _cell_agg(gen_vals[m].get(p, {}))
            for d in range(grid):
                gv[d, p] = c; gi[d, p] = q; gs[d, p] = sd; gn[d, p] = nd
        # probe（d は 0 始まりでそのまま行 index）
        for d in range(grid):
            for p in range(grid):
                per_day = probe_vals[m].get((d, p), {})
                c, q, sd, nd = _cell_agg(per_day)
                pv[d, p] = c; pi[d, p] = q; ps[d, p] = sd; pn[d, p] = nd
                per_repeat_p[(d, p)] = _seed_day_means(per_day)
        # diff は **probe 行列 − 汎用ヘッド行列**（精度の引き算なので seed のペアは取らない。§8.3）。
        # ばらつきは 2 つの行列の差からは求まらないので NaN（必要なら daily_metrics から再集計する）。
        dv = pv - gv
        probe_res[m] = MatrixResult(m, grid, pv, pi, per_repeat_p, ps, pn)
        gen_res[m] = MatrixResult(m, grid, gv, gi, {}, gs, gn)
        diff_res[m] = MatrixResult(f"{m}_diff", grid, dv, _empty_matrix(grid), {},
                                   _empty_matrix(grid), pn)

    return {"probe": probe_res, "general": gen_res, "diff": diff_res}
