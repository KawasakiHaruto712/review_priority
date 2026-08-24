"""距離×時期行列の構築（転移学習フロー。design.md §6, §8）。

各 seed k（＝別の事前学習エンコーダ）について：
  1. 教師あり事前学習でエンコーダ（＋汎用ヘッド）を作る（§6.1）
  2. 各ビンの集合を埋め込みキャッシュ（§6.4）
  3. 学習ビンごとに linear probe を学習（使い回し）→ 評価ビンで予測（§6.2）
  4. 汎用ヘッドも各位置で評価（§8.3）
seed 横断で中央値＋IQR に集約し、以下 3 種の行列（MatrixResult）を返す：
  - probe   : per-bin probe の d×p 行列
  - general : 汎用ヘッドの d×p 行列（距離 d に非依存＝列 p ごとに一定）
  - diff    : probe − 汎用ヘッド（正＝特化が効く／負＝特化が悪化。§8.3）
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.analysis.preliminary_analysis.concept_drift_detection.evaluation import metrics
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import set_builder
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st

logger = logging.getLogger(__name__)


@dataclass
class MatrixResult:
    """1 指標の d×p 行列。value/iqr は [距離 d(0始まり), 位置 p(0始まり)] の 2 次元。"""
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


def build_matrices(bins: dict, *, encoders, scaler, metric_names=None,
                   base_seed=None, bin_count=None, device=None,
                   probe_sink=None, general_sink=None) -> dict:
    """転移学習フローで probe / general / diff の行列（指標ごと）を構築する。

    bins: {ローカルビン index -> レコード列}（binning.make_local_bins の出力）
    encoders: [(凍結エンコーダ, 汎用ヘッド), ...]（seed ごと。全リリース共有。§6.1）
    scaler: 事前学習データで fit した特徴標準化器（§7）
    probe_sink / general_sink: list を渡すと per-Change 予測を追記（保存用）。
    """
    metric_names = metric_names or metrics.metric_columns(constants.K_LIST)
    base_seed = constants.RANDOM_SEED if base_seed is None else base_seed
    bin_count = constants.BIN_COUNT if bin_count is None else bin_count
    device = st.resolve_device() if device is None else device
    max_set = constants.MAX_SET_SIZE
    thr = constants.CLASSIFY_THRESHOLD
    n_repeats = len(encoders)

    # 各ビンの集合（TSet）を用意（0..bin_count-1 が対象リリース、-bin_count..-1 が直前リリース）
    sets_by_bin = {i: set_builder.sets_in_bins(bins, i, max_set) for i in bins}

    # per-k の指標値を貯める（k でペアを取れるよう dict{k: value}）
    probe_vals = {m: defaultdict(dict) for m in metric_names}   # m -> (d,p) -> {k: val}
    gen_vals = {m: defaultdict(dict) for m in metric_names}      # m -> p -> {k: val}

    for k, (encoder, general_head) in enumerate(encoders):
        seed = base_seed + k
        embeds = {i: st.embed_sets(encoder, scaler, sets_by_bin[i], device) for i in sets_by_bin}

        # 学習ビンごとに probe を 1 回学習して使い回す（同一 i を共有するセルで再学習しない）
        probe_cache: dict[int, object] = {}

        def get_probe(i):
            if i not in probe_cache:
                s, e = sets_by_bin.get(i, []), embeds.get(i, [])
                probe_cache[i] = (st.train_probe(e, s, seed, device)
                                  if set_builder.count_records(s) >= constants.MIN_TRAIN else None)
            return probe_cache[i]

        for p in range(bin_count):
            eval_sets = sets_by_bin.get(p, [])
            if set_builder.count_records(eval_sets) < constants.MIN_EVAL:
                continue
            eval_emb = embeds.get(p, [])

            # 汎用ヘッド（§8.3）：位置 p で評価（距離に非依存）
            grows = st.predict(general_head, eval_emb, eval_sets, device)
            gm = metrics.cell_metrics(grows, constants.K_LIST, thr, constants.POOL_AUC)
            for m in metric_names:
                gen_vals[m][p][k] = gm[m]
            if general_sink is not None:
                for yt, yp, cid, t in grows:
                    general_sink.append((p + 1, k, t.date().isoformat(), cid, yt, yp))

            # per-bin probe（§6.2）：距離 d ごとに学習ビン i=p-d の probe で予測
            for d in range(1, bin_count + 1):
                head = get_probe(p - d)
                if head is None:
                    continue
                rows = st.predict(head, eval_emb, eval_sets, device)
                pm = metrics.cell_metrics(rows, constants.K_LIST, thr, constants.POOL_AUC)
                for m in metric_names:
                    probe_vals[m][(d, p)][k] = pm[m]
                if probe_sink is not None:
                    for yt, yp, cid, t in rows:
                        probe_sink.append((d, p + 1, k, t.date().isoformat(), cid, yt, yp))

    # ── seed 横断で集約して MatrixResult に ──
    probe_res, gen_res, diff_res = {}, {}, {}
    for m in metric_names:
        pv = _empty_matrix(bin_count); pi = _empty_matrix(bin_count)
        gv = _empty_matrix(bin_count); gi = _empty_matrix(bin_count)
        dv = _empty_matrix(bin_count); di = _empty_matrix(bin_count)
        per_repeat_p = {}
        # 汎用ヘッド（位置ごと、d に一定でブロードキャスト）
        gen_center_by_p = {}
        for p in range(bin_count):
            gc, gq = _agg(list(gen_vals[m][p].values()))
            gen_center_by_p[p] = (gc, gen_vals[m][p])
            for d in range(bin_count):
                gv[d, p] = gc; gi[d, p] = gq
        # probe と diff
        for d in range(1, bin_count + 1):
            for p in range(bin_count):
                pk = probe_vals[m].get((d, p), {})
                c, q = _agg(list(pk.values()))
                pv[d - 1, p] = c; pi[d - 1, p] = q
                per_repeat_p[(d, p - 1)] = list(pk.values())  # plotter/writer 互換の列は 0 始まり
                # diff は k でペアを取って (probe - 汎用) を計算してから集約
                gp = gen_center_by_p[p][1]
                paired = [pk[kk] - gp[kk] for kk in pk.keys() & gp.keys()
                          if not (np.isnan(pk[kk]) or np.isnan(gp[kk]))]
                dc, dq = _agg(paired)
                dv[d - 1, p] = dc; di[d - 1, p] = dq
        probe_res[m] = MatrixResult(m, bin_count, pv, pi, per_repeat_p)
        gen_res[m] = MatrixResult(m, bin_count, gv, gi, {})
        diff_res[m] = MatrixResult(f"{m}_diff", bin_count, dv, di, {})

    return {"probe": probe_res, "general": gen_res, "diff": diff_res}
