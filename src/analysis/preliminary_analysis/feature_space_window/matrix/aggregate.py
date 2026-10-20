"""精度表のまとめ方（feature_space_window/design.md §9.1。concept_drift_detection/evaluation/drift_matrix.py からの複製。2026-10-21）。

集約の順序は RQ1 と同じ **① 日ごとに seed 中央値 → ② 評価時期（列）の日数で平均**。3 種の表を返す：
  - probe   : 特徴のずれ g × 評価時期 p の表
  - general : 汎用ヘッドの表（行によらない＝列ごとに一定）
  - diff    : probe − 汎用ヘッド（行列どうしの単純な差。seed のペアは取らない）
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.analysis.preliminary_analysis.feature_space_window.evaluation import metrics
from src.analysis.preliminary_analysis.feature_space_window.utils import constants


@dataclass
class MatrixResult:
    """1 指標の g×p の表（RQ1 の MatrixResult と同じ形。bin_count ＝ N_v）。"""
    metric: str
    bin_count: int
    value: np.ndarray
    iqr: np.ndarray
    per_repeat: dict = field(default_factory=dict)
    day_std: np.ndarray | None = None
    n_days: np.ndarray | None = None


def _agg(vals: list[float]) -> tuple[float, float]:
    """1 評価日の seed 横断集約：(代表値, IQR)。NaN は除外。空なら (nan, nan)。"""
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(arr)) if constants.REPEAT_AGG == "median" else float(np.mean(arr))
    q75, q25 = np.percentile(arr, [75, 25]) if arr.size > 1 else (arr[0], arr[0])
    return center, float(q75 - q25)


def cell_agg(per_day: dict) -> tuple[float, float, float, int]:
    """{評価日 -> {seed -> 日次値}} → (セル値, seed 間 IQR, 日間 std, 有効日数)。"""
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
    by_seed: dict = defaultdict(list)
    for by in per_day.values():
        for kk, v in by.items():
            by_seed[kk].append(v)
    return [float(np.mean(vs)) for _kk, vs in sorted(by_seed.items()) if vs]


def _empty(n: int) -> np.ndarray:
    return np.full((n, n), np.nan)


def daily_row(day, row, seed_idx: int, position: int, n_pos_e: int, n_neg_e: int, n_pos_t: int, n_neg_t: int,
              missing: bool, reason: str, mvals: dict | None) -> dict:
    """（評価日, 行, seed）ごとの指標行（RQ1 の `_daily_row` と同じ列。行の列名は distance のまま。row='general' は汎用ヘッド）。"""
    out = {"day": day.isoformat(), "distance": row, "seed": seed_idx, "position": position,
           "n_pos_eval": n_pos_e, "n_neg_eval": n_neg_e, "n_pos_train": n_pos_t, "n_neg_train": n_neg_t,
           "missing": int(missing), "missing_reason": reason}
    for col in metrics.metric_columns(constants.K_LIST):
        out[col] = (mvals or {}).get(col, metrics.NAN)
    return out


def assemble(probe_vals: dict, gen_vals: dict, grid: int, metric_names: list[str]) -> dict:
    """probe_vals[m][(g, p)][日][seed]・gen_vals[m][p][日][seed] → {"probe"|"general"|"diff": {m: MatrixResult}}。"""
    probe_res, gen_res, diff_res = {}, {}, {}
    for m in metric_names:
        pv, pi, ps, pn = _empty(grid), _empty(grid), _empty(grid), _empty(grid)
        gv, gi, gs, gn = _empty(grid), _empty(grid), _empty(grid), _empty(grid)
        per_repeat = {}
        for p in range(grid):
            c, q, sd, nd = cell_agg(gen_vals[m].get(p, {}))
            for g in range(grid):
                gv[g, p], gi[g, p], gs[g, p], gn[g, p] = c, q, sd, nd
        for g in range(grid):
            for p in range(grid):
                per_day = probe_vals[m].get((g, p), {})
                pv[g, p], pi[g, p], ps[g, p], pn[g, p] = cell_agg(per_day)
                per_repeat[(g, p)] = _seed_day_means(per_day)
        probe_res[m] = MatrixResult(m, grid, pv, pi, per_repeat, ps, pn)
        gen_res[m] = MatrixResult(m, grid, gv, gi, {}, gs, gn)
        diff_res[m] = MatrixResult(f"{m}_diff", grid, pv - gv, _empty(grid), {}, _empty(grid), pn)
    return {"probe": probe_res, "general": gen_res, "diff": diff_res}
