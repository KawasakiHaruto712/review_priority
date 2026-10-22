"""5 版をまとめた 3 × 3 の表（design.md §15。pretrained_encoders/design.md §15.9。2026-10-22。重みあり版）。

各版の日ごとの指標（drift_matrix の daily_sink と同じ行：day・distance・seed・position・missing・各指標）を全版ぶんつなぎ、
セル (d, p) に入る全版の評価日について、版ごとの表と同じ集約（日ごとに seed 中央値 → 評価日で平均。± は評価日の間の標準偏差）をする。
評価日の多い版ほど重くなる（本人判断：全評価日の間）。版は周期の 1/3 で列を切っているので、同じ列は「周期の同じ段階」になる。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_detection.evaluation.drift_matrix import (
    MatrixResult, _cell_agg, _empty_matrix, _seed_day_means,
)


def merge_versions(daily: pd.DataFrame, grid: int, metric_names: list[str]) -> dict:
    """{"probe"|"general"|"diff": {指標: MatrixResult}} を返す。daily は全版の日ごとの行をつないだもの。"""
    d = daily[daily["missing"] == 0]
    gen = d[d["distance"].astype(str) == "general"]
    prb = d[d["distance"].astype(str) != "general"].copy()
    prb["distance"] = prb["distance"].astype(int)
    probe_res, gen_res, diff_res = {}, {}, {}
    for m in metric_names:
        pv, pi, ps, pn = (_empty_matrix(grid) for _ in range(4))
        gv, gi, gs, gn = (_empty_matrix(grid) for _ in range(4))
        per_repeat = {}
        for p in range(grid):
            c, q, sd, nd = _cell_agg(_per_day(gen[gen["position"] == p], m))
            gv[:, p], gi[:, p], gs[:, p], gn[:, p] = c, q, sd, nd       # 汎用ヘッドはずれによらない
        for dd in range(grid):
            for p in range(grid):
                per_day = _per_day(prb[(prb["distance"] == dd) & (prb["position"] == p)], m)
                c, q, sd, nd = _cell_agg(per_day)
                pv[dd, p], pi[dd, p], ps[dd, p], pn[dd, p] = c, q, sd, nd
                per_repeat[(dd, p)] = _seed_day_means(per_day)
        probe_res[m] = MatrixResult(m, grid, pv, pi, per_repeat, ps, pn)
        gen_res[m] = MatrixResult(m, grid, gv, gi, {}, gs, gn)
        diff_res[m] = MatrixResult(f"{m}_diff", grid, pv - gv, _empty_matrix(grid), {}, _empty_matrix(grid), pn)
    return {"probe": probe_res, "general": gen_res, "diff": diff_res}


def _per_day(rows: pd.DataFrame, metric: str) -> dict:
    """行 → {評価日 -> {seed -> 値}}（NaN は除く。drift_matrix の集約と同じ入れ物）。"""
    out: dict = defaultdict(dict)
    if metric not in rows.columns:
        return out
    for day, seed, v in zip(rows["day"], rows["seed"], rows[metric]):
        if pd.notna(v):
            out[day][int(seed)] = float(v)
    return out
