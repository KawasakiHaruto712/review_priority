"""RQ2：評価日 × 行ごとに 1 つの seed を選ぶ決まり（design.md §3.2。2026-10-22）。

寄与度分析（concept_drift_cause/main.py の median_seed・select_seeds）と同じ決まりの**複製**（import しない。各ディレクトリを独立させるため）。
2 つが同じ seed を選ぶことはテストで確かめる。

- 欠測（missing ＝ 1）の seed を除く。
- 残りを recall@pos → AUC → seed 番号の昇順に並べ、(件数 − 1) ÷ 2 番目（切り捨て。0 始まり）を選ぶ（5 個なら 3 番目）。
- 全部欠測（または recall@pos が全部 NaN）なら選ばない（その評価日・行は比べない）。
"""
from __future__ import annotations

import pandas as pd

SELECT_METRICS = ("recall@pos", "auc")   # 並べる順（concept_drift_cause の SELECT_METRICS と同じ）


def median_seed(sub: pd.DataFrame, metrics=SELECT_METRICS) -> int | None:
    """その (評価日, 行) で真ん中の seed。なければ None。"""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    s = sub.dropna(subset=[metrics[0]])
    if s.empty:
        return None
    s = s.sort_values(list(metrics) + ["seed"], kind="stable", na_position="last")
    return int(s.iloc[(len(s) - 1) // 2]["seed"])


def select_seeds(daily: pd.DataFrame, metrics=SELECT_METRICS) -> dict:
    """{(評価日(ISO), 行(str)) -> seed}。daily は RQ2 の daily_metrics（行の列名は distance。汎用ヘッドの行 general も入っている）。"""
    d = daily[daily["missing"] == 0].copy()
    d["distance"] = d["distance"].astype(str)
    out = {}
    for (day, dist), sub in d.groupby(["day", "distance"]):
        seed = median_seed(sub, metrics)
        if seed is not None:
            out[(str(day), str(dist))] = seed
    return out
