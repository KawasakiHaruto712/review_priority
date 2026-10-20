"""評価日ごとの比較を、セル（行 r × 評価時期 p）にまとめる（design.md §3.1・§3.3）。

- δ の平均と標準偏差（評価日の間。標準偏差は ddof=1、1 日だけなら NaN）
- 有意（p ＜ ALPHA）だった日の割合（p が出ない日＝値が一定の日は分母に入れない。全部出なければ NaN）
p 値そのものは平均しない（平均した値は確率にならない）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.data_drift.utils import constants

KEYS = ["row", "position", "feature", "group"]


def to_cells(daily: pd.DataFrame, alpha: float = constants.ALPHA) -> pd.DataFrame:
    """daily（評価日ごと）→ cells（セルごと）。"""
    if daily.empty:
        return pd.DataFrame(columns=KEYS + ["delta_mean", "delta_std", "sig_rate", "n_days", "n_days_p"])
    d = daily.copy()
    d["_sig"] = np.where(d["p"].notna(), (d["p"] < alpha).astype(float), np.nan)
    g = d.groupby(KEYS, sort=True)
    out = pd.DataFrame({
        "delta_mean": g["delta"].mean(),
        "delta_std": g["delta"].std(ddof=1),
        "sig_rate": g["_sig"].mean(),            # NaN（p が出ない日）は除いて平均＝有意だった日の割合
        "n_days": g["delta"].count(),
        "n_days_p": g["_sig"].count(),
    }).reset_index()
    return out
