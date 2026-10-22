"""評価日ごとの比較を、セル（行 r × 評価時期 p）にまとめる（design.md §3.1・§3.3）。

- δ の平均と標準偏差（評価日の間。標準偏差は ddof=1、1 日だけなら NaN）
- 有意（p ＜ ALPHA）だった日の割合（p が出ない日＝値が一定の日は分母に入れない。全部出なければ NaN）
p 値そのものは平均しない（平均した値は確率にならない）。
- （2026-10-22 追加）全体のずれの大きさ：セルごと（summarize_cells）と段ごと（summarize_rows）に、|δ| の平均と |δ| ≥ 0.147 の数（§3.4）
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


# ── 全体のずれの大きさ（design.md §3.4。2026-10-22） ─────────────────
def _overall(per_feature: pd.DataFrame, keys: list[str], bound: float) -> pd.DataFrame:
    """per_feature（keys ＋ feature ＋ delta ＋ n_days）を、keys ごとに 55 特徴量でまとめる。"""
    d = per_feature.copy()
    d["_abs"] = d["delta"].abs()
    d["_ge"] = (d["_abs"] >= bound).astype(int)
    g = d.groupby(keys, sort=True)
    return pd.DataFrame({
        "abs_delta_mean": g["_abs"].mean(),      # |δ| の平均（値が一定で δ ＝ 0 の特徴量も含む）
        "n_ge_bound": g["_ge"].sum(),            # |δ| ≥ bound の特徴量の数
        "n_features": g["_abs"].count(),
        "n_days": g["n_days"].max(),
    }).reset_index()


def summarize_cells(cells: pd.DataFrame, bound: float = constants.EFFECT_BOUNDS[0]) -> pd.DataFrame:
    """cells → cells_summary（セル × 群ごとに |δ の平均| の 55 特徴量の平均と、|δ の平均| ≥ 0.147 の特徴量の数）。"""
    cols = ["row", "position", "group", "abs_delta_mean", "n_ge_bound", "n_features", "n_days"]
    if cells.empty:
        return pd.DataFrame(columns=cols)
    pf = cells[["row", "position", "group", "feature", "delta_mean", "n_days"]].rename(columns={"delta_mean": "delta"})
    return _overall(pf.dropna(subset=["delta"]), ["row", "position", "group"], bound)[cols]


def summarize_rows(cells: pd.DataFrame, bound: float = constants.EFFECT_BOUNDS[0]) -> pd.DataFrame:
    """cells → 段（行）× 群ごとの全体のずれの大きさ（RQ1 と RQ2 を比べる折れ線の点。design.md §5）。

    評価時期 p をまとめ、特徴量ごとに「その段の全評価日の δ の平均」（＝セルの δ の平均を評価日の数で重み付けた平均）を作ってから、
    |δ| の平均と |δ| ≥ bound の数にする。評価日ごとの値を平均する重みはセルと同じ。
    """
    cols = ["row", "group", "abs_delta_mean", "n_ge_bound", "n_features", "n_days"]
    c = cells.dropna(subset=["delta_mean"])
    c = c[c["n_days"] > 0]
    if c.empty:
        return pd.DataFrame(columns=cols)
    c = c.assign(_w=c["delta_mean"] * c["n_days"])
    g = c.groupby(["row", "group", "feature"], sort=True)
    pf = pd.DataFrame({"delta": g["_w"].sum() / g["n_days"].sum(), "n_days": g["n_days"].sum()}).reset_index()
    return _overall(pf, ["row", "group"], bound)[cols]
