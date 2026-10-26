"""日ごとの値 → マスごとの平均 → 最良の N（design.md §3・§4）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.release_best_size.utils import constants


def daily_values(metrics: pd.DataFrame) -> pd.DataFrame:
    """評価日 × N ごとの F1 値（5 seed の中央値。欠けた seed・NaN を除く。3.4 と同じ決まり）。列 day・size・value。
    5 seed とも欠けた（評価日 × N）は行を作らない。"""
    d = metrics[(metrics["missing"] == 0) & metrics["value"].notna()]
    return d.groupby(["day", "size"])["value"].median().reset_index()


def attach_position(daily: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """精度表の day → position を付ける。精度表にない日（リリースの日）は落とす（design.md §2）。"""
    out = daily.merge(positions.rename("position"), left_on="day", right_index=True, how="inner")
    return out.assign(position=out["position"].astype(int))


def cell_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """位置 × N ごとの平均・標準偏差（ddof＝1）・使った日数。N ごとに、その N で値が出た日だけを使う（4.1 と同じ）。"""
    g = daily.groupby(["position", "size"])["value"]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n_days": g.size()}).reset_index()


def pick_best(stats: pd.DataFrame) -> str | None:
    """1 マスの N 別の表（列 size・mean・std）から最良の N。平均が最大、同じなら標準偏差が小さい方、
    それでも同じなら N の短い順（lookback_window の select_best.pick_best と同じ決まり。design.md §4）。平均が出ていなければ None。"""
    d = stats.dropna(subset=["mean"])
    if d.empty:
        return None
    tied = d[d["mean"] == d["mean"].max()]
    if len(tied) > 1:
        order = {s: i for i, s in enumerate(constants.SIZES)}
        tied = tied.assign(_o=tied["size"].map(order)).sort_values(["std", "_o"], na_position="last", kind="stable")
    return str(tied["size"].iloc[0])


def release_rows(method: str, project: str, version: str, daily: pd.DataFrame, selected: str) -> list[dict]:
    """1 版の 3 マス分の行（best_size.csv の行。design.md §5）。daily は position 付きの日ごとの値。"""
    stats = cell_stats(daily)
    rows = []
    for position in range(constants.N_COLS):
        s = stats[stats["position"] == position]
        q = constants.position_display(position)
        row = {"method": method, "project": project, "version": version, "position": position,
               "position_display": q, "phase_name": constants.PHASE_NAMES[q]}
        by = s.set_index("size")
        for size in constants.SIZES:
            row[f"mean_{size}"] = by["mean"].get(size, np.nan)
            row[f"std_{size}"] = by["std"].get(size, np.nan)
            row[f"n_days_{size}"] = int(by["n_days"].get(size, 0))
        best = pick_best(s)
        row["best"] = best
        row["best_mean"] = row[f"mean_{best}"] if best else np.nan
        row["selected"] = selected
        row["selected_mean"] = row[f"mean_{selected}"]
        row["diff_best_minus_selected"] = row["best_mean"] - row["selected_mean"]
        row["is_selected_best"] = best == selected
        row["n_days_any"] = int(daily.loc[daily["position"] == position, "day"].nunique())
        rows.append(row)
    return rows
