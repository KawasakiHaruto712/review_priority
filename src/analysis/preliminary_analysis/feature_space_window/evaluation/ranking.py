"""順位の集計（1 位／最下位の回数。design.md §5.1）。lookback_window/evaluation/ranking.py の複製（各ディレクトリ自己完結）。

- （版, 日, 大きさ）で seed 中央値 → 6 つの大きさがそろう日だけで順位づけ。
- 各日の最高 AUC の大きさに「1 位 ＋1」、最低に「最下位 ＋1」。同着は全部に足す（率の合計は 100% を超えうる）。
- 日次のまま（ビン集約はしない）。RQ2 には汎用ヘッドがないので除く処理はない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.feature_space_window.io import result_writer
from src.analysis.preliminary_analysis.feature_space_window.utils import constants


def size_labels_desc() -> list[str]:
    """大きさを大きい順（2M → 1d）にラベル化（要約表と同じ並び）。"""
    return [constants.SIZE_LABELS.get(k, f"{k}d") for k in sorted(constants.SIZES_DAYS, reverse=True)]


def daily_table(project: str, versions: list[str], metric: str) -> pd.DataFrame:
    """全版の（版, 日）× 大きさ の指標表（seed 中央値・6 つの大きさがそろう日だけ）。"""
    sizes = size_labels_desc()
    frames = []
    for v in versions:
        try:
            mdf = result_writer.load_metrics(project, v)
        except FileNotFoundError:
            continue
        mdf = mdf[mdf["size"].isin(sizes)]
        per = mdf.groupby(["day", "size"])[metric].median().reset_index()
        piv = per.pivot(index="day", columns="size", values=metric).reindex(columns=sizes)
        piv.insert(0, "version", v)
        frames.append(piv)
    if not frames:
        return pd.DataFrame(columns=["version"] + sizes)
    out = pd.concat(frames).set_index("version", append=True).reorder_levels(["version", "day"])
    return out.dropna(subset=sizes)


def count_first_last(piv: pd.DataFrame, labels: list[str],
                     lower_is_better: bool = False) -> tuple[pd.Series, pd.Series, int]:
    """各大きさが 1 位／最下位になった日数（同着は全部に足す）と対象日数。lower_is_better なら最小を 1 位（2026-10-21）。"""
    n = len(piv)
    if n == 0:
        zero = pd.Series(0, index=labels, dtype=int)
        return zero, zero.copy(), 0
    vals = piv[labels]
    hi, lo = vals.max(axis=1), vals.min(axis=1)
    best, worst = (lo, hi) if lower_is_better else (hi, lo)
    first = vals.eq(best, axis=0).sum(axis=0).astype(int).reindex(labels)
    last = vals.eq(worst, axis=0).sum(axis=0).astype(int).reindex(labels)
    return first, last, n


def build_ranking(project: str, versions: list[str], metric: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """（表, {版または「全体」: 対象日数}）。表は index＝大きさ、列＝`<版>_1位 / _最下位 / _1位率 / _最下位率`（＋全体）。"""
    sizes = size_labels_desc()
    daily = daily_table(project, versions, metric)
    df = pd.DataFrame(index=sizes)
    ndays: dict[str, int] = {}
    has = isinstance(daily.index, pd.MultiIndex) and "version" in (daily.index.names or [])
    present = set(daily.index.get_level_values("version")) if has else set()
    empty = pd.DataFrame(columns=sizes)
    groups = [(v, daily.xs(v, level="version") if v in present else empty) for v in versions]
    groups.append(("全体", daily if has else empty))
    for label, piv in groups:
        first, last, n = count_first_last(piv, sizes, metric in constants.LOWER_IS_BETTER)
        ndays[label] = n
        df[f"{label}_1位"] = first
        df[f"{label}_最下位"] = last
        df[f"{label}_1位率"] = (100.0 * first / n).round(1) if n else np.nan
        df[f"{label}_最下位率"] = (100.0 * last / n).round(1) if n else np.nan
    return df, ndays
