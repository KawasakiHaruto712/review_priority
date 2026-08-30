"""順位集計（1位/最下位カウント。design.md §11.3）。

要約表（§11.1）は「平均AUCでは1週間が良い」を示すが、平均だけでは"ブレ"が見えない。
本モジュールは日次で各窓を指標で順位づけし、**1位になった回数・最下位になった回数**を
数える。1位だけでなく最下位も併記することで、「平均が良い窓が本当に安定して良いのか
（それともまぐれ勝ちのノイズか）」を評価する。

- `(版, 日, 窓)` で seed 平均 → 全窓が揃う日だけで順位づけ（汎用ヘッド general は除外）。
- 各日の最高指標窓に「1位 +1」、最低指標窓に「最下位 +1」。**同着は両方加算**（率は100%超え可）。
- ビン集約はしない（日次のまま）。集約は step2 側に閉じ込める。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.lookback_window.io import result_io
from src.analysis.preliminary_analysis.lookback_window.utils import constants


def _window_labels_desc() -> list[str]:
    """窓長を大きい順（2M→1d）にラベル化（§11.1 と同じ並び）。"""
    return [constants.WINDOW_LABELS.get(w, f"{w}d") for w in sorted(constants.WINDOWS_DAYS, reverse=True)]


def _daily_table(project: str, versions: list[str], metric: str) -> pd.DataFrame:
    """全版の `(版, 日) × 窓` 指標表を作る（seed 平均・全窓が揃う日だけ）。

    返り値: index=(version, day), columns=窓ラベル（値=seed平均した指標）。
    汎用ヘッド general は含めない。
    """
    windows = _window_labels_desc()
    frames = []
    for v in versions:
        try:
            mdf = result_io.load_metrics(project, v)
        except FileNotFoundError:
            continue
        mdf = mdf[mdf["window"].isin(windows)]
        # (日, 窓) で seed 平均（NaN は除外して平均）
        per = mdf.groupby(["day", "window"])[metric].mean().reset_index()
        piv = per.pivot(index="day", columns="window", values=metric)
        piv = piv.reindex(columns=windows)
        piv.insert(0, "version", v)
        frames.append(piv)
    if not frames:
        return pd.DataFrame(columns=["version"] + windows)
    out = pd.concat(frames)
    out = out.set_index("version", append=True).reorder_levels(["version", "day"])
    # 全窓が揃う日だけ（公平な順位づけのため）
    return out.dropna(subset=windows)


def _count_first_last(piv: pd.DataFrame, windows: list[str]) -> tuple[pd.Series, pd.Series, int]:
    """各窓が「1位/最下位」になった日数を数える。同着は両方加算。

    piv: index=日（or (版,日)）, columns=窓。返り値: (1位カウント, 最下位カウント, 対象日数)。
    """
    first = pd.Series(0, index=windows, dtype=int)
    last = pd.Series(0, index=windows, dtype=int)
    n = len(piv)
    if n == 0:
        return first, last, 0
    vals = piv[windows]
    row_max = vals.max(axis=1)
    row_min = vals.min(axis=1)
    # 厳密一致で同着判定（同着は該当する全窓に加算）
    first = (vals.eq(row_max, axis=0)).sum(axis=0).astype(int).reindex(windows)
    last = (vals.eq(row_min, axis=0)).sum(axis=0).astype(int).reindex(windows)
    return first, last, n


def build_ranking(project: str, versions: list[str], metric: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """順位集計表を作る（§11.3）。

    返り値:
      - df: index=窓（2M→1d 降順）、列=`<版>_1位 / _最下位 / _1位率 / _最下位率`（＋「全体」）。
      - ndays: {版ラベル: 対象日数}（'全体' を含む）。率の分母。
    """
    windows = _window_labels_desc()
    daily = _daily_table(project, versions, metric)

    df = pd.DataFrame(index=windows)
    ndays: dict[str, int] = {}

    groups = [(v, daily.xs(v, level="version") if v in daily.index.get_level_values("version") else
               pd.DataFrame(columns=windows)) for v in versions]
    groups.append(("全体", daily))  # 全版プール

    for label, piv in groups:
        first, last, n = _count_first_last(piv, windows)
        ndays[label] = n
        df[f"{label}_1位"] = first
        df[f"{label}_最下位"] = last
        rate = (lambda s: (100.0 * s / n).round(1) if n else np.nan)
        df[f"{label}_1位率"] = rate(first)
        df[f"{label}_最下位率"] = rate(last)
    return df, ndays
