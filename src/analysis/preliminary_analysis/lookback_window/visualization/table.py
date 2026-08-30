"""要約表（主成果物。design.md §11.1）。

行＝窓長（2M/1M/2w/1w/3d/1d）、列＝バージョン、セル＝指標値 ± 標準偏差（＋小さく欠測量）、
右端 Ave＝全リリースの全評価日をまとめた 平均 ± 標準偏差。
metrics.csv から生成（再実行不要）。
- 値  ＝日ごとに seed 中央値 → 評価日で平均。
- std ＝日ごとに seed 中央値 → その日次系列の評価日方向の標準偏差（ddof=1）＝期間内のブレ。
- Ave ＝版を等重みにせず、全リリースの全評価日を等重みでまとめた 平均・標準偏差。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.preliminary_analysis.lookback_window.io import result_io  # noqa: E402
from src.analysis.preliminary_analysis.lookback_window.utils import constants  # noqa: E402
from src.analysis.preliminary_analysis.lookback_window.visualization._fonts import configure_japanese_font  # noqa: E402

configure_japanese_font()


def _window_labels_desc() -> list[str]:
    """窓長を大きい順（2M→1d）にラベル化。"""
    return [constants.WINDOW_LABELS.get(w, f"{w}d") for w in sorted(constants.WINDOWS_DAYS, reverse=True)]


def _cell_series(mdf: pd.DataFrame, metric: str, window_label: str):
    """(有効な日次系列[seed中央値・NaN除外], 欠測日数, 総日数)。欠測＝その日が全 seed で NaN。"""
    sub = mdf[mdf["window"] == window_label]
    if sub.empty:
        return pd.Series(dtype=float), 0, 0
    per_day = sub.groupby("day")[metric].median()
    total = int(per_day.shape[0])
    valid = per_day.dropna()
    miss = total - int(valid.shape[0])
    return valid, miss, total


def build_summary(project: str, versions: list[str], metric: str):
    """(val_df, std_df, miss_df, total_df) を返す。index=窓長(desc), columns=versions(+'Ave')。

    - 各セル：値＝日次系列（seed中央値）の平均、std＝その系列の標準偏差（ddof=1）。
    - Ave：版を等重みにせず、全リリースの全評価日をまとめた 平均・標準偏差。
    """
    rows = _window_labels_desc()
    val = pd.DataFrame(index=rows, columns=versions, dtype=float)
    std = pd.DataFrame(index=rows, columns=versions, dtype=float)
    miss = pd.DataFrame(index=rows, columns=versions, dtype=float)
    total = pd.DataFrame(index=rows, columns=versions, dtype=float)
    all_days: dict[str, list[pd.Series]] = {wl: [] for wl in rows}  # 窓ごとに全版の日次系列を集める（Ave用）

    for v in versions:
        try:
            mdf = result_io.load_metrics(project, v)
        except FileNotFoundError:
            continue
        for wl in rows:
            series, miss_, total_ = _cell_series(mdf, metric, wl)
            val.loc[wl, v] = float(series.mean()) if len(series) else np.nan
            std.loc[wl, v] = float(series.std(ddof=1)) if len(series) > 1 else np.nan
            miss.loc[wl, v] = miss_
            total.loc[wl, v] = total_
            if len(series):
                all_days[wl].append(series)

    # Ave＝全リリースの全評価日をまとめた 平均・標準偏差（全日を等重み）
    for wl in rows:
        if all_days[wl]:
            combined = pd.concat(all_days[wl])
            val.loc[wl, "Ave"] = float(combined.mean())
            std.loc[wl, "Ave"] = float(combined.std(ddof=1)) if len(combined) > 1 else np.nan
        else:
            val.loc[wl, "Ave"] = np.nan
            std.loc[wl, "Ave"] = np.nan
    return val, std, miss, total


def _fmt(v, s) -> str:
    """`値 ± 標準偏差`（値が NaN なら '-'、std が NaN なら値のみ）。"""
    if pd.isna(v):
        return "-"
    return f"{v:.3f}" if pd.isna(s) else f"{v:.3f} ± {s:.3f}"


def render_table(val: pd.DataFrame, std: pd.DataFrame, miss: pd.DataFrame, total: pd.DataFrame,
                 metric: str, out_path: Path) -> None:
    """要約表を PNG で描く（セル：値 ± 標準偏差＋小さく欠測量）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(val.columns)                 # versions + 'Ave'
    rows = list(val.index)
    cell_text = []
    for wl in rows:
        line = []
        for c in cols:
            vtxt = _fmt(val.loc[wl, c], std.loc[wl, c])
            if c == "Ave":
                line.append(vtxt)            # Ave は欠測併記なし
            else:
                m = miss.loc[wl, c]
                t = total.loc[wl, c]
                sub = "" if pd.isna(m) or pd.isna(t) or t == 0 else f"\n欠測 {int(m)}/{int(t)}"
                line.append(vtxt + sub)
        cell_text.append(line)

    fig, ax = plt.subplots(figsize=(2.2 * len(cols) + 1, 0.9 * len(rows) + 1))
    ax.axis("off")
    ax.set_title(f"{metric}：窓長 × バージョン（値 ± 標準偏差。右端 Ave＝全評価日の平均±標準偏差）", fontsize=11)
    tbl = ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols,
                   cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=constants.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def build_and_save(project: str, versions: list[str], metric: str, out_dir: Path) -> None:
    """要約表を作って CSV＋PNG で保存。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val, std, miss, total = build_summary(project, versions, metric)
    # CSV：各版の値の隣に std 列を置く（window, 26.0.0, 26.0.0_std, …, Ave, Ave_std）
    csv_df = pd.DataFrame(index=val.index)
    for c in val.columns:                    # versions + 'Ave'
        csv_df[c] = val[c]
        csv_df[f"{c}_std"] = std[c]
    csv_df.to_csv(out_dir / f"summary_{metric}.csv", index_label="window")
    miss.to_csv(out_dir / f"summary_{metric}_missing.csv", index_label="window")
    render_table(val, std, miss, total, metric, out_dir / f"summary_{metric}.png")
