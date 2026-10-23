"""要約表（design.md §5。lookback_window/visualization/table.py と同じまとめ方・同じ形）。

行＝大きさ（2M/1M/2w/1w/3d/1d）、列＝版、セル＝値 ± 標準偏差（＋小さく欠測量）、右端 Ave＝全版の全評価日をまとめた平均 ± 標準偏差。
- 値  ＝日ごとに seed 中央値 → 評価日で平均。
- std ＝日ごとに seed 中央値 → その日次系列の評価日方向の標準偏差（ddof=1）。
- 欠測＝その日が全 seed で NaN。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.preliminary_analysis.feature_space_window.io import result_writer  # noqa: E402
from src.analysis.preliminary_analysis.feature_space_window.utils import constants  # noqa: E402
from src.analysis.preliminary_analysis.feature_space_window.visualization._fonts import configure_japanese_font  # noqa: E402

configure_japanese_font()


def size_labels_desc(sizes=None) -> list[str]:
    sizes = constants.SIZES_DAYS if sizes is None else sizes
    return [constants.SIZE_LABELS.get(k, f"{k}d") for k in sorted(sizes, reverse=True)]


def cell_series(mdf: pd.DataFrame, metric: str, size_label: str):
    """(有効な日次系列[seed 中央値・NaN 除外], 欠測日数, 総日数)。"""
    sub = mdf[mdf["size"] == size_label]
    if sub.empty:
        return pd.Series(dtype=float), 0, 0
    per_day = sub.groupby("day")[metric].median()
    total = int(per_day.shape[0])
    valid = per_day.dropna()
    return valid, total - int(valid.shape[0]), total


def build_summary(metrics_by_version: dict, metric: str, sizes=None):
    """(val, std, miss, total)。index＝大きさ（降順）、columns＝版（＋'Ave'）。metrics_by_version は {版 -> metrics の DataFrame}。"""
    rows = size_labels_desc(sizes)
    versions = list(metrics_by_version)
    val = pd.DataFrame(index=rows, columns=versions, dtype=float)
    std = pd.DataFrame(index=rows, columns=versions, dtype=float)
    miss = pd.DataFrame(index=rows, columns=versions, dtype=float)
    total = pd.DataFrame(index=rows, columns=versions, dtype=float)
    all_days: dict[str, list[pd.Series]] = {r: [] for r in rows}
    for v, mdf in metrics_by_version.items():
        for r in rows:
            series, m, t = cell_series(mdf, metric, r)
            val.loc[r, v] = float(series.mean()) if len(series) else np.nan
            std.loc[r, v] = float(series.std(ddof=1)) if len(series) > 1 else np.nan
            miss.loc[r, v] = m
            total.loc[r, v] = t
            if len(series):
                all_days[r].append(series)
    for r in rows:
        if all_days[r]:
            combined = pd.concat(all_days[r])
            val.loc[r, "Ave"] = float(combined.mean())
            std.loc[r, "Ave"] = float(combined.std(ddof=1)) if len(combined) > 1 else np.nan
        else:
            val.loc[r, "Ave"] = np.nan
            std.loc[r, "Ave"] = np.nan
    return val, std, miss, total


def _fmt(v, s) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:.3f}" if pd.isna(s) else f"{v:.3f} ± {s:.3f}"


def render_table(val, std, miss, total, metric: str, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols, rows = list(val.columns), list(val.index)
    cell_text = []
    for r in rows:
        line = []
        for c in cols:
            txt = _fmt(val.loc[r, c], std.loc[r, c])
            if c != "Ave":
                m, t = miss.loc[r, c], total.loc[r, c]
                txt += "" if pd.isna(m) or pd.isna(t) or t == 0 else f"\n欠測 {int(m)}/{int(t)}"
            line.append(txt)
        cell_text.append(line)
    fig, ax = plt.subplots(figsize=(2.2 * len(cols) + 1, 0.9 * len(rows) + 1))
    ax.axis("off")
    # 2026-10-24（design.md §10.2）：図の中だけ f1_plugin を F1-score と書く
    ax.set_title(f"{constants.METRIC_DISPLAY.get(metric, metric)}：特徴空間で近い k 日 × バージョン（値 ± 標準偏差。右端 Ave＝全評価日の平均±標準偏差）", fontsize=11)
    tbl = ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=constants.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def build_and_save(project: str, versions: list[str], metric: str, sizes=None) -> Path:
    """保存済み metrics.csv から要約表を CSV（summary_<metric>.csv・summary_<metric>_missing.csv）と PNG で書く。"""
    mbv = {}
    for v in versions:
        try:
            mbv[v] = result_writer.load_metrics(project, v)
        except FileNotFoundError:
            continue
    out_dir = constants.OUTPUT_ROOT / project / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    val, std, miss, total = build_summary(mbv, metric, sizes)
    csv_df = pd.DataFrame(index=val.index)
    for c in val.columns:
        csv_df[c] = val[c]
        csv_df[f"{c}_std"] = std[c]
    csv_df.to_csv(out_dir / f"summary_{metric}.csv", index_label="size")
    miss.to_csv(out_dir / f"summary_{metric}_missing.csv", index_label="size")
    render_table(val, std, miss, total, metric, out_dir / f"summary_{metric}.png")
    return out_dir
