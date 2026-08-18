"""折れ線グラフ（補助・バージョンごと。design.md §11.2）。

素の折れ線：横=日, 縦=指標。p1〜p6 破線・月次集計テキストは入れない。
窓長ごとに線（小マーカー・欠測は飛ばして結ぶ）。凡例に欠測日数を控えめ併記。
汎用ヘッド（チューニングなし）基準線をデフォルトで重ねる。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.preliminary_analysis.lookback_window.io import result_io  # noqa: E402
from src.analysis.preliminary_analysis.lookback_window.utils import constants  # noqa: E402
from src.analysis.preliminary_analysis.lookback_window.visualization._fonts import configure_japanese_font  # noqa: E402

configure_japanese_font()


def _series(mdf: pd.DataFrame, metric: str, window_label: str):
    """window の (日付, 値) 系列（seed 中央値）と (欠測日数, 総日数)。"""
    sub = mdf[mdf["window"] == window_label]
    if sub.empty:
        return None, None, 0, 0
    per_day = sub.groupby("day")[metric].median().sort_index()
    days = pd.to_datetime(per_day.index)
    total = int(per_day.shape[0])
    miss = int(per_day.isna().sum())
    return days, per_day.values, miss, total


def plot_version(project: str, version: str, metric: str, out_path: Path,
                 windows=None, draw_general: bool | None = None) -> None:
    """1 バージョンの折れ線を描いて保存。"""
    windows = constants.WINDOWS_DAYS if windows is None else windows
    draw_general = constants.DRAW_GENERAL_DEFAULT if draw_general is None else draw_general
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mdf = result_io.load_metrics(project, version)

    fig, ax = plt.subplots(figsize=(11, 5))
    for w in windows:
        wl = constants.WINDOW_LABELS.get(w, f"{w}d")
        days, vals, miss, total = _series(mdf, metric, wl)
        if days is None:
            continue
        # 欠測(NaN)は飛ばして結ぶ：pandas の plot は NaN で自然に途切れるが、
        # ここでは有効点のみ結ぶため dropna して繋ぐ。マーカーは小さめ。
        s = pd.Series(vals, index=days).dropna()
        ax.plot(s.index, s.values, marker="o", markersize=2.5, linewidth=1.2,
                label=f"{wl} (欠測 {miss}/{total})")

    if draw_general:
        days, vals, miss, total = _series(mdf, metric, "general")
        if days is not None:
            s = pd.Series(vals, index=days).dropna()
            ax.plot(s.index, s.values, color="black", linestyle="--", linewidth=1.2,
                    marker="", label=f"general (欠測 {miss}/{total})")

    ax.set_title(f"{project} {version}：{metric}（窓長ごと・日次）")
    ax.set_xlabel("評価日")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=constants.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
