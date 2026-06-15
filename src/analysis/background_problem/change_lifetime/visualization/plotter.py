"""
生存期間の図の描画（§5.3）。
- plot_histogram  … 頻度ヒストグラム（対数ビン・対数軸）
- plot_boxplot    … リリース横断の箱ひげ図（1 リリース＝1 箱）
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI 不要のバックエンド（ファイル出力のみ）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

logger = logging.getLogger(__name__)

# 日本語フォントの自動選択（無ければ既定のまま）
_JP_FONT_CANDIDATES = [
    "Hiragino Sans", "Hiragino Kaku Gothic ProN", "YuGothic", "Yu Gothic",
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic", "AppleGothic",
]


def _configure_japanese_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    logger.warning("日本語フォントが見つかりません。ラベルが文字化けする可能性があります。")


_configure_japanese_font()


def _save_empty(out_path: Path, title: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.warning(f"データが無いため空の図を保存: {out_path}")


def plot_histogram(
    counts, edges, out_path: Path, title: str, xlabel: str, ylabel: str,
    x_log: bool = True, dpi: int = 150,
) -> None:
    """頻度ヒストグラムを描いて保存する。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(counts) == 0 or len(edges) == 0:
        _save_empty(out_path, title, dpi)
        return

    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(edges[:-1], counts, width=widths, align="edge",
           color="#4878a8", edgecolor="white", linewidth=0.3)
    if x_log:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"図を保存: {out_path}")


def plot_boxplot(
    values_by_release: dict, out_path: Path, title: str, ylabel: str,
    y_log: bool = True, dpi: int = 150,
) -> None:
    """リリース横断の箱ひげ図（1 リリース＝1 箱）を描いて保存する。

    values_by_release: {version: [lifetime, ...]}（挿入順 = リリース順）
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # データのあるリリースだけ箱にする
    versions = [v for v, vals in values_by_release.items() if vals]
    data = [np.asarray(values_by_release[v], dtype=float) for v in versions]
    if not data:
        _save_empty(out_path, title, dpi)
        return

    fig, ax = plt.subplots(figsize=(max(8, len(versions) * 1.1), 6))
    ax.boxplot(data, tick_labels=versions, showfliers=True,
               flierprops={"marker": ".", "markersize": 3, "alpha": 0.3})
    if y_log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("リリース")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"図を保存: {out_path}")
