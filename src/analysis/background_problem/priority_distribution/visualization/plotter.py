"""
分布図の描画（§5.7）。2 種類の図を出力する:
- plot_band        … 中心線 + 平均±(1/2)std バンド（distribution_mean_std.png）
- plot_percentiles … 分位点（10/30/50/70/90 など）の折れ線（distribution_percentiles.png）

横軸＝リリースサイクル内の相対位置、縦軸＝所要時間（必要なら対数軸）。
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

# 対数軸で lower<=0 を避けるための下限クリップ値
_LOG_FLOOR = 1e-3

# 日本語ラベルが文字化けしないよう、利用可能な日本語フォントを選ぶ（無ければ既定のまま）
_JP_FONT_CANDIDATES = [
    "Hiragino Sans", "Hiragino Kaku Gothic ProN", "YuGothic", "Yu Gothic",
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic", "AppleGothic",
]


def _configure_japanese_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策
            return
    logger.warning("日本語フォントが見つかりません。ラベルが文字化けする可能性があります。")


_configure_japanese_font()


def _new_axes(title: str, xlabel: str, ylabel: str, y_log: bool):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def _save_empty(out_path: Path, title: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.warning(f"データが無いため空の図を保存: {out_path}")


def plot_band(
    points: list,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    y_log: bool = True,
    dpi: int = 150,
) -> None:
    """中心線 + 平均±(1/2)std バンドの図を描いて保存する。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not points:
        _save_empty(out_path, title, dpi)
        return

    x = np.array([p.x for p in points])
    mean = np.array([p.mean for p in points])
    lower = np.array([p.lower for p in points])
    upper = np.array([p.upper for p in points])

    if y_log:
        # 対数軸では 0 以下を描けないので微小正値にクリップ
        mean = np.clip(mean, _LOG_FLOOR, None)
        lower = np.clip(lower, _LOG_FLOOR, None)
        upper = np.clip(upper, _LOG_FLOOR, None)

    fig, ax = _new_axes(title, xlabel, ylabel, y_log)
    ax.fill_between(x, lower, upper, alpha=0.3, color="#4878a8", linewidth=0)
    ax.plot(x, mean, color="#1f4e79", linewidth=1.0)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"図を保存: {out_path}")


def plot_percentiles(
    points: list,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    y_log: bool = True,
    dpi: int = 150,
) -> None:
    """分位点ごとの折れ線（10/30/50/70/90 など）を描いて保存する。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not points:
        _save_empty(out_path, title, dpi)
        return

    x = np.array([p.x for p in points])
    # 分位点キー（昇順）を最初の点から取得
    pct_keys = sorted(next((p.percentiles for p in points if p.percentiles), {}).keys())
    if not pct_keys:
        _save_empty(out_path, title, dpi)
        return

    fig, ax = _new_axes(title, xlabel, ylabel, y_log)
    cmap = plt.get_cmap("viridis")
    n = len(pct_keys)
    for i, q in enumerate(pct_keys):
        y = np.array([p.percentiles.get(q, np.nan) for p in points], dtype=float)
        if y_log:
            y = np.clip(y, _LOG_FLOOR, None)
        color = cmap(i / max(n - 1, 1))
        label = f"{q}%ile" + ("（中央値）" if q == 50 else "")
        ax.plot(x, y, color=color, linewidth=1.0, label=label)

    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"図を保存: {out_path}")
