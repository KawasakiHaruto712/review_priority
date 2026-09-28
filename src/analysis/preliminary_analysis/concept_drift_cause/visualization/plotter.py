"""寄与度プロファイルの作図（design.md §9.4）。

2 種類をペアで出す。
  個別セル図   (距離, 位置) の組 1 つを 1 枚の横向き発散棒グラフにする
  格子一覧     その個別セル図を距離の行ごと・位置の列ごとに格子状に並べる（集約はしない）

棒は中央 0 の左右振り分けで、15 次元すべてを `FEATURE_NAMES` の定義順に並べる。
軸はプロジェクト × 系列ごとに共通（§9.4.5）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # 画面を持たない環境で描くため
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_cause.utils import constants
from src.analysis.preliminary_analysis.lookback_window.visualization._fonts import configure_japanese_font
from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_builder import FEATURE_NAMES

logger = logging.getLogger(__name__)

POS_COLOR = "#4C72B0"    # 正（右）
NEG_COLOR = "#C44E52"    # 負（左）
SERIES_LABEL = {"diff": "差分（probe − 汎用）", "probe": "probe", "general": "汎用ヘッド"}
GROUP_LABEL = {"pos": "正例", "neg": "負例"}


def _row(stats: pd.DataFrame, distance, position) -> pd.Series | None:
    """そのセルの行を取り出す（無ければ None）。"""
    s = stats[(stats["distance"].astype(str) == str(distance)) & (stats["position"] == position)]
    return None if s.empty else s.iloc[0]


def _draw_bars(ax, row: pd.Series, lim: float, *, show_labels: bool,
               show_error_bars: bool, show_values: bool) -> None:
    """1 つのセルの棒を ax に描く。row が None のときは呼ばない。"""
    y = np.arange(len(FEATURE_NAMES))[::-1]          # 上から定義順に並べる
    mean = np.array([row[f] for f in FEATURE_NAMES], dtype=float)
    std = np.array([row.get(f + "_std", np.nan) for f in FEATURE_NAMES], dtype=float)
    std = np.nan_to_num(std, nan=0.0)
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in mean]

    ax.barh(y, mean, color=colors, height=0.68,
            xerr=std if show_error_bars else None,
            error_kw=dict(ecolor="#555555", elinewidth=0.8, capsize=2) if show_error_bars else None)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(FEATURE_NAMES) - 0.3)

    if show_labels:
        ax.set_yticks(y)
        ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    else:
        ax.set_yticks([])

    # 軸をはみ出した棒は端で切れるので、その先に続いていることを矢印で示す（§9.4.5）。
    # 端は「|平均| ＋ 標準偏差」の 95 パーセンタイルなので、平均が枠外に出る棒が必ず一定数ある。
    # 棒と同じ色で棒の上に置くと隠れるので、**枠の外側**に出す
    for yi, m, c in zip(y, mean, colors):
        if abs(m) > lim:
            ax.plot(np.sign(m) * lim * 1.03, yi, marker=">" if m > 0 else "<",
                    color=c, markersize=7, clip_on=False, linestyle="none",
                    markeredgecolor="white", markeredgewidth=0.5)

    if show_values:
        # **常に右端の外側**に揃える。棒の向きに合わせて左右に振ると目が行き来して読みにくい。
        for yi, m, s in zip(y, mean, std):
            ax.text(lim * 1.09, yi, f"{m:+.2f} (±{s:.2f})",   # はみ出し矢印の分だけ外にずらす
                    va="center", ha="left", fontsize=7, color="#333333", clip_on=False)


def _empty_panel(ax, note: str = "データなし") -> None:
    """空のセル（評価できる日が 1 日も無い）。枠は残して中身だけ「データなし」にする。"""
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#888888")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")


def _title(project, version, series, group, distance=None, position=None,
           info: pd.Series | None = None) -> str:
    cell = []
    if distance is not None:
        cell.append(f"距離{distance}")
    if position is not None:
        cell.append(f"位置{position}")
    head = (f"{project} {version}  {' × '.join(cell)}  "
            f"{SERIES_LABEL.get(series, series)} / {GROUP_LABEL.get(group, group)}")
    if info is None:
        return head
    auc = info.get("auc", np.nan)
    nd = info.get("n_days", np.nan)
    nc = info.get("n_changes", np.nan)
    label = "正例" if group == "pos" else "負例"
    return (f"{head}\nAUC={auc:.3f}  評価日数={nd:.0f}  "
            f"1日あたり{label}Change数={nc:.1f}")


def plot_cell(stats, info, project, version, series, group, distance, position,
              lim: float, out_path: Path) -> bool:
    """個別セル図を 1 枚描く（§9.4.2）。描けたら True。"""
    row = _row(stats, distance, position)
    if row is None:
        return False
    irow = _row(info, distance, position)
    # 数値を出すときは右側に専用の余白を作る（棒と重ならないよう軸の外に置くため）
    width = 9.2 if constants.PLOT_SHOW_VALUES else 7.2
    fig, ax = plt.subplots(figsize=(width, 4.6))
    _draw_bars(ax, row, lim, show_labels=True,
               show_error_bars=constants.PLOT_SHOW_ERROR_BARS,
               show_values=constants.PLOT_SHOW_VALUES)
    merged = row if irow is None else pd.concat([row, irow.drop(labels=["distance", "position"],
                                                               errors="ignore")])
    ax.set_title(_title(project, version, series, group, distance, position, merged), fontsize=9)
    ax.set_xlabel("平均 IG（左＝優先度を下げる / 右＝上げる）", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    if constants.PLOT_SHOW_VALUES:
        fig.subplots_adjust(right=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=constants.PLOT_DPI)
    plt.close(fig)
    return True


def plot_grid(stats, info, project, version, series, group, *, fixed: str, fixed_value,
              panel_values: list, lim: float, out_path: Path) -> bool:
    """格子一覧を 1 枚描く（§9.4.6）。

    fixed が "distance" なら距離を固定して位置を並べ、"position" なら逆。
    中身は個別セル図と同じで、**集約はしない**。ばらつきは描かない（§9.4.4）。
    """
    n = len(panel_values)
    if n == 0:
        return False
    rows, cols = constants.grid_shape(n)
    # 見出しが 2 行になるぶん、パネルの高さを少し広げる
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows), squeeze=False)

    for i, v in enumerate(panel_values):
        ax = axes[i // cols][i % cols]
        d, p = (fixed_value, v) if fixed == "distance" else (v, fixed_value)
        row = _row(stats, d, p)
        # 特徴量名は各行の左端パネルにのみ（§9.4.6）
        if row is None:
            _empty_panel(ax)
        else:
            _draw_bars(ax, row, lim, show_labels=(i % cols == 0),
                       show_error_bars=False, show_values=False)
            ax.tick_params(axis="x", labelsize=6)
        # パネル見出しは 2 行（§9.4.7）。セルの規模が不揃いなので、AUC だけだと
        # 「1 日・正例 1 件のセル」と「7 日・14 件のセル」を同列に比べてしまう（glance で 14 倍の開き）。
        irow = _row(info, d, p)
        head = f"{'位置' if fixed == 'distance' else '距離'}{v}"
        if irow is not None and np.isfinite(irow.get("auc", np.nan)):
            head += f"  AUC={irow['auc']:.3f}"
        label = "正例" if group == "pos" else "負例"
        nd = row["n_days"] if row is not None and "n_days" in row else np.nan
        nc = irow.get("n_changes", np.nan) if irow is not None else np.nan
        scale = (f"{nd:.0f}日 / {label}{nc:.1f}件"
                 if np.isfinite(nd) and np.isfinite(nc) else "—")
        ax.set_title(f"{head}\n{scale}", fontsize=7)

    # 余った枠は消す（枠を確保するのは対象セルだけ）
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    label = "距離" if fixed == "distance" else "位置"
    fig.suptitle(f"{project} {version}  {label}{fixed_value} の一覧  "
                 f"{SERIES_LABEL.get(series, series)} / {GROUP_LABEL.get(group, group)}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=constants.PLOT_DPI)
    plt.close(fig)
    return True


def configure() -> None:
    """日本語フォントを設定する（豆腐対策）。"""
    configure_japanese_font()
