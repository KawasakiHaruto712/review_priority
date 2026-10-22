"""特徴量別の分布の図（design.md §5。寄与度の図 concept_drift_cause/visualization/plotter.py にそろえる）。

- 個別セル図：セル 1 つを 1 枚の横向きの左右の棒グラフ。棒＝δ の平均、セルごとに |δ の平均| の上位 TOP_N 件。
  色は向き（右＝評価日の方が大きい）、効果量の区分の境目に薄い縦線、横軸は −1〜1、エラーバーと数値（評価日の δ の標準偏差）。
- 格子一覧：個別セル図を、行（時期のずれ）を固定して評価時期を並べたもの／評価時期を固定して行を並べたもの。ばらつきは描かない。
- 題：1 行目「リポジトリ 版 行×評価時期 群」、2 行目「AUC・評価日数・1 日あたり正例 Change 数」（どの群にも）、
  3 行目は正例の群なら正例の割合と件数、負例の群なら負例の割合と件数（全体の群はなし）。
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.preliminary_analysis.data_drift.utils import constants  # noqa: E402
from src.analysis.preliminary_analysis.lookback_window.visualization._fonts import configure_japanese_font  # noqa: E402


def configure() -> None:
    configure_japanese_font()


def top_rows(cell: pd.DataFrame, top_n: int = constants.TOP_N) -> pd.DataFrame:
    """1 セル・1 群の行（特徴量ごと）から |δ の平均| の上位 top_n 件（同じ値は特徴量の並びの順）。"""
    c = cell.assign(_abs=cell["delta_mean"].abs().fillna(-1.0), _ord=range(len(cell)))
    return c.sort_values(["_abs", "_ord"], ascending=[False, True]).head(top_n)


def cell_info(class_counts: pd.DataFrame, row: int, position: int) -> dict:
    """そのセルの評価日数・1 日あたりの正例数・正例／負例の割合と件数（評価日の平均）。"""
    s = class_counts[(class_counts["row"] == row) & (class_counts["position"] == position)]
    if s.empty:
        return {}
    return {"n_days": int(s["day"].nunique()), "pos_per_day": float(s["n_pos_eval"].mean()),
            **{k: float(s[k].mean()) for k in ["pos_rate_eval", "pos_rate_train", "neg_rate_eval", "neg_rate_train",
                                               "n_pos_eval", "n_pos_train", "n_neg_eval", "n_neg_train"]}}


def acc_text(accs: dict | None) -> str:
    """精度の表記：recall@pos を先に、AUC を後に（design.md §5。2026-10-21 改めた：旧版は AUC だけ）。"""
    def f(key):
        v = (accs or {}).get(key)
        return "—" if v is None or not np.isfinite(v) else f"{v:.3f}"
    text = f"recall@pos={f('recall@pos')}  AUC={f('auc')}"
    if accs and "f1_plugin" in accs:            # 2026-10-22：重みあり版は F1（plug-in）も出す（design.md §10.1）
        text += f"  F1={f('f1_plugin')}"
    return text


def title(project, version, row_label, row, position, group, info: dict, accs: dict | None) -> str:
    """accs は {"recall@pos": 値, "auc": 値}（同じセルの精度表。なければ None）。"""
    head = f"{project} {version}  {row_label}{row} × 評価時期{position}  {constants.GROUP_LABEL.get(group, group)}"
    nd = info.get("n_days", np.nan)
    pp = info.get("pos_per_day", np.nan)
    line2 = f"{acc_text(accs)}  評価日数={nd:.0f}  1日あたり正例Change数={pp:.1f}" if info else acc_text(accs)
    lines = [head, line2]
    if group in ("pos", "neg") and info:
        lab = constants.GROUP_LABEL[group]
        lines.append(f"{lab}の割合 評価 {info[f'{group}_rate_eval']:.1%} / 学習 {info[f'{group}_rate_train']:.1%}  "
                     f"{lab}の件数 評価 {info[f'n_{group}_eval']:.1f} / 学習 {info[f'n_{group}_train']:.1f}")
    return "\n".join(lines)


def _draw(ax, cell: pd.DataFrame, *, show_err: bool, show_values: bool, label_fontsize: float = 8) -> None:
    top = top_rows(cell)
    names = top["feature"].tolist()
    y = np.arange(len(names))[::-1]
    mean = top["delta_mean"].to_numpy(dtype=float)
    std = np.nan_to_num(top["delta_std"].to_numpy(dtype=float), nan=0.0)
    colors = [constants.POS_COLOR if v >= 0 else constants.NEG_COLOR for v in np.nan_to_num(mean)]
    lim = constants.XLIM
    for b in constants.EFFECT_BOUNDS:                      # 効果量の区分の境目（薄い縦線）
        for s in (-1, 1):
            ax.axvline(s * b, color="#bbbbbb", linewidth=0.6, linestyle="--", zorder=0)
    ax.barh(y, np.nan_to_num(mean), color=colors, height=0.68, xerr=std if show_err else None,
            error_kw=dict(ecolor="#555555", elinewidth=0.8, capsize=2) if show_err else None)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, max(len(names), 1) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=label_fontsize)
    if show_values:
        for yi, m, s in zip(y, mean, std):
            ax.text(lim * 1.04, yi, f"{m:+.2f} (±{s:.2f})", va="center", ha="left", fontsize=7, color="#333333",
                    clip_on=False)


def plot_cell(cells: pd.DataFrame, project, version, row_label, row, position, group, info, accs,
              out_path: Path) -> bool:
    cell = cells[(cells["row"] == row) & (cells["position"] == position) & (cells["group"] == group)]
    if cell.empty:
        return False
    fig, ax = plt.subplots(figsize=(9.2 if constants.SHOW_VALUES else 7.2, 4.8))
    _draw(ax, cell, show_err=constants.SHOW_ERROR_BARS, show_values=constants.SHOW_VALUES)
    ax.set_title(title(project, version, row_label, row, position, group, info, accs), fontsize=9)
    ax.set_xlabel("平均 効果量 δ（左＝評価日の方が小さい / 右＝評価日の方が大きい）", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    if constants.SHOW_VALUES:
        fig.subplots_adjust(right=0.80)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=constants.PLOT_DPI)
    plt.close(fig)
    return True


def _shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return math.ceil(n / cols), cols


def plot_grid(cells: pd.DataFrame, class_counts: pd.DataFrame, aucs, project, version, row_label, group, *,
              fixed: str, fixed_value: int, panel_values: list, out_path: Path) -> bool:
    """fixed が "row" なら行を固定して評価時期を並べ、"position" なら評価時期を固定して行を並べる。"""
    if not panel_values:
        return False
    nr, nc = _shape(len(panel_values))
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.1 * nr), squeeze=False)
    for i, v in enumerate(panel_values):
        ax = axes[i // nc][i % nc]
        r, p = (fixed_value, v) if fixed == "row" else (v, fixed_value)
        cell = cells[(cells["row"] == r) & (cells["position"] == p) & (cells["group"] == group)]
        if cell.empty:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "データなし", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#888888")
        else:
            _draw(ax, cell, show_err=False, show_values=False, label_fontsize=5)
            ax.tick_params(axis="x", labelsize=6)
        info = cell_info(class_counts, r, p)
        accs = aucs(r, p) if aucs else None              # {"recall@pos": …, "auc": …}（2026-10-21）
        head = f"{'評価時期' if fixed == 'row' else row_label}{v}"
        if accs:
            head += f"  {acc_text(accs)}"
        scale = f"{info['n_days']}日 / 正例{info['pos_per_day']:.1f}件" if info else "—"
        ax.set_title(f"{head}\n{scale}", fontsize=7)
    for j in range(len(panel_values), nr * nc):
        axes[j // nc][j % nc].axis("off")
    lab = row_label if fixed == "row" else "評価時期"
    fig.suptitle(f"{project} {version}  {lab}{fixed_value} の一覧  特徴量別の分布 / {constants.GROUP_LABEL.get(group, group)}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=constants.PLOT_DPI)
    plt.close(fig)
    return True
