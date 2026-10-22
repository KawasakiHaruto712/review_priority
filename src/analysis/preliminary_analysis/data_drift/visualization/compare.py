"""RQ1 と RQ2 を比べる折れ線（design.md §3.4・§5。2026-10-22）。

リポジトリ × 版ごとに 1 枚・左右 2 つのグラフ。左＝55 特徴量の |δ| の平均、右＝|δ| ≥ 0.147 の特徴量の数。
横軸＝ずれの段（RQ1 は時期のずれ d、RQ2 は特徴のずれ g）。各点はその段の、その版の全評価日の平均（評価時期 p はまとめる）。群は全体だけ。
どちらも保存済みの cells.csv から作る（計算し直さない）。
読み方：段 0 の高さ＝どちらが評価日の分布に近いか。線の傾き＝ずれの段が進むと分布もずれるか。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.preliminary_analysis.data_drift.stats.aggregate import summarize_rows  # noqa: E402
from src.analysis.preliminary_analysis.data_drift.utils import constants  # noqa: E402

GROUP = "all"                                        # 折れ線は全体の群だけ（正例・負例は数値のファイルに残す）
STYLES = {"time_distance": {"marker": "o", "color": "#4C72B0"},
          "feature_distance": {"marker": "^", "color": "#DD8452"}}


def compare_table(cells_by_kind: dict) -> pd.DataFrame:
    """{kind -> cells} → 段 × kind × 群ごとの全体のずれの大きさ（kind・row・group・abs_delta_mean・n_ge_bound・n_features・n_days）。"""
    parts = []
    for kind, cells in cells_by_kind.items():
        if cells is None or cells.empty:
            continue
        parts.append(summarize_rows(cells).assign(kind=kind))
    if not parts:
        return pd.DataFrame(columns=["kind", "row", "group", "abs_delta_mean", "n_ge_bound", "n_features", "n_days"])
    out = pd.concat(parts, ignore_index=True)
    return out[["kind", "row", "group", "abs_delta_mean", "n_ge_bound", "n_features", "n_days"]]


def plot_compare(table: pd.DataFrame, project: str, version: str, labels: dict, out_path: Path,
                 dpi: int = constants.PLOT_DPI) -> Path:
    """labels は {kind -> 凡例の文字}（学習の日数を含める）。"""
    d = table[table["group"] == GROUP]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [("abs_delta_mean", "55 特徴量の |δ| の平均"),
              ("n_ge_bound", f"|δ| ≥ {constants.EFFECT_BOUNDS[0]} の特徴量の数")]
    for ax, (col, ylabel) in zip(axes, panels):
        for kind, sub in d.groupby("kind", sort=False):
            sub = sub.sort_values("row")
            st = STYLES.get(kind, {})
            ax.plot(sub["row"], sub[col], marker=st.get("marker", "o"), color=st.get("color"), markersize=4,
                    linewidth=1.2, label=labels.get(kind, kind))
        ax.set_xlabel("ずれの段")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if col == "abs_delta_mean":
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(0, max(1, int(d["n_features"].max()) if len(d) else 1))
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{project} {version}　学習に使った日と評価日の、特徴量の分布のずれ（全体の群）", fontsize=11)
    fig.text(0.5, -0.02, "※ 段の意味が違う：RQ1 の d 段 ＝ 刻み × d 日前の窓、RQ2 の g 段 ＝ 特徴空間で近い順の g 番目の k 日。"
                         "各点はその段の全評価日の平均（評価時期はまとめる）",
             ha="center", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
