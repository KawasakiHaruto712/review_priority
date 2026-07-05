"""図の描画（§5.10）。指標ごとに 2 枚。

png には検定の情報（有意マーク・p値・変化点の印）は一切描かない（検定は json のみ）。
図には指標名と軸の意味だけを書き、「良い向き」は記載しない。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 画面なしで保存
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from src.analysis.preliminary_analysis.concept_drift_existence.evaluation.drift_matrix import MatrixResult

logger = logging.getLogger(__name__)

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
            plt.rcParams["axes.unicode_minus"] = False
            return
    logger.warning("日本語フォントが見つかりません。ラベルが文字化けする可能性があります。")


_configure_japanese_font()

# 色は「良い/悪い」で統一する（指標名の“良い向き”は文字では書かない）。
# 良い=緑・悪い=赤。NDCG は大きいほど良い、MAE/RMSE は小さいほど良いので、
# 小さいほど良い指標は colormap を反転して「良い端の色」を全指標でそろえる。
# 大きいほど良い指標（NDCG・R²・F1・QWK）。残り（MAE/RMSE/誤差系）は小さいほど良い。
_HIGHER_IS_BETTER = {"ndcg", "r2_log", "macro_f1", "micro_f1", "qwk"}
_GOOD_BAD_CMAP = "RdYlGn"  # 0→赤(悪い) … 1→緑(良い)


def _cmap_for(metric: str) -> str:
    """良い=緑・悪い=赤に統一する colormap を返す。"""
    return _GOOD_BAD_CMAP if metric in _HIGHER_IS_BETTER else _GOOD_BAD_CMAP + "_r"


# 各指標の表示値域（ヒートマップのカラーバー範囲＝縦軸の範囲にも使う）。3 指標とも 0〜1（理論域いっぱい）。
# 参考: ランダム予測時の平均（理論基準）は MAE≈0.333、RMSE≈0.408、NDCG@10≈0.5（n依存）。
#   観測: MAE 0.197–0.332 / RMSE 0.252–0.403 / NDCG 0.519–0.829。
_AXIS_RANGE = {
    # 順位（0〜1 のサイズ非依存スケール）
    "mae": (0.0, 1.0), "rmse": (0.0, 1.0), "ndcg": (0.0, 1.0),
    # 回帰誤差（log10 時間。0=完全一致、1=10倍ズレ、2=100倍ズレ。上端は表示用の目安）
    "mae_log": (0.0, 1.5), "rmse_log": (0.0, 1.5), "r2_log": (0.0, 1.0),
    # 分類（0〜1）
    "macro_f1": (0.0, 1.0), "micro_f1": (0.0, 1.0), "qwk": (0.0, 1.0),
}
_DEFAULT_RANGE = (0.0, 1.0)


def _range_for(metric: str) -> tuple[float, float]:
    return _AXIS_RANGE.get(metric, _DEFAULT_RANGE)


def plot_heatmap(result: MatrixResult, path: Path, dpi: int = 150) -> None:
    """正方形ヒートマップ（縦=距離 d/滞留期間、横=位置 p、色=指標値）。

    色は良い/悪いで統一（緑=良い・赤=悪い）。良い向きの“文字”は付けない。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = result.bin_count
    fig, ax = plt.subplots(figsize=(6, 5))
    # カラーバーの値域は指標ごとに固定（ランダム基準に基づく。_AXIS_RANGE 参照）。
    # バージョン間で色が比較可能になる。
    vmin, vmax = _range_for(result.metric)
    im = ax.imshow(result.value, origin="lower", aspect="auto",
                   cmap=_cmap_for(result.metric), vmin=vmin, vmax=vmax)
    # 各セルに値を数字で表示（NaN は空欄）。実際のセル色の明るさ（輝度）で文字色を黒/白に切り替える。
    for d in range(n):
        for p in range(n):
            v = result.value[d, p]
            if np.isnan(v):
                continue
            r, g, b, _ = im.cmap(im.norm(v))  # 表示色に一致
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(p, d, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if lum > 0.5 else "white")
    ax.set_xlabel("位置 p（リリース内のどの時点を予測するか）")
    ax.set_ylabel("距離 d（滞留期間, ビン単位）")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"p{p}" for p in range(1, n + 1)])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"d{d}" for d in range(1, n + 1)])
    ax.set_title(f"{result.metric}（学習×予測 行列）")
    fig.colorbar(im, ax=ax, label=result.metric)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_position_lines(result: MatrixResult, path: Path, dpi: int = 150) -> None:
    """距離固定の位置別折れ線（距離 d ごとに 1 本、横=位置 p、縦=指標値）。生の値のみ。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = result.bin_count
    fig, ax = plt.subplots(figsize=(6, 5))
    positions = np.arange(n)
    for d in range(1, n + 1):
        row = result.value[d - 1]
        if np.isnan(row).all():
            continue
        ax.plot(positions, row, marker="o", label=f"d={d}")
    ax.set_xlabel("位置 p（リリース内のどの時点を予測するか）")
    ax.set_ylabel(result.metric)
    ax.set_ylim(*_range_for(result.metric))  # 指標ごとに値域固定（_AXIS_RANGE。バージョン間で縦軸を比較可能に）
    ax.set_xticks(positions); ax.set_xticklabels([f"p{p}" for p in range(1, n + 1)])
    ax.set_title(f"{result.metric}（距離固定の位置別）")
    ax.legend(title="距離 d", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ── 相対評価版（行列内で「最良セル=1」に正規化した別図） ───────────────
# 従来図（実データ・固定スケール）とは別に追加で出す。行列内コントラスト重視。
# 注意: 各行列を自分の最良で正規化するためバージョン間比較は不可。差はノイズを含むので過度に解釈しない。
def _goodness(metric: str, value: np.ndarray) -> np.ndarray:
    """「高いほど良い」に揃えた値（MAE/RMSE は 1-値、NDCG はそのまま）。"""
    return value if metric in _HIGHER_IS_BETTER else 1.0 - value


def _relative_matrix(result: MatrixResult) -> np.ndarray:
    """行列内の最良セル=1 とした相対値（高いほど良い）。"""
    g = _goodness(result.metric, result.value)
    ref = np.nanmax(g)
    return g / ref if (ref and not np.isnan(ref)) else g


def plot_heatmap_relative(result: MatrixResult, path: Path, dpi: int = 150) -> None:
    """相対評価ヒートマップ（行列内 最良セル=1、緑=良い）。従来図とは別ファイル。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = result.bin_count
    rel = _relative_matrix(result)
    # 凡例（カラーバー）の下端は 0 固定（最良セル=1）。0 からの位置を正しく表す。
    vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rel, origin="lower", aspect="auto", cmap=_GOOD_BAD_CMAP, vmin=vmin, vmax=vmax)
    for d in range(n):
        for p in range(n):
            v = rel[d, p]
            if np.isnan(v):
                continue
            r, g, b, _ = im.cmap(im.norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(p, d, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if lum > 0.5 else "white")
    ax.set_xlabel("位置 p（リリース内のどの時点を予測するか）")
    ax.set_ylabel("距離 d（滞留期間, ビン単位）")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"p{p}" for p in range(1, n + 1)])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"d{d}" for d in range(1, n + 1)])
    ax.set_title(f"{result.metric}（相対: 最良セル=1）")
    fig.colorbar(im, ax=ax, label=f"{result.metric} 相対（最良=1, 高いほど良い）")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_position_lines_relative(result: MatrixResult, path: Path, dpi: int = 150) -> None:
    """相対評価の位置別折れ線（行列内 最良セル=1）。従来図とは別ファイル。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = result.bin_count
    rel = _relative_matrix(result)
    fig, ax = plt.subplots(figsize=(6, 5))
    positions = np.arange(n)
    for d in range(1, n + 1):
        row = rel[d - 1]
        if np.isnan(row).all():
            continue
        ax.plot(positions, row, marker="o", label=f"d={d}")
    ax.set_xlabel("位置 p（リリース内のどの時点を予測するか）")
    ax.set_ylabel(f"{result.metric} 相対（最良=1）")
    ax.set_ylim(0.0, 1.0)  # 下端は 0 固定（最良=1）
    ax.set_xticks(positions); ax.set_xticklabels([f"p{p}" for p in range(1, n + 1)])
    ax.set_title(f"{result.metric}（相対: 距離固定の位置別）")
    ax.legend(title="距離 d", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_all(result: MatrixResult, out_dir: Path, dpi: int = 150) -> None:
    out_dir = Path(out_dir)
    # 従来図（実データ・固定スケール）
    plot_heatmap(result, out_dir / "drift_matrix.png", dpi)
    plot_position_lines(result, out_dir / "position_by_distance.png", dpi)
    # 相対評価版（行列内 最良セル=1。別ファイルで追加）
    plot_heatmap_relative(result, out_dir / "drift_matrix_relative.png", dpi)
    plot_position_lines_relative(result, out_dir / "position_by_distance_relative.png", dpi)
