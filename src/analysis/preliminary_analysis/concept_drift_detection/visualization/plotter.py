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

from src.analysis.preliminary_analysis.concept_drift_detection.evaluation.drift_matrix import MatrixResult
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants

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
# 大きいほど良い指標（NDCG・R²・F1・QWK、および二値分類指標）。残り（MAE/RMSE/誤差系）は小さいほど良い。
_HIGHER_IS_BETTER = {"ndcg", "r2_log", "macro_f1", "micro_f1", "qwk",
                     "mcc", "f1", "precision", "recall", "accuracy", "auc", "ap"}
# 符号付き（バイアス）指標: 0 が中立、± どちらに振れても偏り。良い/悪いの一方向ではないので
# 発散カラーマップ（0=白, +=赤, −=青）で「向きと大きさ」を表す。
# 差分行列（"<指標>_diff" = probe − 汎用ヘッド）も符号付き（+=特化が効く/−=悪化。§8.3）。
_SIGNED_METRICS = {"me_log"}
_GOOD_BAD_CMAP = "RdYlGn"  # 0→赤(悪い) … 1→緑(良い)
_SIGNED_CMAP = "coolwarm"  # −→青, 0→白, +→赤


def _is_signed(metric: str) -> bool:
    """符号付き指標（0中心・±両方向）か。差分行列 "*_diff" を含む。"""
    return metric in _SIGNED_METRICS or metric.endswith("_diff")


def _higher_is_better(metric: str) -> bool:
    """大きいほど良い指標か。norm_rank（正規化順位）は小さいほど良い。それ以外（auc/map/mrr/
    precision@k/recall@k/f1@k/precision/recall/f1）は大きいほど良い。"""
    m = metric[:-5] if metric.endswith("_diff") else metric
    return not m.startswith("norm_rank")


def _cmap_for(metric: str) -> str:
    """colormap を返す。符号付きは発散、良い=緑/悪い=赤（小さいほど良いは反転）。"""
    if _is_signed(metric):
        return _SIGNED_CMAP
    return _GOOD_BAD_CMAP if _higher_is_better(metric) else _GOOD_BAD_CMAP + "_r"


# 各指標の表示値域（ヒートマップのカラーバー範囲＝縦軸の範囲にも使う）。3 指標とも 0〜1（理論域いっぱい）。
# 参考: ランダム予測時の平均（理論基準）は MAE≈0.333、RMSE≈0.408、NDCG@10≈0.5（n依存）。
#   観測: MAE 0.197–0.332 / RMSE 0.252–0.403 / NDCG 0.519–0.829。
_AXIS_RANGE = {
    # 順位（0〜1 のサイズ非依存スケール）
    "mae": (0.0, 1.0), "rmse": (0.0, 1.0), "ndcg": (0.0, 1.0),
    # 回帰誤差（log10 時間。0=完全一致、1=10倍ズレ、2=100倍ズレ。上端は表示用の目安）
    "mae_log": (0.0, 1.5), "rmse_log": (0.0, 1.5), "r2_log": (0.0, 1.0),
    # 符号付きバイアス（log10 時間, 0 中心の対称。+=予測が長め / −=短め）
    "me_log": (-1.0, 1.0),
    # 分類（0〜1）
    "macro_f1": (0.0, 1.0), "micro_f1": (0.0, 1.0), "qwk": (0.0, 1.0),
    # 二値分類。mcc は本来 -1〜1 だが表示は 0〜1（負＝偶然以下は下端色）
    "mcc": (0.0, 1.0), "f1": (0.0, 1.0), "precision": (0.0, 1.0),
    "recall": (0.0, 1.0), "accuracy": (0.0, 1.0),
    # しきい値フリー指標（主指標 AUC、補助 AP）
    "auc": (0.0, 1.0), "ap": (0.0, 1.0),
}
_DEFAULT_RANGE = (0.0, 1.0)
_DIFF_RANGE = (-0.3, 0.3)  # 差分行列（probe − 汎用）の表示域（0中心の対称）


def _range_for(metric: str) -> tuple[float, float]:
    if metric.endswith("_diff"):
        return _DIFF_RANGE
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
    # 各セルの数値表示は既定オフ（26×26 では詰まって読みにくい）。constants で切替可。
    if constants.HEATMAP_SHOW_CELL_VALUES:
        for d in range(n):
            for p in range(n):
                v = result.value[d, p]
                if np.isnan(v):
                    continue
                r, g, b, _ = im.cmap(im.norm(v))  # 表示色に一致
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(p, d, constants.HEATMAP_VALUE_FMT.format(v), ha="center", va="center",
                        fontsize=6, color="black" if lum > 0.5 else "white")
    ax.set_xlabel("位置 p（リリース内のどの時点を予測するか）")
    ax.set_ylabel("距離 d（滞留期間, ビン単位）")
    _step = max(1, n // 13)  # ラベルが詰まらないよう間引く（26 なら 2 つおき）
    _ticks = list(range(0, n, _step))
    ax.set_xticks(_ticks); ax.set_xticklabels([f"p{t + 1}" for t in _ticks], fontsize=7)
    ax.set_yticks(_ticks); ax.set_yticklabels([f"d{t + 1}" for t in _ticks], fontsize=7)
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
    """「高いほど良い」に揃えた値（小さいほど良い指標＝norm_rank は 1-値）。"""
    return value if _higher_is_better(metric) else 1.0 - value


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
    if constants.HEATMAP_SHOW_CELL_VALUES:
        for d in range(n):
            for p in range(n):
                v = rel[d, p]
                if np.isnan(v):
                    continue
                r, g, b, _ = im.cmap(im.norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(p, d, constants.HEATMAP_VALUE_FMT.format(v), ha="center", va="center",
                        fontsize=6, color="black" if lum > 0.5 else "white")
    ax.set_xlabel("位置 p（リリース内のどの時点を予測するか）")
    ax.set_ylabel("距離 d（滞留期間, ビン単位）")
    _step = max(1, n // 13)  # ラベルが詰まらないよう間引く（26 なら 2 つおき）
    _ticks = list(range(0, n, _step))
    ax.set_xticks(_ticks); ax.set_xticklabels([f"p{t + 1}" for t in _ticks], fontsize=7)
    ax.set_yticks(_ticks); ax.set_yticklabels([f"d{t + 1}" for t in _ticks], fontsize=7)
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
    """26×26 ヒートマップを出力（色のみ既定）。距離26本の折れ線はスパゲッティになるため作らない。"""
    out_dir = Path(out_dir)
    plot_heatmap(result, out_dir / "drift_matrix.png", dpi)
    # 相対評価版（行列内 最良セル=1）。符号付き・差分指標は「最良=1」正規化が無意味なので作らない。
    if not _is_signed(result.metric):
        plot_heatmap_relative(result, out_dir / "drift_matrix_relative.png", dpi)
