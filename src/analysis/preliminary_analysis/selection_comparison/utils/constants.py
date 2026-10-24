"""selection_comparison の設定値（学習に用いるチケットの選び方の比較。design.md）。

ほかの分析の出力ファイルを読むだけで、ほかの分析のコードは import しない（design.md §0）。
読み先・出力先は版の種類で切り替える（重みあり版は各フォルダの下の posw/）。
"""
from pathlib import Path

from src.config.path import DEFAULT_DATA_DIR

_BASE = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis"

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = _BASE / "selection_comparison"

# ── 読み先（重みなし版の場所。set_variant_roots で版の種類に合わせる） ─────────
BASE_DETECTION_ROOT = _BASE / "concept_drift_detection"     # 時間的な距離（time_distance）の精度表
BASE_FSW_ROOT = _BASE / "feature_space_window"               # 特徴空間上の距離（feature_distance）の精度表
BASE_DRIFT_ROOT = _BASE / "data_drift"                       # 特徴量別の分布
DETECTION_ROOT = BASE_DETECTION_ROOT
FSW_ROOT = BASE_FSW_ROOT
DRIFT_ROOT = BASE_DRIFT_ROOT


def set_variant_roots(variant_name: str) -> None:
    """読み先を版の種類に合わせる（重みあり版は各フォルダの下の posw/）。出力先は variant.apply で切り替える。"""
    from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant
    global DETECTION_ROOT, FSW_ROOT, DRIFT_ROOT
    DETECTION_ROOT = variant.subdir(BASE_DETECTION_ROOT, variant_name)
    FSW_ROOT = variant.subdir(BASE_FSW_ROOT, variant_name)
    DRIFT_ROOT = variant.subdir(BASE_DRIFT_ROOT, variant_name)


# ── 対象 ─────────────────────────────────────────────
# 論文の 6 件（精度表の PROJECTS と同じ並び。同じ差のときの順にも使う。design.md §4）
PROJECTS = ["qtbase", "qtcreator", "wikimedia_mediawiki_core", "qtdeclarative", "go", "typo3_cms"]
KINDS = ("time_distance", "feature_distance")              # 時間的な距離・特徴空間上の距離
KIND_LABEL = {"time_distance": "時間的な距離", "feature_distance": "特徴空間上の距離"}

METRIC = "f1_plugin"          # 比べる指標（F1 値。plug-in）
N_COLS = 3                    # 評価時期の列の数（重みあり版の 3 × 3）
PHASE_NAMES = ("初期", "中期", "後期")   # 表示の向き（p0 ＝ 初期）の列の名前
TOP_K = 3                     # 上回った側・下回った側それぞれ選ぶ列の数（design.md §4）
EFFECT_BOUND = 0.147          # 分布のずれが「無視できない」の境（|δ|。data_drift と同じ）
DRIFT_GROUP = "all"           # 分布は正例・負例をまとめた群で見る

ENCODING = "utf-8-sig"        # Excel で開いても文字化けしないように


def time_daily_path(project: str, version: str) -> Path:
    return DETECTION_ROOT / project / "set_transformer" / version / "daily_metrics.csv.gz"


def space_daily_path(project: str, version: str) -> Path:
    return FSW_ROOT / project / "feature_distance" / version / "daily_metrics.csv.gz"


def matrix_path(kind: str, project: str, version: str) -> Path:
    """リリースごとの精度表（F1 値）。"""
    root = (DETECTION_ROOT / project / "set_transformer" if kind == "time_distance"
            else FSW_ROOT / project / "feature_distance")
    return root / version / "probe" / METRIC / "drift_matrix.json"


def versions_path(project: str) -> Path:
    """版の一覧（古い順）を持つ run_meta.json（特徴空間上の距離の精度表のもの。時間的な距離と同じ版）。"""
    return FSW_ROOT / project / "feature_distance" / "run_meta.json"


def drift_dir(kind: str, project: str, version: str) -> Path:
    return DRIFT_ROOT / kind / project / version


def position_display(position: int) -> int:
    """保存物の評価時期（p0 ＝ リリース直前＝後期）を、表示の向き（p0 ＝ 初期）の番号にする。"""
    return N_COLS - 1 - int(position)


def position_text(position: int) -> str:
    """表示の向きの名前（例「p0（初期）」）。position は保存物の番号。"""
    q = position_display(position)
    return f"p{q}（{PHASE_NAMES[q]}）"
