"""release_best_size の設定値（リリースごとに適した日数 N。design.md）。

ほかの分析の出力ファイルを読むだけで、ほかの分析のコードは import しない（design.md §0）。
読み先・出力先は版の種類で切り替える（重みあり版は各フォルダの下の posw/）。
"""
from pathlib import Path

from src.config.path import DEFAULT_DATA_DIR

_BASE = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis"

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = _BASE / "release_best_size"

# ── 読み先（重みなし版の場所。set_variant_roots で版の種類に合わせる） ─────────
BASE_LOOKBACK_ROOT = _BASE / "lookback_window"               # 時間による選び方の N 別の結果
BASE_FSW_ROOT = _BASE / "feature_space_window"               # 特徴空間による選び方の N 別の結果と精度表
BASE_DETECTION_ROOT = _BASE / "concept_drift_detection"     # 時間による選び方の精度表（評価時期の対応）
LOOKBACK_ROOT = BASE_LOOKBACK_ROOT
FSW_ROOT = BASE_FSW_ROOT
DETECTION_ROOT = BASE_DETECTION_ROOT


def set_variant_roots(variant_name: str) -> None:
    """読み先を版の種類に合わせる（重みあり版は各フォルダの下の posw/）。出力先は variant.apply で切り替える。"""
    from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant
    global LOOKBACK_ROOT, FSW_ROOT, DETECTION_ROOT
    LOOKBACK_ROOT = variant.subdir(BASE_LOOKBACK_ROOT, variant_name)
    FSW_ROOT = variant.subdir(BASE_FSW_ROOT, variant_name)
    DETECTION_ROOT = variant.subdir(BASE_DETECTION_ROOT, variant_name)


# ── 対象 ─────────────────────────────────────────────
# 論文の 6 件（精度表の PROJECTS と同じ並び）
PROJECTS = ["qtbase", "qtcreator", "wikimedia_mediawiki_core", "qtdeclarative", "go", "typo3_cms"]
# 論文の表の並びと名前（表 5 と同じ）
PAPER_ORDER = ["qtbase", "qtcreator", "qtdeclarative", "wikimedia_mediawiki_core", "go", "typo3_cms"]
REPO_TEXT = {"qtbase": "qtbase", "qtcreator": "qtcreator", "qtdeclarative": "qtdeclarative",
             "wikimedia_mediawiki_core": "mediawiki/core", "go": "go", "typo3_cms": "TYPO3.CMS"}
METHODS = ("time_distance", "feature_distance")             # 時間による選び方・特徴空間による選び方
METHOD_LABEL = {"time_distance": "時間", "feature_distance": "特徴空間"}
SIZE_COLUMN = {"time_distance": "window", "feature_distance": "size"}   # metrics.csv の N の列
SELECTED_FILE = {"time_distance": "selected_window.json", "feature_distance": "selected_size.json"}

# N の候補（短い順。同点がそろったときはこの順で最初のもの。design.md §4）
SIZES = ("1d", "3d", "1w", "2w", "1M", "2M")
SIZE_TEXT = {"1d": "1日", "3d": "3日", "1w": "1週", "2w": "2週", "1M": "1ヶ月", "2M": "2ヶ月"}   # 論文 4.1 と同じ書き方

METRIC = "f1_plugin"          # 比べる指標（F1 値。plug-in）
N_COLS = 3                    # 評価時期の数（重みあり版の 3 × 3）
PHASE_NAMES = ("初期", "中期", "後期")   # 表示の向き（p0 ＝ 初期）の名前

ENCODING = "utf-8-sig"        # Excel で開いても文字化けしないように


def metrics_path(method: str, project: str, version: str) -> Path:
    """評価日 × N × seed の指標。"""
    root = LOOKBACK_ROOT if method == "time_distance" else FSW_ROOT
    return root / project / version / "metrics.csv"


def daily_path(method: str, project: str, version: str) -> Path:
    """精度表の評価日 × 行 × seed の指標（day → position の対応を読む）。"""
    if method == "time_distance":
        return DETECTION_ROOT / project / "set_transformer" / version / "daily_metrics.csv.gz"
    return FSW_ROOT / project / "feature_distance" / version / "daily_metrics.csv.gz"


def selected_path(method: str) -> Path:
    """4.1 で選んだ N（リポジトリごと）。"""
    root = LOOKBACK_ROOT if method == "time_distance" else FSW_ROOT
    return root / "report" / SELECTED_FILE[method]


def summary_path(method: str, project: str) -> Path:
    """4.1 の要約表（版ごと・全体の平均。テストで突き合わせる）。"""
    root = LOOKBACK_ROOT if method == "time_distance" else FSW_ROOT
    return root / project / "summary" / f"summary_{METRIC}.csv"


def versions_path(project: str) -> Path:
    """版の一覧（古い順）を持つ run_meta.json（特徴空間上の距離の精度表のもの。時間的な距離と同じ版）。"""
    return FSW_ROOT / project / "feature_distance" / "run_meta.json"


def position_display(position: int) -> int:
    """保存物の評価時期（p0 ＝ リリース直前＝後期）を、表示の向き（p0 ＝ 初期）の番号にする。"""
    return N_COLS - 1 - int(position)
