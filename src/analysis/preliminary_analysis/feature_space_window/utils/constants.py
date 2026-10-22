"""feature_space_window の設定値（RQ2：特徴空間で近い過去の日で probe を学習する。design.md）。

共有部品は pretrained_encoders・feature_table から import（lookback_window・concept_drift_detection には依存しない）。
"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力（design.md §6） ─────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "feature_space_window"

# ── 対象（design.md §2） ─────────────────────────────────
# lookback_window・pretrained_encoders の constants と同じ内容を自己完結で持つ（重複は許容）。
PROJECTS = {
    "qtbase":                   {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"]},
    "qtcreator":                {"cutoff": "13.0.0", "versions": ["14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0"]},
    "wikimedia_mediawiki_core": {"cutoff": "1.40.0", "versions": ["1.41.0", "1.42.0", "1.43.0", "1.44.0", "1.45.0"]},
    "qtdeclarative":            {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"]},
    "go":                       {"cutoff": "go1.20", "versions": ["go1.21", "go1.22", "go1.23", "go1.24", "go1.25"]},
    "typo3_cms":                {"cutoff": "13.0.0", "versions": ["13.1.0", "13.2.0", "13.3.0", "13.4.0", "14.0.0"]},
}


def project_config(project: str) -> dict:
    cfg = PROJECTS.get(project)
    if cfg is None:
        raise KeyError(f"未知のプロジェクトです（PROJECTS に定義してください）: {project}")
    return cfg


def cutoff_for(project: str) -> str:
    return project_config(project)["cutoff"]


def versions_for(project: str) -> list[str]:
    return list(project_config(project)["versions"])


# ── 使う事前学習エンコーダ ──────────────────────────────
N_SEEDS = None                 # None なら保存済み全 seed

# ── 大きさ（design.md §4。RQ1 の窓長と同じ日数・同じ名前） ─────────────
SIZES_DAYS = [1, 3, 7, 14, 30, 60]
SIZE_LABELS = {1: "1d", 3: "3d", 7: "1w", 14: "2w", 30: "1M", 60: "2M"}

# ── 近さ（design.md §3） ────────────────────────────────
STD_EPS = 1e-8                 # 標準偏差がこれ未満の次元は 1 とみなす（事前学習の Scaler と同じ決まり。2026-10-20）

# ── 指標（design.md §4。lookback_window と同じ） ─────────────────
CLASSIFY_THRESHOLD = 0.5
K_LIST = [5, 10, 20]           # top-k（＋その日の正例数 '@pos' も自動で出す）
PLOT_METRIC_DEFAULT = "auc"
PLOT_DPI = 150
LOWER_IS_BETTER = {"norm_rank"}   # 小さいほど良い指標（順位の集計・報告用の表の向き。design.md §5.1。2026-10-21）

# ── 特徴のずれ×評価時期の精度表（design.md §9。2026-10-21 追加。値は concept_drift_detection と同じ設定の複製） ──
# リポジトリごとの k（近い日の群の日数）と刻み（評価時期の列の幅＝RQ1 の窓長）。どちらも結果を見て本人が手で書く。
# size_days：大きさの分析・順位の集計で選んだ k。step_days：RQ1 の時期のずれ×評価時期の精度表と同じ刻み（RQ1 の window_days）。
MATRIX_SETTINGS = {p: {"size_days": None, "step_days": None} for p in PROJECTS}
# 2026-10-21：step_days は RQ1 の窓長（concept_drift_detection の window_days。recall@pos の最良）と同じ値を書いた（design.md §9.1）。
# size_days（RQ2 の k）は大きさの分析の結果を見てから書く。
for _p, _w in {"qtbase": 60, "qtcreator": 30, "wikimedia_mediawiki_core": 14, "qtdeclarative": 14,
               "go": 7, "typo3_cms": 7}.items():
    MATRIX_SETTINGS[_p]["step_days"] = _w
# 2026-10-21：size_days（RQ2 の k）を書いた。報告用の表 best_recall@pos_all.md の最良（recall@pos の 5 版平均が最良の大きさ。
# RQ1 の窓長と同じ決め方。本人判断）。最良と 2 番目の差は 0.0002〜0.005 と小さい。
for _p, _k in {"qtbase": 60, "qtcreator": 14, "wikimedia_mediawiki_core": 30, "qtdeclarative": 14,
               "go": 14, "typo3_cms": 14}.items():
    MATRIX_SETTINGS[_p]["size_days"] = _k

# 重みあり版（posw）の k と刻み（2026-10-22 追加。design.md §10.1）：k は大きさの分析（重みあり版）の F1（plug-in）の最良、
# 刻みは重みあり版の RQ1 の窓長（concept_drift_detection の WINDOW_DAYS_POSW）を、本人が結果を見て書く。None のままなら止める。
MATRIX_SETTINGS_POSW = {p: {"size_days": None, "step_days": None} for p in PROJECTS}


# 重みあり版の表の形（2026-10-22。design.md §10.1・pretrained_encoders/design.md §15.9）：周期を 3 等分した 3 × 3。
# 行 d ＝ 近い順で「刻み × d ＋ 1 〜 刻み × d ＋ k 番目」の日（刻み ＝ 周期 ÷ 3。版ごと）。MATRIX_SETTINGS_POSW は size_days（k）だけを書く。
FRACTION_PARTS = 3
MERGED_DIRNAME = "all_versions"    # 5 版をまとめた表の出力先（<project>/feature_distance/all_versions/）
HEATMAP_SHOW_SPREAD = True         # 3 × 3 のセルに「± 評価日の間の標準偏差」を書くか
SMALL_GRID_MAX = 6                 # この大きさ以下の表は、セルに数字を書く


def fraction_grid() -> bool:
    """今の版の種類で 3 × 3 を使うか（重みあり版だけ）。"""
    from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
    return pre.ENCODER_VARIANT == "posw"


def matrix_settings() -> dict:
    """版の種類（pretrained_encoders の ENCODER_VARIANT）に合う MATRIX_SETTINGS（2026-10-22）。"""
    from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
    return MATRIX_SETTINGS_POSW if pre.ENCODER_VARIANT == "posw" else MATRIX_SETTINGS


RANDOM_SEED = 42             # probe の seed（k 本目のエンコーダ ＝ RANDOM_SEED + k。RQ1 の精度表と同じ）
REPEAT_AGG = "median"          # 評価日ごとの seed 方向の集約
MIN_TRAIN = 30                 # 学習の k 日の最小レコード数（RQ1 と同じ）
MIN_EVAL = 30                  # 評価日の最小レコード数（RQ1 と同じ）
BIN_DAY_ALIGNED = True         # 評価時期の境界を 0 時にそろえる（RQ1 と同じ）
PLOT_METRICS = ["auc", "recall@pos", "f1_plugin"]   # ヒートマップを描く指標（CSV・json は全指標。2026-10-21 に recall@pos を足した。design.md §9.2）
# （2026-10-22：f1_plugin を足した。重みあり版は F1 で見るため。design.md §10.1）
HEATMAP_SHOW_CELL_VALUES = False
HEATMAP_VALUE_FMT = "{:.2f}"
SAVE_PER_REPEAT = True
