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
