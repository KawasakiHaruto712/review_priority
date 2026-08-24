"""concept_drift_detection の設定値（Phase1 / step2：位置×距離ヒートマップ版）。

design.md 参照。事前学習エンコーダ＋汎用ヘッド・データ構築・特徴・label・model は
`pretrained_encoders` から import して使う（本ディレクトリでは事前学習しない）。
ここに置くのは concept_drift 固有の設定（ビン分割・load 対象・指標・図・検定）。
"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_detection"

# ── 分析対象プロジェクトとリリース（§2） ──────────────────────
# nova 26.0.0–30.0.0（5バージョン）。開発サイクルは 2022-03-30 〜 2024-10-02。
TARGET_PROJECTS = {
    "nova": ["26.0.0", "27.0.0", "28.0.0", "29.0.0", "30.0.0"],
}

# ── ビン分割（§4） ───────────────────────────────────
# per_release: 各リリースを、そのリリース自身の期間で BIN_COUNT 等分（リリース境界に整合）。
# BIN_COUNT=26 → 1 コマ ≒ サイクル/26 ≒ 1 週間。d=26 がちょうど 1 リリース（同フェーズ）。境界は 0 時に揃える。
BINNING = "per_release"
BIN_COUNT = 26
BIN_DAY_ALIGNED = True
MAX_SET_SIZE = 512           # 1 集合の最大 Change 数（pretrained_encoders と一致させる）

# ── 事前学習エンコーダの読み込み（§6.1） ─────────────────────
# pretrained_encoders の保存物（project × cutoff × seed）を load。cutoff=25.0.0（26.0.0 の直前）。
PRETRAINED_CUTOFF = "25.0.0"
N_REPEATS = 10               # load するエンコーダ数（不足は pretrained_encoders.build_encoders で自動作成）
RANDOM_SEED = 42             # probe 学習の seed（k 回目 = RANDOM_SEED + k）
REPEAT_AGG = "median"        # 反復方向の集約（"median" / "mean"）

# ── データ量の床（§7） ────────────────────────────────
MIN_TRAIN = 30               # 学習ビンの最小レコード数（下回るセルは NaN）
MIN_EVAL = 30                # 評価ビンの最小レコード数

# ── 評価指標（§8.1。lookback と同一一式・主 AUC） ──────────────
K_LIST = [5, 10, 20]         # top-k の k（＋その日の正例数 '@pos' も自動で出す）
CLASSIFY_THRESHOLD = 0.5     # precision/recall/f1 用の2値しきい値
# 全指標を「日ごと算出→ビン日数平均→seed中央値+IQR」で統一（AUC も日次平均。§8.1）。
# AUC が荒すぎる場合のみ True にするとプール算出に切替（下の POOL_AUC）。
POOL_AUC = False

# ── 変化区間の判定（§8.2）─────────────────────────────
PERMUTATION_N = 1000
SIGNIFICANCE = 0.05

# ── 図（§10） ────────────────────────────────────────
# 26×26 ヒートマップ（色のみ）。数値表示は既定 False（後から切替可）。
HEATMAP_SHOW_CELL_VALUES = False
HEATMAP_VALUE_FMT = "{:.2f}"
PLOT_METRICS = ["auc"]       # ヒートマップを描く指標（既定 AUC のみ）。CSV/json は全指標を保存。

# ── モデル名・出力（§10）──────────────────────────────
MODEL_NAME = ["set_transformer"]
SAVE_PER_REPEAT = True
SAVE_PREDICTIONS = True
PLOT_DPI = 150
