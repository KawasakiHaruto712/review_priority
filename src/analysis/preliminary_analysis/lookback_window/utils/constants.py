"""lookback_window の設定値（事前分析 step1：チューニング窓の長さの調査）。

design.md 参照。共有部品は pretrained_encoders から import（concept_drift_detection には依存しない）。
"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "lookback_window"

# ── 対象 ─────────────────────────────────────────────
PROJECT = "nova"
VERSIONS = ["26.0.0", "27.0.0", "28.0.0", "29.0.0", "30.0.0"]

# ── 使う事前学習エンコーダ（pretrained_encoders の保存物） ────────────
PRETRAINED_CUTOFF = "25.0.0"   # 全リリース共通の共有モデル
N_SEEDS = None                 # None なら保存済み全 seed

# ── 学習窓（design.md §4） ────────────────────────────
# 学習窓 = 評価日の直前 W 日（[X-W, X-1]。末尾＝前日。評価日自身は予測対象なので入れない）。
WINDOWS_DAYS = [1, 3, 7, 14, 30, 60]
WINDOW_LABELS = {1: "1d", 3: "3d", 7: "1w", 14: "2w", 30: "1M", 60: "2M"}

# ── 指標（design.md §7。全部出す → 描画は後から選ぶ） ──────────────
CLASSIFY_THRESHOLD = 0.5
K_LIST = [5, 10, 20]           # top-k（＋その日の正例数 '@pos' も自動で出す）
PLOT_METRIC_DEFAULT = "auc"    # 既定で先に描く指標

# ── 出力・作図 ───────────────────────────────────────
PLOT_DPI = 150
DRAW_GENERAL_DEFAULT = True    # 汎用ヘッド（チューニングなし）基準線をデフォルトで重ねる
