"""concept_drift_detection の設定値（Phase1：集合Transformer × 転移学習版）。

パラメータはすべてこの 1 ファイルに集約する（最初に触る場所）。設計の根拠は同ディレクトリの design.md。
- 目的変数：計測点から Δ 以内にレビューされたか（2値）。§1
- モデル：集合Transformer（自己注意・位置エンコなし）＋転移学習（事前学習→凍結→linear probing）。§5,§6
- 主指標は AUC（不均衡に頑健）。距離×時期行列は AUC と f1 を必ず描く。§8
"""
from src.config.path import DEFAULT_CONFIG, DEFAULT_DATA_DIR

# ── 入力（ボット判定用の一覧） ─────────────────────────────
GERRYMANDER_CONFIG = DEFAULT_CONFIG / "gerrymanderconfig.ini"
BOT_ACCOUNTS_CSV = DEFAULT_CONFIG / "third_party_ci_accounts.csv"
EXTRA_BOTS_FILE = DEFAULT_CONFIG / "extra_bots.txt"

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_detection"

# ── 分析対象プロジェクトとリリース（§2） ──────────────────────
# nova 26.0.0–30.0.0（5バージョン）。開発サイクルは 2022-03-30 〜 2024-10-02。
TARGET_PROJECTS = {
    "nova": ["26.0.0", "27.0.0", "28.0.0", "29.0.0", "30.0.0"],
}

# ── 計測点・アクティブ集合（§3） ────────────────────────────
MEASUREMENT_STEP_DAYS = 1   # 計測点 T = 毎日 0 時（刻み日数。1=毎日）
LOOKBACK_DAYS = 365         # T のアクティブ判定: T-LOOKBACK <= created <= T < decision_time

# ── 目的変数（§1）───────────────────────────────────
TARGET = "reviewed_within_delta"
REVIEW_HORIZON_DAYS = 1      # Δ（レビュー判定の期間, 日）。「1日以内にレビューされるか」
TASK = "classification"      # 将来 "ranking" / "regression" に拡張可（§12-6）

# ── ビン分割（§4） ───────────────────────────────────
# per_release: 各リリースを、そのリリース自身の期間で BIN_COUNT 等分（リリース境界に整合）。
# d=6 がちょうど 1 リリース（同フェーズ）に対応する。境界は 0 時に揃える。
BINNING = "per_release"
BIN_COUNT = 6
BIN_DAY_ALIGNED = True

# ── 事前学習の期間（§6.1）─────────────────────────────
# 共有エンコーダは「プロジェクト初期 〜 分析対象の直前」まで（評価期間を含めない＝リーク防止）。
# None のとき、最も早い対象リリースの cycle_start（＝26.0.0 の直前）を自動で締め切りに使う。
PRETRAIN_CUTOFF = None       # 例: "2022-03-30" と書けば固定も可

# ── モデル：集合Transformer（§5.2）──────────────────────
D_MODEL = 128                # 埋め込み次元
N_LAYERS = 2                 # 自己注意ブロック数
N_HEADS = 4                  # マルチヘッド数
FFN_DIM = 256                # FFN 中間次元（≈2〜4×d_model）
DROPOUT = 0.1
MAX_SET_SIZE = 512           # 1 集合の最大 Change 数（超過分は打ち切り。メモリ対策。None で無制限）

# ── ヘッド（§5.3）─────────────────────────────────
# "linear": 線形1層（linear probe, 既定）。"mlp": 小さなMLPヘッド（非線形probe）。
HEAD_TYPE = "linear"
HEAD_HIDDEN = 64             # HEAD_TYPE="mlp" のときの隠れ次元

# ── チューニング方式（§6.6）───────────────────────────
# "linear_probe": エンコーダ凍結・ヘッドのみ学習（既定）。将来 "fine_tune" / "peft" に拡張可。
TUNING = "linear_probe"

# ── 事前学習方式（§6.1）───────────────────────────────
# "supervised": Δ以内レビューを予測して事前学習（既定）。将来 "ssl（自己教師あり学習）" に拡張可。
PRETRAIN_METHOD = "supervised"
PRETRAIN_EPOCHS = 20         # 事前学習エポック数
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SETS = 16     # 事前学習の 1 バッチに含める集合（T）の数

# ── linear probing（§6.2）─────────────────────────────
PROBE_EPOCHS = 100           # ヘッド学習エポック数（凍結表現上なので軽い）
PROBE_LR = 1e-2

# ── 反復・集約（§6.5, §9）──────────────────────────────
# N_REPEATS = 事前学習の反復回数（＝別 seed のエンコーダ数）。ばらつきの主因は事前学習なので
# 「事前学習を反復 → 各エンコーダで probe → 集約」で意味のある不確かさ（IQR）を得る。
N_REPEATS = 10
REPEAT_AGG = "median"        # 反復方向の集約（"median" / "mean"）
RANDOM_SEED = 42             # k 回目の事前学習 seed = RANDOM_SEED + k

# ── データ量の床（§7）────────────────────────────────
# 固定はしない（フル集合）。信頼性のため、下回るセルは NaN。単位は「評価/学習レコード数」。
MIN_TRAIN = 30
MIN_EVAL = 30

# ── 評価指標（§8）────────────────────────────────────
# AUC を主指標（しきい値フリー・不均衡に頑健）。距離×時期行列は AUC と f1 を必ず描く。
ENABLED_METRICS = ["auc", "f1", "precision", "recall"]
CLASSIFY_THRESHOLD = 0.5     # precision/recall/f1 用の2値しきい値（当面 0.5。per-cell 調整はしない）

# ── 変化区間の判定（§8.2）─────────────────────────────
PERMUTATION_N = 1000
SIGNIFICANCE = 0.05

# ── モデル名・出力（§10）──────────────────────────────
MODEL_NAME = ["set_transformer"]   # 今回は集合Transformer のみ
SAVE_PER_REPEAT = True
SAVE_PREDICTIONS = True
PLOT_DPI = 150

# ── 計算環境 ─────────────────────────────────────
# "auto" で MPS があれば MPS、無ければ CPU。
DEVICE = "auto"
