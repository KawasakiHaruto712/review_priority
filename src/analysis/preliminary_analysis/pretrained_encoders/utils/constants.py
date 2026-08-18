"""pretrained_encoders の設定値（共有基盤：事前学習エンコーダ＋汎用ヘッドの作成・保存）。

design.md 参照。本ディレクトリは concept_drift_detection に依存しない自己完結の基点。
生データ読み込み（changes / release_dates / bot 設定）と src.features・src.config のみ共有インフラを使う。
"""
from src.config.path import DEFAULT_CONFIG, DEFAULT_DATA_DIR

# ── 入力（ボット判定用の一覧） ─────────────────────────────
GERRYMANDER_CONFIG = DEFAULT_CONFIG / "gerrymanderconfig.ini"
BOT_ACCOUNTS_CSV = DEFAULT_CONFIG / "third_party_ci_accounts.csv"
EXTRA_BOTS_FILE = DEFAULT_CONFIG / "extra_bots.txt"

# ── 出力（保存先） ────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "pretrained_encoders"

# ── 事前学習の対象・締め切り（design.md §1） ──────────────────
# 事前学習データ = プロジェクト開始 〜 FIRST_TARGET_VERSION のサイクル開始（＝その直前リリース）まで。
# 全リリースで共有する単一モデル（per-release では作らない）。
TARGET_PROJECT = "nova"
FIRST_TARGET_VERSION = "26.0.0"     # このバージョンのサイクル開始を締め切りに使う
PRETRAIN_CUTOFF = None              # ISO 日付を書けば固定も可（None なら上記から自動）

# ── 計測点・アクティブ集合（design.md §2） ───────────────────
MEASUREMENT_STEP_DAYS = 1   # 計測点 T = 毎日 0 時（刻み日数。1=毎日）
LOOKBACK_DAYS = 365         # T のアクティブ判定: T-LOOKBACK <= created <= T < decision_time

# ── 目的変数（事前学習の教師。design.md §2） ─────────────────
TARGET = "reviewed_within_delta"
REVIEW_HORIZON_DAYS = 1      # Δ（レビュー判定の期間, 日）

# ── モデル：集合Transformer（design.md §3） ──────────────────
D_MODEL = 128                # 埋め込み次元
N_LAYERS = 2                 # 自己注意ブロック数
N_HEADS = 4                  # マルチヘッド数
FFN_DIM = 256                # FFN 中間次元
DROPOUT = 0.1
MAX_SET_SIZE = 512           # 1 集合の最大 Change 数（超過は打ち切り。None で無制限）

# ── ヘッド（汎用ヘッド／下流の probe 用） ───────────────────
HEAD_TYPE = "linear"         # "linear"（線形1層）/ "mlp"（小MLP）
HEAD_HIDDEN = 64             # HEAD_TYPE="mlp" のときの隠れ次元

# ── 事前学習（design.md §4） ─────────────────────────────
PRETRAIN_METHOD = "supervised"
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SETS = 16     # 事前学習の 1 バッチに含める集合（T）の数

# ── linear probing（下流が使う。model.train_probe 用） ──────────
PROBE_EPOCHS = 100
PROBE_LR = 1e-2

# ── 反復（seed ＝ 別エンコーダ。design.md §5） ────────────────
N_REPEATS = 10               # 作るエンコーダ数
RANDOM_SEED = 42             # k 個目の torch seed = RANDOM_SEED + k

# ── 計算環境 ─────────────────────────────────────
DEVICE = "auto"              # "auto" で MPS があれば MPS、無ければ CPU
