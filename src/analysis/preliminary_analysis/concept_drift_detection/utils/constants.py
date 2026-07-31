"""concept_drift_detection の設定値（レビュー優先順位ドリフトの存在確認・再設計版）。

パラメータはすべてこの 1 ファイルに集約する（最初に触る場所）。設計の根拠は同ディレクトリの design.md。
目的変数は「計測点から Δ 以内にレビューされたか」の2値。評価は precision / recall / f1。
"""
from src.config.path import DEFAULT_CONFIG, DEFAULT_DATA_DIR

# ── 入力（ボット判定用の一覧） ─────────────────────────────
GERRYMANDER_CONFIG = DEFAULT_CONFIG / "gerrymanderconfig.ini"
BOT_ACCOUNTS_CSV = DEFAULT_CONFIG / "third_party_ci_accounts.csv"
EXTRA_BOTS_FILE = DEFAULT_CONFIG / "extra_bots.txt"

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_detection"

# ── 分析対象プロジェクトとリリース ────────────────────────
TARGET_PROJECTS = {
    "nova": [
        "2015.1.0",
        "12.0.0", "13.0.0", "14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0", "19.0.0", "20.0.0",
    ],
}

# ── 計測点・アクティブ集合（§2.2, §2.4） ────────────────────
MEASUREMENT_STEP_DAYS = 1   # 計測点 T = 毎日 0 時の定点グリッド（刻み日数。1=毎日）
LOOKBACK_DAYS = 365         # T 時点のアクティブ判定: T-LOOKBACK <= created <= T < decision_time

# ── 目的変数（§2.3）───────────────────────────────────
# 「T から REVIEW_HORIZON_DAYS 以内に人間レビューが付くか」の2値。正例=1 / 負例=0。
TARGET = "reviewed_within_delta"
REVIEW_HORIZON_DAYS = 1      # Δ（レビュー判定の期間, 日）。1週間→1日 等に変更可
OBJECTIVE = "classification"

# ── ビン分割（§2.5） ──────────────────────────────────
BINNING = "equal_time"      # 当該リリースを BIN_COUNT 等分（同幅を前リリースへ延長）
BIN_COUNT = 6
# ビン境界を 0 時に揃える（開始を 0 時に丸め、ビン幅を整数日に）。毎日 0 時の計測点と一致し、
# Δ=MEASUREMENT_STEP 日のとき学習末尾マージンによる除外が厳密に 0 になる。False で従来の秒単位等分。
BIN_DAY_ALIGNED = True

# ── 先読みマージン（§2.3.1） ──────────────────────────────
# 学習: ビン末尾 Δ を除外（学習ラベルが学習期間を超えて未来を覗くリークを防ぐ）。
# 評価: 自主的な末尾除外はしない（先読みでラベル付け）。絶対末尾は物理的に自動除外（ラベル None）。
TRAIN_TAIL_MARGIN = True
EVAL_TAIL_MARGIN = False

# ── 学習/評価データ（§2.7） ───────────────────────────────
N_TRAIN = 500               # 学習 Change 数（全セル共通の固定数）。比較可能性のため
N_EVAL = "all"              # 評価 Change 数。"all"=全 Change、数値なら固定数に切替
MIN_TRAIN = 30              # 学習 Change 数の床（下回るセルは NaN）
MIN_EVAL = 30               # 評価 Change 数の床（下回るセルは信頼性低として NaN）
RANDOM_SEED = 42            # 反復のベース乱数（k 回目は RANDOM_SEED + k）

# 学習の重み: 各行に 1/(その Change の学習内レコード数)。長寿 Change の水増しを是正。
CHANGE_BALANCED_WEIGHT = True
# 評価の重み: 既定なし（実運用の判定品質を測る）。ON にすると Change ごと等価で交絡を抑制。
EVAL_CHANGE_BALANCED_WEIGHT = False

# ── セルの集約（§2.7） ────────────────────────────────
N_REPEATS = 10              # 各セルで学習サンプリングを変えて繰り返す回数
REPEAT_AGG = "median"       # 反復方向の集約＝セル値（"median" / "mean"）
SAVE_PER_REPEAT = True      # 各反復の代表値を json に残す
SAVE_PREDICTIONS = True     # 生予測（評価レコードの y_true/y_pred）を保存（後から再計算用）

# ── モデル（§4） ─────────────────────────────────────
MODEL_NAME = ["lightgbm", "random_forest"]

# ── 評価指標（§2.8） ─────────────────────────────────
# 正例＝Δ以内にレビュー。プールして算出。不均衡ゆえ accuracy は主指標にしない。
ENABLED_METRICS = ["precision", "recall", "f1"]
CLASSIFY_THRESHOLD = 0.5    # 2値判定しきい値（0.5 が既定。後で振れるよう設定に置く）

# ── 変化区間の判定（§5） ─────────────────────────────
PERMUTATION_N = 1000        # 並べ替え検定の反復回数（再学習しない）
SIGNIFICANCE = 0.05
PLOT_DPI = 150
