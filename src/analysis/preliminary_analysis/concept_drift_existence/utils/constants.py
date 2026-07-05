"""concept_drift_existence 分析の設定値（事前分析① 変化区間の存在確認）。

パラメータはすべてこの 1 ファイルに集約する（初学者が最初に触る場所）。
設計の根拠は同ディレクトリの design.md を参照。
"""
from src.config.path import DEFAULT_CONFIG, DEFAULT_DATA_DIR

# ── 入力（ボット判定用の一覧。Change/リリース日は common.data_loader で読む） ──
GERRYMANDER_CONFIG = DEFAULT_CONFIG / "gerrymanderconfig.ini"
BOT_ACCOUNTS_CSV = DEFAULT_CONFIG / "third_party_ci_accounts.csv"
EXTRA_BOTS_FILE = DEFAULT_CONFIG / "extra_bots.txt"

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_existence"

# ── 分析対象プロジェクトとリリース ────────────────────────
TARGET_PROJECTS = {
    "nova": [
        "2015.1.0",
        "12.0.0", "13.0.0", "14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0", "19.0.0", "20.0.0",
    ],
}

# ── 計測時点・アクティブ集合（§2.2） ─────────────────────
MEASUREMENT_STEP_DAYS = 1  # 計測点 T = 毎日 0 時の定点グリッド（刻み日数。1=毎日）
LOOKBACK_DAYS = 365        # T 時点のアクティブ判定: T-LOOKBACK <= created <= T < decision_time

# ── ラベル ───────────────────────────────────────────
LABEL_NAME = "time_to_next_review"  # registry で将来差替可（§5.2）
DURATION_UNIT = "hours"              # ラベルの時間単位
CENSORING_MODE = "drop"             # "drop"(末尾まで未レビューは除外) | "survival"(将来)
# 学習時に target を log1p 変換するか（time は指数的に広がるため heavy-tail 対策）。
# 評価は順位ベースで log は単調変換＝順位不変なので、指標の定義は変わらず「モデルの当てやすさ」だけが上がる。
LABEL_LOG_TRANSFORM = True

# ── ビン分割（§2.5） ──────────────────────────────────
# ビン幅 = 当該リリースを BIN_COUNT 等分。同幅を前リリースへ延長。距離(滞留期間)の最大行数も BIN_COUNT。
# 学習候補プール = [cycle_start-(cycle_end-cycle_start), cycle_end]。
BINNING = "equal_time"  # "equal_time"(当該リリースの時間等分, 既定) | "equal_count"
BIN_COUNT = 6

# ── 学習/評価データ（§2.6。数えるのは「Change 数」。レコード数の不一致は許容） ──
TRAIN_EVAL_RATIO = (8, 2)  # 学習(層1+層2学習側) : 評価(層3+層2評価側) の Change 数比
N_TRAIN = 400              # 全セル共通の学習 Change 数（None=供給から自動決定）
N_EVAL = 100               # 全セル共通の評価 Change 数（N_TRAIN:N_EVAL=8:2）
MIN_TRAIN = 30             # 学習 Change 数の床。供給できないセルは除外（NaN）
MIN_EVAL = 10              # 〃（評価側 Change 数）
RANDOM_SEED = 42           # 反復のベース乱数（k 回目は RANDOM_SEED + k）
# 同一 Change は複数の計測点 T で何度もレコードになる（長寿ほど行数が多い）。学習時に各行へ
# 1/(その Change の学習内レコード数) の重みを与え、Change ごとの合計発言力を 1 に揃える。
# 効果: (1) 長寿 Change が行数で押し切る偏りを除去、(2) 重複の水増しを是正し「Change 数 ≒ 実効標本数」で学習。
# False にすると従来どおり重みなし（全行を等価）。重みあり/なしの精度比較に使う。
CHANGE_BALANCED_WEIGHT = True

# ── セルの集約（2段で中央値。§5.7, §6.2） ──
T_AGG = "median"        # 1反復内で計測点 T 方向に指標_T をまとめる集約（"median" / "mean"）
N_REPEATS = 10          # 各セルで層2振り分け・学習を変えて繰り返す回数
REPEAT_AGG = "median"   # 反復方向の集約＝セル値（"median" / "mean"）
SAVE_PER_REPEAT = True  # 各反復の代表値(T方向集約後)も json に残す
# 各レコードの生の予測（評価セットの (T, change_id, 真の時間, 予測時間)）を保存するか。
# True なら後から「再学習なしで」別の評価指標（回帰精度・分類など）を計算し直せる。
# 保存は <project>/<model>/<version>/predictions.csv.gz（hours 単位。圧縮で全体 ~0.2GB 程度）。
SAVE_PREDICTIONS = True

# ── モデル（§5.3） ───────────────────────────────────
# リストの並び順＝実行順。モデルを最外ループにして「1モデルで全リリース→次モデル」で回す（§5.11）。
# 軽い順に並べるのが目安（LightGBM の方が RandomForest より軽量・高速）。
MODEL_NAME = ["lightgbm", "random_forest"]  # GNN などは将来的に実装予定
OBJECTIVE = "regression"                     # pointwise 回帰（既定）

# ── 評価指標（§2.8。3つとも順位ベース。MAE/RMSE は正規化順位） ──
# NDCG@n の n =「評価集合の中で、次の計測点（翌日0時）までに実際にレビューされる上位件数」に合わせる。
# 注意: ランキング対象は実キュー全部(~600/日)ではなく、固定サンプルした評価集合(N_EVAL=100、
#   うち1日に Open なのは ~60件)。レビュー率は ~11%/日 なので、評価集合 ~60 の中で実際に捌かれるのは ~6件。
#   それに少し余裕を持たせ上位 ~16% の n=10 とする（n=60 は評価集合とほぼ同サイズで識別力ゼロ、n=6 はノイジー）。
NDCG_AT = 10
# 評価指標 3 系統（§2.8）。順位(per-T 集約) / 回帰誤差(プール) / 分類(プール)。
#   順位:   mae / rmse / ndcg（正規化順位ベース）
#   回帰誤差: mae_log / rmse_log / r2_log（log 時間。誤差そのもの）
#   分類:   macro_f1 / micro_f1 / qwk（時間バケツに離散化）
ENABLED_METRICS = ["mae", "rmse", "ndcg",
                   "mae_log", "rmse_log", "r2_log",
                   "macro_f1", "micro_f1", "qwk"]

# 分類のバケツ境界（hours 昇順）。既定 = 1日 / 1週間 / 1ヶ月 → 4 クラス（≤1日/≤1週/≤1ヶ月/>1ヶ月）。
# 後から変更可能（例: 全体四分位で凍結など）。§2.8.3。
CLASS_BUCKETS_HOURS = [24, 168, 720]

# ── 変化区間の判定（§7。検定の情報は png には描かず json のみ） ──
PERMUTATION_N = 1000  # 並べ替え検定の反復回数（計算済み行列の並べ替えのみ。再学習しない）
SIGNIFICANCE = 0.05   # 有意水準
PLOT_DPI = 150
