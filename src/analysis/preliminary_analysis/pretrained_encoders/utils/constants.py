"""pretrained_encoders の設定値（共有基盤：事前学習エンコーダ＋汎用ヘッドの作成・保存）。

design.md 参照。本ディレクトリは concept_drift_detection に依存しない自己完結の基点。
生データ読み込み（changes / release_dates）とボットの判定（src.utils.bot_detection）、
src.features・src.config のみ共有インフラを使う。
"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力（保存先） ────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "pretrained_encoders"

# ── 事前学習の対象・締め切り（design.md §1, §7） ─────────────
# 事前学習データ = プロジェクト開始 〜 そのプロジェクトの cutoff リリース日まで。
# プロジェクトごとに独立の共有モデルを作る（per-release では作らない）。
# プロジェクト別設定：project -> {"cutoff": 版ラベル, "versions": チューニング5版}。
# cutoff は事前学習の締め（サイクル開始）。cutoff の「日付」は major_releases_summary.csv から実行時に引く
# （日付は二重管理しない）。versions は下流のチューニング対象（本ディレクトリでは cutoff のみ使用）。
# 同じ表を lookback_window・concept_drift_detection の constants にも別途保持する（各ディレクトリ自己完結。重複は許容）。
# PROJECTS は今回の論文の 6 件（2025 年末までの直近 5 機能リリース。design.md §1）。何も指定しなければこの 6 件を回す。
# 2026-10-15 に OpenStack 6 件から切り替え、旧 TARGET_PROJECT（既定の 1 件 "nova"）は廃止した（§7）。
PROJECTS = {
    "qtbase":                   {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"]},
    "qtcreator":                {"cutoff": "13.0.0", "versions": ["14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0"]},
    "wikimedia_mediawiki_core": {"cutoff": "1.40.0", "versions": ["1.41.0", "1.42.0", "1.43.0", "1.44.0", "1.45.0"]},
    "qtdeclarative":            {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"]},
    "go":                       {"cutoff": "go1.20", "versions": ["go1.21", "go1.22", "go1.23", "go1.24", "go1.25"]},
    "typo3_cms":                {"cutoff": "13.0.0", "versions": ["13.1.0", "13.2.0", "13.3.0", "13.4.0", "14.0.0"]},
}

# OpenStack 6 件（以前の対象）。今回の論文では分析しないが、設定は残す（design.md §7）。
# 名前を指定したときだけ引ける（project_config は PROJECTS → OPENSTACK_PROJECTS の順に探す）。
OPENSTACK_PROJECTS = {
    "nova":     {"cutoff": "25.0.0", "versions": ["26.0.0", "27.0.0", "28.0.0", "29.0.0", "30.0.0"]},
    "neutron":  {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"]},
    "cinder":   {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"]},
    "glance":   {"cutoff": "24.0.0", "versions": ["25.0.0", "26.0.0", "27.0.0", "28.0.0", "29.0.0"]},
    "keystone": {"cutoff": "21.0.0", "versions": ["22.0.0", "23.0.0", "24.0.0", "25.0.0", "26.0.0"]},
    "swift":    {"cutoff": "2.29.0", "versions": ["2.30.0", "2.31.0", "2.32.0", "2.33.0", "2.34.0"]},
}


def project_config(project: str) -> dict | None:
    """その project の設定（PROJECTS → OPENSTACK_PROJECTS の順に探す）。無ければ None。"""
    return PROJECTS.get(project) or OPENSTACK_PROJECTS.get(project)

# 特徴量 days_to_feature_release（次の機能リリースまでの日数。旧名 days_to_major_release）の「次のリリース」は、サイクル単位のリリースの表
# （major_releases_summary.csv）から日付の順に探す（design.md §11.4）。swift も公式の日付に揃えたので、
# 旧版にあった RELEASE_LEVEL（swift だけ X.Y.0 を区切りとする設定）は廃止した。

# ── 計測点・アクティブ集合（design.md §2） ───────────────────
MEASUREMENT_STEP_DAYS = 1   # 計測点 T = 毎日 0 時（刻み日数。1=毎日）
LOOKBACK_DAYS = 365         # T のアクティブ判定: T-LOOKBACK <= created <= T かつ T に Open（design.md §11.1）

# ── 目的変数（事前学習の教師。design.md §2） ─────────────────
TARGET = "reviewed_within_delta"
REVIEW_HORIZON_DAYS = 1      # Δ（レビュー判定の期間, 日）

# ── モデル：集合Transformer（design.md §3） ──────────────────
D_MODEL = 128                # 埋め込み次元
N_LAYERS = 2                 # 自己注意ブロック数
N_HEADS = 4                  # マルチヘッド数
FFN_DIM = 256                # FFN 中間次元
DROPOUT = 0.1
MAX_SET_SIZE = None          # 1 集合の最大 Change 数（超過は打ち切り。None で無制限）
                             # 旧版は 512。OpenStack では一度も効いておらず、全件を見る方針で None にした（design.md §3）

# ── ヘッド（汎用ヘッド／下流の probe 用） ───────────────────
HEAD_TYPE = "linear"         # "linear"（線形1層）/ "mlp"（小MLP）
HEAD_HIDDEN = 64             # HEAD_TYPE="mlp" のときの隠れ次元

# ── 事前学習（design.md §4） ─────────────────────────────
PRETRAIN_METHOD = "supervised"
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SETS = 16     # 事前学習の 1 バッチに含める集合（T）の数
# 勾配累積（design.md §4。2026-10-19）：1 バッチをこの数の集合ずつに分けて逆伝播し、勾配を足してから 1 回更新する
# （16 日をまとめて計算したときと数学的に同じ更新。qtbase のメモリ不足への対処）。None なら分けない
PRETRAIN_MICRO_SETS = 4
# 2026-10-20：かたまりごとに GPU（MPS）のキャッシュを空にする（design.md §4。キャッシュの積み上がりでメモリ不足になったため。数値には効かない）
# 埋め込み（embed_sets）を一度に計算する集合の数（design.md §6。2026-10-20。旧版は PRETRAIN_BATCH_SETS＝16 日まとめていた。値は変わらない）
EMBED_CHUNK_SETS = 4

# ── linear probing（下流が使う。model.train_probe 用） ──────────
PROBE_EPOCHS = 100
PROBE_LR = 1e-2

# ── 反復（seed ＝ 別エンコーダ。design.md §5） ────────────────
# 事前学習エンコーダを使う分析（lookback_window / concept_drift_detection /
# concept_drift_cause）はすべてこの数で揃える。**奇数**であることに意味がある：
# concept_drift_detection が日ごとに seed 中央値を取るとき、奇数なら中央値が
# 実在する 1 個のモデルの値になり、concept_drift_cause がそのモデルを一意に特定できる。
N_REPEATS = 5                # 作るエンコーダ数
RANDOM_SEED = 42             # k 個目の torch seed = RANDOM_SEED + k

# ── 計算環境 ─────────────────────────────────────
DEVICE = "auto"              # "auto" で CUDA → MPS → CPU の順（2026-10-20 改めた。旧版は MPS → CPU。design.md §7・§14）
CUDA_MEMORY_FRACTION = 0.25  # CUDA のときに使う GPU のメモリの上限（共有のサーバで全体の 1/4。design.md §14.4）
