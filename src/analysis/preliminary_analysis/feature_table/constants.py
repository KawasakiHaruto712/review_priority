"""feature_table の設定値（design.md）。"""
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.status import STATUS_FEATURES
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.ticket import TICKET_FEATURES
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.waiting import WAITING_FEATURES
from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_builder import FEATURE_NAMES
from src.config.path import DEFAULT_DATA_DIR

# ── 置き場所（§1） ─────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "feature_table"
TABLE_ROOT = OUTPUT_ROOT / "table"            # 表：<TABLE_ROOT>/<project>/pretrain.npz と pretrain_meta.json
SELECTION_ROOT = OUTPUT_ROOT / "selection"    # 相関の結果

# ── 表の列（§2.2） ────────────────────────────────────
# 並びは既存の 15 個 → チケット自身 13 → 開発状況 22 → 待っている間 17（絞る前の 67 個）
ALL_FEATURE_NAMES: list[str] = list(FEATURE_NAMES) + list(TICKET_FEATURES) + list(STATUS_FEATURES) + list(WAITING_FEATURES)
# 代わりの値（pretrained_encoders/design.md §13.2）を入れうる 20 個。印（filled）の列の並び
_FILLABLE = {
    "merge_rate", "recent_merge_rate", "days_to_feature_release",
    "merge_ratio_60d", "merge_ratio_total", "ci_last_result", "ci_fail_ratio",
    "prior_reviewer_count", "prior_message_count", "ci_run_count", "recheck_count", "author_open_changes",
    "file_days_since_change", "avg_days_between_revisions", "days_since_last_ci",
    "avg_days_to_merge", "avg_days_to_close", "core_first_response_days", "prior_first_response_days",
    "non_core_change_ratio",
}
FILLABLE_NAMES: list[str] = [n for n in ALL_FEATURE_NAMES if n in _FILLABLE]
# 負の値が正しくありうる特徴量（それ以外で −1 が出たら、代わりの値も出せなかった印として数える）
SIGNED_FEATURES = {"weighted_vote_score"}

CHUNK_ROWS = 100_000          # 配列のかたまりの行数（§2.3）
PERIOD_PRETRAIN = "pretrain"  # 期間の名前（後で評価期間を足すときは別の名前のファイルにする。§2.4）
PERIOD_EVAL = "eval"          # 評価期間の表（5 版をまとめて 1 つ。§2.1・§2.4。2026-10-18 追加）
PERIODS = (PERIOD_PRETRAIN, PERIOD_EVAL)

# ── 相関で絞る（§3） ──────────────────────────────────
SPEARMAN_THRESHOLD = 0.7      # |ρ| ≥ 0.7（AutoSpearman・Kraemer ら 2003）
MAJORITY_KS = (3, 4, 5)       # 過半数の決まり：|ρ| ≥ 0.7 のリポジトリが k 件以上
RATIO = (2, 3)                # 割合の決まり：|ρ| ≥ 0.7 のリポジトリが、ρ が出るリポジトリの 2/3 以上（§3.2 の 4。2026-10-20 追加）
RULES = [f"majority{k}" for k in MAJORITY_KS] + ["mean", "ratio"]
ADOPTED_RULE = "ratio"        # 採った決まり（2026-10-18 本人判断）。ほかは参考
