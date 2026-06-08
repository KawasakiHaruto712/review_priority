"""
shared_ranking 用の定数。

trend_models / daily_regression の双方から再 export される単一の正本。
"""

from pathlib import Path

# ロジック更新時にバンプする。キャッシュファイル名の hash 入力に含まれるため、
# 値を変えると過去キャッシュは自動的に無効化される。
LOGIC_VERSION = "shared_ranking@2"

# 1 スナップショット内の最小チケット数
MIN_QUERY_SIZE = 2

# 打ち切り上限: 1 年（= 365 日 × 86400 秒）
MAX_CENSORING_SECONDS = 31_536_000

# 16 特徴量（順序固定）
FEATURE_NAMES = [
    # Bug Metrics
    "bug_fix_confidence",
    # Change Metrics
    "lines_added",
    "lines_deleted",
    "files_changed",
    "elapsed_time",
    "revision_count",
    "test_code_presence",
    # Developer Metrics
    "past_report_count",
    "recent_report_count",
    "merge_rate",
    "recent_merge_rate",
    # Project Metrics
    "days_to_major_release",
    "open_ticket_count",
    "reviewed_lines_in_period",
    # Refactoring Metrics
    "refactoring_confidence",
    # Review Metrics
    "uncompleted_requests",
]

# 中間データ（pickle）置き場
SHARED_RANKING_CACHE_DIR = Path("data/analysis/shared_ranking")

# キャッシュファイルのプレフィックス
CACHE_FILENAME_PREFIX = "event_ranking"
