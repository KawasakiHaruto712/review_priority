"""
Daily Regression Analysis 用の定数定義
"""

# 分析対象プロジェクトとバージョン
DAILY_REGRESSION_CONFIG = {
    'project': {
        'nova': [
            '2015.1.0',
            '12.0.0',
            '13.0.0',
            '14.0.0',
            '15.0.0',
            '16.0.0',
            '17.0.0',
            '18.0.0',
            '19.0.0',
            '20.0.0',
        ]
    }
}

# 打ち切り最大値（秒）: 1年 = 365日 × 86400秒
MAX_CENSORING_SECONDS = 31_536_000

# 除外ウィンドウ（秒）: 1年以上レビューなしは除外
EXCLUSION_WINDOW_SECONDS = 31_536_000

# 日ごとの最小サンプル数
MIN_SAMPLES = 2

# 回帰の目的変数設定
# - rank: レビュー優先順位（1が最優先）
# - time: 次レビューまでの秒数
TARGET_COLUMN_BY_MODE = {
    'rank': 'review_priority_rank',
    'time': 'time_to_review_seconds',
}

# デフォルトは順位回帰
DEFAULT_TARGET_MODE = 'rank'
DEFAULT_TARGET_COLUMN = TARGET_COLUMN_BY_MODE[DEFAULT_TARGET_MODE]

# メトリクス一覧（src/features で抽出する16種類）
METRIC_COLUMNS = [
    'bug_fix_confidence',
    'lines_added',
    'lines_deleted',
    'files_changed',
    'elapsed_time',
    'revision_count',
    'test_code_presence',
    'past_report_count',
    'recent_report_count',
    'merge_rate',
    'recent_merge_rate',
    'days_to_major_release',
    'open_ticket_count',
    'reviewed_lines_in_period',
    'refactoring_confidence',
    'uncompleted_requests'
]

# メトリクスの表示名
METRIC_DISPLAY_NAMES = {
    'bug_fix_confidence': 'Bug Fix Confidence',
    'lines_added': 'Lines Added',
    'lines_deleted': 'Lines Deleted',
    'files_changed': 'Files Changed',
    'elapsed_time': 'Elapsed Time',
    'revision_count': 'Revision Count',
    'test_code_presence': 'Test Code Presence',
    'past_report_count': 'Past Report Count',
    'recent_report_count': 'Recent Report Count',
    'merge_rate': 'Merge Rate',
    'recent_merge_rate': 'Recent Merge Rate',
    'days_to_major_release': 'Days to Major Release',
    'open_ticket_count': 'Open Ticket Count',
    'reviewed_lines_in_period': 'Reviewed Lines in Period',
    'refactoring_confidence': 'Refactoring Confidence',
    'uncompleted_requests': 'Uncompleted Requests'
}

# 出力ディレクトリのベースパス
OUTPUT_DIR_BASE = 'data/analysis/daily_regression'

# データ読み込み設定
DATA_LOAD_CONFIG = {
    'major_releases_file': 'data/openstack/major_releases_summary.csv',
    'core_developers_file': 'data/openstack_collected/core_developers.json',
    'changes_dir_template': 'data/openstack_collected/{project}/changes',
    'bot_config_file': 'src/config/gerrymanderconfig.ini'
}

# 可視化設定
VISUALIZATION_CONFIG = {
    'figure_size': (16, 8),
    'dpi': 300,
    'save_format': 'png',
    'significance_level': 0.05,
    'line_color': '#1f77b4',
    'line_alpha': 0.8,
    'line_width': 1.0,
}
