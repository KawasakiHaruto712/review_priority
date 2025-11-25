"""
Trend Metrics Analysis用の定数定義
分析対象リリースと期間の定義
"""

# 分析対象プロジェクトとリリース
TREND_ANALYSIS_CONFIG = {
    'project': 'nova',
    'target_releases': ['20.0.0', '21.0.0'],  # [current_release, next_release]
}

# 分析期間の定義（日数）
ANALYSIS_PERIODS = {
    'early': {
        'base_date': 'current_release',  # 基準日：現在のリリース日
        'offset_start': 0,                # リリース日当日から
        'offset_end': 30,                 # 30日後まで
        'description': 'リリース直後30日間'
    },
    'late': {
        'base_date': 'next_release',      # 基準日：次のリリース日
        'offset_start': -30,              # 30日前から
        'offset_end': 0,                  # リリース日当日まで
        'description': '次リリース直前30日間'
    }
}

# レビューアタイプの定義
REVIEWER_TYPES = {
    'core_reviewed': {
        'label': 'core_reviewed',
        'description': 'コアレビューアがレビューした',
        'condition': 'has_core_review == True'
    },
    'core_not_reviewed': {
        'label': 'core_not_reviewed',
        'description': 'コアレビューアがレビューしていない',
        'condition': 'has_core_review == False'
    },
    'non_core_reviewed': {
        'label': 'non_core_reviewed',
        'description': '非コアレビューアのみがレビューした',
        'condition': 'has_core_review == False and has_non_core_review == True'
    },
    'non_core_not_reviewed': {
        'label': 'non_core_not_reviewed',
        'description': '誰もレビューしていない',
        'condition': 'has_core_review == False and has_non_core_review == False'
    }
}

# 分析グループの定義（期間 × レビューアタイプ）
ANALYSIS_GROUPS = [
    'early_core_reviewed',
    'early_core_not_reviewed',
    'early_non_core_reviewed',
    'early_non_core_not_reviewed',
    'late_core_reviewed',
    'late_core_not_reviewed',
    'late_non_core_reviewed',
    'late_non_core_not_reviewed'
]

# メトリクス一覧
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

# メトリクスのデータ範囲設定
# 'period_only': 分析期間内のChangeデータのみ使用
# 'all_data': 収集した全Changeデータを使用（開発者の累積実績評価用）
# 'recent_data': 3ヶ月前までのデータを使用
METRIC_DATA_SCOPE = {
    'bug_fix_confidence': 'period_only',
    'lines_added': 'period_only',
    'lines_deleted': 'period_only',
    'files_changed': 'period_only',
    'elapsed_time': 'period_only',
    'revision_count': 'period_only',
    'test_code_presence': 'period_only',
    'past_report_count': 'all_data',          # 全期間（開発者の累積実績）
    'recent_report_count': 'recent_data',     # 3ヶ月間
    'merge_rate': 'all_data',                 # 全期間（開発者の累積実績）
    'recent_merge_rate': 'recent_data',       # 3ヶ月間
    'days_to_major_release': 'period_only',
    'open_ticket_count': 'period_only',
    'reviewed_lines_in_period': 'period_only',
    'refactoring_confidence': 'period_only',
    'uncompleted_requests': 'period_only'
}

# recent_dataで使用する期間（日数）
RECENT_DATA_PERIOD_DAYS = 90  # 3ヶ月

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
OUTPUT_DIR_BASE = 'data/analysis/trend_metrics'

# 統計検定の設定
STATISTICAL_TEST_CONFIG = {
    'alpha': 0.05,                    # 有意水準
    'use_bonferroni': True,           # Bonferroni補正を使用するか
    'test_method': 'mann_whitney_u',  # 使用する検定手法
    'effect_size_method': 'cohen_d'   # 効果量の計算方法
}

# データ読み込み設定
DATA_LOAD_CONFIG = {
    'major_releases_file': 'data/openstack/major_releases_summary.csv',
    'core_developers_file': 'data/openstack_collected/core_developers.json',
    'changes_dir_template': 'data/openstack_collected/{project}/changes',
    'bot_config_file': 'src/config/gerrymanderconfig.ini'
}

# 可視化設定
VISUALIZATION_CONFIG = {
    'figure_size': (16, 12),
    'dpi': 300,
    'color_palette': 'Set2',
    'boxplot_grid_size': (2, 4),  # 8グループを2行×4列で表示
    'save_format': 'pdf'
}
