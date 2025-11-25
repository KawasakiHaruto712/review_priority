"""
Analysis パッケージ
統計分析とトレンド比較
"""

from src.analysis.trend_metrics.analysis.statistical_analyzer import (
    calculate_summary_statistics,
    mann_whitney_u_test,
    calculate_effect_size_cohens_d,
    perform_pairwise_tests,
    perform_all_statistical_tests,
    summarize_significant_results,
    interpret_effect_size
)

from src.analysis.trend_metrics.analysis.trend_comparator import (
    extract_period_from_group_name,
    extract_reviewer_type_from_group_name,
    calculate_metric_change,
    compare_early_vs_late_by_reviewer_type,
    identify_diverging_trends,
    calculate_group_size_changes,
    generate_trend_summary
)

__all__ = [
    # statistical_analyzer
    'calculate_summary_statistics',
    'mann_whitney_u_test',
    'calculate_effect_size_cohens_d',
    'perform_pairwise_tests',
    'perform_all_statistical_tests',
    'summarize_significant_results',
    'interpret_effect_size',
    # trend_comparator
    'extract_period_from_group_name',
    'extract_reviewer_type_from_group_name',
    'calculate_metric_change',
    'compare_early_vs_late_by_reviewer_type',
    'identify_diverging_trends',
    'calculate_group_size_changes',
    'generate_trend_summary',
]
