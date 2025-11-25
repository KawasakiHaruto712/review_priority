"""
メトリクス計算モジュール
各Changeに対して16種類のメトリクスを計算する
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.features.change_metrics import (
    calculate_lines_added,
    calculate_lines_deleted,
    calculate_files_changed,
    calculate_elapsed_time,
    calculate_revision_count,
    check_test_code_presence
)
from src.features.bug_metrics import calculate_bug_fix_confidence
from src.features.refactoring_metrics import calculate_refactoring_confidence
from src.features.developer_metrics import (
    calculate_past_report_count,
    calculate_recent_report_count,
    calculate_merge_rate,
    calculate_recent_merge_rate,
    get_owner_email
)
from src.features.project_metrics import (
    calculate_days_to_major_release,
    calculate_predictive_target_ticket_count,
    calculate_reviewed_lines_in_period
)
from src.features.review_metrics import ReviewStatusAnalyzer
from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# ReviewStatusAnalyzerのインスタンスを保持（再利用のため）
_review_analyzer = None

def get_review_analyzer():
    global _review_analyzer
    if _review_analyzer is None:
        keywords_path = DEFAULT_DATA_DIR / "processed" / "review_keywords.json"
        label_path = DEFAULT_DATA_DIR / "processed" / "review_label.json"
        config_path = DEFAULT_CONFIG / "gerrymanderconfig.ini"
        
        if keywords_path.exists() and label_path.exists():
            _review_analyzer = ReviewStatusAnalyzer(keywords_path, config_path, label_path)
        else:
            logger.warning("Review metrics configuration files not found.")
            _review_analyzer = None
    return _review_analyzer

def calculate_metrics(
    change: Dict[str, Any],
    all_changes_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    project_name: str,
    period_start: datetime
) -> Dict[str, Any]:
    """
    Changeに対する全メトリクスを計算する
    
    Args:
        change: 対象のChangeデータ
        all_changes_df: 全Change履歴（開発者メトリクス計算用）
        releases_df: リリース情報（プロジェクトメトリクス計算用）
        project_name: プロジェクト名
        period_start: 分析期間の開始日時
        
    Returns:
        Dict: メトリクス名と値の辞書
    """
    metrics = {}
    
    # 分析時点の決定
    # 基本はChangeの作成日時だが、期間開始時点で既にOpenだった場合は期間開始日時を基準とする
    created_str = change.get('created')
    if not created_str:
        return metrics
        
    try:
        created_dt = pd.to_datetime(created_str).to_pydatetime()
        
        # Change作成日時と期間開始日時の遅い方を分析時点とする
        # created < period_start の場合 -> period_start (期間開始時点でOpen)
        # created >= period_start の場合 -> created (期間中にOpen)
        if created_dt < period_start:
            analysis_time = period_start
        else:
            analysis_time = created_dt
            
    except Exception:
        return metrics

    # 1. Change Metrics
    metrics['lines_added'] = calculate_lines_added(change, analysis_time)
    metrics['lines_deleted'] = calculate_lines_deleted(change, analysis_time)
    metrics['files_changed'] = calculate_files_changed(change, analysis_time)
    metrics['elapsed_time'] = calculate_elapsed_time(change, analysis_time)
    metrics['revision_count'] = calculate_revision_count(change, analysis_time)
    metrics['test_code_presence'] = check_test_code_presence(change)

    # 2. Bug Metrics
    metrics['bug_fix_confidence'] = calculate_bug_fix_confidence(change)

    # 3. Refactoring Metrics
    metrics['refactoring_confidence'] = calculate_refactoring_confidence(change)

    # 4. Developer Metrics
    owner_email = get_owner_email(change)
    if owner_email:
        metrics['past_report_count'] = calculate_past_report_count(owner_email, all_changes_df, analysis_time)
        metrics['recent_report_count'] = calculate_recent_report_count(owner_email, all_changes_df, analysis_time)
        metrics['merge_rate'] = calculate_merge_rate(owner_email, all_changes_df, analysis_time)
        metrics['recent_merge_rate'] = calculate_recent_merge_rate(owner_email, all_changes_df, analysis_time)
    else:
        metrics['past_report_count'] = 0
        metrics['recent_report_count'] = 0
        metrics['merge_rate'] = 0.0
        metrics['recent_merge_rate'] = 0.0

    # 5. Project Metrics
    metrics['days_to_major_release'] = calculate_days_to_major_release(analysis_time, project_name, releases_df)
    metrics['open_ticket_count'] = calculate_predictive_target_ticket_count(all_changes_df, analysis_time)
    metrics['reviewed_lines_in_period'] = calculate_reviewed_lines_in_period(all_changes_df, analysis_time)

    # 6. Review Metrics
    analyzer = get_review_analyzer()
    if analyzer:
        # 辞書をコピーして渡す（副作用防止）
        change_copy = change.copy()
        metrics['uncompleted_requests'] = analyzer.analyze_pr_status(change_copy, analysis_time)
    else:
        metrics['uncompleted_requests'] = 0

    return metrics

def enrich_changes_with_line_metrics(changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Changeリストに対して行数情報（lines_added, lines_deleted）を付与する
    DataFrame作成前の前処理として使用
    
    Args:
        changes: Changeデータのリスト
        
    Returns:
        List[Dict]: 行数情報が付与されたChangeリスト
    """
    for change in changes:
        if 'lines_added' not in change:
            change['lines_added'] = calculate_lines_added(change)
        if 'lines_deleted' not in change:
            change['lines_deleted'] = calculate_lines_deleted(change)
    return changes

def enrich_changes_with_owner_email(changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Changeリストに対してowner_emailを付与する
    DataFrame作成前の前処理として使用
    
    Args:
        changes: Changeデータのリスト
        
    Returns:
        List[Dict]: owner_emailが付与されたChangeリスト
    """
    for change in changes:
        if 'owner_email' not in change:
            change['owner_email'] = get_owner_email(change)
    return changes
