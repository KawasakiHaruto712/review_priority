"""
レビューアタイプ分類モジュール
Changeをコアレビューア・非コアレビューアの分類で振り分ける
"""

import logging
from typing import Dict, List

from src.analysis.trend_metrics.utils.core_reviewer_checker import get_project_core_reviewers, is_core_reviewer
from src.analysis.trend_metrics.utils.constants import SEPARATE_CORE_REVIEWERS, ANALYSIS_GROUPS

logger = logging.getLogger(__name__)


def classify_by_reviewer_type(
    change: Dict,
    core_reviewers_data: Dict,
    project_name: str
) -> List[str]:
    """
    Changeをレビューアタイプで分類
    
    Args:
        change: Changeデータ（'reviewers'キーを持つ）
        core_reviewers_data: コアレビューア情報（全プロジェクト）
        project_name: 分析対象プロジェクト名
    
    Returns:
        List[str]: レビューアタイプのリスト
    """
    reviewer_emails = change.get('reviewers', [])
    
    # コアレビューアと非コアレビューアをそれぞれチェック
    has_core_reviewer = False
    has_non_core_reviewer = False
    
    for email in reviewer_emails:
        if is_core_reviewer(email, project_name, core_reviewers_data):
            has_core_reviewer = True
        else:
            has_non_core_reviewer = True
    
    # レビューアタイプを決定
    result = []
    
    if SEPARATE_CORE_REVIEWERS:
        # コアレビューアの観点
        if has_core_reviewer:
            result.append('core_reviewed')
        else:
            result.append('core_not_reviewed')
            
        # 非コアレビューアの観点
        if has_non_core_reviewer:
            result.append('non_core_reviewed')
        else:
            result.append('non_core_not_reviewed')
    else:
        # 単純にレビューされたかどうか
        if has_core_reviewer or has_non_core_reviewer:
            result.append('reviewed')
        else:
            result.append('not_reviewed')
    
    return result


def classify_changes_into_groups(
    early_changes: List[Dict],
    late_changes: List[Dict],
    core_reviewers_data: Dict,
    project_name: str
) -> Dict[str, List[Dict]]:
    """
    前期・後期のChangeをグループに分類
    
    Args:
        early_changes: 前期のChangeリスト
        late_changes: 後期のChangeリスト
        core_reviewers_data: コアレビューア情報
        project_name: プロジェクト名
    
    Returns:
        Dict[str, List[Dict]]: 分類されたChange
    """
    # ANALYSIS_GROUPSに基づいて初期化
    groups = {group: [] for group in ANALYSIS_GROUPS}
    
    # 前期の分類
    for change in early_changes:
        reviewer_types = classify_by_reviewer_type(change, core_reviewers_data, project_name)
        for reviewer_type in reviewer_types:
            group_key = f"early_{reviewer_type}"
            if group_key in groups:
                groups[group_key].append(change)
            
    # 後期の分類
    for change in late_changes:
        reviewer_types = classify_by_reviewer_type(change, core_reviewers_data, project_name)
        for reviewer_type in reviewer_types:
            group_key = f"late_{reviewer_type}"
            if group_key in groups:
                groups[group_key].append(change)
                
    return groups

    logger.info("=" * 60)
    
    return groups
