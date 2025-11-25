"""
レビューアタイプ分類モジュール
Changeをコアレビューア・非コアレビューアの分類で振り分ける
"""

import logging
from typing import Dict, List

from src.analysis.trend_metrics.utils.core_reviewer_checker import get_project_core_reviewers, is_core_reviewer

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
                   常に以下の組み合わせで返される：
                   - 'core_reviewed' または 'core_not_reviewed'
                   - 'non_core_reviewed' または 'non_core_not_reviewed'
    
    重要な注意点:
    - botのレビューは除外済み（period_extractorで除外）
    - コアレビューアの判定は、分析対象プロジェクトのコアレビューアリストのみを使用
    - レビューアのメールアドレスで判定
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
    
    return result


def classify_changes_into_groups(
    early_changes: List[Dict],
    late_changes: List[Dict],
    core_reviewers_data: Dict,
    project_name: str
) -> Dict[str, List[Dict]]:
    """
    前期・後期のChangeを8グループに分類
    
    Args:
        early_changes: 前期のChangeリスト
        late_changes: 後期のChangeリスト
        core_reviewers_data: コアレビューア情報
        project_name: プロジェクト名
    
    Returns:
        Dict[str, List[Dict]]: 8グループに分類されたChange
    """
    groups = {
        'early_core_reviewed': [],
        'early_core_not_reviewed': [],
        'early_non_core_reviewed': [],
        'early_non_core_not_reviewed': [],
        'late_core_reviewed': [],
        'late_core_not_reviewed': [],
        'late_non_core_reviewed': [],
        'late_non_core_not_reviewed': []
    }
    
    # 前期の分類
    for change in early_changes:
        reviewer_types = classify_by_reviewer_type(change, core_reviewers_data, project_name)
        for reviewer_type in reviewer_types:
            group_key = f'early_{reviewer_type}'
            groups[group_key].append(change)
    
    # 後期の分類
    for change in late_changes:
        reviewer_types = classify_by_reviewer_type(change, core_reviewers_data, project_name)
        for reviewer_type in reviewer_types:
            group_key = f'late_{reviewer_type}'
            groups[group_key].append(change)
    
    # ログ出力
    logger.info("=" * 60)
    logger.info("Changeの分類結果:")
    for group_name, changes in groups.items():
        logger.info(f"  {group_name}: {len(changes)} 件")
    logger.info("=" * 60)
    
    return groups
