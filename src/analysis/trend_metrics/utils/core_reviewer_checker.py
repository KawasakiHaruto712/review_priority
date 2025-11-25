"""
コアレビューア判定モジュール
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def is_core_reviewer(reviewer_email: str, project_name: str, core_reviewers_data: Dict) -> bool:
    """
    レビューアがコアレビューアかどうかを判定
    
    Args:
        reviewer_email: レビューアのメールアドレス
        project_name: 分析対象プロジェクト名（例: 'nova'）
        core_reviewers_data: core_developers.jsonから読み込んだ全データ
    
    Returns:
        bool: コアレビューアの場合True
    
    重要な注意点:
    - core_developers.jsonはプロジェクトごとに構造化されている
    - 例: {'nova': {'core_reviewers': [...]}, 'neutron': {'core_reviewers': [...]}}
    - 分析対象プロジェクトのリストのみを参照すること
    - 他プロジェクトのコアレビューアは対象外
    """
    if not reviewer_email:
        return False
    
    # core_developers.jsonの構造: {'project': {'nova': {'members': [{'name': ..., 'email': ...}]}}}
    project_data = core_reviewers_data.get('project', {}).get(project_name, {})
    members = project_data.get('members', [])
    core_emails = [member.get('email', '') for member in members if member.get('email')]
    
    # メールアドレスでマッチング
    return reviewer_email in core_emails


def get_project_core_reviewers(core_reviewers_data: Dict, project_name: str) -> List[str]:
    """
    指定プロジェクトのコアレビューアリストを取得
    
    Args:
        core_reviewers_data: core_developers.jsonから読み込んだ全データ
        project_name: プロジェクト名
    
    Returns:
        List[str]: コアレビューアメールアドレスのリスト
    """
    # core_developers.jsonの構造: {'project': {'nova': {'members': [{'name': ..., 'email': ...}]}}}
    project_data = core_reviewers_data.get('project', {}).get(project_name, {})
    members = project_data.get('members', [])
    core_reviewers = [member.get('email', '') for member in members if member.get('email')]
    
    logger.info(f"{project_name}のコアレビューア: {len(core_reviewers)} 名")
    
    return core_reviewers
