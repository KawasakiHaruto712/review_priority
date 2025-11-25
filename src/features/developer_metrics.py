import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ロギング設定の例 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_owner_email(change_data: Dict[str, Any]) -> str:
    """
    Changeデータからオーナーのメールアドレスを取得する
    
    Args:
        change_data (Dict[str, Any]): Changeデータ
        
    Returns:
        str: オーナーのメールアドレス。見つからない場合は空文字
    """
    # 1. トップレベルのownerフィールドを確認
    owner = change_data.get('owner', {})
    if isinstance(owner, dict) and owner.get('email'):
        return owner.get('email')
    
    # 2. revisionsからauthorを確認
    revisions = change_data.get('revisions', {})
    if revisions:
        # 最初に見つかったリビジョンのauthorを使用
        for rev_id, rev_data in revisions.items():
            commit = rev_data.get('commit', {})
            author = commit.get('author', {})
            if author.get('email'):
                return author.get('email')
                
    return ''

def calculate_past_report_count(
    developer_email: str, 
    all_prs_df: pd.DataFrame, 
    analysis_time: datetime
) -> int:
    """
    開発者が指定された分析時点までに過去に報告したチケット数（Change数）を計算
    
    Args:
        developer_email (str): 対象の開発者のメールアドレス．
        all_prs_df (pd.DataFrame): 全てのChange履歴を含むDataFrame
                                   'owner_email'と'created'カラムが必要
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻

    Returns:
        int: 報告されたChangeの総数
    """
    # 対象開発者のChangeをフィルタリングし，分析時点以前に作成されたものをカウント
    reported_changes = all_prs_df[
        (all_prs_df['owner_email'] == developer_email) & 
        (all_prs_df['created'] <= analysis_time)
    ]
    return len(reported_changes)

def calculate_recent_report_count(
    developer_email: str, 
    all_prs_df: pd.DataFrame, 
    analysis_time: datetime, 
    lookback_months: int = 3
) -> int:
    """
    開発者が指定された分析時点から過去指定月数以内に報告したチケット数（Change数）を計算

    Args:
        developer_email (str): 対象の開発者のメールアドレス．
        all_prs_df (pd.DataFrame): 全てのChange履歴を含むDataFrame
                                   'owner_email'と'created'カラムが必要
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻
        lookback_months (int): 過去に遡る月数（デフォルト: 3ヶ月）

    Returns:
        int: 直近で報告されたChangeの総数
    """
    # 3ヶ月前の日付を計算
    start_of_period = analysis_time - timedelta(days=30 * lookback_months) # 簡易的に30日/月

    reported_changes_recent = all_prs_df[
        (all_prs_df['owner_email'] == developer_email) & 
        (all_prs_df['created'] <= analysis_time) & 
        (all_prs_df['created'] >= start_of_period)
    ]
    return len(reported_changes_recent)

def calculate_merge_rate(
    developer_email: str, 
    all_prs_df: pd.DataFrame, 
    analysis_time: datetime
) -> float:
    """
    開発者が指定された分析時点までに報告したチケットのマージ率を計算
    
    Args:
        developer_email (str): 対象の開発者のメールアドレス．
        all_prs_df (pd.DataFrame): 全てのChange履歴を含むDataFrame
                                   'owner_email', 'created', 'merged'カラムが必要
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻

    Returns:
        float: マージ率 (0.0〜1.0) 報告実績がない場合は0.0
    """
    reported_changes = all_prs_df[
        (all_prs_df['owner_email'] == developer_email) & 
        (all_prs_df['created'] <= analysis_time)
    ]

    if len(reported_changes) == 0:
        return 0.0

    # マージされたChangeをフィルタリング (mergedカラムにタイムスタンプがあり，かつ分析時点以前にマージされている)
    merged_changes = reported_changes[
        (reported_changes['merged'].notna()) & 
        (reported_changes['merged'] <= analysis_time)
    ]

    return len(merged_changes) / len(reported_changes)

def calculate_recent_merge_rate(
    developer_email: str, 
    all_prs_df: pd.DataFrame, 
    analysis_time: datetime, 
    lookback_months: int = 3
) -> float:
    """
    開発者が指定された分析時点から過去指定月数以内に報告したチケットのマージ率を計算
    
    Args:
        developer_email (str): 対象の開発者のメールアドレス．
        all_prs_df (pd.DataFrame): 全てのChange履歴を含むDataFrame
                                   'owner_email', 'created', 'merged'カラムが必要です
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻．
        lookback_months (int): 過去に遡る月数（デフォルト: 3ヶ月）

    Returns:
        float: 直近のマージ率 (0.0〜1.0) 直近の報告実績がない場合は0.0
    """
    start_of_period = analysis_time - timedelta(days=30 * lookback_months) # 簡易的に30日/月

    reported_changes_recent = all_prs_df[
        (all_prs_df['owner_email'] == developer_email) & 
        (all_prs_df['created'] <= analysis_time) & 
        (all_prs_df['created'] >= start_of_period)
    ]

    if len(reported_changes_recent) == 0:
        return 0.0

    merged_changes_recent = reported_changes_recent[
        (reported_changes_recent['merged'].notna()) & 
        (reported_changes_recent['merged'] <= analysis_time)
    ]

    return len(merged_changes_recent) / len(reported_changes_recent)