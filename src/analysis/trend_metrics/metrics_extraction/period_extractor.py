"""
期間別データ抽出モジュール
指定期間内のChangeを抽出し、レビューア情報を付与する
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd

from src.analysis.trend_metrics.utils.data_loader import is_bot

logger = logging.getLogger(__name__)


def calculate_periods(
    current_release_date: pd.Timestamp,
    next_release_date: pd.Timestamp,
    period_config: Dict
) -> Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]:
    """
    前期・後期の期間を計算
    
    Args:
        current_release_date: 現在のリリース日
        next_release_date: 次のリリース日
        period_config: 期間設定（ANALYSIS_PERIODS）
    
    Returns:
        Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]: 
            (前期の開始・終了, 後期の開始・終了)
    """
    # 前期（early）の計算
    early_config = period_config['early']
    early_start = current_release_date + timedelta(days=early_config['offset_start'])
    early_end = current_release_date + timedelta(days=early_config['offset_end'])
    
    # 後期（late）の計算
    late_config = period_config['late']
    late_start = next_release_date + timedelta(days=late_config['offset_start'])
    late_end = next_release_date + timedelta(days=late_config['offset_end'])
    
    logger.info(f"前期: {early_start} ~ {early_end}")
    logger.info(f"後期: {late_start} ~ {late_end}")
    
    return (early_start, early_end), (late_start, late_end)


def extract_changes_in_period(
    all_changes: List[Dict],
    period: Tuple[datetime, datetime],
    next_release_date: pd.Timestamp = None
) -> List[Dict]:
    """
    指定期間内にopenだったChangeを抽出
    
    期間内にopenだったことのあるChangeを対象とする:
    - created < end_time (期間終了前に作成された)
    - created >= next_release_date - 1 year (次のリリースから1年以内に作成された)
    - AND (merged/abandoned > start_time OR status == 'NEW'/'MERGED') (期間開始後にクローズされた、またはまだopen)
    - AND (merged/abandoned <= next_release_date) (次のリリース日までに判断が下された)
    
    Args:
        all_changes: 全Changeデータ
        period: 期間（開始日時, 終了日時）
        next_release_date: 次のリリース日（1年以内フィルタ用、および判断日フィルタ用）
    
    Returns:
        List[Dict]: 期間内にopenだったChangeリスト
    """
    start_date, end_date = period
    period_changes = []
    
    # 次のリリース日から1年前の日時を計算
    min_created_date = None
    if next_release_date is not None:
        min_created_date = next_release_date - timedelta(days=365)
    
    for change in all_changes:
        created_str = change.get('created')
        if not created_str:
            continue
        
        try:
            # ISO形式の日時文字列をパース
            created_date = pd.to_datetime(created_str)
            
            # 条件0: 次のリリース日から1年以内に作成された
            if min_created_date is not None and created_date < min_created_date:
                continue
            
            # 条件1: 期間終了前に作成された
            if created_date >= end_date:
                continue
            
            # 条件2: 期間開始後にクローズされた、またはまだopen
            # submittedまたはupdatedを確認（MERGEDの場合はsubmitted、ABANDONEDの場合はupdated）
            status = change.get('status', '')
            
            # MERGEDまたはABANDONEDの場合、クローズ日時を確認
            close_date = None
            
            if status == 'MERGED':
                submitted_str = change.get('submitted')
                if submitted_str:
                    close_date = pd.to_datetime(submitted_str)
            elif status == 'ABANDONED':
                # ABANDONEDの場合、updatedをクローズ日時として使用
                updated_str = change.get('updated')
                if updated_str:
                    close_date = pd.to_datetime(updated_str)
            
            # 追加条件: 次のリリース日までに判断が下されたChangeのみを対象とする
            if next_release_date is not None:
                # NEWは判断が下されていないので除外
                if status == 'NEW':
                    continue
                # 判断日が次のリリース日より後の場合は除外
                if close_date and close_date > next_release_date:
                    continue

            # NEWの場合は期間内にopenだった（next_release_dateがない場合のみここに来る）
            if status == 'NEW':
                period_changes.append(change)
                continue
            
            # クローズ日時が期間開始より後なら、期間内にopenだった
            if close_date and close_date > start_date:
                period_changes.append(change)
            elif not close_date:
                # クローズ日時が不明な場合も含める（安全側に倒す）
                period_changes.append(change)
                
        except Exception as e:
            logger.warning(f"日付解析エラー: {created_str}, エラー: {e}")
            continue
    
    logger.info(f"期間内にopenだったChange: {len(period_changes)} 件 ({start_date} ~ {end_date})")
    
    return period_changes


def extract_reviewers_from_messages(
    messages: List[Dict], 
    bot_names: List[str],
    period: Tuple[datetime, datetime] = None
) -> List[str]:
    """
    メッセージからレビューアのメールアドレスを抽出（bot除外）
    
    Args:
        messages: Changeのメッセージリスト
        bot_names: 除外するbot名のリスト
        period: 期間指定（開始日時, 終了日時）。指定された場合、期間内のレビューのみを抽出
    
    Returns:
        List[str]: レビューアのメールアドレスリスト（bot除外済み、重複なし）
    """
    reviewers = set()
    
    for message in messages:
        author = message.get('author', {})
        author_name = author.get('name', '')
        author_email = author.get('email', '')
        
        # 期間指定がある場合は期間内のメッセージのみを対象
        if period is not None:
            message_date_str = message.get('date')
            if message_date_str:
                try:
                    message_date = pd.to_datetime(message_date_str)
                    start_date, end_date = period
                    if not (start_date <= message_date < end_date):
                        continue
                except Exception:
                    continue
        
        # メールアドレスが存在し、botでない場合のみレビューアとして追加
        if author_email and not is_bot(author_name, bot_names):
            reviewers.add(author_email)
    
    return list(reviewers)


def add_reviewer_info_to_changes(
    changes: List[Dict],
    bot_names: List[str],
    period: Tuple[datetime, datetime]
) -> List[Dict]:
    """
    各Changeにレビューア情報を追加
    
    Args:
        changes: Changeリスト
        bot_names: bot名のリスト
        period: 期間（開始日時, 終了日時）。この期間内のレビューのみを抽出
    
    Returns:
        List[Dict]: レビューア情報付きChangeリスト
    """
    for change in changes:
        messages = change.get('messages', [])
        reviewers = extract_reviewers_from_messages(messages, bot_names, period)
        change['reviewers'] = reviewers
    
    logger.info(f"{len(changes)} 件のChangeにレビューア情報を追加しました")
    
    return changes


def get_changes_for_metric_calculation(
    all_changes: List[Dict],
    period_changes: List[Dict],
    metric_name: str,
    metric_data_scope: Dict,
    period_start: datetime,
    recent_period_days: int = 90
) -> List[Dict]:
    """
    メトリクス計算用のChangeデータを取得
    メトリクスのデータ範囲設定に応じて適切なデータを返す
    
    Args:
        all_changes: 全Changeデータ
        period_changes: 期間内のChangeデータ
        metric_name: メトリクス名
        metric_data_scope: メトリクスのデータ範囲設定
        period_start: 期間の開始日時
        recent_period_days: recent_dataで使用する日数
    
    Returns:
        List[Dict]: メトリクス計算用のChangeデータ
    """
    scope = metric_data_scope.get(metric_name, 'period_only')
    
    if scope == 'period_only':
        return period_changes
    elif scope == 'all_data':
        return all_changes
    elif scope == 'recent_data':
        # 期間開始日からrecent_period_days日前までのデータ
        recent_start = period_start - timedelta(days=recent_period_days)
        recent_changes = []
        
        for change in all_changes:
            created_str = change.get('created')
            if not created_str:
                continue
            
            try:
                created_date = pd.to_datetime(created_str)
                if recent_start <= created_date < period_start:
                    recent_changes.append(change)
            except Exception:
                continue
        
        return recent_changes + period_changes
    else:
        return period_changes
