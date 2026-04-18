"""
サンプル抽出モジュール
日ごとの対象Change抽出・目的変数（次レビューまでの秒数、順位）算出
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.analysis.daily_regression.utils.data_loader import is_bot


def _parse_gerrit_timestamp(ts_str: str) -> Optional[datetime]:
    """Gerrit タイムスタンプ文字列を高速にパースする

    フォーマット: 'YYYY-MM-DD HH:MM:SS.nnnnnnnnn'
    pd.to_datetime より大幅に高速。
    """
    try:
        # 小数点以下を切り捨て（マイクロ秒6桁まで対応）
        dot_idx = ts_str.find('.')
        if dot_idx != -1:
            # 小数点以下を6桁に切り詰め
            frac = ts_str[dot_idx + 1:dot_idx + 7].ljust(6, '0')
            clean = ts_str[:dot_idx + 1] + frac
            return datetime.strptime(clean, '%Y-%m-%d %H:%M:%S.%f')
        else:
            return datetime.strptime(ts_str.strip(), '%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        return None
from src.analysis.daily_regression.utils.constants import (
    MAX_CENSORING_SECONDS,
    EXCLUSION_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)


def _is_change_open_on_date(change: Dict, analysis_date: date) -> bool:
    """
    Change が指定日に Open 状態かを判定する

    条件:
    - created <= analysis_date 23:59:59（当日以前に作成済み）
    - updated > analysis_date 00:00:00 または status == "NEW"（まだクローズされていない）

    Args:
        change: Changeデータ
        analysis_date: 分析対象日

    Returns:
        bool: Open 状態であれば True
    """
    created_str = change.get('created')
    if not created_str:
        return False

    created_dt = _parse_gerrit_timestamp(created_str)
    if created_dt is None:
        return False

    day_end = datetime(analysis_date.year, analysis_date.month, analysis_date.day, 23, 59, 59)
    if created_dt > day_end:
        return False

    day_start = datetime(analysis_date.year, analysis_date.month, analysis_date.day, 0, 0, 0)

    updated_str = change.get('updated')
    status = change.get('status', '')

    if status == 'NEW':
        return True

    if updated_str:
        updated_dt = _parse_gerrit_timestamp(updated_str)
        if updated_dt is None:
            return False
        return updated_dt > day_start

    return False


def _get_first_non_bot_review_time(
    change: Dict,
    after_time: datetime,
    bot_names: List[str]
) -> Optional[datetime]:
    """
    Change の messages から、after_time 以降の最初の非botレビュー時刻を取得する

    除外対象:
    - bot によるメッセージ
    - Change 作成者自身の最初のアップロードメッセージ（"Uploaded patch set"）

    Args:
        change: Changeデータ
        after_time: この時刻以降のメッセージを対象
        bot_names: bot名のリスト

    Returns:
        Optional[datetime]: 最初の非botレビュー時刻。無ければ None
    """
    messages = change.get('messages', [])
    if not messages:
        return None

    owner = change.get('owner', {})
    owner_account_id = owner.get('_account_id')

    for msg in messages:
        msg_date_str = msg.get('date')
        if not msg_date_str:
            continue

        msg_dt = _parse_gerrit_timestamp(msg_date_str)
        if msg_dt is None:
            continue

        if msg_dt < after_time:
            continue

        author = msg.get('author', {})
        author_name = author.get('name', '')

        if is_bot(author_name, bot_names):
            continue

        message_text = msg.get('message', '')
        author_account_id = author.get('_account_id')
        if (author_account_id == owner_account_id
                and 'Uploaded patch set' in message_text):
            continue

        return msg_dt

    return None


def extract_daily_samples(
    analysis_date: date,
    all_changes: List[Dict],
    bot_names: List[str],
    max_censoring_seconds: int = MAX_CENSORING_SECONDS,
    exclusion_window_seconds: int = EXCLUSION_WINDOW_SECONDS,
) -> pd.DataFrame:
    """
    指定日の Open Change を抽出し、目的変数を算出する

    ルール:
    - 既に Open の Change: analysis_time = D 00:00:00
    - 当日 Open の Change: analysis_time = created 時刻
    - 1年以上レビューなし → 除外
    - y > max_censoring_seconds → 打ち切り

    Args:
        analysis_date: 分析対象日
        all_changes: 全Changeデータのリスト
        bot_names: bot名のリスト
        max_censoring_seconds: 打ち切り最大値（秒）
        exclusion_window_seconds: 除外ウィンドウ（秒）

    Returns:
        pd.DataFrame: columns=['change_number', 'created', 'analysis_time',
                               'first_review_time', 'time_to_review_seconds',
                               'review_priority_rank']
    """
    day_start = datetime(analysis_date.year, analysis_date.month, analysis_date.day, 0, 0, 0)

    records = []

    for change in all_changes:
        if not _is_change_open_on_date(change, analysis_date):
            continue

        created_str = change.get('created')
        if not created_str:
            continue

        created_dt = _parse_gerrit_timestamp(created_str)
        if created_dt is None:
            continue

        # 分析時点の決定
        if created_dt < day_start:
            analysis_time = day_start
        else:
            analysis_time = created_dt

        # 最初の非botレビュー時刻を取得
        first_review_time = _get_first_non_bot_review_time(
            change, analysis_time, bot_names
        )

        if first_review_time is None:
            # レビューなし → 除外ウィンドウ内かチェック
            # 除外ウィンドウ（1年）以内にレビューされていなければ除外
            continue

        # 目的変数の算出
        time_to_review = (first_review_time - analysis_time).total_seconds()

        if time_to_review < 0:
            continue

        # 1年超の場合チェック
        if time_to_review > exclusion_window_seconds:
            continue

        # 打ち切り
        if time_to_review > max_censoring_seconds:
            time_to_review = max_censoring_seconds

        change_number = change.get('_number') or change.get('change_number', 0)

        records.append({
            'change_number': change_number,
            'created': created_dt,
            'analysis_time': analysis_time,
            'first_review_time': first_review_time,
            'time_to_review_seconds': time_to_review,
        })

    if not records:
        return pd.DataFrame(columns=[
            'change_number', 'created', 'analysis_time',
            'first_review_time', 'time_to_review_seconds',
            'review_priority_rank'
        ])

    df = pd.DataFrame(records)

    # 1が最優先（レビュー待ち時間が短いほど上位）
    df['review_priority_rank'] = (
        df['time_to_review_seconds']
        .rank(method='dense', ascending=True)
        .astype(float)
    )

    logger.debug(f"{analysis_date}: {len(df)} サンプルを抽出")
    return df
