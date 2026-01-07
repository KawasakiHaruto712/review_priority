"""
Trend Models Analysis - データローダー

本モジュールでは、分析に必要なデータの読み込みと期間フィルタリング機能を提供します。
"""

import os
import json
import logging
import configparser
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import pandas as pd

from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG
from src.analysis.trend_models.utils.constants import (
    ANALYSIS_PERIODS,
    PERIOD_DURATION_DAYS,
)

logger = logging.getLogger(__name__)

# ロギング設定
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _parse_datetime(dt_str: Any) -> Optional[datetime]:
    """
    日時文字列をdatetimeオブジェクトに変換するヘルパー関数
    
    Args:
        dt_str: 日時文字列またはdatetimeオブジェクト
        
    Returns:
        datetime: 変換されたdatetimeオブジェクト、失敗時はNone
    """
    if isinstance(dt_str, datetime):
        return dt_str
    if not dt_str:
        return None
    try:
        # OpenStack Gerritのタイムスタンプ形式を処理
        dt_str = str(dt_str).replace('.000000000', '').replace(' ', 'T')
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


def load_major_releases_summary(data_dir: Path = None) -> pd.DataFrame:
    """
    メジャーリリースサマリーCSVを読み込む
    
    Args:
        data_dir: データディレクトリのパス
        
    Returns:
        pd.DataFrame: リリース情報のDataFrame
            - component: プロジェクト名
            - version: バージョン番号
            - release_date: リリース日（datetime型）
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    csv_path = data_dir / 'openstack' / 'major_releases_summary.csv'
    
    if not csv_path.exists():
        logger.error(f"リリースサマリーファイルが見つかりません: {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # カラム名をリネーム（projectをcomponentに）
    if 'project' in df.columns:
        df = df.rename(columns={'project': 'component'})
    
    # release_dateをdatetime型に変換
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    logger.info(f"リリースサマリーをロード: {len(df)}件")
    return df


def load_all_changes(
    project_name: str,
    data_dir: Path = None,
    use_collected: bool = False
) -> List[Dict[str, Any]]:
    """
    指定プロジェクトの全Changeデータを読み込む
    
    Args:
        project_name: プロジェクト名（例: 'nova', 'neutron'）
        data_dir: データディレクトリのパス
        use_collected: 収集済みデータを使用するかどうか
        
    Returns:
        List[Dict]: Changeデータのリスト
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    if use_collected:
        changes_dir = data_dir / 'openstack_collected' / project_name / 'changes'
    else:
        changes_dir = data_dir / 'openstack' / project_name / 'changes'
    
    if not changes_dir.exists():
        logger.error(f"Changesディレクトリが見つかりません: {changes_dir}")
        return []
    
    changes = []
    change_files = list(changes_dir.glob('change_*.json'))
    
    for change_file in change_files:
        try:
            with open(change_file, 'r', encoding='utf-8') as f:
                change_data = json.load(f)
                changes.append(change_data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Changeファイルの読み込みに失敗: {change_file}, エラー: {e}")
    
    logger.info(f"プロジェクト '{project_name}' のChangeをロード: {len(changes)}件")
    return changes


def load_core_developers(
    project_name: str = None,
    data_dir: Path = None
) -> Dict[str, List[Dict[str, str]]]:
    """
    コア開発者情報を読み込む
    
    Args:
        project_name: プロジェクト名（Noneの場合は全プロジェクト）
        data_dir: データディレクトリのパス
        
    Returns:
        Dict: プロジェクト名をキー、メンバー情報リストを値とする辞書
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    json_path = data_dir / 'openstack_collected' / 'core_developers.json'
    
    if not json_path.exists():
        logger.warning(f"コア開発者ファイルが見つかりません: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"コア開発者ファイルの読み込みに失敗: {e}")
        return {}
    
    projects = data.get('project', {})
    
    if project_name:
        if project_name in projects:
            return {project_name: projects[project_name].get('members', [])}
        else:
            logger.warning(f"プロジェクト '{project_name}' がコア開発者データに見つかりません")
            return {}
    
    # 全プロジェクトの場合
    result = {}
    for proj, proj_data in projects.items():
        result[proj] = proj_data.get('members', [])
    
    return result


def load_bot_names_from_config(config_path: Path = None) -> List[str]:
    """
    gerrymanderconfig.iniからボット名のリストを読み込む
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        List[str]: ボット名のリスト
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG / 'gerrymanderconfig.ini'
    
    bot_names = []
    config = configparser.ConfigParser()
    
    try:
        config.read(config_path)
        if 'organization' in config and 'bots' in config['organization']:
            bot_names = [name.strip() for name in config['organization']['bots'].split(',')]
            logger.info(f"ボット名をロード: {len(bot_names)}件")
        else:
            logger.warning("gerrymanderconfig.iniに'organization'セクションまたは'bots'エントリが見つかりません")
    except configparser.Error as e:
        logger.error(f"設定ファイルのパース中にエラー: {e}")
    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {config_path}")
    
    return bot_names


def get_period_dates(
    current_release_date: datetime,
    next_release_date: datetime,
    period_type: str
) -> Tuple[datetime, datetime]:
    """
    期間タイプに応じた開始日・終了日を計算する
    
    Args:
        current_release_date: 現在のリリース日
        next_release_date: 次のリリース日
        period_type: 期間タイプ（'early', 'late', 'all'）
        
    Returns:
        Tuple[datetime, datetime]: (開始日, 終了日)
    """
    period_config = ANALYSIS_PERIODS.get(period_type)
    if not period_config:
        raise ValueError(f"無効な期間タイプ: {period_type}")
    
    base_date_type = period_config['base_date']
    offset_start = period_config['offset_start']
    offset_end = period_config['offset_end']
    
    # 基準日を決定
    if base_date_type == 'current_release':
        base_date = current_release_date
    else:  # 'next_release'
        base_date = next_release_date
    
    # 開始日を計算
    period_start = base_date + timedelta(days=offset_start)
    
    # 終了日を計算
    if offset_end == 'next_release':
        period_end = next_release_date
    else:
        period_end = base_date + timedelta(days=offset_end)
    
    return period_start, period_end


def filter_changes_by_period(
    changes: List[Dict[str, Any]],
    period_start: datetime,
    period_end: datetime,
    next_release_date: datetime = None,
    exclude_post_release: bool = True
) -> List[Dict[str, Any]]:
    """
    期間内にオープンだったChangeをフィルタリングする
    
    Args:
        changes: Changeデータのリスト
        period_start: 期間開始日
        period_end: 期間終了日
        next_release_date: 次のリリース日（除外判定に使用）
        exclude_post_release: 次リリース後に判定されたChangeを除外するか
        
    Returns:
        List[Dict]: フィルタリングされたChangeデータのリスト
    """
    filtered_changes = []
    
    # 条件0: 次のリリース日から1年以内に作成されたChangeに絞る
    min_created_date = None
    if next_release_date:
        min_created_date = next_release_date - timedelta(days=365)
    
    for change in changes:
        # 作成日時をパース
        created = _parse_datetime(change.get('created'))
        if not created:
            continue
        
        # 次のリリース日から1年以内に作成されたかチェック
        if min_created_date is not None and created < min_created_date:
            continue
        
        # 期間終了前に作成されたかチェック
        if created >= period_end:
            continue
        
        # クローズ日時を取得（submitted, merged, updated のいずれか）
        status = change.get('status', '')
        closed = None
        
        if status == 'MERGED':
            closed = _parse_datetime(change.get('submitted') or change.get('merged'))
        elif status == 'ABANDONED':
            closed = _parse_datetime(change.get('updated'))
        
        # 期間開始後にクローズまたはまだオープンかチェック
        if closed and closed < period_start:
            continue  # 期間開始前にクローズされている
        
        # 次リリース後に判定されたChangeを除外
        if exclude_post_release and next_release_date and status in ['MERGED', 'ABANDONED']:
            submitted = _parse_datetime(change.get('submitted'))
            if submitted and submitted > next_release_date:
                continue
        
        filtered_changes.append(change)
    
    logger.info(f"期間フィルタリング: {len(changes)}件 → {len(filtered_changes)}件 "
                f"({period_start.date()} ～ {period_end.date()})")
    return filtered_changes


def get_reviewers_in_period(
    change: Dict[str, Any],
    period_start: datetime,
    period_end: datetime,
    bot_names: List[str] = None
) -> List[str]:
    """
    期間内にレビューしたレビューアのリストを取得する
    
    Args:
        change: Changeデータ
        period_start: 期間開始日
        period_end: 期間終了日
        bot_names: ボット名のリスト（除外対象）
        
    Returns:
        List[str]: レビューアのメールアドレスまたはユーザー名のリスト
    """
    if bot_names is None:
        bot_names = []
    
    # Change作成者を取得
    owner = change.get('owner', {})
    owner_email = owner.get('email', '')
    owner_username = owner.get('username', '')
    
    reviewers = set()
    messages = change.get('messages', [])
    
    for message in messages:
        # メッセージ日時をパース
        msg_date = _parse_datetime(message.get('date'))
        if not msg_date:
            continue
        
        # 期間内かチェック
        if msg_date < period_start or msg_date > period_end:
            continue
        
        # 著者情報を取得
        author = message.get('author', {})
        author_email = author.get('email', '')
        author_username = author.get('username', '')
        
        # ボットを除外
        if author_username in bot_names or author_email in bot_names:
            continue
        
        # SERVICE_USERタグがあれば除外（自動化されたシステム）
        if 'SERVICE_USER' in author.get('tags', []):
            continue
        
        # Change作成者（オーナー）を除外
        if author_email == owner_email or author_username == owner_username:
            continue
        
        # 自動生成されたメッセージを除外
        tag = message.get('tag', '')
        if tag.startswith('autogenerated:'):
            continue
        
        # レビューアとして追加
        reviewer_id = author_email or author_username
        if reviewer_id:
            reviewers.add(reviewer_id)
    
    return list(reviewers)


def changes_to_dataframe(changes: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    ChangeリストをDataFrameに変換する
    
    Args:
        changes: Changeデータのリスト
        
    Returns:
        pd.DataFrame: Change情報のDataFrame
    """
    records = []
    
    for change in changes:
        # 基本情報を抽出
        owner = change.get('owner', {})
        
        # insertions/deletionsの取得（file_changesまたはトップレベルから）
        insertions = change.get('insertions', 0)
        deletions = change.get('deletions', 0)
        
        # file_changesから計算する場合
        file_changes = change.get('file_changes', {})
        if isinstance(file_changes, dict) and file_changes:
            insertions = sum(fc.get('lines_inserted', 0) for fc in file_changes.values())
            deletions = sum(fc.get('lines_deleted', 0) for fc in file_changes.values())
        
        record = {
            'change_number': change.get('change_number'),
            'owner_email': owner.get('email', ''),
            'created': _parse_datetime(change.get('created')),
            'updated': _parse_datetime(change.get('updated')),
            'status': change.get('status', ''),
            'merged': _parse_datetime(change.get('submitted') or change.get('merged')),
            'lines_added': insertions,
            'lines_deleted': deletions,
            'subject': change.get('subject', ''),
            'project': change.get('project', ''),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # 日時カラムをdatetime型に変換
    datetime_cols = ['created', 'updated', 'merged']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def get_release_pairs(
    releases_df: pd.DataFrame,
    project_name: str,
    target_releases: List[str] = None
) -> List[Tuple[Dict, Dict]]:
    """
    連続するリリースペアを取得する
    
    Args:
        releases_df: リリース情報のDataFrame
        project_name: プロジェクト名
        target_releases: 対象リリースのリスト（Noneの場合は全て）
        
    Returns:
        List[Tuple]: (現在のリリース情報, 次のリリース情報) のタプルのリスト
    """
    # プロジェクトでフィルタリング
    project_releases = releases_df[releases_df['component'] == project_name].copy()
    
    # リリース日でソート
    project_releases = project_releases.sort_values('release_date').reset_index(drop=True)
    
    # 全リリースでペアを作成（次のリリース日をreleases_dfから取得するため）
    all_pairs = []
    for i in range(len(project_releases) - 1):
        current = project_releases.iloc[i].to_dict()
        next_rel = project_releases.iloc[i + 1].to_dict()
        all_pairs.append((current, next_rel))
    
    # 対象リリースでフィルタリング
    if target_releases:
        pairs = [(current, next_rel) for current, next_rel in all_pairs 
                 if current['version'] in target_releases]
    else:
        pairs = all_pairs
    
    return pairs
