import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import json
import os
from src.config.path import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# ロギング設定の例 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_lines_info_to_dataframe(df: pd.DataFrame, project_name: str) -> pd.DataFrame:
    """
    DataFrameにchangeデータから行数情報（lines_added, lines_deleted, files_changed）を追加します。
    
    Args:
        df (pd.DataFrame): 'change_number'カラムを含むDataFrame
        project_name (str): プロジェクト名（例: 'neutron', 'nova'）
    
    Returns:
        pd.DataFrame: 行数情報が追加されたDataFrame
    """
    lines_added_list = []
    lines_deleted_list = []
    files_changed_list = []
    
    for idx, row in df.iterrows():
        change_number = row.get('change_number')
        if pd.isna(change_number):
            lines_added_list.append(0)
            lines_deleted_list.append(0)
            files_changed_list.append(0)
            continue
            
        # changeファイルのパスを構築
        change_file_path = os.path.join(
            DEFAULT_DATA_DIR, 'openstack', project_name, 'changes', 
            f'change_{int(change_number)}.json'
        )
        
        try:
            if os.path.exists(change_file_path):
                with open(change_file_path, 'r', encoding='utf-8') as f:
                    change_data = json.load(f)
                
                # change_metrics関数を使用して行数を計算
                from src.features.change_metrics import (
                    calculate_lines_added, 
                    calculate_lines_deleted, 
                    calculate_files_changed
                )
                
                lines_added = calculate_lines_added(change_data)
                lines_deleted = calculate_lines_deleted(change_data)
                files_changed = calculate_files_changed(change_data)
                
                lines_added_list.append(lines_added)
                lines_deleted_list.append(lines_deleted)
                files_changed_list.append(files_changed)
            else:
                logger.warning(f"Change file not found: {change_file_path}")
                lines_added_list.append(0)
                lines_deleted_list.append(0)
                files_changed_list.append(0)
                
        except Exception as e:
            logger.error(f"Error reading change file {change_file_path}: {e}")
            lines_added_list.append(0)
            lines_deleted_list.append(0)
            files_changed_list.append(0)
    
    # DataFrameに新しいカラムを追加
    df_with_lines = df.copy()
    df_with_lines['lines_added'] = lines_added_list
    df_with_lines['lines_deleted'] = lines_deleted_list
    df_with_lines['files_changed'] = files_changed_list
    
    return df_with_lines

def _get_major_version(version_string: str) -> int | None:
    """
    バージョン文字列からメジャーバージョン番号を抽出します。
    例: "13.0.0.0b3" -> 13
    """
    if not isinstance(version_string, str):
        return None
    try:
        # バージョン文字列の最初の数値部分をメジャーバージョンとして抽出
        return int(version_string.split('.')[0])
    except (ValueError, IndexError):
        return None

def calculate_days_to_major_release(
    analysis_time: datetime, 
    component_name: str, 
    all_releases_df: pd.DataFrame
) -> float:
    """
    指定された分析時点から、対象コンポーネントの次のメジャーバージョンアップリリース日までの
    残り日数を計算

    Args:
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻
        component_name (str): 対象のOpenStackコンポーネント名
        all_releases_df (pd.DataFrame): 全てのリリース履歴を含むDataFrame
                                        'component', 'version', 'release_date' (datetime型) カラムが必要

    Returns:
        float: 次のメジャーリリース日までの残り日数 見つからない場合は-1.0
               分析時点がリリース日より後の場合は0.0
    """
    # 対象コンポーネントのリリースをフィルタリング
    component_releases = all_releases_df[all_releases_df['component'] == component_name].copy()
    
    # メジャーバージョンを抽出し、Noneは除外
    component_releases['major_version_num'] = component_releases['version'].apply(_get_major_version)
    component_releases = component_releases.dropna(subset=['major_version_num'])
    
    # リリース日とメジャーバージョン番号でソート (昇順)
    component_releases = component_releases.sort_values(by=['release_date', 'major_version_num'])

    next_major_release_date = None
    
    # 分析時点以前の最新のメジャーバージョン番号を特定
    current_base_major_version = -1 # 初期値としてありえない低い数値を設定
    
    # analysis_time 以前の最も新しいリリースを取得し、そのメジャーバージョンを基準とする
    past_releases_at_analysis_time = component_releases[component_releases['release_date'] <= analysis_time]
    if not past_releases_at_analysis_time.empty:
        latest_past_release = past_releases_at_analysis_time.iloc[-1]
        current_base_major_version = latest_past_release['major_version_num']

    # 分析時点より後のリリースを順に見ていき、メジャーバージョン番号が増加した最初のリリースを探す
    for idx, row in component_releases[component_releases['release_date'] > analysis_time].iterrows():
        if row['major_version_num'] > current_base_major_version:
            next_major_release_date = row['release_date']
            break
        # もしanalysis_timeより後のリリースで、まだmajor_versionが上がっていない場合、
        # そのリリースが新たな基準となりうる（例: 1.0 -> 1.1 -> 2.0 で、analysis_timeが1.0と1.1の間の場合）
        current_base_major_version = row['major_version_num']


    if next_major_release_date:
        time_difference = next_major_release_date - analysis_time
        return max(0.0, time_difference.total_seconds() / (24 * 3600)) # 日数に変換
    else:
        logger.warning(f"No upcoming major version increment release found for component '{component_name}' after {analysis_time}.")
        return -1.0 # 今後のメジャーバージョンアップが見つからない場合

def calculate_predictive_target_ticket_count(
    all_prs_df: pd.DataFrame, 
    analysis_time: datetime
) -> int:
    """
    指定された分析時点においてオープンであるChangeの数を算出
    オープンとは、分析時点までに作成され、かつ分析時点までにマージされていないChangeを指す

    Args:
        all_prs_df (pd.DataFrame): 全てのChange履歴を含むDataFrame。
                                   'created', 'merged' (datetime型) カラムが必要です。
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻。

    Returns:
        int: 分析時点においてオープンなChangeの総数。
    """
    # 分析時点までに作成されたChange
    created_by_analysis_time = all_prs_df[all_prs_df['created'] <= analysis_time]

    # その中で、分析時点までにマージされていないChange
    # 'merged' がNaNである（まだマージされていない）か、
    # 'merged' が analysis_time より後である（分析時点ではまだマージされていない）
    open_changes_at_analysis_time = created_by_analysis_time[
        (created_by_analysis_time['merged'].isna()) | 
        (created_by_analysis_time['merged'] > analysis_time)
    ]

    return len(open_changes_at_analysis_time)

def calculate_reviewed_lines_in_period(
    all_prs_df: pd.DataFrame, 
    analysis_time: datetime, 
    lookback_days: int = 14 # 過去2週間
) -> int:
    """
    指定された分析時点から過去指定日数以内（デフォルト2週間）にレビューされた（活動があった）
    行数の合計を計算します。
    ここで「レビューされた」とは、その期間内にPRが更新されたことを指します。
    
    Args:
        all_prs_df (pd.DataFrame): 全てのPR履歴を含むDataFrame。
                                   'updated' (datetime型), 'lines_added', 'lines_deleted' カラムが必要です。
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻。
        lookback_days (int): 遡る日数（デフォルト: 14日）。

    Returns:
        int: 指定期間内にレビューされた総行数（追加+削除）。
    """
    start_of_period = analysis_time - timedelta(days=lookback_days)

    # lookback_days期間内に更新があったPRをフィルタリング
    active_prs_in_period = all_prs_df[
        (all_prs_df['updated'] >= start_of_period) & 
        (all_prs_df['updated'] <= analysis_time)
    ]
    
    # 行数情報が利用可能な場合は実際の行数を計算
    if 'lines_added' in active_prs_in_period.columns and 'lines_deleted' in active_prs_in_period.columns:
        lines_added = active_prs_in_period['lines_added'].fillna(0).astype(int).sum()
        lines_deleted = active_prs_in_period['lines_deleted'].fillna(0).astype(int).sum()
        return lines_added + lines_deleted
    else:
        # 行数情報が利用できない場合は0を返す
        logger.warning("行数情報が利用できないため、0を返します。")
        return 0