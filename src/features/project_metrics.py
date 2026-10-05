import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import json
import os
import re
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

def calculate_days_to_major_release(
    analysis_time: datetime,
    component_name: str,
    all_releases_df: pd.DataFrame,
) -> float:
    """
    指定された分析時点から、対象コンポーネントの次のリリース日までの残り日数を計算

    all_releases_df はサイクル単位のリリースの表（major_releases_summary.csv。1 サイクル 1 行。
    作り方は src/collectors/README.md の「メジャーリリースの表」）。その component の
    release_date を日付順に並べ、分析時点より後の最初の日付までの日数を返す。
    status が planned（まだ出ていないサイクルの予定日）の行も使う。

    旧版は版番号の形（X.0.0、swift は 2.Y.0）で区切りを拾っていたため、版の付け方が違う
    2015 年秋より前（2011.2・2015.1.0 など）が抜けていた（pretrained_encoders/design.md §11.4）。
    表がサイクル単位になったので、版番号は見ない。

    Args:
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻
        component_name (str): 対象のコンポーネント名
        all_releases_df (pd.DataFrame): 'component', 'release_date'（datetime 型）カラムが必要

    Returns:
        float: 次のリリース日までの残り日数。見つからない場合は -1.0
               分析時点がリリース日の 0 時ちょうどのときは、そのリリースは済んだものとみなす（旧版と同じ）
    """
    dates = pd.to_datetime(
        all_releases_df.loc[all_releases_df['component'] == component_name, 'release_date']
    ).dropna()
    upcoming = dates[dates > pd.Timestamp(analysis_time)]
    if upcoming.empty:
        logger.warning(f"No upcoming release found for component '{component_name}' after {analysis_time}.")
        return -1.0
    time_difference = upcoming.min() - pd.Timestamp(analysis_time)
    return max(0.0, time_difference.total_seconds() / (24 * 3600))  # 日数に変換


def calculate_predictive_target_ticket_count(
    all_prs_df: pd.DataFrame, 
    analysis_time: datetime
) -> int:
    """
    指定された分析時点においてオープン（まだ決着していない）であるChangeの数を算出

    オープンとは、分析時点までに作成され、かつ分析時点までに **決着（マージまたは放棄）していない**
    Changeを指す。マージだけでなく放棄(ABANDONED)も「閉じた」とみなす。

    Args:
        all_prs_df (pd.DataFrame): 全てのChange履歴を含むDataFrame。
                                   'created', 'decision_time' (datetime型。マージ/放棄の時刻。
                                   未決は NaT) カラムが必要です。
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻。

    Returns:
        int: 分析時点においてオープンなChangeの総数。
    """
    # 分析時点までに作成されたChange
    created_by_analysis_time = all_prs_df[all_prs_df['created'] <= analysis_time]

    # その中で、分析時点までに決着（マージまたは放棄）していないChange
    # 'decision_time' が NaT（まだ未決）か、'decision_time' が analysis_time より後（その時点では未決）
    open_changes_at_analysis_time = created_by_analysis_time[
        (created_by_analysis_time['decision_time'].isna()) |
        (created_by_analysis_time['decision_time'] > analysis_time)
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