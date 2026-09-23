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

# リリース境界とみなすバージョン形式。呼び出し側が release_level で選ぶ
#   "major" -> X.0.0    OpenStackの協調リリース（nova, neutron, cinder, glance, keystone）
#   "minor" -> X.Y.0    協調リリースに乗らず独自にリリースするもの（swift は 2.0.0 以降 2.Y.0）
_RELEASE_LEVEL_REGEX = {
    "major": re.compile(r"^\d+\.0\.0$"),
    "minor": re.compile(r"^\d+\.\d+\.0$"),
}


def _release_ordinal(version_string: str, release_level: str = "major") -> tuple | None:
    """
    バージョン文字列から「リリースの世代」を表すタプルを抽出します。
    例: release_level="major" なら "13.0.0" -> (13, 0)
        release_level="minor" なら "2.31.0" -> (2, 31)

    タプルなのは、"minor" のときにメジャー番号の繰り上がり（2.36.0 -> 3.0.0）も
    正しく「次の世代」と判定できるようにするためです。

    注意: この関数は世代番号を抽出するだけです。
    リリース境界かどうかの判定には _is_boundary_release を使用してください。
    """
    if not isinstance(version_string, str):
        return None
    try:
        parts = version_string.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if release_level == "minor" else 0
        return (major, minor)
    except (ValueError, IndexError):
        return None


def _is_boundary_release(version_string: str, release_level: str = "major") -> bool:
    """
    バージョン文字列がリリース境界の形式かどうかを判定します。

    release_level="major" のとき
      許可: "20.0.0", "7.0.0"
      除外: "20.0.0.0rc1", "20.0.0.0b1", "20.0.1", "20.1.0"

    release_level="minor" のとき（swift のように協調リリースに乗らないプロジェクト用）
      許可: "2.31.0", "2.0.0"
      除外: "2.31.1", "2.31.0rc1"

    Args:
        version_string: バージョン文字列
        release_level: "major"（X.0.0）または "minor"（X.Y.0）

    Returns:
        True  -> 指定した形式に正確に一致
        False -> それ以外
    """
    if not isinstance(version_string, str):
        return False

    regex = _RELEASE_LEVEL_REGEX.get(release_level)
    if regex is None:
        raise ValueError(f"release_level は {sorted(_RELEASE_LEVEL_REGEX)} のいずれかです: {release_level!r}")

    # 正確に3パートで、指定形式に一致し、かつ追加のサフィックス（rc/b等）がないもののみ許可
    return bool(regex.match(version_string.strip()))


def calculate_days_to_major_release(
    analysis_time: datetime,
    component_name: str,
    all_releases_df: pd.DataFrame,
    release_level: str = "major"
) -> float:
    """
    指定された分析時点から、対象コンポーネントの次のリリース日までの残り日数を計算

    どのバージョン形式をリリース境界とみなすかは release_level で指定します
    （rc/beta等のサフィックス付きはどちらの場合も除外）。

    Args:
        analysis_time (datetime): メトリクスを計算する基準となる分析時点の時刻
        component_name (str): 対象のOpenStackコンポーネント名
        all_releases_df (pd.DataFrame): 全てのリリース履歴を含むDataFrame
                                        'component', 'version', 'release_date' (datetime型) カラムが必要
        release_level (str): "major" なら X.0.0 を境界とする（OpenStackの協調リリース）
                             "minor" なら X.Y.0 を境界とする（swift のように独自にリリースするもの）

    Returns:
        float: 次のリリース日までの残り日数 見つからない場合は-1.0
               分析時点がリリース日より後の場合は0.0
    """
    # 対象コンポーネントのリリースをフィルタリング
    component_releases = all_releases_df[all_releases_df['component'] == component_name].copy()

    # リリース境界の形式のものだけをフィルタリング
    component_releases['is_major'] = component_releases['version'].apply(
        lambda v: _is_boundary_release(v, release_level))
    major_releases_only = component_releases[component_releases['is_major']].copy()

    if major_releases_only.empty:
        logger.warning(f"No major releases found for component '{component_name}' "
                       f"(release_level={release_level}).")
        return -1.0

    # リリースの世代番号を抽出
    major_releases_only['major_version_num'] = major_releases_only['version'].apply(
        lambda v: _release_ordinal(v, release_level))
    major_releases_only = major_releases_only.dropna(subset=['major_version_num'])
    
    # リリース日とメジャーバージョン番号でソート (昇順)
    major_releases_only = major_releases_only.sort_values(by=['release_date', 'major_version_num'])

    next_major_release_date = None
    
    # 分析時点以前の最新のリリース世代を特定
    # 初期値はありえない低い値。_release_ordinal と同じ (major, minor) のタプルにしておかないと、
    # 分析時点が最初のリリースより前のとき（＝この初期値のまま比較に入るとき）に
    # tuple と int の比較になって落ちる
    current_base_major_version = (-1, -1)
    
    # analysis_time 以前の最も新しいメジャーリリースを取得し、そのメジャーバージョンを基準とする
    past_releases_at_analysis_time = major_releases_only[major_releases_only['release_date'] <= analysis_time]
    if not past_releases_at_analysis_time.empty:
        latest_past_release = past_releases_at_analysis_time.iloc[-1]
        current_base_major_version = latest_past_release['major_version_num']

    # 分析時点より後のメジャーリリースを順に見ていき、メジャーバージョン番号が増加した最初のリリースを探す
    for idx, row in major_releases_only[major_releases_only['release_date'] > analysis_time].iterrows():
        if row['major_version_num'] > current_base_major_version:
            next_major_release_date = row['release_date']
            break
        # もしanalysis_timeより後のリリースで、まだmajor_versionが上がっていない場合、
        # そのリリースが新たな基準となりうる（例: 7.0.0 -> 8.0.0 -> 9.0.0 で、analysis_timeが7.0.0と8.0.0の間の場合）
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