import logging
from datetime import datetime
from typing import Dict, Any
from typing import Dict, Any
import json
from src.config.path import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

# ロギング設定の例 (必要に応じて調整)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_lines_added(change_data: Dict[str, Any]) -> int:
    """
    Change（PR）におけるファイルごとの追加行数の合計を計算
    
    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ

    Returns:
        int: 追加された総行数
    """
    total_lines_added = 0
    file_changes = change_data.get("file_changes", {})
    for file_info in file_changes.values():
        total_lines_added += file_info.get("lines_inserted", 0)
    return total_lines_added

def calculate_lines_deleted(change_data: Dict[str, Any]) -> int:
    """
    Change（PR）におけるファイルごとの削除行数の合計を計算

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ

    Returns:
        int: 削除された総行数
    """
    total_lines_deleted = 0
    file_changes = change_data.get("file_changes", {})
    for file_info in file_changes.values():
        total_lines_deleted += file_info.get("lines_deleted", 0)
    return total_lines_deleted

def calculate_files_changed(change_data: Dict[str, Any]) -> int:
    """
    Change（PR）における変更ファイルの総数を計算

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ

    Returns:
        int: 変更されたファイルの総数
    """
    files = change_data.get("files", [])
    return len(files)

def calculate_elapsed_time(change_data: Dict[str, Any], analysis_time: datetime) -> float:
    """
    Change（PR）が作成されてから指定された分析時点までの経過時間（分数）を計算

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ
        analysis_time (datetime): 経過時間を計算する基準となる分析時点の時刻。

    Returns:
        float: 経過時間（分数），計算できない場合は-1.0
    """
    created_str = change_data.get("created")
    if not created_str:
        logger.warning(f"Change {change_data.get('change_number', 'N/A')} has no 'created' timestamp. Cannot calculate elapsed time.")
        return -1.0
    
    try:
        # 時刻部分にスペースが含まれる場合は'T'に変換してからパース
        created_dt = datetime.fromisoformat(created_str.replace(' ', 'T'))
        
        # ここでは、0または負の値（まだ作成されていない、あるいは未来の分析時点）を返す
        if analysis_time < created_dt:
            logger.warning(f"Analysis time {analysis_time} is earlier than change creation time {created_dt} for change {change_data.get('change_number', 'N/A')}. Elapsed time set to 0.0.")
            return -1.0
            
        time_difference = analysis_time - created_dt
        return time_difference.total_seconds() / 60  # 分数に変換
    except ValueError as e:
        logger.error(f"Error parsing 'created' timestamp '{created_str}' for change {change_data.get('change_number', 'N/A')}: {e}")
        return -1.0

def calculate_revision_count(change_data: Dict[str, Any], analysis_time: datetime) -> int:
    """
    Change（PR）が指定された分析時点までに持っていたリビジョン数を計算
    個別のコミット情報ファイルからタイムスタンプを取得

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ
        analysis_time (datetime): リビジョン数を計算する基準となる分析時点の時刻

    Returns:
        int: 分析時点までのリビジョンの総数
    """
    change_number = change_data.get("change_number")
    component_name = change_data.get("project") # プロジェクト名を取得

    if not change_number or not component_name:
        logger.error(f"Missing 'change_number' or 'project' in change_data: {change_data}")
        return 0

    commit_ids = change_data.get("commit_ids", [])
    count = 0

    # コミット情報が保存されているディレクトリパスを構築
    commits_dir = DEFAULT_DATA_DIR / "openstack" / component_name / "commits"

    for commit_id in commit_ids:
        # 短縮されたコミットIDでファイル名が保存されている可能性を考慮
        short_commit_id = commit_id[:8] if len(commit_id) > 8 else commit_id
        commit_file_name = f"commit_{change_number}_{short_commit_id}.json"
        commit_file_path = commits_dir / commit_file_name

        if commit_file_path.exists():
            try:
                with open(commit_file_path, "r", encoding="utf-8") as f:
                    commit_info = json.load(f)
                
                # コミットの作成日時（committerのdate）を取得
                commit_created_str = commit_info.get("commit", {}).get("committer", {}).get("date")
                
                if commit_created_str:
                    try:
                        # 時刻部分にスペースが含まれる場合は'T'に変換してからパース
                        commit_created_dt = datetime.fromisoformat(commit_created_str.split('.')[0].replace(' ', 'T'))
                        
                        if commit_created_dt <= analysis_time:
                            count += 1
                    except ValueError as e:
                        logger.warning(f"Error parsing commit created timestamp '{commit_created_str}' for commit {commit_id} in change {change_number}: {e}")
            except Exception as e:
                logger.warning(f"Error loading or parsing commit file {commit_file_path}: {e}")
        else:
            logger.debug(f"Commit file not found: {commit_file_path}") # ファイルがない場合はデバッグログに留める
            
    return count

def check_test_code_presence(change_data: Dict[str, Any]) -> int:
    """
    変更（PR）にテストコードが含まれているかを判定
    ファイルパスに 'test' や 'tests' が含まれるファイルをテストコードと見なす

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ

    Returns:
        int: テストコードが含まれている場合は1、含まれていない場合は0。
    """
    files = change_data.get("files", [])
    for file_path in files:
        if "/test/" in file_path.lower() or "/tests/" in file_path.lower() or file_path.lower().endswith(('_test.py', 'test_.py')):
            return 1
    return 0