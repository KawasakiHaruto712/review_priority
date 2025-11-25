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

def _parse_datetime(dt_input: Any) -> datetime:
    """
    日時文字列またはdatetimeオブジェクトをdatetimeに変換するヘルパー関数
    """
    if isinstance(dt_input, datetime):
        return dt_input
    if not dt_input:
        return None
    try:
        return datetime.fromisoformat(str(dt_input).replace('.000000000', '').replace(' ', 'T'))
    except (ValueError, TypeError):
        return None

def _get_files_at_analysis_time(change_data: Dict[str, Any], analysis_time: datetime = None) -> Dict[str, Any]:
    """
    分析時点での最新リビジョンからファイル情報を取得するヘルパー関数
    analysis_timeが指定されない場合は、最新のリビジョンを使用する
    """
    revisions = change_data.get('revisions', {})
    if not revisions:
        # 古い形式またはデータがない場合のフォールバック
        # file_changes (dict) または files (list/dict) を確認
        return change_data.get('file_changes', {}) or change_data.get('files', {})

    if analysis_time:
        target_revision = None
        latest_date = None
        
        for rev_data in revisions.values():
            created_dt = _parse_datetime(rev_data.get('created'))
            if not created_dt:
                continue
                
            if created_dt <= analysis_time:
                if latest_date is None or created_dt > latest_date:
                    latest_date = created_dt
                    target_revision = rev_data
        
        if target_revision:
            return target_revision.get('files', {})
        
        # 分析時点より前のリビジョンが見つからない場合（通常ありえないが）
        return {}

    current_revision = change_data.get('current_revision')
    if current_revision and current_revision in revisions:
        return revisions[current_revision].get('files', {})
    
    # current_revisionがない場合は最後のリビジョンを使用
    if revisions:
        # _numberでソートして最新を取得
        sorted_revs = sorted(revisions.values(), key=lambda x: x.get('_number', 0))
        if sorted_revs:
            return sorted_revs[-1].get('files', {})
    
    return {}

def calculate_lines_added(change_data: Dict[str, Any], analysis_time: datetime = None) -> int:
    """
    Change（PR）におけるファイルごとの追加行数の合計を計算
    
    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ
        analysis_time (datetime, optional): 分析時点。指定された場合、その時点でのリビジョンを使用する。

    Returns:
        int: 追加された総行数
    """
    file_changes = _get_files_at_analysis_time(change_data, analysis_time)
    total_lines_added = 0
    
    # file_changesがリストの場合（古い形式の一部）の対応
    if isinstance(file_changes, list):
        # リストの場合は詳細な行数情報がない可能性があるため0を返すか、
        # 別の方法で取得する必要があるが、ここでは辞書形式を前提とする
        return 0

    for file_path, file_change in file_changes.items():
        if isinstance(file_change, dict):
            lines_inserted = file_change.get('lines_inserted', 0)
            total_lines_added += lines_inserted
    
    return total_lines_added

def calculate_lines_deleted(change_data: Dict[str, Any], analysis_time: datetime = None) -> int:
    """
    Change（PR）におけるファイルごとの削除行数の合計を計算

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ
        analysis_time (datetime, optional): 分析時点。指定された場合、その時点でのリビジョンを使用する。

    Returns:
        int: 削除された総行数
    """
    file_changes = _get_files_at_analysis_time(change_data, analysis_time)
    total_lines_deleted = 0
    
    if isinstance(file_changes, list):
        return 0

    for file_path, file_change in file_changes.items():
        if isinstance(file_change, dict):
            lines_deleted = file_change.get('lines_deleted', 0)
            total_lines_deleted += lines_deleted
    
    return total_lines_deleted

def calculate_files_changed(change_data: Dict[str, Any], analysis_time: datetime = None) -> int:
    """
    Change（PR）における変更ファイルの総数を計算

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ
        analysis_time (datetime, optional): 分析時点。指定された場合、その時点でのリビジョンを使用する。

    Returns:
        int: 変更されたファイルの総数
    """
    files = _get_files_at_analysis_time(change_data, analysis_time)
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
    created = change_data.get("created")
    if not created:
        logger.warning(f"Change {change_data.get('change_number', 'N/A')} has no 'created' timestamp. Cannot calculate elapsed time.")
        return -1.0
    
    try:
        # datetimeオブジェクトの場合はそのまま使用、文字列の場合は変換
        if isinstance(created, datetime):
            created_dt = created
        else:
            # 時刻部分にスペースが含まれる場合は'T'に変換してからパース
            created_str = str(created).replace('.000000000', '').replace(' ', 'T')
            created_dt = datetime.fromisoformat(created_str)
        
        # ここでは、0または負の値（まだ作成されていない、あるいは未来の分析時点）を返す
        if analysis_time < created_dt:
            logger.warning(f"Analysis time {analysis_time} is earlier than change creation time {created_dt} for change {change_data.get('change_number', 'N/A')}. Elapsed time set to 0.0.")
            return -1.0
            
        time_difference = analysis_time - created_dt
        return time_difference.total_seconds() / 60  # 分数に変換
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing 'created' timestamp '{created}' for change {change_data.get('change_number', 'N/A')}: {e}")
        return -1.0

def calculate_revision_count(change_data: Dict[str, Any], analysis_time: datetime) -> int:
    """
    Change（PR）が指定された分析時点までに持っていたリビジョン数を計算
    revisionsデータがある場合はそれを使用し、ない場合は個別のコミット情報ファイルから取得

    Args:
        change_data (Dict[str, Any]): OpenStack Gerritから収集されたChange（PR）のデータ
        analysis_time (datetime): リビジョン数を計算する基準となる分析時点の時刻

    Returns:
        int: 分析時点までのリビジョンの総数
    """
    # 1. revisionsデータから計算（高速・推奨）
    revisions = change_data.get('revisions')
    if revisions:
        count = 0
        for rev_data in revisions.values():
            created_dt = _parse_datetime(rev_data.get('created'))
            if created_dt and created_dt <= analysis_time:
                count += 1
        return count

    # 2. フォールバック: 外部ファイルから計算
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
    # 現在のデータにはfilesフィールドが含まれていないため、デフォルト値を返す
    # 実際のプロジェクトでは、コミット情報から計算するか、別途収集が必要
    # PRのタイトルや説明からテストの存在を推測する簡易的な方法を使用
    subject = change_data.get("subject", "").lower()
    message = change_data.get("message", "").lower()
    
    test_keywords = ["test", "unittest", "pytest", "testing"]
    
    for keyword in test_keywords:
        if keyword in subject or keyword in message:
            return 1
    
    return 0

def get_change_text_data(change_data: Dict[str, Any]) -> tuple[str, str]:
    """
    Changeデータからタイトル(subject)と説明(message)を抽出する
    
    Args:
        change_data (Dict[str, Any]): Changeデータ

    Returns:
        tuple[str, str]: (subject, message)
    """
    subject = change_data.get('subject', '')
    
    # 1. トップレベルのmessageを確認 (古い形式や一部のデータ)
    message = change_data.get('message', '')
    
    if not message:
        # 2. revisionsからmessageを確認
        revisions = change_data.get('revisions', {})
        if revisions:
            current_revision = change_data.get('current_revision')
            if current_revision and current_revision in revisions:
                message = revisions[current_revision].get('commit', {}).get('message', '')
            else:
                # current_revisionがない場合は最後のリビジョンを使用
                message = list(revisions.values())[-1].get('commit', {}).get('message', '')
                
    return subject, message