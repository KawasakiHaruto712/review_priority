"""
OpenStackのGerritからデータを収集するためのモジュール

このモジュールはOpenStackのGerritからPull Request（PR）データとコミット情報を収集し、
分析のためにローカルに保存します。PRとコミット情報は別々に保存され、関連付けるためのIDを持ちます。
"""

import os
import json
import requests
import logging
import time
import random
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from src.config.path import DEFAULT_DATA_DIR
from src.utils.constants import OPENSTACK_CORE_COMPONENTS, START_DATE, END_DATE
from src.utils.lang_identifiyer import identify_lang_from_file

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{DEFAULT_DATA_DIR}/openstack_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetryConfig:
    """リトライ設定クラス"""
    def __init__(self, max_retries=10, base_delay=30.0, max_delay=3840.0, backoff_factor=2.0, jitter=True):
        """
        Args:
            max_retries (int): 最大リトライ回数
            base_delay (float): 初期待機時間（秒）
            max_delay (float): 最大待機時間（秒）
            backoff_factor (float): バックオフ係数（指数的増加の倍率）
            jitter (bool): ジッターを有効にするか（ランダムな揺らぎを追加）
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, retry_count):
        """
        リトライ回数に基づいて待機時間を計算
        
        Args:
            retry_count (int): 現在のリトライ回数（0から開始）
            
        Returns:
            float: 待機時間（秒）
        """
        # 指数バックオフによる待機時間の計算
        delay = self.base_delay * (self.backoff_factor ** retry_count)
        
        # 最大待機時間を超えないように制限
        delay = min(delay, self.max_delay)
        
        # ジッターを追加（ランダムな揺らぎで同時リクエストの衝突を避ける）
        if self.jitter:
            # ±25%のランダムな揺らぎを追加
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)

def retry_with_backoff(retry_config=None):
    """
    指数バックオフを用いたリトライデコレータ
    
    Args:
        retry_config (RetryConfig): リトライ設定
        
    Returns:
        function: デコレートされた関数
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for retry_count in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    
                    # 最後のリトライの場合は例外を再発生
                    if retry_count >= retry_config.max_retries:
                        logger.error(f"最大リトライ回数({retry_config.max_retries})に達しました: {e}")
                        raise
                    
                    # 待機時間を計算
                    delay = retry_config.get_delay(retry_count)
                    
                    # エラーの種類によってログレベルを調整
                    if isinstance(e, requests.exceptions.Timeout):
                        log_level = logging.WARNING
                        error_type = "タイムアウト"
                    elif isinstance(e, requests.exceptions.ConnectionError):
                        log_level = logging.WARNING
                        error_type = "接続エラー"
                    elif hasattr(e, 'response') and e.response is not None:
                        if e.response.status_code == 429:
                            log_level = logging.WARNING
                            error_type = "レート制限"
                            # 429エラーの場合は追加の待機時間を設ける
                            delay = max(delay, 30)
                        elif 500 <= e.response.status_code < 600:
                            log_level = logging.WARNING
                            error_type = f"サーバーエラー({e.response.status_code})"
                        else:
                            log_level = logging.ERROR
                            error_type = f"HTTPエラー({e.response.status_code})"
                    else:
                        log_level = logging.ERROR
                        error_type = "ネットワークエラー"
                    
                    logger.log(
                        log_level,
                        f"{error_type}が発生しました。{delay:.1f}秒後にリトライします "
                        f"(試行回数: {retry_count + 1}/{retry_config.max_retries + 1}): {e}"
                    )
                    
                    # 指定された時間だけ待機
                    time.sleep(delay)
                except Exception as e:
                    # RequestException以外の例外は即座に再発生
                    logger.error(f"予期しないエラーが発生しました: {e}")
                    raise
            
            # ここに到達することはないはずだが、念のため
            raise last_exception
        
        return wrapper
    return decorator

class OpenStackGerritCollector:
    """OpenStackのGerritからデータを収集するクラス"""
    
    BASE_URL = "https://review.opendev.org/a"

    def __init__(self, components=None, start_date=None, end_date=None, username=None, password=None, retry_config=None):
        """
        コンストラクタ
        
        Args:
            components (list): 取得対象のOpenStackコンポーネントリスト
            start_date (str): 取得対象の開始日 (YYYY-MM-DD)
            end_date (str): 取得対象の終了日 (YYYY-MM-DD)
            username (str): Gerritのユーザー名
            password (str): Gerritのパスワード
            retry_config (RetryConfig): リトライ設定
        """

        load_dotenv()

        self.components = components if components is not None else OPENSTACK_CORE_COMPONENTS
        self.start_date = start_date if start_date is not None else START_DATE
        self.end_date = end_date if end_date is not None else END_DATE
        self.username = username if username is not None else os.getenv("GERRIT_USERNAME")
        self.password = password if password is not None else os.getenv("GERRIT_PASSWORD")
        
        # リトライ設定
        self.retry_config = retry_config if retry_config is not None else RetryConfig(
            max_retries=10,
            base_delay=30.0,
            max_delay=3840.0,
            backoff_factor=2.0,
            jitter=True
        )

        # 認証ヘッダーを明示的に設定
        self.headers = {
            "Authorization": f"Basic {self._base64_encode(f'{self.username}:{self.password}')}",
            "Accept": "application/json"
        }

        # リクエストセッションの設定（接続プーリングとタイムアウト設定）
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # タイムアウト設定（接続タイムアウト, 読み取りタイムアウト）
        self.timeout = (30, 120)

        # 保存先ディレクトリの作成
        self.output_dir = DEFAULT_DATA_DIR / "openstack"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for component in self.components:
            # コンポーネントディレクトリ
            (self.output_dir / component).mkdir(exist_ok=True)
            # PR情報用のディレクトリ
            (self.output_dir / component / "changes").mkdir(exist_ok=True)
            # コミット情報用のディレクトリ
            (self.output_dir / component / "commits").mkdir(exist_ok=True)

    def _base64_encode(self, text):
        """
        文字列をBase64エンコードする
        
        Args:
            text (str): エンコードする文字列
            
        Returns:
            str: Base64エンコードされた文字列
        """
        import base64
        return base64.b64encode(text.encode()).decode()
    
    @retry_with_backoff()
    def _make_request(self, endpoint, params=None):
        """
        GerritのREST APIにリクエストを送信（リトライ機能付き）
        
        Args:
            endpoint (str): APIエンドポイント
            params (dict): クエリパラメータ
            
        Returns:
            dict: レスポンスデータ
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        # リクエストを実行（タイムアウト設定付き）
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        # Gerrit APIのレスポンスの処理
        data = response.text
        if data.startswith(")]}'"):
            # 確実に特殊プレフィックスを除去
            if '\n' in data:
                data = data[data.find('\n') + 1:]
            else:
                data = data[4:]  # ')]}\'の4文字を除去

        # JSON形式でない場合のエラーハンドリング
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSONデコードエラー: {e}")
            logger.debug(f"受信したレスポンスデータ: {data[:200]}...")
            raise
        
        # API制限回避のための基本待機
        time.sleep(1)  # 基本的な待機時間
        
        return parsed_data
    
    def get_merged_changes(self, component, limit=100, skip=0):
        """
        マージ済みの変更を取得
        
        Args:
            component (str): OpenStackのコンポーネント名
            limit (int): 一度に取得する変更の数
            skip (int): スキップする変更の数
            
        Returns:
            list: 変更のリスト
        """
        # GerritのSearchエンドポイントを使用するためのクエリを構築
        query = f"project:openstack/{component} status:merged after:{self.start_date} before:{self.end_date}"
        
        endpoint = "changes/"
        params = {
            "q": query,
            "n": limit,
            "S": skip,
            "o": [
                "CURRENT_REVISION",
                "ALL_REVISIONS",
                "CURRENT_COMMIT",
                "ALL_COMMITS",
                "DETAILED_ACCOUNTS",
                "DETAILED_LABELS",
                "MESSAGES",
                "CURRENT_FILES",
                "ALL_FILES",
                "REVIEWED",
                "SUBMITTABLE"
            ]
        }
        
        return self._make_request(endpoint, params)
    
    def get_change_detail(self, change_id):
        """
        特定の変更の詳細情報を取得
        
        Args:
            change_id (str): 変更ID
            
        Returns:
            dict: 変更の詳細情報
        """
        endpoint = f"changes/{change_id}/detail"
        return self._make_request(endpoint)
    
    @retry_with_backoff()
    def get_file_content(self, change_id, revision_id, file_path):
        from urllib.parse import quote
        encoded_path = quote(file_path, safe='')
        
        endpoint = f"changes/{change_id}/revisions/{revision_id}/files/{encoded_path}/content"

        try:
            # Pythonファイルのみ処理する
            file_lang = identify_lang_from_file(file_path)
            if file_lang != "Python":
                logger.info(f"Pythonファイル以外なのでスキップします: {file_path} (言語: {file_lang})")
                return None
        except ValueError:
            # 未知の拡張子の場合はスキップ
            logger.info(f"未知のファイル拡張子なのでスキップします: {file_path}")
            return None
        
        # 認証ヘッダーを使用してリクエスト
        url = f"{self.BASE_URL}/{endpoint}"
        
        response = self.session.get(url, timeout=self.timeout)

        if response.status_code == 404:
            # 404エラーは正常なケースとして処理
            logger.info(f"ファイル {file_path} はこのリビジョンでは見つかりませんでした（スキップします）")
            return None

        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "")

        # 基本的な待機時間
        time.sleep(1)

        if "application/json" in content_type:
            # JSONレスポンス
            return json.loads(response.text.replace(")]}'", ""))
        else:
            # JSONでない場合（Base64エンコードされたファイル内容等）
            return response.text
    
    def get_comments(self, change_id):
        """
        変更に対するコメントを取得
        
        Args:
            change_id (str): 変更ID
            
        Returns:
            dict: コメント情報
        """
        endpoint = f"changes/{change_id}/comments"
        return self._make_request(endpoint)

    def get_reviewers(self, change_id):
        """
        レビュワー情報を取得
        
        Args:
            change_id (str): 変更ID
            
        Returns:
            list: レビュワー情報のリスト
        """
        endpoint = f"changes/{change_id}/reviewers"
        return self._make_request(endpoint)
    
    def get_diff(self, change_id, revision_id, file_path):
        """
        特定のファイルの差分を取得
        
        Args:
            change_id (str): 変更ID
            revision_id (str): リビジョンID
            file_path (str): ファイルパス
            
        Returns:
            str: ファイルの差分
        """
        from urllib.parse import quote
        encoded_path = quote(file_path, safe='')
        
        endpoint = f"changes/{change_id}/revisions/{revision_id}/files/{encoded_path}/diff"
        diff_data = self._make_request(endpoint)
        
        if diff_data:
            return diff_data.get("content", "")
        return None
    
    def get_revision_commits(self, change_id, revision_id):
        """
        特定のリビジョンのコミット情報を取得
        
        Args:
            change_id (str): 変更ID
            revision_id (str): リビジョンID
            
        Returns:
            dict: コミット情報
        """
        endpoint = f"changes/{change_id}/revisions/{revision_id}/commit"
        return self._make_request(endpoint)
    
    def get_commit_parents(self, change_id, revision_id):
        """
        コミットの親（Parent）情報を取得
        
        Args:
            change_id (str): 変更ID
            revision_id (str): リビジョンID
            
        Returns:
            list: 親コミットのリスト
        """
        endpoint = f"changes/{change_id}/revisions/{revision_id}/commit"
        commit_data = self._make_request(endpoint)
        
        if commit_data and "parents" in commit_data:
            return commit_data["parents"]
        return []
    
    def collect_data(self, batch_size=100):
        """
        OpenStackのすべての対象コンポーネントからデータを収集
        
        Args:
            batch_size (int): 一度に取得する変更の数
        """
        logger.info(f"データ収集を開始します（リトライ設定: 最大{self.retry_config.max_retries}回）")
        
        all_data = {
            "changes": {},
            "commits": {}
        }

        # 日付フォーマット用のヘルパー関数
        def format_date(date_str):
            if not date_str or date_str == "不明":
                return "不明"
            # ISO形式の日付文字列から分数以降を削除
            if "." in date_str:
                date_str = date_str.split(".")[0]
            return date_str
        
        for component in self.components:
            logger.info(f"{component}のデータ収集を開始します...")
            component_changes = []
            component_commits = []
            skip = 0
            
            while True:
                try:
                    changes = self.get_merged_changes(component, batch_size, skip)
                    
                    if not changes or len(changes) == 0:
                        logger.info(f"{component}の全ての変更を取得しました（合計: {len(component_changes)}）")
                        break
                    
                    # 各変更の詳細情報を取得
                    for change in changes:
                        change_id = change["id"]
                        change_number = change["_number"]
                        pr_date = format_date(change.get("created", "不明"))

                        # 進捗を表示（日付から分数以降を削除）
                        logger.info(f"処理中: {component} - PR #{change_number} ({pr_date})")
                        
                        try:
                            # 詳細情報の取得
                            detail = self.get_change_detail(change_id)
                            if not detail:
                                continue
                            
                            # コメント情報の取得
                            comments = self.get_comments(change_id)
                            
                            # レビュワー情報の取得
                            reviewers = self.get_reviewers(change_id)
                            
                            # 現在のリビジョン情報
                            current_revision = change.get("current_revision")
                            revisions = change.get("revisions", {})
                            
                            # PRに関連するコミットIDのリストを保持
                            commit_ids = []
                            
                            # すべてのリビジョン（コミット）情報を収集
                            for revision_id, revision_info in revisions.items():
                                try:
                                    # コミット情報を取得
                                    commit_info = self.get_revision_commits(change_id, revision_id)
                                    
                                    if not commit_info:
                                        continue
                                    
                                    # 変更されたファイル情報
                                    files = revision_info.get("files", {})
                                    
                                    # ファイルの差分情報を取得
                                    file_diffs = {}
                                    for file_path in files.keys():
                                        # ファイルパスが長すぎる場合やバイナリファイルはスキップ
                                        if len(file_path) > 255 or files[file_path].get("binary", False):
                                            continue
                                        
                                        try:
                                            # ファイル差分を取得
                                            diff = self.get_diff(change_id, revision_id, file_path)
                                            if diff:
                                                file_diffs[file_path] = diff
                                        except Exception as e:
                                            logger.warning(f"ファイル差分の取得に失敗しました ({file_path}): {e}")
                                    
                                    # 親コミット情報の取得
                                    parents = self.get_commit_parents(change_id, revision_id)
                                    
                                    # コミット詳細情報を構造化
                                    commit_data = {
                                        "revision_id": revision_id,
                                        "change_id": change_id,
                                        "change_number": change_number,
                                        "project": change.get("project", ""),
                                        "branch": change.get("branch", ""),
                                        "commit": {
                                            "message": commit_info.get("message", ""),
                                            "author": commit_info.get("author", {}),
                                            "committer": commit_info.get("committer", {}),
                                            "commit_id": revision_id,
                                            "parents": parents
                                        },
                                        "files": list(files.keys()),
                                        "file_changes": {
                                            path: {
                                                "lines_inserted": info.get("lines_inserted", 0),
                                                "lines_deleted": info.get("lines_deleted", 0),
                                                "size_delta": info.get("size_delta", 0)
                                            } for path, info in files.items()
                                        },
                                        "file_diffs": file_diffs,
                                        "created": revision_info.get("created", ""),
                                        "uploader": revision_info.get("uploader", {})
                                    }
                                    
                                    component_commits.append(commit_data)
                                    
                                    # コミットIDをリストに追加（PRとコミットの関連付け用）
                                    commit_ids.append(revision_id)
                                    
                                    # 各コミットのデータを個別に保存
                                    self._save_commit_data(component, change_number, revision_id, commit_data)
                                    
                                except Exception as e:
                                    logger.warning(f"コミット情報の取得に失敗しました ({revision_id}): {e}")
                                    continue
                            
                            # 現在のリビジョン情報を使用してファイル内容を取得
                            if current_revision:
                                revision_info = revisions.get(current_revision, {})
                                files = revision_info.get("files", {})
                                commit_info = revision_info.get("commit", {})
                            else:
                                files = {}
                                commit_info = {}
                            
                            # ファイルの内容を取得
                            file_contents = {}
                            if current_revision:
                                for file_path in files.keys():
                                    # ファイルパスが長すぎる場合やバイナリファイルはスキップ
                                    if len(file_path) > 255 or files[file_path].get("binary", False):
                                        continue
                                    
                                    # コード差分と内容の取得
                                    try:
                                        # 変更後のファイル内容を取得
                                        content = self.get_file_content(change_id, current_revision, file_path)
                                        # ファイル差分も取得
                                        diff = self.get_diff(change_id, current_revision, file_path)
                                        
                                        file_data = {
                                            "content": content,
                                            "diff": diff
                                        }
                                        if content or diff:
                                            file_contents[file_path] = file_data
                                    except Exception as e:
                                        logger.warning(f"ファイル内容の取得に失敗しました ({file_path}): {e}")
                            
                            # 収集したデータを構造化（ただしコミット詳細は含めない）
                            pr_data = {
                                "change_id": change_id,
                                "change_number": change_number,
                                "project": change.get("project", ""),
                                "branch": change.get("branch", ""),
                                "subject": change.get("subject", ""),
                                "created": change.get("created", ""),
                                "updated": change.get("updated", ""),
                                "submitted": change.get("submitted", ""),
                                "merged": change.get("submitted", ""),  # submittedがマージ日時と同じ
                                "owner": change.get("owner", {}),
                                "labels": change.get("labels", {}),
                                "current_revision": current_revision,
                                "commit_message": commit_info.get("message", ""),
                                "files": list(files.keys()),
                                "file_changes": {
                                    path: {
                                        "lines_inserted": info.get("lines_inserted", 0),
                                        "lines_deleted": info.get("lines_deleted", 0),
                                        "size_delta": info.get("size_delta", 0)
                                    } for path, info in files.items()
                                },
                                "file_contents": file_contents,  # ファイルの内容を追加
                                "comments": comments,
                                "reviewers": reviewers,
                                "messages": change.get("messages", []),
                                "commit_ids": commit_ids  # PRに関連するコミットIDのリストのみを保持
                            }
                            
                            component_changes.append(pr_data)
                            
                            # 各変更ごとのデータを個別に保存
                            self._save_change_data(component, change_number, pr_data)
                            
                        except Exception as e:
                            logger.error(f"変更 #{change_number} の処理中にエラーが発生しました: {e}")
                            continue
                    
                    skip += len(changes)
                    
                    # バッチ間の待機時間
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"コンポーネント {component} のバッチ処理中にエラーが発生しました: {e}")
                    # バッチレベルでの軽微なエラーは続行
                    break
            
            # コンポーネントごとのデータをまとめて保存
            self._save_component_summary(component, component_changes)
            # コミット情報のサマリーも保存
            self._save_component_commit_summary(component, component_commits)
            all_data["changes"][component] = component_changes
            all_data["commits"][component] = component_commits
            
        # 全体のサマリーデータを保存
        self._save_summary(all_data)
        
        logger.info("全てのコンポーネントのデータ収集が完了しました")
    
    def _save_change_data(self, component, change_number, data):
        """個別の変更データを保存"""
        output_path = self.output_dir / component / "changes" / f"change_{change_number}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_commit_data(self, component, change_number, revision_id, data):
        """個別のコミットデータを保存"""
        short_revision_id = revision_id[:8] if len(revision_id) > 8 else revision_id
        output_path = self.output_dir / component / "commits" / f"commit_{change_number}_{short_revision_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_component_summary(self, component, changes):
        """コンポーネントごとのサマリーデータを保存"""
        summary = {
            "total_changes": len(changes),
            "date_range": {
                "start": self.start_date,
                "end": self.end_date
            }
        }
        
        output_path = self.output_dir / component / "changes" / "summary.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        if changes:
            flat_data = []
            for change in changes:
                flat_change = {
                    "change_id": change["change_id"],
                    "change_number": change["change_number"],
                    "project": change["project"],
                    "branch": change["branch"],
                    "subject": change["subject"],
                    "created": change["created"],
                    "updated": change["updated"],
                    "merged": change["merged"],
                    "owner_name": change["owner"].get("name", ""),
                    "owner_email": change["owner"].get("email", ""),
                    "files_changed": len(change["files"]),
                    "lines_inserted": sum(f.get("lines_inserted", 0) for f in change["file_changes"].values()),
                    "lines_deleted": sum(f.get("lines_deleted", 0) for f in change["file_changes"].values()),
                    "size_delta": sum(f.get("size_delta", 0) for f in change["file_changes"].values()),
                    "comments_count": sum(len(comments) for comments in change["comments"].values()) if change["comments"] else 0,
                    "messages_count": len(change["messages"]),
                    "reviewers_count": len(change["reviewers"]),
                    "commit_count": len(change.get("commit_ids", []))
                }
                flat_data.append(flat_change)
            
            df = pd.DataFrame(flat_data)
            csv_path = self.output_dir / component / "changes" / "changes.csv"
            df.to_csv(csv_path, index=False)
    
    def _save_component_commit_summary(self, component, commits):
        """コンポーネントごとのコミットサマリーデータを保存"""
        if commits:
            flat_commits = []
            for commit in commits:
                flat_commit = {
                    "change_id": commit["change_id"],
                    "change_number": commit["change_number"],
                    "revision_id": commit["revision_id"],
                    "commit_message": commit["commit"]["message"],
                    "author_name": commit["commit"]["author"].get("name", ""),
                    "author_email": commit["commit"]["author"].get("email", ""),
                    "commit_time": commit["commit"]["committer"].get("date", ""),
                    "files_changed": len(commit["files"]),
                    "lines_inserted": sum(f.get("lines_inserted", 0) for f in commit["file_changes"].values()),
                    "lines_deleted": sum(f.get("lines_deleted", 0) for f in commit["file_changes"].values()),
                    "size_delta": sum(f.get("size_delta", 0) for f in commit["file_changes"].values()),
                    "parent_count": len(commit["commit"].get("parents", [])),
                }
                flat_commits.append(flat_commit)
            
            df = pd.DataFrame(flat_commits)
            csv_path = self.output_dir / component / "commits" / "commits.csv"
            df.to_csv(csv_path, index=False)
            
            # JSONサマリーも保存
            summary = {
                "total_commits": len(commits),
                "date_range": {
                    "start": self.start_date,
                    "end": self.end_date
                }
            }
            
            output_path = self.output_dir / component / "commits" / "summary.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self, all_data):
        """全体のサマリーデータを保存"""
        components_summary = {}
        total_changes = 0
        total_commits = 0
        
        for component, changes in all_data["changes"].items():
            component_changes = len(changes)
            total_changes += component_changes
            
            component_commits = len(all_data["commits"].get(component, []))
            total_commits += component_commits
            
            components_summary[component] = {
                "total_changes": component_changes,
                "total_commits": component_commits
            }
        
        summary = {
            "collection_date": datetime.now().isoformat(),
            "date_range": {
                "start": self.start_date,
                "end": self.end_date
            },
            "components": self.components,
            "total_components": len(self.components),
            "total_changes": total_changes,
            "total_commits": total_commits,
            "components_summary": components_summary,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay,
                "backoff_factor": self.retry_config.backoff_factor
            }
        }
        
        output_path = self.output_dir / "summary.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        summary_data = [
            {
                "component": comp,
                "changes": stats["total_changes"],
                "commits": stats["total_commits"]
            }
            for comp, stats in components_summary.items()
        ]
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / "component_summary.csv"
            df.to_csv(csv_path, index=False)

def main():
    """メイン関数"""
    logger.info("OpenStack Gerritデータ収集を開始します")
    
    # カスタムリトライ設定の例
    custom_retry_config = RetryConfig(
        max_retries=10,          # 最大10回までリトライ
        base_delay=30.0,         # 初期待機時間30秒
        max_delay=3840.0,         # 最大待機時間3840秒
        backoff_factor=2.0,     # 2倍ずつ増加
        jitter=True             # ジッター有効
    )
    
    collector = OpenStackGerritCollector(retry_config=custom_retry_config)
    
    try:
        collector.collect_data()
        logger.info("データ収集が正常に完了しました")
    except Exception as e:
        logger.error(f"データ収集中に回復不可能なエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()