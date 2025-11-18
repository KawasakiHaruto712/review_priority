"""
変更データ収集オーケストレーター

OpenStack Gerritから変更データを収集するメインクラスです。
"""

import os
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import requests

from src.collectors.config.collector_config import CollectorConfig
from src.collectors.base.retry_handler import RetryConfig
from src.collectors.storage.change_storage import ChangeStorage
from src.collectors.storage.commit_storage import CommitStorage

# エンドポイントクラスのインポート
from src.collectors.endpoints.changes_endpoint import ChangesEndpoint
from src.collectors.endpoints.change_detail_endpoint import ChangeDetailEndpoint
from src.collectors.endpoints.comments_endpoint import CommentsEndpoint
from src.collectors.endpoints.reviewers_endpoint import ReviewersEndpoint
from src.collectors.endpoints.file_content_endpoint import FileContentEndpoint
from src.collectors.endpoints.file_diff_endpoint import FileDiffEndpoint
from src.collectors.endpoints.commit_endpoint import CommitEndpoint
from src.collectors.endpoints.commit_parents_endpoint import CommitParentsEndpoint

logger = logging.getLogger(__name__)


class ChangeCollector:
    """OpenStack Gerrit変更データ収集オーケストレーター"""
    
    # エンドポイントクラスマッピング
    ENDPOINT_CLASSES = {
        'ChangesEndpoint': ChangesEndpoint,
        'ChangeDetailEndpoint': ChangeDetailEndpoint,
        'CommentsEndpoint': CommentsEndpoint,
        'ReviewersEndpoint': ReviewersEndpoint,
        'FileContentEndpoint': FileContentEndpoint,
        'FileDiffEndpoint': FileDiffEndpoint,
        'CommitEndpoint': CommitEndpoint,
        'CommitParentsEndpoint': CommitParentsEndpoint,
    }
    
    def __init__(self, config_path: Optional[Path] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルパス
            username: Gerritユーザー名
            password: Gerritパスワード
        """
        load_dotenv()
        
        # 設定読み込み
        self.config = CollectorConfig(config_path)
        
        # 認証情報
        self.username = username or os.getenv("GERRIT_USERNAME")
        self.password = password or os.getenv("GERRIT_PASSWORD")
        
        # セッション設定
        self.session = self._create_session()
        
        # リトライ設定
        retry_config_dict = self.config.get_retry_config()
        self.retry_config = RetryConfig(**retry_config_dict)
        
        # エンドポイントインスタンスを作成
        self.endpoints = self._initialize_endpoints()
        
        # ストレージ初期化
        storage_config = self.config.get_storage_config()
        self.change_storage = ChangeStorage(Path(storage_config['output_dir']))
        self.commit_storage = CommitStorage(Path(storage_config['output_dir']))
        
        logger.info("ChangeCollector初期化完了")
    
    def _create_session(self) -> requests.Session:
        """HTTPセッションを作成"""
        session = requests.Session()
        auth_string = base64.b64encode(
            f"{self.username}:{self.password}".encode()
        ).decode()
        
        session.headers.update({
            "Authorization": f"Basic {auth_string}",
            "Accept": "application/json"
        })
        
        return session
    
    def _initialize_endpoints(self) -> Dict[str, Any]:
        """エンドポイントインスタンスを初期化"""
        endpoints = {}
        enabled_endpoints = self.config.get_enabled_endpoints()
        
        for endpoint_info in enabled_endpoints:
            name = endpoint_info['name']
            endpoint_config = endpoint_info['config']
            class_name = endpoint_config['class']
            
            if class_name in self.ENDPOINT_CLASSES:
                endpoint_class = self.ENDPOINT_CLASSES[class_name]
                endpoints[name] = endpoint_class(
                    username=self.username,
                    password=self.password,
                    session=self.session,
                    timeout=(30, 120)
                )
                logger.info(f"エンドポイント初期化: {name} ({class_name})")
            else:
                logger.warning(f"未知のエンドポイントクラス: {class_name}")
        
        return endpoints
    
    def collect_all_components(self):
        """全コンポーネントのデータを収集"""
        collection_config = self.config.get_collection_config()
        components = collection_config['components']
        
        logger.info(f"データ収集開始: {len(components)}コンポーネント")
        
        for component in components:
            self.collect_component(component)
        
        logger.info("全コンポーネントのデータ収集完了")
    
    def collect_component(self, component: str):
        """特定コンポーネントのデータを収集"""
        logger.info(f"{component} の収集開始")
        
        collection_config = self.config.get_collection_config()
        batch_size = collection_config['batch_size']
        start_date = collection_config['start_date']
        end_date = collection_config['end_date']
        
        skip = 0
        component_changes = []
        component_commits = []
        
        # 1. 変更リストを取得
        while True:
            if not self.config.is_endpoint_enabled('changes'):
                logger.warning("changesエンドポイントが無効です")
                break
            
            changes = self.endpoints['changes'].fetch(
                component=component,
                start_date=start_date,
                end_date=end_date,
                limit=batch_size,
                skip=skip
            )
            
            if not changes:
                break
            
            # 2. 各変更の詳細データを収集
            for change in changes:
                change_data = self._collect_change_details(change, component)
                
                if change_data:
                    component_changes.append(change_data['change'])
                    component_commits.extend(change_data['commits'])
            
            skip += len(changes)
        
        # 3. データを保存
        self.change_storage.save_component_data(component, component_changes)
        self.commit_storage.save_component_data(component, component_commits)
        
        logger.info(f"{component} の収集完了: {len(component_changes)}変更, {len(component_commits)}コミット")
    
    def _collect_change_details(self, change: Dict[str, Any], 
                                component: str) -> Optional[Dict[str, Any]]:
        """変更の詳細データを収集"""
        change_id = change['id']
        change_number = change['_number']
        
        try:
            result = {
                'change': change.copy(),
                'commits': []
            }
            
            # _number を change_number として保存（ストレージでの検索用）
            result['change']['change_number'] = change_number
            
            # 変更詳細
            if self.config.is_endpoint_enabled('change_detail'):
                detail = self.endpoints['change_detail'].fetch(change_id=change_id)
                result['change'].update(detail)
            
            # コメント
            if self.config.is_endpoint_enabled('comments'):
                comments = self.endpoints['comments'].fetch(change_id=change_id)
                result['change']['comments'] = comments
            
            # レビュワー
            if self.config.is_endpoint_enabled('reviewers'):
                reviewers = self.endpoints['reviewers'].fetch(change_id=change_id)
                result['change']['reviewers'] = reviewers
            
            # コミット情報
            if self.config.is_endpoint_enabled('commit'):
                revisions = change.get('revisions', {})
                for revision_id in revisions.keys():
                    commit_data = self._collect_commit_details(
                        change_id, revision_id, change_number
                    )
                    if commit_data:
                        result['commits'].append(commit_data)
            
            logger.info(f"処理中: {component} - PR #{change_number}")
            
            return result
            
        except Exception as e:
            logger.error(f"変更 #{change_number} の収集エラー: {e}")
            return None
    
    def _collect_commit_details(self, change_id: str, revision_id: str,
                                change_number: int) -> Optional[Dict[str, Any]]:
        """コミットの詳細データを収集"""
        try:
            commit_data = {
                'change_id': change_id,
                'revision_id': revision_id,
                'change_number': change_number,
            }
            
            # コミット情報
            commit_info = self.endpoints['commit'].fetch(
                change_id=change_id,
                revision_id=revision_id
            )
            commit_data['commit'] = commit_info
            
            # ファイル情報
            files = commit_info.get('files', {})
            commit_data['files'] = list(files.keys())
            commit_data['file_changes'] = {
                path: {
                    "lines_inserted": info.get("lines_inserted", 0),
                    "lines_deleted": info.get("lines_deleted", 0),
                    "size_delta": info.get("size_delta", 0)
                } for path, info in files.items()
            }
            
            # 親コミット
            if self.config.is_endpoint_enabled('commit_parents'):
                parents = self.endpoints['commit_parents'].fetch(
                    change_id=change_id,
                    revision_id=revision_id
                )
                commit_data['commit']['parents'] = parents
            
            # ファイル差分
            if self.config.is_endpoint_enabled('file_diff'):
                file_diffs = {}
                
                for file_path in files.keys():
                    try:
                        diff = self.endpoints['file_diff'].fetch(
                            change_id=change_id,
                            revision_id=revision_id,
                            file_path=file_path
                        )
                        if diff:
                            file_diffs[file_path] = diff
                    except Exception as e:
                        logger.warning(f"ファイル差分取得エラー ({file_path}): {e}")
                
                commit_data['file_diffs'] = file_diffs
            
            return commit_data
            
        except Exception as e:
            logger.error(f"コミット {revision_id[:8]} の収集エラー: {e}")
            return None


def main():
    """メイン関数"""
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger.info("OpenStack Gerritデータ収集を開始します（リファクタリング版）")
    
    try:
        collector = ChangeCollector()
        collector.collect_all_components()
        logger.info("データ収集が正常に完了しました")
    except Exception as e:
        logger.error(f"データ収集中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
