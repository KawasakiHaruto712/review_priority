"""
コレクター設定管理モジュール

エンドポイント設定ファイルを読み込み、管理します。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.constants import OPENSTACK_CORE_COMPONENTS, START_DATE, END_DATE
from src.config.path import DEFAULT_DATA_DIR


class CollectorConfig:
    """コレクター設定管理クラス"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: 設定ファイルパス（省略時はデフォルト）
        """
        if config_path is None:
            config_path = Path(__file__).parent / "endpoint_config.yaml"
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # constants.pyからデフォルト値を読み込み
        if config['collection']['components'] is None:
            config['collection']['components'] = OPENSTACK_CORE_COMPONENTS
        
        if config['collection']['start_date'] is None:
            config['collection']['start_date'] = START_DATE
        
        if config['collection']['end_date'] is None:
            config['collection']['end_date'] = END_DATE
        
        if config['storage']['output_dir'] is None:
            # デフォルトはdataディレクトリ配下のopenstack_collectedディレクトリ
            # 既存のdata/openstack/ディレクトリとは別に管理
            config['storage']['output_dir'] = str(DEFAULT_DATA_DIR / "openstack_collected")
        
        return config
    
    def get_enabled_endpoints(self) -> List[Dict[str, Any]]:
        """有効なエンドポイントをリストで取得（優先度順）"""
        endpoints = []
        for name, endpoint_config in self.config['endpoints'].items():
            if endpoint_config.get('enabled', False):
                endpoints.append({
                    'name': name,
                    'config': endpoint_config
                })
        
        # 優先度順にソート
        endpoints.sort(key=lambda x: x['config'].get('priority', 999))
        return endpoints
    
    def get_endpoint_config(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """特定のエンドポイント設定を取得"""
        return self.config['endpoints'].get(endpoint_name)
    
    def is_endpoint_enabled(self, endpoint_name: str) -> bool:
        """エンドポイントが有効かチェック"""
        endpoint = self.get_endpoint_config(endpoint_name)
        return endpoint.get('enabled', False) if endpoint else False
    
    def get_collection_config(self) -> Dict[str, Any]:
        """収集設定を取得"""
        return self.config['collection']
    
    def get_retry_config(self) -> Dict[str, Any]:
        """リトライ設定を取得"""
        return self.config['retry']
    
    def get_storage_config(self) -> Dict[str, Any]:
        """ストレージ設定を取得"""
        return self.config['storage']
