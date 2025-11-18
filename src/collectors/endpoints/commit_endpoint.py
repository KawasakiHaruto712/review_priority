"""
コミット情報取得エンドポイント

特定のリビジョンのコミット情報を取得します。
"""

from typing import Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


class CommitEndpoint(BaseAPIClient):
    """コミット情報取得エンドポイント"""
    
    def get_endpoint_path(self, change_id: str, revision_id: str, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return f"changes/{change_id}/revisions/{revision_id}/commit"
    
    def fetch(self, change_id: str, revision_id: str) -> Dict[str, Any]:
        """
        コミット情報を取得
        
        Args:
            change_id: 変更ID
            revision_id: リビジョンID
            
        Returns:
            コミット情報
        """
        endpoint = self.get_endpoint_path(change_id=change_id, revision_id=revision_id)
        return self.make_request(endpoint)
