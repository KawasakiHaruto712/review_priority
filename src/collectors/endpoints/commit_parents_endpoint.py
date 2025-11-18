"""
親コミット取得エンドポイント

コミットの親コミット情報を取得します。
"""

from typing import List, Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


class CommitParentsEndpoint(BaseAPIClient):
    """親コミット取得エンドポイント"""
    
    def get_endpoint_path(self, change_id: str, revision_id: str, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return f"changes/{change_id}/revisions/{revision_id}/commit"
    
    def fetch(self, change_id: str, revision_id: str) -> List[Dict[str, Any]]:
        """
        親コミット情報を取得
        
        Args:
            change_id: 変更ID
            revision_id: リビジョンID
            
        Returns:
            親コミットのリスト
        """
        endpoint = self.get_endpoint_path(change_id=change_id, revision_id=revision_id)
        commit_data = self.make_request(endpoint)
        
        if commit_data and "parents" in commit_data:
            return commit_data["parents"]
        return []
