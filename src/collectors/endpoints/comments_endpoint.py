"""
コメント取得エンドポイント

変更に対するコメント情報を取得します。
"""

from typing import Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


class CommentsEndpoint(BaseAPIClient):
    """コメント取得エンドポイント"""
    
    def get_endpoint_path(self, change_id: str, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return f"changes/{change_id}/comments"
    
    def fetch(self, change_id: str) -> Dict[str, Any]:
        """
        コメント情報を取得
        
        Args:
            change_id: 変更ID
            
        Returns:
            コメント情報
        """
        endpoint = self.get_endpoint_path(change_id=change_id)
        return self.make_request(endpoint)
