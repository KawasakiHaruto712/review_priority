"""
レビュワー取得エンドポイント

変更のレビュワー情報を取得します。
"""

from typing import List, Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


class ReviewersEndpoint(BaseAPIClient):
    """レビュワー取得エンドポイント"""
    
    def get_endpoint_path(self, change_id: str, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return f"changes/{change_id}/reviewers"
    
    def fetch(self, change_id: str) -> List[Dict[str, Any]]:
        """
        レビュワー情報を取得
        
        Args:
            change_id: 変更ID
            
        Returns:
            レビュワー情報のリスト
        """
        endpoint = self.get_endpoint_path(change_id=change_id)
        return self.make_request(endpoint)
