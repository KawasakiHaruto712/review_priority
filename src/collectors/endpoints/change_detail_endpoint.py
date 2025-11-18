"""
変更詳細取得エンドポイント

特定の変更の詳細情報を取得します。
"""

from typing import Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


class ChangeDetailEndpoint(BaseAPIClient):
    """変更詳細取得エンドポイント"""
    
    def get_endpoint_path(self, change_id: str, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return f"changes/{change_id}/detail"
    
    def fetch(self, change_id: str) -> Dict[str, Any]:
        """
        変更の詳細情報を取得
        
        Args:
            change_id: 変更ID
            
        Returns:
            変更詳細情報
        """
        endpoint = self.get_endpoint_path(change_id=change_id)
        return self.make_request(endpoint)
