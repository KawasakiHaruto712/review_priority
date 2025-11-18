"""
変更リスト取得エンドポイント

OpenStack Gerritの変更リストを取得します。
"""

from typing import List, Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


class ChangesEndpoint(BaseAPIClient):
    """変更リスト取得エンドポイント"""
    
    def get_endpoint_path(self, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return "changes/"
    
    def fetch(self, component: str, start_date: str, end_date: str,
              limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """
        変更リストを取得
        
        Args:
            component: OpenStackコンポーネント名
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            limit: 取得件数
            skip: スキップ件数
            
        Returns:
            変更リスト
        """
        query = f"project:openstack/{component} after:{start_date} before:{end_date}"
        
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
        
        endpoint = self.get_endpoint_path()
        return self.make_request(endpoint, params)
