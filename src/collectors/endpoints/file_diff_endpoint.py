"""
ファイル差分取得エンドポイント

特定のファイルの差分情報を取得します。
"""

from typing import Optional, Dict, Any
from urllib.parse import quote

from src.collectors.base.base_api_client import BaseAPIClient


class FileDiffEndpoint(BaseAPIClient):
    """ファイル差分取得エンドポイント"""
    
    def get_endpoint_path(self, change_id: str, revision_id: str, 
                         file_path: str, **kwargs) -> str:
        """エンドポイントパスを返す"""
        encoded_path = quote(file_path, safe='')
        return f"changes/{change_id}/revisions/{revision_id}/files/{encoded_path}/diff"
    
    def fetch(self, change_id: str, revision_id: str, 
              file_path: str) -> Optional[Dict[str, Any]]:
        """
        ファイル差分を取得
        
        Args:
            change_id: 変更ID
            revision_id: リビジョンID
            file_path: ファイルパス
            
        Returns:
            ファイル差分情報
        """
        endpoint = self.get_endpoint_path(
            change_id=change_id, 
            revision_id=revision_id, 
            file_path=file_path
        )
        diff_data = self.make_request(endpoint)
        
        if diff_data:
            return diff_data.get("content", "")
        return None
