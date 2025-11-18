"""
ファイル内容取得エンドポイント

特定のリビジョンのファイル内容を取得します。
"""

import json
import time
import logging
from typing import Optional
from urllib.parse import quote

from src.collectors.base.base_api_client import BaseAPIClient
from src.collectors.base.retry_handler import retry_with_backoff
from src.utils.lang_identifiyer import identify_lang_from_file

logger = logging.getLogger(__name__)


class FileContentEndpoint(BaseAPIClient):
    """ファイル内容取得エンドポイント"""
    
    def get_endpoint_path(self, change_id: str, revision_id: str, 
                         file_path: str, **kwargs) -> str:
        """エンドポイントパスを返す"""
        encoded_path = quote(file_path, safe='')
        return f"changes/{change_id}/revisions/{revision_id}/files/{encoded_path}/content"
    
    @retry_with_backoff()
    def fetch(self, change_id: str, revision_id: str, 
              file_path: str) -> Optional[str]:
        """
        ファイル内容を取得
        
        Args:
            change_id: 変更ID
            revision_id: リビジョンID
            file_path: ファイルパス
            
        Returns:
            ファイル内容（取得できない場合はNone）
        """
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
        
        endpoint = self.get_endpoint_path(
            change_id=change_id, 
            revision_id=revision_id, 
            file_path=file_path
        )
        url = f"{self.BASE_URL}/{endpoint}"
        
        response = self.session.get(url, timeout=self.timeout)
        
        if response.status_code == 404:
            # 404エラーは正常なケースとして処理
            logger.info(f"ファイル {file_path} はこのリビジョンでは見つかりませんでした")
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
