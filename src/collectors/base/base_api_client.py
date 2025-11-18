"""
基底APIクライアントモジュール

Gerrit APIクライアントの基底クラスを提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
import json
import time
import logging

from src.collectors.base.retry_handler import retry_with_backoff

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """Gerrit APIクライアントの基底クラス"""
    
    BASE_URL = "https://review.opendev.org/a"
    
    def __init__(self, username: str, password: str, session: requests.Session, 
                 timeout: tuple = (30, 120)):
        """
        Args:
            username: Gerritユーザー名
            password: Gerritパスワード
            session: リクエストセッション
            timeout: タイムアウト設定 (接続, 読み取り)
        """
        self.username = username
        self.password = password
        self.session = session
        self.timeout = timeout
    
    @abstractmethod
    def get_endpoint_path(self, **kwargs) -> str:
        """
        エンドポイントのパスを返す
        
        Returns:
            エンドポイントパス
        """
        pass
    
    @retry_with_backoff()
    def make_request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        APIリクエストを実行
        
        Args:
            endpoint: APIエンドポイント
            params: クエリパラメータ
            
        Returns:
            レスポンスデータ
        """
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        return self._parse_response(response)
    
    def _parse_response(self, response: requests.Response) -> Any:
        """
        Gerrit APIレスポンスをパース
        
        Args:
            response: HTTPレスポンス
            
        Returns:
            パースされたデータ
        """
        data = response.text
        if data.startswith(")]}'"):
            if '\n' in data:
                data = data[data.find('\n') + 1:]
            else:
                data = data[4:]
        
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSONデコードエラー: {e}")
            logger.debug(f"受信したレスポンスデータ: {data[:200]}...")
            raise ValueError(f"JSONデコードエラー: {e}")
        
        time.sleep(1)  # API制限回避
        return parsed_data
    
    @abstractmethod
    def fetch(self, **kwargs) -> Any:
        """
        データを取得する（各エンドポイントで実装）
        
        Returns:
            取得したデータ
        """
        pass
