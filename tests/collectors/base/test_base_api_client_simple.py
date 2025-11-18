"""
BaseAPIClientの簡略化されたテスト
"""
import pytest
import requests
from unittest.mock import Mock, patch
from src.collectors.base.base_api_client import BaseAPIClient


class ConcreteAPIClient(BaseAPIClient):
    """テスト用の具象クラス"""
    
    def get_endpoint_path(self, **kwargs) -> str:
        """エンドポイントパスを返す"""
        return "test/endpoint"
    
    def fetch(self, **kwargs):
        """fetchメソッドの実装"""
        return {}


class TestBaseAPIClient:
    """BaseAPIClientクラスのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        session = requests.Session()
        session.auth = ("test_user", "test_pass")
        
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session
        )
        
        assert client.username == "test_user"
        assert client.password == "test_pass"
        assert client.session == session
    
    @patch('requests.Session.get')
    def test_make_request_success(self, mock_get):
        """正常なリクエストのテスト"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ')]}\'\n{"status": "success"}'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        session = requests.Session()
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session
        )
        
        result = client.make_request("test")
        
        assert result == {"status": "success"}
        mock_response.raise_for_status.assert_called_once()
    
    def test_parse_response_with_prefix(self):
        """Gerritプレフィックス付きレスポンスのパース"""
        session = requests.Session()
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session
        )
        
        mock_response = Mock()
        mock_response.text = ')]}\'\n{"key": "value"}'
        
        result = client._parse_response(mock_response)
        assert result == {"key": "value"}
    
    def test_parse_response_without_prefix(self):
        """プレフィックスなしレスポンスのパース"""
        session = requests.Session()
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session
        )
        
        mock_response = Mock()
        mock_response.text = '{"key": "value"}'
        
        result = client._parse_response(mock_response)
        assert result == {"key": "value"}
    
    def test_cannot_instantiate_abstract_class(self):
        """抽象クラスを直接インスタンス化できないことのテスト"""
        session = requests.Session()
        
        with pytest.raises(TypeError):
            BaseAPIClient(
                username="test",
                password="test",
                session=session
            )
