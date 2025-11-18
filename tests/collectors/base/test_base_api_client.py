"""
BaseAPIClientの抽象基底クラスのテスト
"""
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from src.collectors.base.base_api_client import BaseAPIClient
from src.collectors.base.retry_handler import RetryConfig


class ConcreteAPIClient(BaseAPIClient):
    """テスト用の具象クラス"""
    
    def __init__(self, username: str, password: str, session: requests.Session,
                 timeout: tuple = (30, 120)):
        super().__init__(username, password, session, timeout)
    
    def get_endpoint_path(self, **kwargs) -> str:
        """エンドポイントパスを返す"""
        param = kwargs.get('param', 'default')
        return f"test/{param}"
    
    def fetch(self, param: str) -> dict:
        """テスト用のメソッド"""
        endpoint = self.get_endpoint_path(param=param)
        return self.make_request(endpoint)


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
        assert client.timeout == (30, 120)
    
    def test_initialization_with_custom_timeout(self):
        """カスタムタイムアウトでの初期化のテスト"""
        session = requests.Session()
        
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session,
            timeout=(10, 60)
        )
        
        assert client.timeout == (10, 60)
    
    def test_get_endpoint_path(self):
        """エンドポイントパス取得のテスト"""
        session = requests.Session()
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session
        )
        
        path = client.get_endpoint_path(param="test123")
        assert path == "test/test123"


class TestMakeRequest:
    """make_requestメソッドのテスト"""
    
    @patch('requests.Session.get')
    def test_successful_request(self, mock_get):
        """成功するリクエストのテスト"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ')]}\'\n{"status": "success", "data": "test"}'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        session = requests.Session()
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session
        )
        
        result = client.fetch("param1")
        
        assert result == {"status": "success", "data": "test"}
        mock_get.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
    
    @patch('requests.Session.get')
    def test_request_with_params(self, mock_get):
        """パラメータ付きリクエストのテスト"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ')]}\'\n{"result": "ok"}'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        session = requests.Session()
        client = ConcreteAPIClient(
            username="test_user",
            password="test_pass",
            session=session
        )
        
        result = client.make_request(
            "endpoint",
            params={"key": "value", "limit": 10}
        )
        
        assert result == {"result": "ok"}
        # URLに BASE_URL が含まれることを確認
        call_args = mock_get.call_args
        assert "endpoint" in call_args[0][0]
        assert call_args[1]["params"] == {"key": "value", "limit": 10}
    
    @patch('requests.Session.get')
    def test_request_timeout(self, mock_get):
        """タイムアウトエラーのテスト"""
        mock_get.side_effect = requests.Timeout("Request timeout")
        
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass",
            retry_config=RetryConfig(max_retries=2, initial_wait=0.01)
        )
        
        with pytest.raises(requests.Timeout):
            client.get_test_data("param1")
        
        # 初回 + 2回のリトライ = 3回
        assert mock_get.call_count == 3
    
    @patch('requests.Session.get')
    def test_request_connection_error(self, mock_get):
        """接続エラーのテスト"""
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass",
            retry_config=RetryConfig(max_retries=2, initial_wait=0.01)
        )
        
        with pytest.raises(requests.ConnectionError):
            client.get_test_data("param1")
        
        assert mock_get.call_count == 3
    
    @patch('requests.Session.get')
    def test_request_http_error(self, mock_get):
        """HTTPエラーのテスト"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass",
            retry_config=RetryConfig(max_retries=2, initial_wait=0.01)
        )
        
        with pytest.raises(requests.HTTPError):
            client.get_test_data("param1")
        
        assert mock_get.call_count == 3
    
    @patch('requests.Session.get')
    @patch('time.sleep')
    def test_retry_on_server_error(self, mock_sleep, mock_get):
        """サーバーエラー時のリトライのテスト"""
        # 最初の2回は500エラー、3回目は成功
        mock_response_error = Mock()
        mock_response_error.status_code = 500
        mock_response_error.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.text = ')]}\'\n{"status": "success"}'
        mock_response_success.raise_for_status = Mock()
        
        mock_get.side_effect = [
            mock_response_error,
            mock_response_error,
            mock_response_success
        ]
        
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass",
            retry_config=RetryConfig(max_retries=3, initial_wait=0.01, jitter=False)
        )
        
        result = client.get_test_data("param1")
        
        assert result == {"status": "success"}
        assert mock_get.call_count == 3


class TestParseResponse:
    """_parse_responseメソッドのテスト"""
    
    def test_parse_normal_json(self):
        """通常のJSON形式のパースのテスト"""
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass"
        )
        
        mock_response = Mock()
        mock_response.text = ')]}\'\n{"key": "value", "number": 123}'
        
        result = client._parse_response(mock_response)
        
        assert result == {"key": "value", "number": 123}
    
    def test_parse_json_without_prefix(self):
        """プレフィックスなしのJSON形式のパースのテスト"""
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass"
        )
        
        mock_response = Mock()
        mock_response.text = '{"key": "value", "number": 123}'
        
        result = client._parse_response(mock_response)
        
        assert result == {"key": "value", "number": 123}
    
    def test_parse_complex_json(self):
        """複雑なJSON構造のパースのテスト"""
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass"
        )
        
        mock_response = Mock()
        mock_response.text = ')]}\'\n{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "total": 2}'
        
        result = client._parse_response(mock_response)
        
        assert result == {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "total": 2
        }
    
    def test_parse_empty_response(self):
        """空のレスポンスのパースのテスト"""
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass"
        )
        
        mock_response = Mock()
        mock_response.text = ')]}\'\n{}'
        
        result = client._parse_response(mock_response)
        
        assert result == {}
    
    def test_parse_invalid_json(self):
        """不正なJSON形式のパースエラーのテスト"""
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass"
        )
        
        mock_response = Mock()
        mock_response.text = ')]}\'\nInvalid JSON'
        
        with pytest.raises(ValueError):
            client._parse_response(mock_response)
    
    def test_parse_json_with_unicode(self):
        """Unicode文字を含むJSONのパースのテスト"""
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass"
        )
        
        mock_response = Mock()
        mock_response.text = ')]}\'\n{"message": "こんにちは", "emoji": "🚀"}'
        
        result = client._parse_response(mock_response)
        
        assert result == {"message": "こんにちは", "emoji": "🚀"}


class TestAbstractMethods:
    """抽象クラスのテスト"""
    
    def test_cannot_instantiate_base_class(self):
        """BaseAPIClientを直接インスタンス化できないことのテスト"""
        with pytest.raises(TypeError):
            BaseAPIClient(
                base_url="https://api.example.com",
                username="test_user",
                password="test_pass"
            )
    
    def test_concrete_class_can_be_instantiated(self):
        """具象クラスはインスタンス化できることのテスト"""
        client = ConcreteAPIClient(
            base_url="https://api.example.com",
            username="test_user",
            password="test_pass"
        )
        
        assert isinstance(client, BaseAPIClient)
        assert isinstance(client, ConcreteAPIClient)


class TestIntegration:
    """統合テスト"""
    
    @patch('requests.Session.get')
    def test_full_request_flow(self, mock_get):
        """完全なリクエストフローのテスト"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ')]}\'\n{"id": "change123", "status": "MERGED", "subject": "Fix bug"}'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = ConcreteAPIClient(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass",
            retry_config=RetryConfig(max_retries=3)
        )
        
        result = client.get_test_data("change123")
        
        assert result["id"] == "change123"
        assert result["status"] == "MERGED"
        assert result["subject"] == "Fix bug"
        assert mock_get.call_count == 1
    
    @patch('requests.Session.get')
    @patch('time.sleep')
    def test_retry_and_recovery(self, mock_sleep, mock_get):
        """リトライと回復のテスト"""
        # 最初の2回は失敗、3回目は成功
        mock_get.side_effect = [
            requests.ConnectionError("Failed 1"),
            requests.Timeout("Failed 2"),
            Mock(
                status_code=200,
                text=')]}\'\n{"status": "recovered"}',
                raise_for_status=Mock()
            )
        ]
        
        client = ConcreteAPIClient(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass",
            retry_config=RetryConfig(
                max_retries=3,
                initial_wait=0.01,
                jitter=False
            )
        )
        
        result = client.get_test_data("test")
        
        assert result == {"status": "recovered"}
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
