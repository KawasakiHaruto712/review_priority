"""
エンドポイントクラスの簡略化されたテスト
"""
import pytest
import requests
from unittest.mock import Mock, patch
from src.collectors.endpoints.changes_endpoint import ChangesEndpoint
from src.collectors.endpoints.change_detail_endpoint import ChangeDetailEndpoint
from src.collectors.endpoints.comments_endpoint import CommentsEndpoint


class TestChangesEndpoint:
    """ChangesEndpointのテスト"""
    
    def test_get_endpoint_path(self):
        """エンドポイントパスのテスト"""
        session = requests.Session()
        endpoint = ChangesEndpoint(
            username="test",
            password="test",
            session=session
        )
        
        path = endpoint.get_endpoint_path()
        assert path == "changes/"
    
    @patch('src.collectors.base.base_api_client.BaseAPIClient.make_request')
    def test_fetch(self, mock_request):
        """fetchメソッドのテスト"""
        mock_request.return_value = [
            {"id": "change1", "project": "openstack/nova"}
        ]
        
        session = requests.Session()
        endpoint = ChangesEndpoint(
            username="test",
            password="test",
            session=session
        )
        
        result = endpoint.fetch(
            component="nova",
            start_date="2024-01-01",
            end_date="2024-12-31",
            limit=100
        )
        
        assert len(result) == 1
        assert result[0]["id"] == "change1"
        mock_request.assert_called_once()


class TestChangeDetailEndpoint:
    """ChangeDetailEndpointのテスト"""
    
    def test_get_endpoint_path(self):
        """エンドポイントパスのテスト"""
        session = requests.Session()
        endpoint = ChangeDetailEndpoint(
            username="test",
            password="test",
            session=session
        )
        
        path = endpoint.get_endpoint_path(change_id="12345")
        assert "12345" in path
        assert "detail" in path
    
    @patch('src.collectors.base.base_api_client.BaseAPIClient.make_request')
    def test_fetch(self, mock_request):
        """fetchメソッドのテスト"""
        mock_request.return_value = {
            "id": "change1",
            "project": "openstack/nova",
            "subject": "Fix bug"
        }
        
        session = requests.Session()
        endpoint = ChangeDetailEndpoint(
            username="test",
            password="test",
            session=session
        )
        
        result = endpoint.fetch(change_id="12345")
        
        assert result["id"] == "change1"
        assert result["subject"] == "Fix bug"
        mock_request.assert_called_once()


class TestCommentsEndpoint:
    """CommentsEndpointクラスのテスト"""
    
    def test_get_endpoint_path(self):
        """エンドポイントパスのテスト"""
        session = requests.Session()
        endpoint = CommentsEndpoint(
            username="test",
            password="test",
            session=session
        )
        
        path = endpoint.get_endpoint_path(
            change_id="12345"
        )
        assert "12345" in path
        assert "comments" in path
    
    @patch('src.collectors.base.base_api_client.BaseAPIClient.make_request')
    def test_fetch(self, mock_request):
        """fetchメソッドのテスト"""
        mock_request.return_value = {
            "file.py": [{"line": 10, "message": "Good"}]
        }
        
        session = requests.Session()
        endpoint = CommentsEndpoint(
            username="test",
            password="test",
            session=session
        )
        
        result = endpoint.fetch(
            change_id="12345"
        )
        
        assert result is not None
        mock_request.assert_called_once()
