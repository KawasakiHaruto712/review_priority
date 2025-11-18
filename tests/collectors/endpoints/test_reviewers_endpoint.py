"""
ReviewersEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from src.collectors.endpoints.reviewers_endpoint import ReviewersEndpoint


class TestReviewersEndpoint:
    """ReviewersEndpointクラスのテスト"""
    
    @patch.object(ReviewersEndpoint, 'make_request')
    def test_get_reviewers_basic(self, mock_request):
        """基本的なレビュアー取得のテスト"""
        mock_request.return_value = [
            {
                "_account_id": 1000001,
                "name": "Alice",
                "email": "alice@example.com",
                "username": "alice"
            },
            {
                "_account_id": 1000002,
                "name": "Bob",
                "email": "bob@example.com",
                "username": "bob"
            }
        ]
        
        endpoint = ReviewersEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_reviewers(change_id="12345")
        
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"
        
        call_args = mock_request.call_args
        assert "changes/12345/reviewers" in call_args[0][0]
    
    @patch.object(ReviewersEndpoint, 'make_request')
    def test_get_reviewers_empty(self, mock_request):
        """レビュアーがいない場合のテスト"""
        mock_request.return_value = []
        
        endpoint = ReviewersEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_reviewers(change_id="12345")
        
        assert result == []
