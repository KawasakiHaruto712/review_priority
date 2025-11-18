"""
CommentsEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from src.collectors.endpoints.comments_endpoint import CommentsEndpoint


class TestCommentsEndpoint:
    """CommentsEndpointクラスのテスト"""
    
    @patch.object(CommentsEndpoint, 'make_request')
    def test_get_comments_basic(self, mock_request):
        """基本的なコメント取得のテスト"""
        mock_request.return_value = {
            "nova/compute.py": [
                {
                    "id": "comment1",
                    "line": 42,
                    "message": "Consider using a different approach",
                    "updated": "2024-01-01 10:00:00.000000000",
                    "author": {
                        "name": "Reviewer1"
                    }
                }
            ],
            "nova/api.py": [
                {
                    "id": "comment2",
                    "line": 100,
                    "message": "This looks good",
                    "updated": "2024-01-01 11:00:00.000000000",
                    "author": {
                        "name": "Reviewer2"
                    }
                }
            ]
        }
        
        endpoint = CommentsEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_comments(
            change_id="12345",
            revision_id="current"
        )
        
        assert len(result) == 2
        assert "nova/compute.py" in result
        assert "nova/api.py" in result
        assert len(result["nova/compute.py"]) == 1
        
        call_args = mock_request.call_args
        assert "changes/12345/revisions/current/comments" in call_args[0][0]
    
    @patch.object(CommentsEndpoint, 'make_request')
    def test_get_comments_empty(self, mock_request):
        """コメントがない場合のテスト"""
        mock_request.return_value = {}
        
        endpoint = CommentsEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_comments(
            change_id="12345",
            revision_id="current"
        )
        
        assert result == {}
    
    @patch.object(CommentsEndpoint, 'make_request')
    def test_get_comments_specific_revision(self, mock_request):
        """特定リビジョンのコメント取得のテスト"""
        mock_request.return_value = {}
        
        endpoint = CommentsEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_comments(
            change_id="12345",
            revision_id="abc123def456"
        )
        
        call_args = mock_request.call_args
        assert "revisions/abc123def456/comments" in call_args[0][0]
