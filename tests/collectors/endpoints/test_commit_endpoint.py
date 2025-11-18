"""
CommitEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from src.collectors.endpoints.commit_endpoint import CommitEndpoint


class TestCommitEndpoint:
    """CommitEndpointクラスのテスト"""
    
    @patch.object(CommitEndpoint, 'make_request')
    def test_get_commit_basic(self, mock_request):
        """基本的なコミット情報取得のテスト"""
        mock_request.return_value = {
            "commit": "abc123def456",
            "parents": [
                {"commit": "parent123"}
            ],
            "author": {
                "name": "John Doe",
                "email": "john@example.com",
                "date": "2024-01-01 10:00:00.000000000"
            },
            "committer": {
                "name": "Gerrit Code Review",
                "email": "gerrit@openstack.org",
                "date": "2024-01-01 10:05:00.000000000"
            },
            "subject": "Fix bug in compute module",
            "message": "Fix bug in compute module\n\nDetailed description of the fix."
        }
        
        endpoint = CommitEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_commit(
            change_id="12345",
            revision_id="current"
        )
        
        assert result["commit"] == "abc123def456"
        assert result["author"]["name"] == "John Doe"
        assert result["subject"] == "Fix bug in compute module"
        
        call_args = mock_request.call_args
        assert "changes/12345/revisions/current/commit" in call_args[0][0]
    
    @patch.object(CommitEndpoint, 'make_request')
    def test_get_commit_specific_revision(self, mock_request):
        """特定リビジョンのコミット情報取得のテスト"""
        mock_request.return_value = {
            "commit": "specific123",
            "author": {"name": "Alice"},
            "subject": "Specific revision"
        }
        
        endpoint = CommitEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_commit(
            change_id="12345",
            revision_id="abc123def456"
        )
        
        call_args = mock_request.call_args
        assert "revisions/abc123def456/commit" in call_args[0][0]
    
    @patch.object(CommitEndpoint, 'make_request')
    def test_get_commit_not_found(self, mock_request):
        """存在しないコミットのテスト"""
        mock_request.side_effect = Exception("404 Not Found")
        
        endpoint = CommitEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        with pytest.raises(Exception, match="404 Not Found"):
            endpoint.get_commit(
                change_id="99999",
                revision_id="current"
            )
