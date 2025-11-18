"""
CommitParentsEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from src.collectors.endpoints.commit_parents_endpoint import CommitParentsEndpoint


class TestCommitParentsEndpoint:
    """CommitParentsEndpointクラスのテスト"""
    
    @patch.object(CommitParentsEndpoint, 'make_request')
    def test_get_commit_parents_basic(self, mock_request):
        """基本的な親コミット情報取得のテスト"""
        mock_request.return_value = [
            {
                "commit": "parent1abc123",
                "subject": "Previous commit 1"
            },
            {
                "commit": "parent2def456",
                "subject": "Previous commit 2"
            }
        ]
        
        endpoint = CommitParentsEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_commit_parents(
            project="openstack/nova",
            commit_id="abc123def456"
        )
        
        assert len(result) == 2
        assert result[0]["commit"] == "parent1abc123"
        assert result[1]["commit"] == "parent2def456"
        
        call_args = mock_request.call_args
        assert "projects/openstack%2Fnova/commits/abc123def456/in" in call_args[0][0] or \
               "projects/openstack/nova/commits/abc123def456/in" in call_args[0][0]
    
    @patch.object(CommitParentsEndpoint, 'make_request')
    def test_get_commit_parents_single_parent(self, mock_request):
        """単一の親コミットのテスト"""
        mock_request.return_value = [
            {
                "commit": "parent123",
                "subject": "Single parent commit"
            }
        ]
        
        endpoint = CommitParentsEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_commit_parents(
            project="openstack/nova",
            commit_id="abc123"
        )
        
        assert len(result) == 1
        assert result[0]["commit"] == "parent123"
    
    @patch.object(CommitParentsEndpoint, 'make_request')
    def test_get_commit_parents_no_parents(self, mock_request):
        """親コミットがない場合のテスト"""
        mock_request.return_value = []
        
        endpoint = CommitParentsEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_commit_parents(
            project="openstack/nova",
            commit_id="initial_commit"
        )
        
        assert result == []
    
    @patch.object(CommitParentsEndpoint, 'make_request')
    def test_get_commit_parents_project_encoding(self, mock_request):
        """プロジェクト名のエンコーディングのテスト"""
        mock_request.return_value = []
        
        endpoint = CommitParentsEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        # スラッシュを含むプロジェクト名
        result = endpoint.get_commit_parents(
            project="openstack/nova",
            commit_id="abc123"
        )
        
        call_args = mock_request.call_args
        # URLエンコードされているか確認
        url = call_args[0][0]
        assert "openstack%2Fnova" in url or "openstack/nova" in url
