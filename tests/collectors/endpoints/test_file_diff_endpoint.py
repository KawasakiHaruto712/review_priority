"""
FileDiffEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from src.collectors.endpoints.file_diff_endpoint import FileDiffEndpoint


class TestFileDiffEndpoint:
    """FileDiffEndpointクラスのテスト"""
    
    @patch.object(FileDiffEndpoint, 'make_request')
    def test_get_file_diff_basic(self, mock_request):
        """基本的なファイル差分取得のテスト"""
        mock_request.return_value = {
            "change_type": "MODIFIED",
            "old_path": "nova/compute.py",
            "new_path": "nova/compute.py",
            "lines_inserted": 10,
            "lines_deleted": 5,
            "size_delta": 250,
            "size": 5000
        }
        
        endpoint = FileDiffEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_file_diff(
            change_id="12345",
            revision_id="current",
            file_path="nova/compute.py"
        )
        
        assert result["change_type"] == "MODIFIED"
        assert result["lines_inserted"] == 10
        assert result["lines_deleted"] == 5
        
        call_args = mock_request.call_args
        assert "changes/12345/revisions/current/files/" in call_args[0][0]
        assert "diff" in call_args[0][0]
    
    @patch.object(FileDiffEndpoint, 'make_request')
    def test_get_file_diff_added_file(self, mock_request):
        """追加されたファイルの差分のテスト"""
        mock_request.return_value = {
            "change_type": "ADDED",
            "new_path": "nova/new_feature.py",
            "lines_inserted": 100,
            "lines_deleted": 0
        }
        
        endpoint = FileDiffEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_file_diff(
            change_id="12345",
            revision_id="current",
            file_path="nova/new_feature.py"
        )
        
        assert result["change_type"] == "ADDED"
        assert result["lines_inserted"] == 100
        assert result["lines_deleted"] == 0
