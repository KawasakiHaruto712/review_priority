"""
ChangeDetailEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from src.collectors.endpoints.change_detail_endpoint import ChangeDetailEndpoint


class TestChangeDetailEndpoint:
    """ChangeDetailEndpointクラスのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        endpoint = ChangeDetailEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        assert endpoint.base_url == "https://review.openstack.org"
    
    @patch.object(ChangeDetailEndpoint, 'make_request')
    def test_get_change_detail_basic(self, mock_request):
        """基本的な変更詳細取得のテスト"""
        mock_request.return_value = {
            "id": "nova~master~I1234567890abcdef",
            "project": "openstack/nova",
            "branch": "master",
            "change_id": "I1234567890abcdef",
            "subject": "Fix bug",
            "status": "MERGED",
            "created": "2024-01-01 10:00:00.000000000",
            "updated": "2024-01-05 15:30:00.000000000",
            "owner": {
                "name": "John Doe",
                "email": "john@example.com"
            }
        }
        
        endpoint = ChangeDetailEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_change_detail(change_id="12345")
        
        assert result["change_id"] == "I1234567890abcdef"
        assert result["project"] == "openstack/nova"
        assert result["status"] == "MERGED"
        
        # URLの確認
        call_args = mock_request.call_args
        assert "changes/12345/detail" in call_args[0][0]
    
    @patch.object(ChangeDetailEndpoint, 'make_request')
    def test_get_change_detail_with_options(self, mock_request):
        """オプション指定での変更詳細取得のテスト"""
        mock_request.return_value = {
            "id": "nova~master~I1234567890abcdef",
            "messages": [
                {"message": "Patch Set 1: Code-Review+2"}
            ],
            "current_revision": "abc123",
            "revisions": {
                "abc123": {
                    "files": {
                        "nova/compute.py": {"lines_inserted": 10}
                    }
                }
            }
        }
        
        endpoint = ChangeDetailEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        options = ["CURRENT_REVISION", "CURRENT_FILES", "MESSAGES"]
        result = endpoint.get_change_detail(
            change_id="12345",
            options=options
        )
        
        assert "messages" in result
        assert "revisions" in result
        
        call_args = mock_request.call_args
        params = call_args[1]["params"]
        assert len(params["o"]) == 3
        assert "CURRENT_REVISION" in params["o"]
    
    @patch.object(ChangeDetailEndpoint, 'make_request')
    def test_get_change_detail_not_found(self, mock_request):
        """存在しない変更のテスト"""
        mock_request.side_effect = Exception("404 Not Found")
        
        endpoint = ChangeDetailEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        with pytest.raises(Exception, match="404 Not Found"):
            endpoint.get_change_detail(change_id="99999")
