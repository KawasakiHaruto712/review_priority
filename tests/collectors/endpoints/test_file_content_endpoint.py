"""
FileContentEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from src.collectors.endpoints.file_content_endpoint import FileContentEndpoint


class TestFileContentEndpoint:
    """FileContentEndpointクラスのテスト"""
    
    @patch.object(FileContentEndpoint, 'make_request')
    def test_get_file_content_basic(self, mock_request):
        """基本的なファイル内容取得のテスト"""
        mock_request.return_value = "ZGVmIG1haW4oKToKICAgIHByaW50KCJIZWxsbyBXb3JsZCIp"
        
        endpoint = FileContentEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_file_content(
            change_id="12345",
            revision_id="current",
            file_path="nova/compute.py"
        )
        
        # Base64デコード後の内容
        assert "def main()" in result
        assert "Hello World" in result
        
        call_args = mock_request.call_args
        assert "changes/12345/revisions/current/files/" in call_args[0][0]
        assert "nova%2Fcompute.py" in call_args[0][0] or "nova/compute.py" in call_args[0][0]
    
    @patch.object(FileContentEndpoint, 'make_request')
    def test_get_file_content_not_found(self, mock_request):
        """存在しないファイルのテスト"""
        mock_request.side_effect = Exception("404 Not Found")
        
        endpoint = FileContentEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        with pytest.raises(Exception, match="404 Not Found"):
            endpoint.get_file_content(
                change_id="12345",
                revision_id="current",
                file_path="non_existent.py"
            )
