"""
ChangesEndpointのテスト
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.collectors.endpoints.changes_endpoint import ChangesEndpoint
from src.collectors.base.retry_handler import RetryConfig


class TestChangesEndpoint:
    """ChangesEndpointクラスのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        assert endpoint.base_url == "https://review.openstack.org"
        assert endpoint.username == "test_user"
        assert endpoint.password == "test_pass"
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_basic(self, mock_request):
        """基本的な変更取得のテスト"""
        mock_request.return_value = [
            {
                "id": "change1",
                "project": "openstack/nova",
                "subject": "Fix bug",
                "status": "MERGED"
            },
            {
                "id": "change2",
                "project": "openstack/nova",
                "subject": "Add feature",
                "status": "NEW"
            }
        ]
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_changes(project="openstack/nova")
        
        assert len(result) == 2
        assert result[0]["id"] == "change1"
        assert result[1]["id"] == "change2"
        
        # URLとパラメータの確認
        call_args = mock_request.call_args
        assert "changes/" in call_args[0][0]
        assert call_args[1]["params"]["q"] == "project:openstack/nova"
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_with_status(self, mock_request):
        """ステータス指定での変更取得のテスト"""
        mock_request.return_value = [
            {
                "id": "change1",
                "project": "openstack/nova",
                "status": "MERGED"
            }
        ]
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_changes(
            project="openstack/nova",
            status="merged"
        )
        
        assert len(result) == 1
        
        call_args = mock_request.call_args
        query = call_args[1]["params"]["q"]
        assert "project:openstack/nova" in query
        assert "status:merged" in query
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_with_date_range(self, mock_request):
        """日付範囲指定での変更取得のテスト"""
        mock_request.return_value = []
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        since = datetime(2024, 1, 1)
        until = datetime(2024, 12, 31)
        
        result = endpoint.get_changes(
            project="openstack/nova",
            since=since,
            until=until
        )
        
        call_args = mock_request.call_args
        query = call_args[1]["params"]["q"]
        assert "project:openstack/nova" in query
        assert "after:2024-01-01" in query
        assert "before:2024-12-31" in query
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_with_branch(self, mock_request):
        """ブランチ指定での変更取得のテスト"""
        mock_request.return_value = []
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_changes(
            project="openstack/nova",
            branch="stable/yoga"
        )
        
        call_args = mock_request.call_args
        query = call_args[1]["params"]["q"]
        assert "project:openstack/nova" in query
        assert "branch:stable/yoga" in query
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_with_limit(self, mock_request):
        """制限数指定での変更取得のテスト"""
        mock_request.return_value = [{"id": f"change{i}"} for i in range(50)]
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_changes(
            project="openstack/nova",
            limit=50
        )
        
        assert len(result) == 50
        
        call_args = mock_request.call_args
        assert call_args[1]["params"]["n"] == 50
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_with_start(self, mock_request):
        """開始位置指定での変更取得のテスト"""
        mock_request.return_value = []
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_changes(
            project="openstack/nova",
            start=100
        )
        
        call_args = mock_request.call_args
        assert call_args[1]["params"]["S"] == 100
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_multiple_conditions(self, mock_request):
        """複数条件での変更取得のテスト"""
        mock_request.return_value = []
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_changes(
            project="openstack/nova",
            status="merged",
            branch="master",
            since=datetime(2024, 1, 1),
            until=datetime(2024, 12, 31),
            limit=100,
            start=0
        )
        
        call_args = mock_request.call_args
        query = call_args[1]["params"]["q"]
        
        assert "project:openstack/nova" in query
        assert "status:merged" in query
        assert "branch:master" in query
        assert "after:2024-01-01" in query
        assert "before:2024-12-31" in query
        assert call_args[1]["params"]["n"] == 100
        assert call_args[1]["params"]["S"] == 0
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_empty_result(self, mock_request):
        """空の結果のテスト"""
        mock_request.return_value = []
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        result = endpoint.get_changes(project="openstack/nova")
        
        assert result == []
        assert isinstance(result, list)
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_get_changes_pagination(self, mock_request):
        """ページネーションのテスト"""
        # 1ページ目
        mock_request.side_effect = [
            [{"id": f"change{i}"} for i in range(500)],
            [{"id": f"change{i}"} for i in range(500, 750)],
            []
        ]
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        # 複数回呼び出してページネーション
        page1 = endpoint.get_changes(project="openstack/nova", limit=500, start=0)
        page2 = endpoint.get_changes(project="openstack/nova", limit=500, start=500)
        page3 = endpoint.get_changes(project="openstack/nova", limit=500, start=1000)
        
        assert len(page1) == 500
        assert len(page2) == 250
        assert len(page3) == 0
    
    def test_build_query_basic(self):
        """基本的なクエリ構築のテスト"""
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        query = endpoint._build_query(project="openstack/nova")
        assert query == "project:openstack/nova"
    
    def test_build_query_with_all_params(self):
        """全パラメータでのクエリ構築のテスト"""
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass"
        )
        
        query = endpoint._build_query(
            project="openstack/nova",
            status="merged",
            branch="master",
            since=datetime(2024, 1, 1),
            until=datetime(2024, 12, 31)
        )
        
        assert "project:openstack/nova" in query
        assert "status:merged" in query
        assert "branch:master" in query
        assert "after:2024-01-01" in query
        assert "before:2024-12-31" in query
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_error_handling(self, mock_request):
        """エラーハンドリングのテスト"""
        mock_request.side_effect = Exception("API Error")
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="test_user",
            password="test_pass",
            retry_config=RetryConfig(max_retries=0)
        )
        
        with pytest.raises(Exception, match="API Error"):
            endpoint.get_changes(project="openstack/nova")


class TestChangesEndpointIntegration:
    """ChangesEndpointの統合テスト"""
    
    @patch.object(ChangesEndpoint, 'make_request')
    def test_realistic_query(self, mock_request):
        """実際のクエリシナリオのテスト"""
        mock_request.return_value = [
            {
                "id": "I1234567890abcdef",
                "project": "openstack/nova",
                "branch": "master",
                "subject": "Fix instance creation bug",
                "status": "MERGED",
                "created": "2024-06-15 10:30:00.000000000",
                "updated": "2024-06-20 15:45:00.000000000",
                "insertions": 25,
                "deletions": 10
            }
        ]
        
        endpoint = ChangesEndpoint(
            base_url="https://review.openstack.org",
            username="reviewer",
            password="secret"
        )
        
        result = endpoint.get_changes(
            project="openstack/nova",
            status="merged",
            since=datetime(2024, 6, 1),
            until=datetime(2024, 6, 30),
            limit=1000
        )
        
        assert len(result) == 1
        assert result[0]["project"] == "openstack/nova"
        assert result[0]["status"] == "MERGED"
