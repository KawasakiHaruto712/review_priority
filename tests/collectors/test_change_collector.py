"""
ChangeCollectorの統合テスト
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.collectors.change_collector import ChangeCollector


class TestChangeCollectorInitialization:
    """ChangeCollectorの初期化のテスト"""
    
    @patch.dict('os.environ', {
        'GERRIT_USERNAME': 'test_user',
        'GERRIT_PASSWORD': 'test_pass'
    })
    def test_initialization_from_env(self):
        """環境変数からの初期化のテスト"""
        collector = ChangeCollector()
        
        assert collector.username == 'test_user'
        assert collector.password == 'test_pass'
        assert collector.base_url == "https://review.openstack.org"
    
    def test_initialization_with_params(self):
        """パラメータ指定での初期化のテスト"""
        collector = ChangeCollector(
            username="direct_user",
            password="direct_pass",
            base_url="https://custom.review.com"
        )
        
        assert collector.username == "direct_user"
        assert collector.password == "direct_pass"
        assert collector.base_url == "https://custom.review.com"
    
    def test_initialization_missing_credentials(self):
        """認証情報が不足している場合のテスト"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="username and password"):
                ChangeCollector()
    
    def test_endpoints_initialization(self):
        """エンドポイントの初期化のテスト"""
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        # 各エンドポイントが初期化されているか確認
        assert collector.changes_endpoint is not None
        assert collector.change_detail_endpoint is not None
        assert collector.comments_endpoint is not None
        assert collector.reviewers_endpoint is not None
        assert collector.file_content_endpoint is not None
        assert collector.file_diff_endpoint is not None
        assert collector.commit_endpoint is not None
        assert collector.commit_parents_endpoint is not None


class TestChangeCollectorCollection:
    """ChangeCollectorの収集機能のテスト"""
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    def test_collect_changes_basic(self, mock_get_changes):
        """基本的な変更収集のテスト"""
        mock_get_changes.return_value = [
            {"id": "change1", "project": "openstack/nova"},
            {"id": "change2", "project": "openstack/nova"}
        ]
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        result = collector.collect_changes(project="openstack/nova")
        
        assert len(result) == 2
        assert result[0]["id"] == "change1"
        mock_get_changes.assert_called_once()
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    @patch('src.collectors.endpoints.change_detail_endpoint.ChangeDetailEndpoint.get_change_detail')
    def test_collect_changes_with_details(self, mock_get_detail, mock_get_changes):
        """詳細情報付きの変更収集のテスト"""
        mock_get_changes.return_value = [
            {"id": "change1", "project": "openstack/nova"}
        ]
        
        mock_get_detail.return_value = {
            "id": "change1",
            "project": "openstack/nova",
            "subject": "Fix bug",
            "owner": {"name": "John Doe"}
        }
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        result = collector.collect_changes_with_details(
            project="openstack/nova",
            include_details=True
        )
        
        assert len(result) == 1
        assert result[0]["subject"] == "Fix bug"
        assert result[0]["owner"]["name"] == "John Doe"
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    def test_collect_changes_with_filters(self, mock_get_changes):
        """フィルター付きの変更収集のテスト"""
        mock_get_changes.return_value = []
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        collector.collect_changes(
            project="openstack/nova",
            status="merged",
            branch="master",
            since=datetime(2024, 1, 1),
            until=datetime(2024, 12, 31)
        )
        
        # 正しいパラメータで呼ばれたか確認
        call_args = mock_get_changes.call_args
        assert call_args[1]["project"] == "openstack/nova"
        assert call_args[1]["status"] == "merged"
        assert call_args[1]["branch"] == "master"
    
    @patch('src.collectors.endpoints.comments_endpoint.CommentsEndpoint.get_comments')
    def test_collect_comments(self, mock_get_comments):
        """コメント収集のテスト"""
        mock_get_comments.return_value = {
            "file.py": [
                {"line": 10, "message": "Good code"}
            ]
        }
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        result = collector.collect_comments(
            change_id="12345",
            revision_id="current"
        )
        
        assert "file.py" in result
        assert len(result["file.py"]) == 1
    
    @patch('src.collectors.endpoints.reviewers_endpoint.ReviewersEndpoint.get_reviewers')
    def test_collect_reviewers(self, mock_get_reviewers):
        """レビュアー収集のテスト"""
        mock_get_reviewers.return_value = [
            {"name": "Alice", "_account_id": 1001},
            {"name": "Bob", "_account_id": 1002}
        ]
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        result = collector.collect_reviewers(change_id="12345")
        
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
    
    @patch('src.collectors.endpoints.commit_endpoint.CommitEndpoint.get_commit')
    def test_collect_commit_info(self, mock_get_commit):
        """コミット情報収集のテスト"""
        mock_get_commit.return_value = {
            "commit": "abc123",
            "author": {"name": "John Doe"},
            "subject": "Fix bug"
        }
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        result = collector.collect_commit_info(
            change_id="12345",
            revision_id="current"
        )
        
        assert result["commit"] == "abc123"
        assert result["author"]["name"] == "John Doe"


class TestChangeCollectorStorage:
    """ChangeCollectorのストレージ機能のテスト"""
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    @patch('src.collectors.storage.change_storage.ChangeStorage.save_json')
    @patch('src.collectors.storage.change_storage.ChangeStorage.save_csv')
    def test_collect_and_save(self, mock_save_csv, mock_save_json, mock_get_changes):
        """収集と保存の統合テスト"""
        mock_get_changes.return_value = [
            {"id": "change1", "project": "openstack/nova"},
            {"id": "change2", "project": "openstack/nova"}
        ]
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        collector.collect_and_save_changes(
            project="openstack/nova",
            filename="nova_changes"
        )
        
        # 保存メソッドが呼ばれたか確認
        mock_save_json.assert_called_once()
        mock_save_csv.assert_called_once()
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    @patch('src.collectors.storage.change_storage.ChangeStorage.save_summary')
    def test_collect_with_summary(self, mock_save_summary, mock_get_changes):
        """サマリー付き収集のテスト"""
        mock_get_changes.return_value = [
            {"id": "change1", "status": "MERGED"},
            {"id": "change2", "status": "NEW"},
            {"id": "change3", "status": "MERGED"}
        ]
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        collector.collect_and_save_changes_with_summary(
            project="openstack/nova",
            filename="nova_changes"
        )
        
        # サマリー保存が呼ばれたか確認
        mock_save_summary.assert_called_once()
        
        # サマリーの内容を確認
        summary_call = mock_save_summary.call_args[0][0]
        assert summary_call["total"] == 3
        assert summary_call["merged"] == 2
        assert summary_call["new"] == 1


class TestChangeCollectorError:
    """ChangeCollectorのエラーハンドリングのテスト"""
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    def test_handle_api_error(self, mock_get_changes):
        """APIエラーのハンドリングのテスト"""
        mock_get_changes.side_effect = Exception("API Error")
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        with pytest.raises(Exception, match="API Error"):
            collector.collect_changes(project="openstack/nova")
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    def test_handle_empty_result(self, mock_get_changes):
        """空の結果のハンドリングのテスト"""
        mock_get_changes.return_value = []
        
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        result = collector.collect_changes(project="openstack/nova")
        
        assert result == []
        assert isinstance(result, list)


class TestChangeCollectorIntegration:
    """ChangeCollectorの完全な統合テスト"""
    
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    @patch('src.collectors.endpoints.change_detail_endpoint.ChangeDetailEndpoint.get_change_detail')
    @patch('src.collectors.endpoints.comments_endpoint.CommentsEndpoint.get_comments')
    @patch('src.collectors.endpoints.reviewers_endpoint.ReviewersEndpoint.get_reviewers')
    @patch('src.collectors.endpoints.commit_endpoint.CommitEndpoint.get_commit')
    @patch('src.collectors.storage.change_storage.ChangeStorage.save_json')
    @patch('src.collectors.storage.change_storage.ChangeStorage.save_csv')
    @patch('src.collectors.storage.change_storage.ChangeStorage.save_summary')
    def test_complete_workflow(
        self,
        mock_save_summary,
        mock_save_csv,
        mock_save_json,
        mock_get_commit,
        mock_get_reviewers,
        mock_get_comments,
        mock_get_detail,
        mock_get_changes
    ):
        """完全なワークフローの統合テスト"""
        # モックの設定
        mock_get_changes.return_value = [
            {"id": "change1", "_number": 12345}
        ]
        
        mock_get_detail.return_value = {
            "id": "change1",
            "project": "openstack/nova",
            "subject": "Fix bug",
            "status": "MERGED"
        }
        
        mock_get_comments.return_value = {
            "file.py": [{"line": 10, "message": "Good"}]
        }
        
        mock_get_reviewers.return_value = [
            {"name": "Alice"}
        ]
        
        mock_get_commit.return_value = {
            "commit": "abc123",
            "author": {"name": "John"}
        }
        
        # コレクターの実行
        collector = ChangeCollector(
            username="test_user",
            password="test_pass"
        )
        
        # 変更を収集
        changes = collector.collect_changes(project="openstack/nova")
        assert len(changes) == 1
        
        # 詳細情報を収集
        details = collector.collect_change_detail(change_id="change1")
        assert details["subject"] == "Fix bug"
        
        # コメントを収集
        comments = collector.collect_comments(
            change_id="change1",
            revision_id="current"
        )
        assert "file.py" in comments
        
        # レビュアーを収集
        reviewers = collector.collect_reviewers(change_id="change1")
        assert len(reviewers) == 1
        
        # コミット情報を収集
        commit = collector.collect_commit_info(
            change_id="change1",
            revision_id="current"
        )
        assert commit["commit"] == "abc123"
    
    @patch.dict('os.environ', {
        'GERRIT_USERNAME': 'env_user',
        'GERRIT_PASSWORD': 'env_pass'
    })
    @patch('src.collectors.endpoints.changes_endpoint.ChangesEndpoint.get_changes')
    def test_env_file_integration(self, mock_get_changes):
        """環境変数ファイルとの統合テスト"""
        mock_get_changes.return_value = []
        
        # 環境変数から認証情報を取得
        collector = ChangeCollector()
        
        assert collector.username == 'env_user'
        assert collector.password == 'env_pass'
        
        # 収集を実行
        result = collector.collect_changes(project="openstack/nova")
        
        assert result == []
        mock_get_changes.assert_called_once()
