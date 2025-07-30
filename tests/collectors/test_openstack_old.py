"""
OpenStackDataCollectorのテスト

このモジュールはOpenStackのGerritからデータを収集する機能をテストします。
ネットワーク依存部分はモックを使用してテストします。
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import os

# テスト対象のモジュールをインポート
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.collectors.openstack import (
    RetryConfig,
    OpenStackGerritCollector
)


class TestRetryConfig:
    """RetryConfigクラスのテスト"""
    
    def test_init_default_values(self):
        """デフォルト値での初期化テスト"""
        config = RetryConfig()
        assert config.max_retries == 10
        assert config.base_delay == 30.0
        assert config.max_delay == 3840.0
        assert config.backoff_factor == 2.0
        assert config.jitter == True
    
    def test_init_custom_values(self):
        """カスタム値での初期化テスト"""
        config = RetryConfig(
            max_retries=5,
            base_delay=10.0,
            max_delay=1000.0,
            backoff_factor=1.5,
            jitter=False
        )
        assert config.max_retries == 5
        assert config.base_delay == 10.0
        assert config.max_delay == 1000.0
        assert config.backoff_factor == 1.5
        assert config.jitter == False
    
    def test_get_delay_no_jitter(self):
        """ジッターなしでの遅延時間計算テスト"""
        config = RetryConfig(base_delay=10.0, backoff_factor=2.0, jitter=False)
        
        # 指数的バックオフのテスト
        assert config.get_delay(0) == 10.0
        assert config.get_delay(1) == 20.0
        assert config.get_delay(2) == 40.0
        assert config.get_delay(3) == 80.0
    
    def test_get_delay_with_max_limit(self):
        """最大遅延時間制限のテスト"""
        config = RetryConfig(base_delay=10.0, max_delay=50.0, backoff_factor=2.0, jitter=False)
        
        # 最大値を超えない
        assert config.get_delay(10) == 50.0  # 本来なら10240.0だが制限される
    
    def test_get_delay_with_jitter(self):
        """ジッター付き遅延時間テスト"""
        config = RetryConfig(base_delay=10.0, backoff_factor=2.0, jitter=True)
        
        delay = config.get_delay(1)
        # ジッターありの場合、基本値から±25%の範囲内
        expected_base = 20.0
        assert expected_base * 0.75 <= delay <= expected_base * 1.25


class TestOpenStackGerritCollector:
    """OpenStackGerritCollectorクラスのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        # 一時ディレクトリでテスト
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir)
        
        # モックされた環境でコレクターを初期化
        with patch('src.collectors.openstack.DEFAULT_DATA_DIR', self.test_data_dir):
            self.collector = OpenStackGerritCollector()
    
    def teardown_method(self):
        """各テストの後に実行されるクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """初期化テスト"""
        assert self.collector.components == ["nova", "neutron", "cinder", "glance", "keystone", "swift"]
        assert isinstance(self.collector.retry_config, RetryConfig)
        assert self.collector.session is not None
        assert hasattr(self.collector, 'output_dir')
    
    @patch('src.collectors.openstack.requests.Session.get')
    def test_make_request_success(self, mock_get):
        """成功するAPIリクエストのテスト"""
        # モックレスポンスの設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # リクエスト実行
        result = self.collector._make_request("https://test.example.com/api")
        
        # 結果検証
        assert result == {"test": "data"}
        mock_get.assert_called_once()
    
    @patch('src.collectors.openstack.requests.Session.get')
    @patch('src.collectors.openstack.time.sleep')
    def test_make_request_retry(self, mock_sleep, mock_get):
        """リトライ機能のテスト"""
        # 最初は失敗、2回目は成功するレスポンス
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = Exception("Server Error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"success": True}
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]
        
        # リクエスト実行
        result = self.collector._make_request("https://test.example.com/api")
        
        # 結果検証
        assert result == {"success": True}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()
    
    @patch('src.collectors.openstack.requests.Session.get')
    def test_get_merged_changes(self, mock_get):
        """get_merged_changesメソッドのテスト"""
        # モックレスポンスの設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ")]}'\\n[{\"_number\": 12345, \"subject\": \"Test change\"}]"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # メソッド実行
        result = self.collector.get_merged_changes("nova", limit=10)
        
        # 結果検証
        assert isinstance(result, list)
        mock_get.assert_called_once()
    
    def test_base64_encode(self):
        """Base64エンコードテスト"""
        test_string = "test:password"
        encoded = self.collector._base64_encode(test_string)
        
        import base64
        expected = base64.b64encode(test_string.encode('utf-8')).decode('ascii')
        assert encoded == expected
    
    def test_save_data_to_file(self):
        """データファイル保存のテスト"""
        test_data = [
            {"change_number": 12345, "subject": "Test change 1"},
            {"change_number": 12346, "subject": "Test change 2"}
        ]
        
        file_path = self.test_data_dir / "test_changes.json"
        self.collector._save_data_to_file(test_data, file_path)
        
        # ファイルが作成されたことを確認
        assert file_path.exists()
        
        # ファイル内容の確認
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 2
        assert saved_data[0]["change_number"] == 12345
        assert saved_data[1]["subject"] == "Test change 2"
    
    @patch.object(OpenStackGerritCollector, '_make_request')
    def test_collect_changes_for_project(self, mock_request):
        """プロジェクト別Change収集のテスト"""
        # モックレスポンスデータ
        mock_changes = [
            {
                "_number": 12345,
                "id": "I1234567890abcdef",
                "subject": "Test change",
                "status": "NEW",
                "owner": {"name": "test_user"},
                "created": "2024-01-01T10:00:00Z",
                "updated": "2024-01-01T15:00:00Z",
                "messages": []
            }
        ]
        
        mock_request.return_value = mock_changes
        
        # Changes収集実行
        with patch.object(self.collector, '_save_data_to_file') as mock_save:
            result = self.collector.collect_changes_for_project(
                project="nova",
                start_date="2024-01-01",
                end_date="2024-01-31"
            )
        
        # 結果検証
        assert result == 1  # 1件のChangeが収集された
        mock_request.assert_called()
        mock_save.assert_called()
    
    def test_get_project_data_path(self):
        """プロジェクトデータパス取得のテスト"""
        path = self.collector._get_project_data_path("nova", "changes")
        expected_path = self.test_data_dir / "openstack" / "nova" / "changes"
        assert path == expected_path
    
    def test_create_summary_stats(self):
        """サマリー統計作成のテスト"""
        test_stats = {
            "nova": {"changes": 100, "commits": 80},
            "neutron": {"changes": 150, "commits": 120}
        }
        
        with patch.object(self.collector, '_save_data_to_file') as mock_save:
            self.collector._create_summary_stats(test_stats)
        
        # サマリーファイル保存が呼ばれたことを確認
        mock_save.assert_called()
        args, kwargs = mock_save.call_args
        summary_data = args[0]
        
        assert "timestamp" in summary_data
        assert "projects" in summary_data
        assert summary_data["projects"]["nova"]["changes"] == 100


class TestOpenStackDataCollectorIntegration:
    """統合テスト"""
    
    def setup_method(self):
        """統合テストの初期化"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir)
    
    def teardown_method(self):
        """統合テストのクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.collectors.openstack.DEFAULT_DATA_DIR')
    @patch.object(OpenStackGerritCollector, 'collect_changes_for_project')
    @patch.object(OpenStackGerritCollector, 'collect_commits_for_project')
    def test_collect_all_data_workflow(self, mock_collect_commits, mock_collect_changes, mock_data_dir):
        """全データ収集ワークフローのテスト"""
        mock_data_dir.return_value = self.test_data_dir
        
        # モック設定
        mock_collect_changes.return_value = 50  # 50件のChanges
        mock_collect_commits.return_value = 40   # 40件のCommits
        
        collector = OpenStackGerritCollector()
        
        # 小規模なプロジェクトリストでテスト
        test_projects = ["nova", "neutron"]
        
        with patch('src.utils.constants.OPENSTACK_CORE_COMPONENTS', test_projects):
            results = collector.collect_all_data(
                start_date="2024-01-01",
                end_date="2024-01-31"
            )
        
        # 結果検証
        assert "nova" in results
        assert "neutron" in results
        assert mock_collect_changes.call_count == 2
        assert mock_collect_commits.call_count == 2


if __name__ == '__main__':
    # テストの実行
    pytest.main([__file__, '-v'])
