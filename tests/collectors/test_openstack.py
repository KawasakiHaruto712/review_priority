"""
OpenStackGerritCollectorのテスト

このモジュールは、src/collectors/openstack.pyのOpenStackGerritCollectorクラスと
関連する機能をテストします。
"""

import pytest
import requests
import json
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import logging

# テスト対象のモジュールをインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.collectors.openstack import (
    RetryConfig,
    OpenStackGerritCollector
)


class TestRetryConfig:
    """RetryConfigクラスのテスト"""
    
    def test_default_initialization(self):
        """デフォルト設定での初期化テスト"""
        config = RetryConfig()
        
        assert config.max_retries == 10
        assert config.base_delay == 30.0
        assert config.max_delay == 3840.0
        assert config.backoff_factor == 2.0
        assert config.jitter == True
    
    def test_custom_initialization(self):
        """カスタム設定での初期化テスト"""
        config = RetryConfig(
            max_retries=5,
            base_delay=10.0,
            max_delay=1000.0,
            backoff_factor=3.0,
            jitter=False
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 10.0
        assert config.max_delay == 1000.0
        assert config.backoff_factor == 3.0
        assert config.jitter == False
    
    def test_get_delay(self):
        """遅延時間計算のテスト"""
        config = RetryConfig(base_delay=10.0, backoff_factor=2.0, jitter=False)
        
        # 1回目のリトライ
        delay1 = config.get_delay(1)
        assert delay1 == 10.0
        
        # 2回目のリトライ
        delay2 = config.get_delay(2)
        assert delay2 == 20.0
        
        # 3回目のリトライ
        delay3 = config.get_delay(3)
        assert delay3 == 40.0
    
    def test_get_delay_with_max_limit(self):
        """最大遅延時間制限のテスト"""
        config = RetryConfig(base_delay=100.0, max_delay=150.0, backoff_factor=2.0, jitter=False)
        
        # 最大値を超えない場合
        delay1 = config.get_delay(1)
        assert delay1 == 100.0
        
        # 最大値を超える場合
        delay2 = config.get_delay(2)
        assert delay2 == 150.0  # max_delayで制限される
    
    def test_get_delay_with_jitter(self):
        """ジッター付き遅延時間のテスト"""
        config = RetryConfig(base_delay=10.0, backoff_factor=2.0, jitter=True)
        
        # ジッターがあるので値は変動するが、範囲内であることを確認
        delay1 = config.get_delay(1)
        assert 5.0 <= delay1 <= 15.0  # ±50%の範囲
        
        delay2 = config.get_delay(2)
        assert 10.0 <= delay2 <= 30.0  # ±50%の範囲


class TestOpenStackGerritCollector:
    """OpenStackGerritCollectorクラスのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        self.collector = OpenStackGerritCollector()
        
        # テスト用の一時ディレクトリ
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """各テストの後に実行されるクリーンアップ"""
        # 一時ディレクトリの削除
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """コレクターの初期化テスト"""
        assert self.collector.base_url == "https://review.opendev.org"
        assert self.collector.retry_config is not None
        assert isinstance(self.collector.retry_config, RetryConfig)
    
    def test_custom_initialization(self):
        """カスタム設定でのコレクター初期化テスト"""
        custom_retry = RetryConfig(max_retries=5)
        collector = OpenStackGerritCollector(
            base_url="https://custom.gerrit.com",
            retry_config=custom_retry
        )
        
        assert collector.base_url == "https://custom.gerrit.com"
        assert collector.retry_config.max_retries == 5
    
    @patch('requests.get')
    def test_make_request_success(self, mock_get):
        """正常なAPIリクエストのテスト"""
        # モックレスポンスの設定
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = ")]}'\n{\"test\": \"data\"}"
        mock_get.return_value = mock_response
        
        result = self.collector._make_request("/test/endpoint")
        
        assert result == {"test": "data"}
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_make_request_with_retries(self, mock_get):
        """リトライ機能のテスト"""
        # 最初の2回は失敗、3回目は成功
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = requests.RequestException("Network error")
        
        mock_response_success = Mock()
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.text = ")]}'\n{\"success\": true}"
        
        mock_get.side_effect = [mock_response_fail, mock_response_fail, mock_response_success]
        
        with patch('time.sleep'):  # sleepをモック化して高速化
            result = self.collector._make_request("/test/endpoint")
        
        assert result == {"success": True}
        assert mock_get.call_count == 3
    
    @patch('requests.get')
    def test_make_request_max_retries_exceeded(self, mock_get):
        """最大リトライ回数超過のテスト"""
        # 常に失敗するレスポンス
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Persistent error")
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):  # sleepをモック化
            with pytest.raises(requests.RequestException):
                self.collector._make_request("/test/endpoint")
        
        # max_retries + 1回呼び出される
        assert mock_get.call_count == self.collector.retry_config.max_retries + 1
    
    def test_parse_gerrit_response(self):
        """Gerritレスポンスのパースのテスト"""
        # 正常なGerritレスポンス
        gerrit_response = ")]}'\n{\"test\": \"data\", \"number\": 12345}"
        result = self.collector._parse_gerrit_response(gerrit_response)
        
        assert result == {"test": "data", "number": 12345}
    
    def test_parse_gerrit_response_invalid_json(self):
        """不正なJSONのパースエラーテスト"""
        invalid_response = ")]}'\n{invalid json"
        
        with pytest.raises(json.JSONDecodeError):
            self.collector._parse_gerrit_response(invalid_response)
    
    def test_parse_gerrit_response_missing_prefix(self):
        """Gerrit接頭辞なしのレスポンステスト"""
        response_without_prefix = "{\"test\": \"data\"}"
        result = self.collector._parse_gerrit_response(response_without_prefix)
        
        assert result == {"test": "data"}
    
    @patch.object(OpenStackGerritCollector, '_make_request')
    def test_get_changes(self, mock_request):
        """Changes取得のテスト"""
        # モックレスポンス
        mock_changes = [
            {"_number": 12345, "subject": "Test change 1"},
            {"_number": 12346, "subject": "Test change 2"}
        ]
        mock_request.return_value = mock_changes
        
        result = self.collector.get_changes(
            project="nova",
            after="2024-01-01",
            before="2024-01-31"
        )
        
        assert len(result) == 2
        assert result[0]["_number"] == 12345
        assert result[1]["subject"] == "Test change 2"
        
        # APIが正しいパラメータで呼び出されることを確認
        expected_query = "project:nova+after:2024-01-01+before:2024-01-31"
        mock_request.assert_called_once()
        call_args = mock_request.call_args[0][0]
        assert expected_query in call_args
    
    @patch.object(OpenStackGerritCollector, '_make_request')
    def test_get_change_details(self, mock_request):
        """Change詳細取得のテスト"""
        mock_detail = {
            "_number": 12345,
            "subject": "Test change",
            "status": "MERGED",
            "messages": [{"message": "Looks good"}]
        }
        mock_request.return_value = mock_detail
        
        result = self.collector.get_change_details(12345)
        
        assert result["_number"] == 12345
        assert result["status"] == "MERGED"
        assert len(result["messages"]) == 1
        
        mock_request.assert_called_once_with("/changes/12345/detail?o=DETAILED_LABELS&o=DETAILED_ACCOUNTS&o=MESSAGES&o=CURRENT_REVISION&o=CURRENT_FILES")
    
    @patch.object(OpenStackGerritCollector, 'get_changes')
    @patch.object(OpenStackGerritCollector, 'get_change_details')
    def test_collect_changes_data(self, mock_get_details, mock_get_changes):
        """Changes データ収集のテスト"""
        # モック設定
        mock_get_changes.return_value = [
            {"_number": 12345},
            {"_number": 12346}
        ]
        
        mock_get_details.side_effect = [
            {"_number": 12345, "subject": "Change 1"},
            {"_number": 12346, "subject": "Change 2"}
        ]
        
        result = self.collector.collect_changes_data(
            project="nova",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        assert len(result) == 2
        assert result[0]["_number"] == 12345
        assert result[1]["subject"] == "Change 2"
        
        # メソッドが正しく呼び出されることを確認
        mock_get_changes.assert_called_once_with("nova", "2024-01-01", "2024-01-31", limit=None)
        assert mock_get_details.call_count == 2
    
    @patch('pandas.DataFrame.to_json')
    @patch('pathlib.Path.mkdir')
    @patch.object(OpenStackGerritCollector, 'collect_changes_data')
    def test_save_changes_to_file(self, mock_collect, mock_mkdir, mock_to_json):
        """Changes データのファイル保存テスト"""
        # モックデータ
        mock_changes = [
            {"_number": 12345, "subject": "Test change"}
        ]
        mock_collect.return_value = mock_changes
        
        # ファイル保存の実行
        self.collector.save_changes_to_file(
            project="nova",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir=self.temp_path
        )
        
        # メソッドが呼び出されることを確認
        mock_collect.assert_called_once()
        mock_mkdir.assert_called()
        mock_to_json.assert_called()
    
    def test_build_query_string(self):
        """クエリ文字列構築のテスト"""
        query = self.collector._build_query_string(
            project="nova",
            after="2024-01-01",
            before="2024-01-31",
            status="merged"
        )
        
        expected = "project:nova+after:2024-01-01+before:2024-01-31+status:merged"
        assert query == expected
    
    def test_build_query_string_with_none_values(self):
        """None値を含むクエリ文字列のテスト"""
        query = self.collector._build_query_string(
            project="nova",
            after=None,
            before="2024-01-31",
            status=None
        )
        
        expected = "project:nova+before:2024-01-31"
        assert query == expected


class TestIntegration:
    """統合テスト"""
    
    @patch('requests.get')
    def test_full_workflow_with_mock_data(self, mock_get):
        """モックデータを使用した全体ワークフローのテスト"""
        # Change一覧のモックレスポンス
        changes_response = Mock()
        changes_response.raise_for_status.return_value = None
        changes_response.text = ")]}'\n[{\"_number\": 12345}]"
        
        # Change詳細のモックレスポンス
        detail_response = Mock()
        detail_response.raise_for_status.return_value = None
        detail_response.text = ")]}'\n{\"_number\": 12345, \"subject\": \"Test change\", \"status\": \"MERGED\"}"
        
        # レスポンスの順序を設定
        mock_get.side_effect = [changes_response, detail_response]
        
        collector = OpenStackGerritCollector()
        
        # データ収集の実行
        result = collector.collect_changes_data(
            project="nova",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        # 結果の検証
        assert len(result) == 1
        assert result[0]["_number"] == 12345
        assert result[0]["subject"] == "Test change"
        assert result[0]["status"] == "MERGED"
        
        # APIが2回呼び出される（一覧取得 + 詳細取得）
        assert mock_get.call_count == 2


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def setup_method(self):
        self.collector = OpenStackGerritCollector()
    
    @patch('requests.get')
    def test_network_error_handling(self, mock_get):
        """ネットワークエラーのハンドリングテスト"""
        mock_get.side_effect = requests.ConnectionError("Network unreachable")
        
        with patch('time.sleep'):
            with pytest.raises(requests.ConnectionError):
                self.collector._make_request("/test")
    
    @patch('requests.get')
    def test_http_error_handling(self, mock_get):
        """HTTPエラーのハンドリングテスト"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):
            with pytest.raises(requests.HTTPError):
                self.collector._make_request("/test")
    
    @patch('requests.get')
    def test_json_decode_error_handling(self, mock_get):
        """JSON解析エラーのハンドリングテスト"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = ")]}'\n{invalid json}"
        mock_get.return_value = mock_response
        
        with pytest.raises(json.JSONDecodeError):
            self.collector._make_request("/test")


if __name__ == '__main__':
    # テストの実行
    pytest.main([__file__, '-v'])
