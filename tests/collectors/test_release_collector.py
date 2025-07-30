"""
ReleaseCollectorのテスト

このモジュールは、src/collectors/release_collector.pyのReleaseCollectorクラスと
関連する機能をテストします。
"""

import pytest
import tempfile
import subprocess
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import yaml
import sys
import os

# テスト対象のモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.collectors.release_collector import ReleaseCollector


class TestReleaseCollector:
    """ReleaseCollectorクラスのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        # テスト用の一時ディレクトリ
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.collector = ReleaseCollector(data_dir=self.temp_path)
    
    def teardown_method(self):
        """各テストの後に実行されるクリーンアップ"""
        # 一時ディレクトリの削除
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """コレクターの初期化テスト"""
        assert self.collector.data_dir == self.temp_path
        assert self.collector.local_repo_path == self.temp_path / "releases_repo"
        assert self.collector.output_path == self.temp_path / "releases_summary.csv"
        assert len(self.collector.target_components) == 6  # OPENSTACK_CORE_COMPONENTS
        assert "nova" in self.collector.target_components
        assert "neutron" in self.collector.target_components
    
    @patch('subprocess.run')
    def test_run_git_command_success(self, mock_run):
        """正常なGitコマンド実行のテスト"""
        # モックの設定
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "git command output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.collector._run_git_command(["git", "status"], self.temp_path)
        
        assert result == "git command output"
        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd=self.temp_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_run_git_command_failure(self, mock_run):
        """Gitコマンド失敗のテスト"""
        # コマンド失敗のモック
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["git", "status"], "error output"
        )
        
        with pytest.raises(subprocess.CalledProcessError):
            self.collector._run_git_command(["git", "status"], self.temp_path)
    
    @patch.object(ReleaseCollector, '_run_git_command')
    @patch('pathlib.Path.exists')
    def test_prepare_repository_clone(self, mock_exists, mock_git):
        """リポジトリクローンのテスト"""
        # リポジトリが存在しない場合
        mock_exists.return_value = False
        mock_git.return_value = "Cloning into repository..."
        
        self.collector._prepare_repository()
        
        # cloneコマンドが実行されることを確認
        mock_git.assert_called_once()
        args = mock_git.call_args[0][0]
        assert "clone" in args
        assert "https://opendev.org/openstack/releases" in args
    
    @patch.object(ReleaseCollector, '_run_git_command')
    @patch('pathlib.Path.exists')
    def test_prepare_repository_update(self, mock_exists, mock_git):
        """リポジトリ更新のテスト"""
        # リポジトリが既に存在する場合
        mock_exists.return_value = True
        mock_git.return_value = "Already up to date."
        
        self.collector._prepare_repository()
        
        # pullコマンドが実行されることを確認
        mock_git.assert_called_once()
        args = mock_git.call_args[0][0]
        assert "pull" in args
    
    @patch.object(ReleaseCollector, '_run_git_command')
    def test_get_release_date_success(self, mock_git):
        """リリース日取得成功のテスト"""
        # Gitログのモック出力
        mock_git.return_value = "2024-01-15 10:30:45 +0000"
        
        # テスト用のYAMLファイルを作成
        yaml_file = self.temp_path / "test.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text("dummy content")
        
        result = self.collector._get_release_date(yaml_file, "1.0.0")
        
        assert result == "2024-01-15"
        mock_git.assert_called_once()
        args = mock_git.call_args[0][0]
        assert "log" in args
        assert "1.0.0" in " ".join(args)
    
    @patch.object(ReleaseCollector, '_run_git_command')
    def test_get_release_date_failure(self, mock_git):
        """リリース日取得失敗のテスト"""
        # Gitコマンド失敗のモック
        mock_git.side_effect = subprocess.CalledProcessError(1, ["git", "log"], "not found")
        
        yaml_file = self.temp_path / "test.yaml"
        yaml_file.write_text("dummy content")
        
        result = self.collector._get_release_date(yaml_file, "1.0.0")
        
        assert result is None
    
    def test_get_release_date_invalid_output(self):
        """不正なGit出力のテスト"""
        with patch.object(self.collector, '_run_git_command') as mock_git:
            mock_git.return_value = "invalid date format"
            
            yaml_file = self.temp_path / "test.yaml"
            yaml_file.write_text("dummy content")
            
            result = self.collector._get_release_date(yaml_file, "1.0.0")
            
            assert result is None
    
    @patch.object(ReleaseCollector, '_prepare_repository')
    @patch.object(ReleaseCollector, '_get_release_date')
    @patch('pandas.DataFrame.to_csv')
    def test_collect_and_save_releases(self, mock_to_csv, mock_get_date, mock_prepare):
        """リリース情報収集・保存のテスト"""
        # モックの設定
        mock_prepare.return_value = None
        mock_get_date.return_value = "2024-01-15"
        
        # テスト用のYAMLファイルを作成
        deliverables_dir = self.collector.local_repo_path / "deliverables"
        deliverables_dir.mkdir(parents=True, exist_ok=True)
        
        # 各シリーズディレクトリを作成
        for series in ["zed", "antelope", "bobcat"]:
            series_dir = deliverables_dir / series
            series_dir.mkdir(exist_ok=True)
            
            # テスト用のYAMLファイル
            yaml_content = {
                "repository-settings": {
                    "openstack/nova": {}
                },
                "releases": [
                    {
                        "version": "1.0.0",
                        "projects": [
                            {
                                "repo": "openstack/nova",
                                "hash": "abc123"
                            }
                        ]
                    }
                ]
            }
            
            yaml_file = series_dir / "nova.yaml"
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_content, f)
        
        # テスト実行
        self.collector.collect_and_save_releases()
        
        # メソッドが呼び出されることを確認
        mock_prepare.assert_called_once()
        mock_to_csv.assert_called_once()
        
        # _get_release_dateが呼び出されることを確認（各シリーズ×リリース数）
        assert mock_get_date.call_count >= 3  # 最低3回（各シリーズで1回）
    
    def test_yaml_parsing_edge_cases(self):
        """YAML解析のエッジケースのテスト"""
        deliverables_dir = self.collector.local_repo_path / "deliverables" / "test"
        deliverables_dir.mkdir(parents=True, exist_ok=True)
        
        # 不正なYAMLファイル
        invalid_yaml = deliverables_dir / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content:")
        
        # 空のYAMLファイル
        empty_yaml = deliverables_dir / "empty.yaml"
        empty_yaml.write_text("")
        
        # リリース情報がないYAMLファイル
        no_releases_yaml = deliverables_dir / "no_releases.yaml"
        no_releases_yaml.write_text("repository-settings:\n  openstack/test: {}")
        
        with patch.object(self.collector, '_prepare_repository'):
            with patch.object(self.collector, '_get_release_date', return_value="2024-01-01"):
                with patch('pandas.DataFrame.to_csv'):
                    # エラーが発生せずに実行完了することを確認
                    self.collector.collect_and_save_releases()


class TestYamlProcessing:
    """YAML処理のテスト"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.collector = ReleaseCollector(data_dir=self.temp_path)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_valid_yaml_processing(self):
        """有効なYAMLファイルの処理テスト"""
        # 有効なYAMLコンテンツ
        yaml_content = {
            "repository-settings": {
                "openstack/nova": {}
            },
            "releases": [
                {
                    "version": "27.0.0",
                    "projects": [
                        {
                            "repo": "openstack/nova",
                            "hash": "abc123def456"
                        }
                    ]
                },
                {
                    "version": "27.1.0", 
                    "projects": [
                        {
                            "repo": "openstack/nova",
                            "hash": "def456ghi789"
                        }
                    ]
                }
            ]
        }
        
        # テストファイル作成
        deliverables_dir = self.collector.local_repo_path / "deliverables" / "zed"
        deliverables_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_file = deliverables_dir / "nova.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        # YAMLの読み込みテスト
        with open(yaml_file, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert "repository-settings" in loaded_data
        assert "releases" in loaded_data
        assert len(loaded_data["releases"]) == 2
        assert loaded_data["releases"][0]["version"] == "27.0.0"
    
    def test_malformed_yaml(self):
        """不正な形式のYAMLのテスト"""
        deliverables_dir = self.collector.local_repo_path / "deliverables" / "test"
        deliverables_dir.mkdir(parents=True, exist_ok=True)
        
        # 不正なYAMLファイル
        malformed_yaml = deliverables_dir / "malformed.yaml"
        malformed_yaml.write_text("""
        invalid:
          - yaml
            content:
        without: proper indentation
        """)
        
        # YAML解析でエラーが発生することを確認
        with pytest.raises(yaml.YAMLError):
            with open(malformed_yaml, 'r') as f:
                yaml.safe_load(f)


class TestIntegration:
    """統合テスト"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.collector = ReleaseCollector(data_dir=self.temp_path)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.object(ReleaseCollector, '_run_git_command')
    @patch('pandas.DataFrame.to_csv')
    def test_full_workflow_simulation(self, mock_to_csv, mock_git):
        """全体ワークフローのシミュレーションテスト"""
        # Gitコマンドのモック（リポジトリクローン/更新）
        mock_git.side_effect = [
            "Cloning into repository...",  # clone
            "2024-01-15 10:30:45 +0000"   # log（リリース日取得）
        ]
        
        # テスト用のディレクトリ構造とファイルを作成
        deliverables_dir = self.collector.local_repo_path / "deliverables"
        
        # 複数のシリーズを作成
        for series in ["zed", "antelope"]:
            series_dir = deliverables_dir / series
            series_dir.mkdir(parents=True, exist_ok=True)
            
            # 複数のコンポーネントファイルを作成
            for component in ["nova", "neutron"]:
                yaml_content = {
                    "repository-settings": {
                        f"openstack/{component}": {}
                    },
                    "releases": [
                        {
                            "version": "1.0.0",
                            "projects": [
                                {
                                    "repo": f"openstack/{component}",
                                    "hash": "abc123"
                                }
                            ]
                        }
                    ]
                }
                
                yaml_file = series_dir / f"{component}.yaml"
                with open(yaml_file, 'w') as f:
                    yaml.dump(yaml_content, f)
        
        # ワークフロー実行
        self.collector.collect_and_save_releases()
        
        # 結果の検証
        mock_to_csv.assert_called_once()
        
        # DataFrameのto_csvが呼び出された際の引数を確認
        call_args = mock_to_csv.call_args
        assert call_args[0][0] == self.collector.output_path  # 出力パス
        
        # Gitコマンドが適切に呼び出されることを確認
        assert mock_git.call_count >= 1


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.collector = ReleaseCollector(data_dir=self.temp_path)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.object(ReleaseCollector, '_run_git_command')
    def test_git_clone_failure(self, mock_git):
        """Gitクローン失敗のテスト"""
        mock_git.side_effect = subprocess.CalledProcessError(1, ["git", "clone"], "clone failed")
        
        with pytest.raises(subprocess.CalledProcessError):
            self.collector._prepare_repository()
    
    @patch.object(ReleaseCollector, '_prepare_repository')
    def test_missing_deliverables_directory(self, mock_prepare):
        """deliverables ディレクトリが存在しない場合のテスト"""
        mock_prepare.return_value = None
        
        # deliverables ディレクトリを作成しない
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            self.collector.collect_and_save_releases()
            
            # 空のDataFrameでも保存が実行されることを確認
            mock_to_csv.assert_called_once()
    
    def test_permission_error(self):
        """ファイル書き込み権限エラーのテスト"""
        # 読み取り専用ディレクトリを作成
        readonly_dir = self.temp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # 読み取り専用
        
        collector = ReleaseCollector(data_dir=readonly_dir)
        
        with patch.object(collector, '_prepare_repository'):
            # PermissionErrorが適切に処理されることを確認
            # （実際の実装では例外処理が必要）
            try:
                collector.collect_and_save_releases()
            except PermissionError:
                pass  # 期待される例外


if __name__ == '__main__':
    # テストの実行
    pytest.main([__file__, '-v'])
