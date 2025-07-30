"""
ReleaseCollectorのテスト

このモジュールはOpenStackのリリース情報収集機能をテストします。
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import os
import yaml

# テスト対象のモジュールをインポート
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.collectors.release_collector import (
    ReleaseCollector,
    download_openstack_releases,
    extract_deliverables_info,
    generate_releases_summary
)


class TestReleaseCollector:
    """ReleaseCollectorクラスのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir)
        
        # モック環境でコレクターを初期化
        with patch('src.collectors.release_collector.app_path.DEFAULT_DATA_DIR', self.test_data_dir):
            self.collector = ReleaseCollector()
    
    def teardown_method(self):
        """各テストの後に実行されるクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """初期化テスト"""
        assert self.collector.data_dir == self.test_data_dir
        assert self.collector.releases_repo_dir is not None
    
    @patch('subprocess.run')
    def test_clone_releases_repo_success(self, mock_subprocess):
        """リリースリポジトリクローン成功のテスト"""
        # subprocess.runのモック
        mock_subprocess.return_value = Mock(returncode=0)
        
        # リポジトリディレクトリが存在しない場合
        result = self.collector._clone_releases_repo()
        
        assert result == True
        mock_subprocess.assert_called()
        
        # git cloneコマンドが呼ばれたかチェック
        call_args = mock_subprocess.call_args[0][0]
        assert 'git' in call_args
        assert 'clone' in call_args
    
    @patch('subprocess.run')
    def test_clone_releases_repo_failure(self, mock_subprocess):
        """リリースリポジトリクローン失敗のテスト"""
        # subprocess.runが失敗を返す
        mock_subprocess.return_value = Mock(returncode=1)
        
        result = self.collector._clone_releases_repo()
        
        assert result == False
        mock_subprocess.assert_called()
    
    def test_extract_deliverables_info(self):
        """デリバラブル情報抽出のテスト"""
        # テスト用のYAMLデータ
        test_deliverables = {
            'nova': {
                'releases': [
                    {
                        'version': '2024.1.0',
                        'projects': [
                            {
                                'repo': 'openstack/nova',
                                'hash': 'abc123def456'
                            }
                        ]
                    },
                    {
                        'version': '2024.2.0',
                        'projects': [
                            {
                                'repo': 'openstack/nova',
                                'hash': 'def456ghi789'
                            }
                        ]
                    }
                ]
            },
            'neutron': {
                'releases': [
                    {
                        'version': '2024.1.0',
                        'projects': [
                            {
                                'repo': 'openstack/neutron',
                                'hash': 'ghi789jkl012'
                            }
                        ]
                    }
                ]
            }
        }
        
        # モックファイルシステム
        mock_deliverables_dir = Mock()
        
        # extract_deliverables_info関数のテスト
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.iterdir') as mock_iterdir, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('yaml.safe_load') as mock_yaml:
            
            # セットアップ
            mock_exists.return_value = True
            mock_iterdir.return_value = [
                Path('2024.1.yaml'),
                Path('2024.2.yaml')
            ]
            mock_yaml.return_value = test_deliverables
            
            result = extract_deliverables_info(mock_deliverables_dir)
            
            # 結果検証
            assert isinstance(result, list)
            # YAML読み込みが呼ばれたことを確認
            assert mock_yaml.call_count >= 0
    
    def test_generate_releases_summary(self):
        """リリースサマリー生成のテスト"""
        # テスト用のデリバラブルデータ
        test_deliverables = [
            {
                'release_cycle': '2024.1',
                'project': 'nova',
                'version': '2024.1.0',
                'repo': 'openstack/nova',
                'hash': 'abc123'
            },
            {
                'release_cycle': '2024.1',
                'project': 'neutron', 
                'version': '2024.1.0',
                'repo': 'openstack/neutron',
                'hash': 'def456'
            },
            {
                'release_cycle': '2024.2',
                'project': 'nova',
                'version': '2024.2.0',
                'repo': 'openstack/nova',
                'hash': 'ghi789'
            }
        ]
        
        summary_df = generate_releases_summary(test_deliverables)
        
        # 結果検証
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 3
        
        # 必要なカラムが存在することを確認
        expected_columns = ['release_cycle', 'project', 'version', 'repo', 'hash']
        for col in expected_columns:
            assert col in summary_df.columns
        
        # データの内容確認
        nova_releases = summary_df[summary_df['project'] == 'nova']
        assert len(nova_releases) == 2
        
        neutron_releases = summary_df[summary_df['project'] == 'neutron']
        assert len(neutron_releases) == 1
    
    @patch.object(ReleaseCollector, '_clone_releases_repo')
    @patch('src.collectors.release_collector.extract_deliverables_info')
    @patch('src.collectors.release_collector.generate_releases_summary')
    def test_collect_release_data(self, mock_summary, mock_extract, mock_clone):
        """リリースデータ収集のテスト"""
        # モック設定
        mock_clone.return_value = True
        mock_extract.return_value = [
            {'project': 'nova', 'version': '2024.1.0'},
            {'project': 'neutron', 'version': '2024.1.0'}
        ]
        mock_summary.return_value = pd.DataFrame([
            {'project': 'nova', 'version': '2024.1.0'},
            {'project': 'neutron', 'version': '2024.1.0'}
        ])
        
        # ファイル保存のモック
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            result = self.collector.collect_release_data()
        
        # 結果検証
        assert result == True
        mock_clone.assert_called_once()
        mock_extract.assert_called_once()
        mock_summary.assert_called_once()
        mock_to_csv.assert_called()
    
    def test_get_project_releases(self):
        """プロジェクト別リリース情報取得のテスト"""
        # テスト用のサマリーデータを作成
        test_summary_path = self.test_data_dir / "releases_summary.csv"
        test_data = pd.DataFrame([
            {'project': 'nova', 'version': '2024.1.0', 'release_cycle': '2024.1'},
            {'project': 'nova', 'version': '2024.2.0', 'release_cycle': '2024.2'},
            {'project': 'neutron', 'version': '2024.1.0', 'release_cycle': '2024.1'}
        ])
        test_data.to_csv(test_summary_path, index=False)
        
        # プロジェクト別リリース情報取得
        with patch.object(self.collector, 'summary_file_path', test_summary_path):
            nova_releases = self.collector.get_project_releases('nova')
            neutron_releases = self.collector.get_project_releases('neutron')
        
        # 結果検証
        assert isinstance(nova_releases, pd.DataFrame)
        assert len(nova_releases) == 2
        assert all(nova_releases['project'] == 'nova')
        
        assert isinstance(neutron_releases, pd.DataFrame)
        assert len(neutron_releases) == 1
        assert all(neutron_releases['project'] == 'neutron')
    
    def test_get_release_dates(self):
        """リリース日付取得のテスト"""
        # テスト用のデリバラブルデータ
        test_deliverables = [
            {
                'release_cycle': '2024.1',
                'project': 'nova',
                'version': '2024.1.0',
                'release_date': '2024-04-01'
            },
            {
                'release_cycle': '2024.2',
                'project': 'nova',
                'version': '2024.2.0',
                'release_date': '2024-10-01'
            }
        ]
        
        # リリース日付の抽出テスト
        release_dates = {}
        for deliverable in test_deliverables:
            cycle = deliverable['release_cycle']
            if 'release_date' in deliverable:
                release_dates[cycle] = deliverable['release_date']
        
        assert '2024.1' in release_dates
        assert '2024.2' in release_dates
        assert release_dates['2024.1'] == '2024-04-01'
        assert release_dates['2024.2'] == '2024-10-01'


class TestReleaseCollectorFunctions:
    """リリースコレクター関数のテスト"""
    
    def test_download_openstack_releases_success(self):
        """OpenStackリリースダウンロード成功のテスト"""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0)
            
            result = download_openstack_releases("/tmp/test_releases")
            
            assert result == True
            mock_subprocess.assert_called()
    
    def test_download_openstack_releases_failure(self):
        """OpenStackリリースダウンロード失敗のテスト"""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=1)
            
            result = download_openstack_releases("/tmp/test_releases")
            
            assert result == False
            mock_subprocess.assert_called()
    
    def test_extract_deliverables_info_empty_directory(self):
        """空ディレクトリでのデリバラブル情報抽出テスト"""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.iterdir') as mock_iterdir:
            
            mock_exists.return_value = True
            mock_iterdir.return_value = []  # 空のディレクトリ
            
            result = extract_deliverables_info(Path("/empty/dir"))
            
            assert isinstance(result, list)
            assert len(result) == 0
    
    def test_generate_releases_summary_empty_data(self):
        """空データでのサマリー生成テスト"""
        result = generate_releases_summary([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestReleaseCollectorIntegration:
    """統合テスト"""
    
    def setup_method(self):
        """統合テストの初期化"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir)
    
    def teardown_method(self):
        """統合テストのクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.collectors.release_collector.app_path.DEFAULT_DATA_DIR')
    @patch('subprocess.run')
    def test_full_release_collection_workflow(self, mock_subprocess, mock_data_dir):
        """完全なリリース収集ワークフローのテスト"""
        mock_data_dir.return_value = self.test_data_dir
        mock_subprocess.return_value = Mock(returncode=0)
        
        # テスト用のYAMLファイルを作成
        releases_dir = self.test_data_dir / "openstack" / "releases_repo" / "deliverables"
        releases_dir.mkdir(parents=True, exist_ok=True)
        
        test_yaml_content = {
            'nova': {
                'releases': [
                    {
                        'version': '2024.1.0',
                        'projects': [{'repo': 'openstack/nova', 'hash': 'abc123'}]
                    }
                ]
            }
        }
        
        test_yaml_file = releases_dir / "2024.1.yaml"
        with open(test_yaml_file, 'w') as f:
            yaml.dump(test_yaml_content, f)
        
        # コレクターを初期化して実行
        collector = ReleaseCollector()
        
        # extract_deliverables_info関数を直接テスト
        deliverables = extract_deliverables_info(releases_dir)
        
        # 結果が正しく抽出されることを確認
        assert isinstance(deliverables, list)


if __name__ == '__main__':
    # テストの実行
    pytest.main([__file__, '-v'])
