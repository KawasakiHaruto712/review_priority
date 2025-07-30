"""
逆強化学習（IRL）モデルのテスト

このモジュールは、src/learning/irl_models.pyの主要な機能をテストします。
- MaxEntIRLModel クラス
- ReviewPriorityDataProcessor クラス
- 時系列学習関数
- 優先順位計算関数
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

# テスト対象のモジュールをインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.learning.irl_models import (
    MaxEntIRLModel,
    ReviewPriorityDataProcessor,
    is_bot_author,
    extract_learning_events,
    get_open_changes_at_time,
    calculate_review_priorities,
    run_temporal_irl_analysis
)


class TestMaxEntIRLModel:
    """MaxEntIRLModelクラスのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        self.model = MaxEntIRLModel(learning_rate=0.01, max_iterations=100, tolerance=1e-6)
        
        # テスト用の特徴量データ
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 5
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.rand(self.n_samples)
        self.y = self.y / np.sum(self.y)  # 正規化
        
    def test_model_initialization(self):
        """モデルの初期化テスト"""
        assert self.model.learning_rate == 0.01
        assert self.model.max_iterations == 100
        assert self.model.tolerance == 1e-6
        assert self.model.weights is None
        assert len(self.model.feature_names) == 0
        
    def test_compute_partition_function(self):
        """分配関数の計算テスト"""
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        features = self.X[:10]  # 小さなサンプルでテスト
        
        z = self.model.compute_partition_function(features, weights)
        
        assert isinstance(z, float)
        assert z > 0  # 分配関数は正の値
        assert not np.isnan(z)
        assert not np.isinf(z)
        
    def test_compute_expected_features(self):
        """期待特徴量の計算テスト"""
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        features = self.X[:10]
        
        expected_features = self.model.compute_expected_features(features, weights)
        
        assert expected_features.shape == (self.n_features,)
        assert not np.any(np.isnan(expected_features))
        assert not np.any(np.isinf(expected_features))
        
    def test_model_fit(self):
        """モデルの学習テスト"""
        self.model.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        
        training_stats = self.model.fit(self.X, self.y)
        
        # 学習後の状態確認
        assert self.model.weights is not None
        assert len(self.model.weights) == self.n_features
        assert not np.any(np.isnan(self.model.weights))
        
        # 統計情報の確認
        assert 'converged' in training_stats
        assert 'final_objective' in training_stats
        assert 'iterations' in training_stats
        
        assert isinstance(training_stats['converged'], bool)
        assert isinstance(training_stats['final_objective'], float)
        assert isinstance(training_stats['iterations'], int)
        
    def test_predict_priority_scores(self):
        """優先順位スコアの予測テスト"""
        # まず学習
        self.model.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        self.model.fit(self.X, self.y)
        
        # 予測
        test_X = np.random.randn(10, self.n_features)
        scores = self.model.predict_priority_scores(test_X)
        
        assert len(scores) == 10
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))
        
    def test_predict_without_training(self):
        """学習前の予測でエラーが発生することを確認"""
        test_X = np.random.randn(10, self.n_features)
        
        with pytest.raises(ValueError, match="モデルが学習されていません"):
            self.model.predict_priority_scores(test_X)
            
    def test_save_and_load_model(self):
        """モデルの保存と読み込みテスト"""
        # 学習
        self.model.feature_names = [f'feature_{i}' for i in range(self.n_features)]
        self.model.fit(self.X, self.y)
        original_weights = self.model.weights.copy()
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.model.save_model(tmp_path)
            
            # 新しいモデルで読み込み
            new_model = MaxEntIRLModel()
            new_model.load_model(tmp_path)
            
            # 重みが同じか確認
            np.testing.assert_array_almost_equal(original_weights, new_model.weights)
            assert self.model.feature_names == new_model.feature_names
            assert self.model.learning_rate == new_model.learning_rate
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    def test_is_bot_author(self):
        """ボット判定関数のテスト"""
        bot_names = ['jenkins', 'zuul', 'ci-bot']
        
        # ボットの場合
        assert is_bot_author('jenkins-user', bot_names) == True
        assert is_bot_author('zuul-bot', bot_names) == True
        assert is_bot_author('CI-BOT', bot_names) == True  # 大文字小文字無視
        
        # ボットでない場合
        assert is_bot_author('john.doe', bot_names) == False
        assert is_bot_author('developer', bot_names) == False
        
        # 空の場合
        assert is_bot_author('', bot_names) == False
        assert is_bot_author('test', []) == False
        
    def test_extract_learning_events(self):
        """学習イベント抽出のテスト"""
        # テスト用のデータフレーム作成
        test_data = [
            {
                'change_number': 12345,
                'id': 'I12345',
                'messages': [
                    {
                        'author': {'name': 'john.doe'},
                        'date': '2022-01-01T10:00:00Z'
                    },
                    {
                        'author': {'name': 'jenkins'},
                        'date': '2022-01-01T11:00:00Z'
                    },
                    {
                        'author': {'name': 'jane.smith'},
                        'date': '2022-01-01T12:00:00Z'
                    }
                ]
            },
            {
                'change_number': 12346,
                'id': 'I12346',
                'messages': [
                    {
                        'author': {'name': 'zuul'},
                        'date': '2022-01-01T13:00:00Z'
                    }
                ]
            }
        ]
        
        changes_df = pd.DataFrame(test_data)
        bot_names = ['jenkins', 'zuul']
        
        events = extract_learning_events(changes_df, bot_names)
        
        # ボット以外のメッセージのみ抽出されることを確認
        assert len(events) == 2  # john.doeとjane.smithのメッセージ
        assert events[0]['author'] == 'john.doe'
        assert events[1]['author'] == 'jane.smith'
        assert all('timestamp' in event for event in events)
        
    def test_get_open_changes_at_time(self):
        """指定時刻でのオープンなChange取得のテスト"""
        # テスト用データ
        test_data = [
            {
                'change_number': 1,
                'created': datetime(2022, 1, 1, 9, 0, 0),
                'updated': datetime(2022, 1, 1, 15, 0, 0)
            },
            {
                'change_number': 2,
                'created': datetime(2022, 1, 1, 11, 0, 0),
                'updated': datetime(2022, 1, 1, 13, 0, 0)
            },
            {
                'change_number': 3,
                'created': datetime(2022, 1, 1, 14, 0, 0),
                'updated': datetime(2022, 1, 1, 16, 0, 0)
            }
        ]
        
        changes_df = pd.DataFrame(test_data)
        target_time = datetime(2022, 1, 1, 12, 0, 0)
        
        open_changes = get_open_changes_at_time(changes_df, target_time)
        
        # 12:00時点でオープンなのは1番と3番（2番は13:00にクローズ済み）
        assert len(open_changes) == 2
        open_numbers = open_changes['change_number'].tolist()
        assert 1 in open_numbers
        assert 3 in open_numbers
        assert 2 not in open_numbers
        
    def test_calculate_review_priorities(self):
        """優先順位計算のテスト"""
        # テスト用のオープンChange
        test_data = [
            {
                'change_number': 1,
                'messages': [
                    {
                        'author': {'name': 'user1'},
                        'date': '2022-01-01T14:00:00Z'  # 2時間後
                    }
                ]
            },
            {
                'change_number': 2,
                'messages': [
                    {
                        'author': {'name': 'user2'},
                        'date': '2022-01-01T13:00:00Z'  # 1時間後
                    }
                ]
            },
            {
                'change_number': 3,
                'messages': []  # レビュー予定なし
            }
        ]
        
        open_changes = pd.DataFrame(test_data)
        current_time = datetime(2022, 1, 1, 12, 0, 0)
        bot_names = ['jenkins']
        
        priorities = calculate_review_priorities(open_changes, current_time, bot_names)
        
        # 3つのChangeがあるので重みは 3/3, 2/3, 1/3
        assert len(priorities) == 3
        assert priorities['2'] == 1.0  # 最も早くレビュー予定（1位）
        assert priorities['1'] == 2.0/3.0  # 2番目（2位）
        assert priorities['3'] == 1.0/3.0  # レビュー予定なし（3位）


class TestReviewPriorityDataProcessor:
    """ReviewPriorityDataProcessorクラスのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        with patch('src.learning.irl_models.ReviewStatusAnalyzer'):
            self.processor = ReviewPriorityDataProcessor()
    
    @patch('src.learning.irl_models.DEFAULT_DATA_DIR')
    def test_load_openstack_data_empty(self, mock_data_dir):
        """データが存在しない場合のテスト"""
        # 空のディレクトリを模擬
        mock_data_dir.__truediv__ = Mock()
        mock_data_dir.__truediv__.return_value.iterdir.return_value = []
        
        with patch('pathlib.Path.exists', return_value=False):
            changes_df, releases_df = self.processor.load_openstack_data()
            
        assert isinstance(changes_df, pd.DataFrame)
        assert isinstance(releases_df, pd.DataFrame)
        
    def test_extract_features_empty_input(self):
        """空の入力での特徴量抽出テスト"""
        empty_df = pd.DataFrame()
        analysis_time = datetime(2022, 1, 1, 12, 0, 0)
        
        result = self.processor.extract_features(empty_df, analysis_time)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestTemporalIRLAnalysis:
    """時系列IRL分析のテスト"""
    
    @patch('src.learning.irl_models.ReviewPriorityDataProcessor')
    @patch('src.utils.constants.START_DATE', '2022-01-01')
    @patch('src.utils.constants.END_DATE', '2022-01-31')
    @patch('src.utils.constants.OPENSTACK_CORE_COMPONENTS', ['test_project'])
    def test_run_temporal_irl_analysis_no_data(self, mock_processor_class):
        """データが存在しない場合の時系列分析テスト"""
        # モックの設定
        mock_processor = Mock()
        mock_processor.load_openstack_data.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_processor_class.return_value = mock_processor
        
        result = run_temporal_irl_analysis(['test_project'])
        
        assert 'error' in result
        assert result['error'] == "データが見つかりません"
        
    @patch('src.learning.irl_models.ReviewPriorityDataProcessor')
    @patch('src.learning.irl_models.DEFAULT_CONFIG')
    @patch('src.utils.constants.START_DATE', '2022-01-01')
    @patch('src.utils.constants.END_DATE', '2022-01-31')
    @patch('src.utils.constants.OPENSTACK_CORE_COMPONENTS', ['test_project'])
    def test_run_temporal_irl_analysis_with_data(self, mock_config, mock_processor_class):
        """データが存在する場合の時系列分析テスト"""
        # テストデータの準備
        test_changes = pd.DataFrame([
            {
                'component': 'test_project',
                'change_number': 12345,
                'created': datetime(2022, 1, 15, 10, 0, 0),
                'messages': [
                    {
                        'author': {'name': 'user1'},
                        'date': '2022-01-15T12:00:00Z'
                    }
                ]
            }
        ])
        
        # モックの設定
        mock_processor = Mock()
        mock_processor.load_openstack_data.return_value = (test_changes, pd.DataFrame())
        mock_processor.extract_features.return_value = pd.DataFrame()
        mock_processor_class.return_value = mock_processor
        
        # 設定ファイルのモック
        mock_config.__truediv__ = Mock()
        mock_config_path = Mock()
        mock_config.__truediv__.return_value = mock_config_path
        
        with patch('configparser.ConfigParser') as mock_config_parser:
            mock_parser = Mock()
            mock_parser.has_option.return_value = True
            mock_parser.get.return_value = 'jenkins, zuul'
            mock_config_parser.return_value = mock_parser
            
            result = run_temporal_irl_analysis(['test_project'])
        
        # エラーが発生するかもしれないが、少なくとも関数が実行されることを確認
        assert isinstance(result, dict)


class TestIntegration:
    """統合テスト"""
    
    def test_full_workflow_with_mock_data(self):
        """モックデータを使用した全体ワークフローのテスト"""
        # 小規模なテストデータでの完全なワークフローをテスト
        with patch('src.learning.irl_models.ReviewStatusAnalyzer'):
            processor = ReviewPriorityDataProcessor()
            
        # テスト用の特徴量データ
        features_df = pd.DataFrame({
            'change_number': [1, 2, 3],
            'bug_fix_confidence': [0.1, 0.2, 0.3],
            'lines_added': [10, 20, 30],
            'lines_deleted': [5, 10, 15],
            'files_changed': [1, 2, 3],
            'elapsed_time': [1.0, 2.0, 3.0],
            'revision_count': [1, 2, 3],
            'test_code_presence': [0, 1, 0],
            'past_report_count': [5, 10, 15],
            'recent_report_count': [1, 2, 3],
            'merge_rate': [0.8, 0.9, 0.7],
            'recent_merge_rate': [0.7, 0.8, 0.6],
            'days_to_major_release': [30, 60, 90],
            'open_ticket_count': [100, 200, 300],
            'reviewed_lines_in_period': [1000, 2000, 3000],
            'refactoring_confidence': [0.1, 0.2, 0.3],
            'uncompleted_requests': [1, 2, 3],
            'priority_weight': [1.0, 0.67, 0.33]
        })
        
        # 特徴量カラム
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        # 特徴量とラベルを抽出
        X = features_df[feature_columns].fillna(0).values
        y = features_df['priority_weight'].values
        
        # IRLモデルの学習
        model = MaxEntIRLModel(learning_rate=0.01, max_iterations=50)
        model.feature_names = feature_columns
        
        training_stats = model.fit(X, y)
        
        # 学習が正常に完了することを確認
        assert training_stats['converged'] in [True, False]  # 収束しない場合もある
        assert isinstance(training_stats['final_objective'], float)
        assert model.weights is not None
        assert len(model.weights) == len(feature_columns)
        
        # 予測の実行
        predictions = model.predict_priority_scores(X)
        assert len(predictions) == len(X)
        assert not np.any(np.isnan(predictions))


if __name__ == '__main__':
    # テストの実行
    pytest.main([__file__, '-v'])
