"""
時系列モデル評価のテスト

このモジュールは、src/learning/temporal_model_evaluation.pyの主要な機能をテストします。
- TemporalModelEvaluator クラス
- ウィンドウごとの正負例定義
- Balanced Random Forestによる分類評価
- 結果の保存機能
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path
import shutil

# テスト対象のモジュールをインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.learning.temporal_model_evaluation import (
    TemporalModelEvaluator,
    run_temporal_model_evaluation
)


# モックデータとヘルパー関数
def create_mock_changes_data(n_changes=100, start_date="2024-01-01", end_date="2024-01-31"):
    """テスト用のChangeデータを作成"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    np.random.seed(42)
    changes = []
    
    for i in range(n_changes):
        # ランダムな日時を生成
        random_days = np.random.randint(0, (end_dt - start_dt).days)
        created_time = start_dt + timedelta(days=random_days)
        
        # メッセージ（レビューコメント）を生成
        messages = []
        n_messages = np.random.randint(0, 5)
        for j in range(n_messages):
            msg_date = created_time + timedelta(hours=np.random.randint(1, 72))
            messages.append({
                'date': msg_date.isoformat(),
                'author': {
                    'name': 'jenkins' if j == 0 else f'reviewer_{j}',
                    'email': f'reviewer{j}@example.com'
                },
                'message': f'Review comment {j}'
            })
        
        change = {
            'change_number': f"12345{i}",
            'id': f"12345{i}",
            'component': np.random.choice(['nova', 'neutron', 'keystone']),
            'created': created_time,
            'updated': created_time + timedelta(hours=np.random.randint(1, 48)),
            'merged': created_time + timedelta(days=np.random.randint(1, 10)) if np.random.random() > 0.3 else None,
            'status': np.random.choice(['NEW', 'MERGED', 'ABANDONED']),
            'owner': {
                'name': f"user_{i % 10}",
                'email': f'user{i % 10}@example.com'
            },
            'owner_email': f'user{i % 10}@example.com',
            'lines_added': np.random.randint(1, 500),
            'lines_deleted': np.random.randint(0, 200),
            'files_changed': np.random.randint(1, 20),
            'messages': messages
        }
        changes.append(change)
    
    return pd.DataFrame(changes)


def create_mock_releases_data():
    """テスト用のリリースデータを作成"""
    releases = [
        {
            'version': '2024.1',
            'release_date': datetime(2024, 3, 1),
            'component': 'nova'
        },
        {
            'version': '2024.2',
            'release_date': datetime(2024, 6, 1),
            'component': 'nova'
        }
    ]
    return pd.DataFrame(releases)


def create_mock_features_data(n_samples=50):
    """テスト用の特徴量データを作成"""
    np.random.seed(42)
    
    feature_columns = [
        'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
        'elapsed_time', 'revision_count', 'test_code_presence',
        'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
        'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
        'refactoring_confidence', 'uncompleted_requests'
    ]
    
    data = {
        'change_number': [f"12345{i}" for i in range(n_samples)]
    }
    
    for col in feature_columns:
        data[col] = np.random.random(n_samples)
    
    return pd.DataFrame(data)


class TestTemporalModelEvaluator:
    """TemporalModelEvaluatorクラスのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        self.evaluator = TemporalModelEvaluator(window_size=14, sliding_step=1, random_state=42)
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """各テスト後のクリーンアップ"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """初期化のテスト"""
        assert self.evaluator.window_size == 14
        assert self.evaluator.sliding_step == 1
        assert self.evaluator.random_state == 42
        assert len(self.evaluator.feature_columns) == 16
        assert isinstance(self.evaluator.evaluation_results, dict)
    
    def test_generate_time_windows(self):
        """時間ウィンドウ生成のテスト"""
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        windows = self.evaluator.generate_time_windows(start_date, end_date)
        
        # ウィンドウ数の検証（実際の実装に合わせる）
        assert len(windows) == 17  # 実際の生成数
        
        # 最初と最後のウィンドウの検証
        first_window = windows[0]
        assert first_window[0] == datetime(2024, 1, 1)
        assert first_window[1] == datetime(2024, 1, 15)
        
        last_window = windows[-1]
        assert last_window[0] == datetime(2024, 1, 17)
        assert last_window[1] == datetime(2024, 1, 31)
    
    def test_load_bot_names(self):
        """ボット名読み込みのテスト"""
        # モックの設定ファイルを使用する場合のテスト
        bot_names = self.evaluator.load_bot_names()
        
        # 少なくともデフォルトのボット名が含まれているか確認
        assert isinstance(bot_names, list)
        assert len(bot_names) > 0
    
    def test_extract_window_labels_positive_case(self):
        """正例（レビューされたPR）の抽出テスト"""
        # ウィンドウ期間中にレビューされたChangeを作成
        window_start = datetime(2024, 1, 5)
        window_end = datetime(2024, 1, 10)
        
        changes = pd.DataFrame([{
            'change_number': '123456',
            'created': datetime(2024, 1, 1),
            'merged': datetime(2024, 1, 15),
            'messages': [
                {
                    'date': datetime(2024, 1, 7).isoformat(),
                    'author': {'name': 'reviewer1'},
                    'message': 'LGTM'
                }
            ]
        }])
        
        bot_names = ['jenkins', 'zuul']
        labels = self.evaluator.extract_window_labels(changes, window_start, window_end, bot_names)
        
        # レビューされているので正例（ラベル=1）
        assert labels.get('123456') == 1
    
    def test_extract_window_labels_negative_case(self):
        """負例（レビューされなかったPR）の抽出テスト"""
        # ウィンドウ期間中にレビューされなかったChangeを作成
        window_start = datetime(2024, 1, 5)
        window_end = datetime(2024, 1, 10)
        
        changes = pd.DataFrame([{
            'change_number': '123457',
            'created': datetime(2024, 1, 1),
            'merged': datetime(2024, 1, 15),
            'messages': []  # レビューなし
        }])
        
        bot_names = ['jenkins', 'zuul']
        labels = self.evaluator.extract_window_labels(changes, window_start, window_end, bot_names)
        
        # レビューされていないので負例（ラベル=0）
        assert labels.get('123457') == 0
    
    def test_extract_window_labels_bot_review(self):
        """ボットによるレビューは無視されることのテスト"""
        window_start = datetime(2024, 1, 5)
        window_end = datetime(2024, 1, 10)
        
        changes = pd.DataFrame([{
            'change_number': '123458',
            'created': datetime(2024, 1, 1),
            'merged': datetime(2024, 1, 15),
            'messages': [
                {
                    'date': datetime(2024, 1, 7).isoformat(),
                    'author': {'name': 'jenkins'},  # ボット
                    'message': 'Build succeeded'
                }
            ]
        }])
        
        bot_names = ['jenkins', 'zuul']
        labels = self.evaluator.extract_window_labels(changes, window_start, window_end, bot_names)
        
        # ボットのみのレビューなので負例（ラベル=0）
        assert labels.get('123458') == 0
    
    @patch('src.learning.temporal_model_evaluation.get_open_changes_at_time')
    @patch.object(TemporalModelEvaluator, 'extract_window_labels')
    def test_evaluate_window_success(self, mock_extract_labels, mock_get_open_changes):
        """ウィンドウ評価の成功ケーステスト"""
        # モックデータの準備
        window_start = datetime(2024, 1, 1)
        window_end = datetime(2024, 1, 15)
        
        # オープンなChangeのモック（十分なデータ数）
        mock_changes = create_mock_changes_data(n_changes=100)
        mock_get_open_changes.return_value = mock_changes
        
        # ラベルのモック（正負例が混在し、十分な数）
        labels = {}
        for i in range(100):
            labels[f"12345{i}"] = 1 if i < 60 else 0  # 60個が正例、40個が負例
        mock_extract_labels.return_value = labels
        
        # 特徴量抽出のモック
        with patch.object(self.evaluator.data_processor, 'extract_features') as mock_extract:
            mock_features = create_mock_features_data(n_samples=100)
            mock_extract.return_value = mock_features
            
            # 評価実行
            result = self.evaluator.evaluate_window(
                mock_changes, window_start, window_end,
                ['jenkins'], 'nova', pd.DataFrame()
            )
        
        # 結果の検証
        assert result is not None
        assert 'window_start' in result
        assert 'window_end' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
        assert 'evaluation_status' in result
        assert result['evaluation_status'] == 'success'
    
    @patch('src.learning.temporal_model_evaluation.get_open_changes_at_time')
    def test_evaluate_window_no_open_changes(self, mock_get_open_changes):
        """オープンなChangeがない場合のテスト"""
        window_start = datetime(2024, 1, 1)
        window_end = datetime(2024, 1, 15)
        
        # 空のDataFrameを返す
        mock_get_open_changes.return_value = pd.DataFrame()
        
        result = self.evaluator.evaluate_window(
            pd.DataFrame(), window_start, window_end,
            ['jenkins'], 'nova', pd.DataFrame()
        )
        
        # Noneが返されることを確認
        assert result is None
    
    def test_save_results_csv(self):
        """CSV保存機能のテスト"""
        # テスト用の評価結果を作成
        self.evaluator.evaluation_results = {
            'nova': [
                {
                    'window_start': datetime(2024, 1, 1),
                    'window_end': datetime(2024, 1, 15),
                    'total_samples': 100,
                    'train_samples': 80,
                    'test_samples': 20,
                    'positive_samples': 60,
                    'negative_samples': 40,
                    'precision': 0.85,
                    'recall': 0.75,
                    'f1_score': 0.80,
                    'evaluation_status': 'success'
                }
            ]
        }
        
        # 結果を保存
        saved_files = self.evaluator.save_results(self.temp_dir)
        
        # CSV保存の確認
        assert len(saved_files['csv']) > 0
        csv_path = Path(saved_files['csv'][0])
        assert csv_path.exists()
        
        # CSVの内容を確認
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert 'f1_score' in df.columns
        assert df['f1_score'].iloc[0] == 0.80
    
    def test_save_results_json(self):
        """JSON保存機能のテスト"""
        # テスト用の評価結果を作成
        self.evaluator.evaluation_results = {
            'nova': [
                {
                    'window_start': datetime(2024, 1, 1),
                    'window_end': datetime(2024, 1, 15),
                    'total_samples': 100,
                    'train_samples': 80,
                    'test_samples': 20,
                    'positive_samples': 60,
                    'negative_samples': 40,
                    'precision': 0.85,
                    'recall': 0.75,
                    'f1_score': 0.80,
                    'evaluation_status': 'success'
                }
            ]
        }
        
        # 結果を保存
        saved_files = self.evaluator.save_results(self.temp_dir)
        
        # JSON保存の確認
        assert len(saved_files['json']) > 0
        json_path = Path(saved_files['json'][0])
        assert json_path.exists()
        
        # JSONの内容を確認
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert data['project'] == 'nova'
        assert 'summary_statistics' in data
        assert 'mean_f1' in data['summary_statistics']
        assert data['summary_statistics']['mean_f1'] == 0.80
    
    def test_save_results_summary_statistics(self):
        """サマリー統計の計算テスト"""
        # 複数ウィンドウの評価結果を作成
        self.evaluator.evaluation_results = {
            'nova': [
                {
                    'window_start': datetime(2024, 1, 1),
                    'window_end': datetime(2024, 1, 15),
                    'total_samples': 100,
                    'train_samples': 80,
                    'test_samples': 20,
                    'positive_samples': 60,
                    'negative_samples': 40,
                    'precision': 0.80,
                    'recall': 0.70,
                    'f1_score': 0.75,
                    'evaluation_status': 'success'
                },
                {
                    'window_start': datetime(2024, 1, 2),
                    'window_end': datetime(2024, 1, 16),
                    'total_samples': 100,
                    'train_samples': 80,
                    'test_samples': 20,
                    'positive_samples': 60,
                    'negative_samples': 40,
                    'precision': 0.90,
                    'recall': 0.80,
                    'f1_score': 0.85,
                    'evaluation_status': 'success'
                }
            ]
        }
        
        # 結果を保存
        saved_files = self.evaluator.save_results(self.temp_dir)
        
        # JSON読み込み
        json_path = Path(saved_files['json'][0])
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # サマリー統計の検証（浮動小数点の誤差を考慮）
        stats = data['summary_statistics']
        assert stats['successful_windows'] == 2
        assert stats['total_windows'] == 2
        assert abs(stats['mean_f1'] - 0.80) < 0.001  # (0.75 + 0.85) / 2
        assert abs(stats['mean_precision'] - 0.85) < 0.001  # (0.80 + 0.90) / 2
    
    @patch.object(TemporalModelEvaluator, 'create_evaluation_visualization')
    def test_save_results_pdf(self, mock_create_viz):
        """PDF生成のテスト"""
        # PDFパスのモック
        mock_create_viz.return_value = str(self.temp_dir / "temporal_evaluation_nova.pdf")
        
        # テスト用の評価結果を作成
        self.evaluator.evaluation_results = {
            'nova': [
                {
                    'window_start': datetime(2024, 1, 1),
                    'window_end': datetime(2024, 1, 15),
                    'total_samples': 100,
                    'train_samples': 80,
                    'test_samples': 20,
                    'positive_samples': 60,
                    'negative_samples': 40,
                    'precision': 0.85,
                    'recall': 0.75,
                    'f1_score': 0.80,
                    'evaluation_status': 'success'
                }
            ]
        }
        
        # 結果を保存
        saved_files = self.evaluator.save_results(self.temp_dir)
        
        # PDF生成関数が呼ばれたことを確認
        mock_create_viz.assert_called_once_with('nova', self.temp_dir)
        
        # PDFリストに追加されたことを確認
        assert len(saved_files['pdf']) > 0


class TestRunTemporalModelEvaluation:
    """run_temporal_model_evaluation関数のテスト"""
    
    @patch('src.learning.temporal_model_evaluation.TemporalModelEvaluator')
    def test_run_temporal_model_evaluation_success(self, mock_evaluator_class):
        """評価実行の成功ケーステスト"""
        # モックの設定
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        
        # run_temporal_evaluationの戻り値をモック
        mock_evaluator.run_temporal_evaluation.return_value = {
            'evaluation_config': {
                'window_size': 14,
                'sliding_step': 1
            },
            'projects_evaluated': 2,
            'total_windows': 10
        }
        
        # save_resultsの戻り値をモック
        mock_evaluator.save_results.return_value = {
            'csv': ['test.csv'],
            'json': ['test.json'],
            'pdf': ['test.pdf']
        }
        
        # 実行
        result = run_temporal_model_evaluation(['nova', 'neutron'])
        
        # 検証
        assert 'evaluation_config' in result
        assert result['projects_evaluated'] == 2
        mock_evaluator.run_temporal_evaluation.assert_called_once()
        mock_evaluator.save_results.assert_called_once()
    
    @patch('src.learning.temporal_model_evaluation.TemporalModelEvaluator')
    def test_run_temporal_model_evaluation_error(self, mock_evaluator_class):
        """評価実行のエラーケーステスト"""
        # エラーを発生させるモック
        mock_evaluator_class.side_effect = Exception("Test error")
        
        # 実行
        result = run_temporal_model_evaluation()
        
        # エラーが返されることを確認
        assert 'error' in result
        assert 'Test error' in result['error']


class TestEdgeCases:
    """エッジケースのテスト"""
    
    def setup_method(self):
        """各テストの前に実行される初期化"""
        self.evaluator = TemporalModelEvaluator(window_size=14, sliding_step=1, random_state=42)
    
    def test_empty_evaluation_results(self):
        """評価結果が空の場合のテスト"""
        self.evaluator.evaluation_results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = self.evaluator.save_results(Path(temp_dir))
            
            # 空の結果でもエラーが発生しないことを確認
            assert saved_files['csv'] == []
            assert saved_files['json'] == []
            assert saved_files['pdf'] == []
    
    def test_all_failed_evaluations(self):
        """全てのウィンドウ評価が失敗した場合のテスト"""
        self.evaluator.evaluation_results = {
            'nova': [
                {
                    'window_start': datetime(2024, 1, 1),
                    'window_end': datetime(2024, 1, 15),
                    'total_samples': 0,
                    'train_samples': 0,
                    'test_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'evaluation_status': 'failure',
                    'error_message': 'Test error'
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = self.evaluator.save_results(Path(temp_dir))
            
            # JSON保存は実行されることを確認
            assert len(saved_files['json']) > 0
            
            # JSONの内容を確認
            json_path = Path(saved_files['json'][0])
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 全て失敗の場合、サマリー統計はゼロ
            assert data['summary_statistics']['successful_windows'] == 0
            assert data['summary_statistics']['mean_f1'] == 0.0
    
    def test_single_window(self):
        """単一ウィンドウのみの場合のテスト"""
        windows = self.evaluator.generate_time_windows("2024-01-01", "2024-01-15")
        
        # ウィンドウサイズと同じ期間の場合、1つのウィンドウのみ
        assert len(windows) == 1
        assert windows[0][0] == datetime(2024, 1, 1)
        assert windows[0][1] == datetime(2024, 1, 15)


if __name__ == "__main__":
    """テストの実行"""
    pytest.main([__file__, '-v'])
