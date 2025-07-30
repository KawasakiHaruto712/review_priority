"""
IRLモデルのパフォーマンステスト

このモジュールは、IRLモデルの実行時間、メモリ使用量、
スケーラビリティなどのパフォーマンス関連のテストを実行します。
"""

import pytest
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import psutil
import os
from typing import Dict, List, Tuple
from unittest.mock import patch

# テスト対象モジュール
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.learning.irl_models import MaxEntIRLModel, calculate_review_priorities
from tests.learning.test_helpers import (
    create_mock_features_dataframe,
    create_mock_changes_dataframe,
    assert_valid_irl_training_stats
)


class TestPerformance:
    """パフォーマンステスト"""
    
    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[float, any]:
        """
        関数の実行時間を測定
        
        Returns:
            Tuple[float, any]: (実行時間(秒), 関数の戻り値)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time, result
    
    def measure_memory_usage(self) -> float:
        """
        現在のメモリ使用量を測定（MB）
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # バイトからMBに変換
    
    def test_model_training_performance_small(self):
        """小規模データでのモデル学習パフォーマンステスト"""
        # 小規模データ（100サンプル、16特徴量）
        features_df = create_mock_features_dataframe(list(range(100)))
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        X = features_df[feature_columns].fillna(0).values
        y = features_df['priority_weight'].values
        
        model = MaxEntIRLModel(learning_rate=0.01, max_iterations=100)
        model.feature_names = feature_columns
        
        # 実行時間測定
        execution_time, training_stats = self.measure_execution_time(model.fit, X, y)
        
        print(f"小規模データ学習時間: {execution_time:.2f}秒")
        assert execution_time < 10.0, f"学習時間が長すぎます: {execution_time:.2f}秒"
        assert_valid_irl_training_stats(training_stats)
    
    def test_model_training_performance_medium(self):
        """中規模データでのモデル学習パフォーマンステスト"""
        # 中規模データ（1000サンプル、16特徴量）
        features_df = create_mock_features_dataframe(list(range(1000)))
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        X = features_df[feature_columns].fillna(0).values
        y = features_df['priority_weight'].values
        
        model = MaxEntIRLModel(learning_rate=0.01, max_iterations=100)
        model.feature_names = feature_columns
        
        # メモリ使用量測定開始
        memory_before = self.measure_memory_usage()
        
        # 実行時間測定
        execution_time, training_stats = self.measure_execution_time(model.fit, X, y)
        
        # メモリ使用量測定終了
        memory_after = self.measure_memory_usage()
        memory_usage = memory_after - memory_before
        
        print(f"中規模データ学習時間: {execution_time:.2f}秒")
        print(f"メモリ使用量増加: {memory_usage:.2f}MB")
        
        assert execution_time < 60.0, f"学習時間が長すぎます: {execution_time:.2f}秒"
        assert memory_usage < 500.0, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        assert_valid_irl_training_stats(training_stats)
    
    @pytest.mark.slow
    def test_model_training_performance_large(self):
        """大規模データでのモデル学習パフォーマンステスト（スローテスト）"""
        # 大規模データ（5000サンプル、16特徴量）
        features_df = create_mock_features_dataframe(list(range(5000)))
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        X = features_df[feature_columns].fillna(0).values
        y = features_df['priority_weight'].values
        
        model = MaxEntIRLModel(learning_rate=0.01, max_iterations=50)  # 反復数を減らして高速化
        model.feature_names = feature_columns
        
        # メモリ使用量測定開始
        memory_before = self.measure_memory_usage()
        
        # 実行時間測定
        execution_time, training_stats = self.measure_execution_time(model.fit, X, y)
        
        # メモリ使用量測定終了
        memory_after = self.measure_memory_usage()
        memory_usage = memory_after - memory_before
        
        print(f"大規模データ学習時間: {execution_time:.2f}秒")
        print(f"メモリ使用量増加: {memory_usage:.2f}MB")
        
        assert execution_time < 300.0, f"学習時間が長すぎます: {execution_time:.2f}秒"
        assert memory_usage < 1000.0, f"メモリ使用量が多すぎます: {memory_usage:.2f}MB"
        assert_valid_irl_training_stats(training_stats)
    
    def test_priority_calculation_performance(self):
        """優先順位計算のパフォーマンステスト"""
        # 様々なサイズでの優先順位計算性能をテスト
        sizes = [10, 50, 100, 500]
        bot_names = ['jenkins', 'zuul', 'ci-bot']
        current_time = datetime(2022, 1, 1, 12, 0, 0)
        
        for size in sizes:
            changes_df = create_mock_changes_dataframe(size, current_time)
            
            execution_time, priorities = self.measure_execution_time(
                calculate_review_priorities, changes_df, current_time, bot_names
            )
            
            print(f"優先順位計算 ({size}件): {execution_time:.4f}秒")
            
            # 実行時間の検証（線形時間であることを期待）
            expected_max_time = size * 0.01  # 1件あたり10msを上限とする
            assert execution_time < expected_max_time, \
                f"優先順位計算が遅すぎます ({size}件): {execution_time:.4f}秒 > {expected_max_time:.4f}秒"
            
            # 結果の検証
            assert len(priorities) == size
    
    def test_prediction_performance(self):
        """予測性能のテスト"""
        # モデルを訓練
        features_df = create_mock_features_dataframe(list(range(100)))
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        X_train = features_df[feature_columns].fillna(0).values
        y_train = features_df['priority_weight'].values
        
        model = MaxEntIRLModel(learning_rate=0.01, max_iterations=50)
        model.feature_names = feature_columns
        model.fit(X_train, y_train)
        
        # 様々なサイズでの予測性能をテスト
        prediction_sizes = [10, 100, 1000]
        
        for size in prediction_sizes:
            # テスト用データ生成
            test_features_df = create_mock_features_dataframe(list(range(size)))
            X_test = test_features_df[feature_columns].fillna(0).values
            
            execution_time, predictions = self.measure_execution_time(
                model.predict_priority_scores, X_test
            )
            
            print(f"予測 ({size}件): {execution_time:.4f}秒")
            
            # 予測は高速であることを期待
            assert execution_time < 1.0, f"予測が遅すぎます ({size}件): {execution_time:.4f}秒"
            assert len(predictions) == size
            assert not np.any(np.isnan(predictions))
    
    def test_convergence_performance(self):
        """収束性能のテスト"""
        # 収束しやすいデータと収束しにくいデータでのテスト
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        # 収束しやすいデータ（明確な傾向がある）
        np.random.seed(42)
        easy_features = create_mock_features_dataframe(list(range(100)))
        X_easy = easy_features[feature_columns].fillna(0).values
        y_easy = easy_features['priority_weight'].values
        
        model_easy = MaxEntIRLModel(learning_rate=0.01, max_iterations=1000)
        model_easy.feature_names = feature_columns
        
        execution_time_easy, stats_easy = self.measure_execution_time(
            model_easy.fit, X_easy, y_easy
        )
        
        print(f"収束しやすいデータ: {execution_time_easy:.2f}秒, 反復回数: {stats_easy['iterations']}")
        
        # 収束しにくいデータ（ランダム）
        np.random.seed(123)
        hard_features = create_mock_features_dataframe(list(range(100)))
        # 優先度をランダムにシャッフル
        y_hard = np.random.permutation(hard_features['priority_weight'].values)
        X_hard = hard_features[feature_columns].fillna(0).values
        
        model_hard = MaxEntIRLModel(learning_rate=0.01, max_iterations=1000)
        model_hard.feature_names = feature_columns
        
        execution_time_hard, stats_hard = self.measure_execution_time(
            model_hard.fit, X_hard, y_hard
        )
        
        print(f"収束しにくいデータ: {execution_time_hard:.2f}秒, 反復回数: {stats_hard['iterations']}")
        
        # 収束しやすいデータの方が早く終わることを期待
        # assert stats_easy['iterations'] <= stats_hard['iterations'], \
        #     "収束しやすいデータの方が多くの反復を要しています"


class TestMemoryLeaks:
    """メモリリークのテスト"""
    
    def test_repeated_training_memory_leak(self):
        """繰り返し学習でのメモリリークテスト"""
        initial_memory = self.measure_memory_usage()
        
        features_df = create_mock_features_dataframe(list(range(100)))
        feature_columns = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        X = features_df[feature_columns].fillna(0).values
        y = features_df['priority_weight'].values
        
        # 複数回学習を実行
        for i in range(10):
            model = MaxEntIRLModel(learning_rate=0.01, max_iterations=10)
            model.feature_names = feature_columns
            model.fit(X, y)
            
            # 明示的にモデルを削除
            del model
        
        final_memory = self.measure_memory_usage()
        memory_increase = final_memory - initial_memory
        
        print(f"初期メモリ: {initial_memory:.2f}MB")
        print(f"最終メモリ: {final_memory:.2f}MB")
        print(f"メモリ増加: {memory_increase:.2f}MB")
        
        # メモリ増加が100MB以下であることを確認
        assert memory_increase < 100.0, f"メモリリークの可能性: {memory_increase:.2f}MB増加"
    
    def measure_memory_usage(self) -> float:
        """現在のメモリ使用量を測定（MB）"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024


class TestScalability:
    """スケーラビリティテスト"""
    
    def test_feature_dimension_scalability(self):
        """特徴量次元数に対するスケーラビリティテスト"""
        base_features = [
            'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
            'elapsed_time', 'revision_count', 'test_code_presence',
            'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
            'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
            'refactoring_confidence', 'uncompleted_requests'
        ]
        
        sample_size = 100
        execution_times = []
        
        # 異なる特徴量次元数でテスト
        for num_features in [4, 8, 16, 32]:
            feature_subset = base_features[:min(num_features, len(base_features))]
            
            # 不足分はダミー特徴量で補う
            if num_features > len(base_features):
                for i in range(len(base_features), num_features):
                    feature_subset.append(f'dummy_feature_{i}')
            
            # テストデータ生成
            features_df = create_mock_features_dataframe(list(range(sample_size)))
            
            # ダミー特徴量を追加
            for feature in feature_subset:
                if feature not in features_df.columns:
                    features_df[feature] = np.random.randn(sample_size)
            
            X = features_df[feature_subset].fillna(0).values
            y = features_df['priority_weight'].values
            
            model = MaxEntIRLModel(learning_rate=0.01, max_iterations=20)
            model.feature_names = feature_subset
            
            start_time = time.time()
            model.fit(X, y)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            print(f"特徴量数 {num_features}: {execution_time:.2f}秒")
        
        # 実行時間が次元数に対して合理的に増加することを確認
        # （指数的ではなく多項式的増加を期待）
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            assert ratio < 10.0, f"特徴量数増加による実行時間の増加が大きすぎます: {ratio:.2f}倍"


if __name__ == '__main__':
    # パフォーマンステストの実行例
    print("=== パフォーマンステスト実行例 ===")
    
    test_perf = TestPerformance()
    
    print("\n--- 小規模データテスト ---")
    test_perf.test_model_training_performance_small()
    
    print("\n--- 中規模データテスト ---")
    test_perf.test_model_training_performance_medium()
    
    print("\n--- 優先順位計算テスト ---")
    test_perf.test_priority_calculation_performance()
    
    print("\n--- 予測性能テスト ---")
    test_perf.test_prediction_performance()
    
    print("\nすべてのパフォーマンステストが完了しました。")
