"""
IRLモデルのテスト実行スクリプト

pytest がインストールされていない環境でも実行可能な
シンプルなテストランナーを提供します。
"""

import sys
import os
import traceback
from typing import List, Callable, Tuple

# パスの追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# テストモジュールのインポート
try:
    from tests.learning.test_helpers import (
        create_mock_change_data,
        create_mock_features_dataframe,
        create_learning_event_scenario,
        assert_valid_irl_training_stats,
        assert_valid_priority_weights
    )
    print("✓ テストヘルパーモジュールのインポート成功")
except ImportError as e:
    print(f"✗ テストヘルパーモジュールのインポートエラー: {e}")

# テスト対象モジュールのインポート
try:
    from src.learning.irl_models import (
        MaxEntIRLModel,
        ReviewPriorityDataProcessor,
        is_bot_author,
        calculate_review_priorities
    )
    print("✓ IRLモデルモジュールのインポート成功")
except ImportError as e:
    print(f"✗ IRLモデルモジュールのインポートエラー: {e}")


class SimpleTestRunner:
    """シンプルなテストランナー"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_func: Callable, test_name: str):
        """単一のテストを実行"""
        try:
            print(f"実行中: {test_name}")
            test_func()
            self.passed += 1
            print(f"✓ {test_name} - PASSED")
        except AssertionError as e:
            self.failed += 1
            error_msg = f"✗ {test_name} - FAILED: {str(e)}"
            print(error_msg)
            self.errors.append(error_msg)
        except Exception as e:
            self.failed += 1
            error_msg = f"✗ {test_name} - ERROR: {str(e)}"
            print(error_msg)
            self.errors.append(error_msg)
            print(traceback.format_exc())
    
    def print_summary(self):
        """テスト結果のサマリーを表示"""
        total = self.passed + self.failed
        print(f"\n=== テスト結果サマリー ===")
        print(f"実行: {total}")
        print(f"成功: {self.passed}")
        print(f"失敗: {self.failed}")
        
        if self.errors:
            print(f"\n=== エラー詳細 ===")
            for error in self.errors:
                print(error)


def test_is_bot_author():
    """ボット判定関数のテスト"""
    bot_names = ['jenkins', 'zuul', 'ci-bot']
    
    # ボットの場合
    assert is_bot_author('jenkins-user', bot_names) == True
    assert is_bot_author('zuul-bot', bot_names) == True
    assert is_bot_author('CI-BOT', bot_names) == True
    
    # ボットでない場合
    assert is_bot_author('john.doe', bot_names) == False
    assert is_bot_author('developer', bot_names) == False
    
    # 空の場合
    assert is_bot_author('', bot_names) == False
    assert is_bot_author('test', []) == False


def test_maxent_irl_model_basic():
    """MaxEntIRLModelの基本機能テスト"""
    import numpy as np
    
    # モデル初期化
    model = MaxEntIRLModel(learning_rate=0.01, max_iterations=10, tolerance=1e-6)
    assert model.learning_rate == 0.01
    assert model.max_iterations == 10
    assert model.weights is None
    
    # テストデータ作成
    np.random.seed(42)
    n_samples, n_features = 20, 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples)
    y = y / np.sum(y)  # 正規化
    
    # 学習
    model.feature_names = [f'feature_{i}' for i in range(n_features)]
    training_stats = model.fit(X, y)
    
    # 学習結果の検証
    assert model.weights is not None
    assert len(model.weights) == n_features
    assert_valid_irl_training_stats(training_stats)
    
    # 予測
    predictions = model.predict_priority_scores(X)
    assert len(predictions) == n_samples
    assert not np.any(np.isnan(predictions))


def test_calculate_review_priorities():
    """優先順位計算のテスト"""
    import pandas as pd
    from datetime import datetime, timezone
    
    # テスト用データ（UTCタイムゾーン付き）
    test_data = [
        {
            'change_number': 1,
            'messages': [
                {
                    'author': {'name': 'user1'},
                    'date': '2022-01-01T14:00:00+00:00'  # 2時間後
                }
            ]
        },
        {
            'change_number': 2,
            'messages': [
                {
                    'author': {'name': 'user2'},
                    'date': '2022-01-01T13:00:00+00:00'  # 1時間後
                }
            ]
        },
        {
            'change_number': 3,
            'messages': []  # レビュー予定なし
        }
    ]
    
    open_changes = pd.DataFrame(test_data)
    current_time = datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    bot_names = ['jenkins']
    
    priorities = calculate_review_priorities(open_changes, current_time, bot_names)
    
    # 検証
    assert_valid_priority_weights(priorities, 3)
    
    # 順序の確認
    assert priorities['2'] > priorities['1']  # user2が早くレビュー予定
    assert priorities['1'] > priorities['3']  # user1がレビュー予定なしより高い


def test_mock_data_generation():
    """モックデータ生成のテスト"""
    from datetime import datetime
    
    # Changeデータの生成
    change_data = create_mock_change_data(
        change_number=12345,
        created_time=datetime(2022, 1, 1, 10, 0, 0),
        component="test_component"
    )
    
    assert change_data['change_number'] == 12345
    assert change_data['component'] == "test_component"
    assert 'owner' in change_data
    assert 'messages' in change_data
    
    # 特徴量データフレームの生成
    features_df = create_mock_features_dataframe([1001, 1002, 1003])
    
    assert len(features_df) == 3
    assert 'change_number' in features_df.columns
    assert 'priority_weight' in features_df.columns
    assert all(0 <= w <= 1 for w in features_df['priority_weight'])


def test_learning_event_scenario():
    """学習イベントシナリオのテスト"""
    changes_df, expected_events = create_learning_event_scenario()
    
    assert len(changes_df) == 3
    assert len(expected_events) == 3
    
    # 全てのChangeにメッセージが含まれることを確認
    for _, change in changes_df.iterrows():
        assert 'messages' in change
        assert isinstance(change['messages'], list)


def test_data_processor_initialization():
    """データプロセッサーの初期化テスト"""
    from unittest.mock import patch
    
    with patch('src.learning.irl_models.ReviewStatusAnalyzer'):
        processor = ReviewPriorityDataProcessor()
        assert processor is not None


def run_all_tests():
    """全テストの実行"""
    runner = SimpleTestRunner()
    
    # 基本機能テスト
    runner.run_test(test_is_bot_author, "ボット判定機能")
    runner.run_test(test_maxent_irl_model_basic, "MaxEntIRLModel基本機能")
    runner.run_test(test_calculate_review_priorities, "優先順位計算")
    runner.run_test(test_mock_data_generation, "モックデータ生成")
    runner.run_test(test_learning_event_scenario, "学習イベントシナリオ")
    runner.run_test(test_data_processor_initialization, "データプロセッサー初期化")
    
    runner.print_summary()
    return runner.failed == 0


if __name__ == '__main__':
    print("=== IRLモデル テスト実行 ===")
    print("このスクリプトは pytest がなくても実行可能です。\n")
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 すべてのテストが成功しました！")
        sys.exit(0)
    else:
        print("\n❌ いくつかのテストが失敗しました。")
        sys.exit(1)
