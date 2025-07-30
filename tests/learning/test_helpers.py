"""
IRLモデルのテスト用ヘルパー関数とモックデータ生成

このモジュールは、test_irl_models.pyで使用するテストデータの生成と
ヘルパー関数を提供します。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json


def create_mock_change_data(
    change_number: int,
    created_time: datetime,
    messages: List[Dict] = None,
    component: str = "test_component",
    updated_time: datetime = None
) -> Dict[str, Any]:
    """
    テスト用のChangeデータを作成
    
    Args:
        change_number (int): Change番号
        created_time (datetime): 作成時刻
        messages (List[Dict]): メッセージリスト
        component (str): コンポーネント名
        updated_time (datetime): 更新時刻
        
    Returns:
        Dict[str, Any]: Changeデータ
    """
    if messages is None:
        messages = []
    
    if updated_time is None:
        updated_time = created_time + timedelta(hours=24)
    
    return {
        'change_number': change_number,
        'id': f'I{change_number}',
        'component': component,
        'created': created_time,
        'updated': updated_time,
        'subject': f'Test change {change_number}',
        'message': f'This is a test change {change_number}',
        'messages': messages,
        'owner': {
            'name': f'user{change_number}',
            'email': f'user{change_number}@example.com'
        },
        'status': 'NEW'
    }


def create_mock_message(
    author_name: str,
    message_time: datetime,
    message_text: str = "Test review comment"
) -> Dict[str, Any]:
    """
    テスト用のメッセージデータを作成
    
    Args:
        author_name (str): 著者名
        message_time (datetime): メッセージ時刻
        message_text (str): メッセージ内容
        
    Returns:
        Dict[str, Any]: メッセージデータ
    """
    return {
        'author': {
            'name': author_name,
            'email': f'{author_name}@example.com',
            'username': author_name
        },
        'date': message_time.isoformat() + 'Z',
        'message': message_text
    }


def create_mock_changes_dataframe(
    num_changes: int = 10,
    start_time: datetime = None,
    time_interval_hours: int = 1
) -> pd.DataFrame:
    """
    テスト用のChangesデータフレームを作成
    
    Args:
        num_changes (int): 作成するChange数
        start_time (datetime): 開始時刻
        time_interval_hours (int): Change間の時間間隔（時間）
        
    Returns:
        pd.DataFrame: Changesデータフレーム
    """
    if start_time is None:
        start_time = datetime(2022, 1, 1, 10, 0, 0)
    
    changes = []
    for i in range(num_changes):
        created_time = start_time + timedelta(hours=i * time_interval_hours)
        
        # いくつかのChangeにメッセージを追加
        messages = []
        if i % 3 == 0:  # 3つに1つの割合でレビューメッセージを追加
            review_time = created_time + timedelta(hours=2)
            messages.append(create_mock_message(f'reviewer{i}', review_time))
        
        change_data = create_mock_change_data(
            change_number=1000 + i,
            created_time=created_time,
            messages=messages,
            component='test_project'
        )
        changes.append(change_data)
    
    return pd.DataFrame(changes)


def create_mock_features_dataframe(
    change_numbers: List[int],
    priority_weights: List[float] = None
) -> pd.DataFrame:
    """
    テスト用の特徴量データフレームを作成
    
    Args:
        change_numbers (List[int]): Change番号のリスト
        priority_weights (List[float]): 優先度重みのリスト
        
    Returns:
        pd.DataFrame: 特徴量データフレーム
    """
    num_changes = len(change_numbers)
    
    if priority_weights is None:
        priority_weights = [(num_changes - i) / num_changes for i in range(num_changes)]
    
    # ランダムな特徴量データを生成
    np.random.seed(42)
    
    features_data = {
        'change_number': change_numbers,
        'component': ['test_project'] * num_changes,
        'created': [datetime(2022, 1, 1, 10, i) for i in range(num_changes)],
        'bug_fix_confidence': np.random.uniform(0, 1, num_changes),
        'lines_added': np.random.randint(1, 100, num_changes),
        'lines_deleted': np.random.randint(0, 50, num_changes),
        'files_changed': np.random.randint(1, 10, num_changes),
        'elapsed_time': np.random.uniform(0.1, 48.0, num_changes),
        'revision_count': np.random.randint(1, 5, num_changes),
        'test_code_presence': np.random.randint(0, 2, num_changes),
        'past_report_count': np.random.randint(0, 50, num_changes),
        'recent_report_count': np.random.randint(0, 10, num_changes),
        'merge_rate': np.random.uniform(0.5, 1.0, num_changes),
        'recent_merge_rate': np.random.uniform(0.4, 1.0, num_changes),
        'days_to_major_release': np.random.randint(1, 365, num_changes),
        'open_ticket_count': np.random.randint(50, 500, num_changes),
        'reviewed_lines_in_period': np.random.randint(100, 10000, num_changes),
        'refactoring_confidence': np.random.uniform(0, 1, num_changes),
        'uncompleted_requests': np.random.randint(0, 5, num_changes),
        'priority_weight': priority_weights
    }
    
    return pd.DataFrame(features_data)


class MockConfigParser:
    """テスト用のConfigParserモック"""
    
    def __init__(self, bot_names: List[str] = None):
        if bot_names is None:
            bot_names = ['jenkins', 'zuul', 'ci-bot']
        self.bot_names = bot_names
    
    def read(self, config_path):
        """設定ファイル読み込みのモック"""
        pass
    
    def has_option(self, section: str, option: str) -> bool:
        """オプション存在確認のモック"""
        if section == 'gerrit' and option == 'bot_name':
            return True
        return False
    
    def get(self, section: str, option: str) -> str:
        """設定値取得のモック"""
        if section == 'gerrit' and option == 'bot_name':
            return ', '.join(self.bot_names)
        return ''


def create_learning_event_scenario():
    """
    学習イベントのシナリオデータを作成
    複数のChangeとそれらのレビュータイミングを含む複雑なシナリオ
    
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: (changes_df, expected_events)
    """
    base_time = datetime(2022, 1, 1, 10, 0, 0)
    
    # Change 1: 早期にレビューされる
    change1_messages = [
        create_mock_message('bot_jenkins', base_time + timedelta(minutes=30)),
        create_mock_message('reviewer1', base_time + timedelta(hours=2)),
        create_mock_message('reviewer2', base_time + timedelta(hours=4))
    ]
    
    # Change 2: 後でレビューされる
    change2_messages = [
        create_mock_message('reviewer3', base_time + timedelta(hours=6)),
        create_mock_message('ci_bot', base_time + timedelta(hours=8))
    ]
    
    # Change 3: レビューされない
    change3_messages = [
        create_mock_message('zuul_bot', base_time + timedelta(hours=1))
    ]
    
    changes = [
        create_mock_change_data(1001, base_time, change1_messages),
        create_mock_change_data(1002, base_time + timedelta(minutes=30), change2_messages),
        create_mock_change_data(1003, base_time + timedelta(hours=1), change3_messages)
    ]
    
    changes_df = pd.DataFrame(changes)
    
    # 期待される学習イベント（bot以外のメッセージのみ）
    expected_events = [
        {
            'change_id': '1001',
            'author': 'reviewer1',
            'timestamp': base_time + timedelta(hours=2)
        },
        {
            'change_id': '1001', 
            'author': 'reviewer2',
            'timestamp': base_time + timedelta(hours=4)
        },
        {
            'change_id': '1002',
            'author': 'reviewer3',
            'timestamp': base_time + timedelta(hours=6)
        }
    ]
    
    return changes_df, expected_events


def assert_valid_irl_training_stats(training_stats: Dict[str, Any]):
    """
    IRL学習統計が有効かどうかを検証
    
    Args:
        training_stats (Dict[str, Any]): 学習統計
    """
    required_keys = ['converged', 'final_objective', 'iterations']
    
    for key in required_keys:
        assert key in training_stats, f"学習統計に必要なキー '{key}' が見つかりません"
    
    assert isinstance(training_stats['converged'], bool)
    assert isinstance(training_stats['final_objective'], (int, float))
    assert isinstance(training_stats['iterations'], int)


def assert_valid_priority_weights(priorities: Dict[str, float], num_changes: int):
    """
    優先度重みが有効かどうかを検証
    
    Args:
        priorities (Dict[str, float]): 優先度重みの辞書
        num_changes (int): 期待されるChange数
    """
    assert len(priorities) == num_changes, f"期待されるChange数: {num_changes}, 実際: {len(priorities)}"
    
    # 全ての重みが0-1の範囲内にあることを確認
    for change_id, weight in priorities.items():
        assert 0 <= weight <= 1, f"Change {change_id} の重みが範囲外: {weight}"
        assert np.isfinite(weight), f"Change {change_id} の重みが無限値またはNaN: {weight}"
    
    # 重みがユニークであることを確認（順位が正しく計算されている）
    weights = list(priorities.values())
    assert len(set(weights)) == len(weights), "重みに重複があります"
    
    # 最高重みが1.0に近いことを確認（正規化が正しく行われている）
    max_weight = max(priorities.values())
    expected_max = num_changes / num_changes  # = 1.0
    assert abs(max_weight - expected_max) < 1e-10, f"最大重みが期待値と異なります: {max_weight} != {expected_max}"


if __name__ == '__main__':
    # ヘルパー関数のテスト実行例
    print("=== テストデータ生成例 ===")
    
    # Changesデータフレームの生成
    changes_df = create_mock_changes_dataframe(5)
    print(f"Changes データフレーム: {len(changes_df)} 件")
    print(changes_df[['change_number', 'component', 'created']].head())
    
    # 特徴量データフレームの生成
    features_df = create_mock_features_dataframe([1001, 1002, 1003])
    print(f"\n特徴量データフレーム: {len(features_df)} 件")
    print(features_df[['change_number', 'priority_weight', 'lines_added']].head())
    
    # 学習イベントシナリオの生成
    scenario_changes, expected_events = create_learning_event_scenario()
    print(f"\n学習イベントシナリオ: {len(scenario_changes)} Changes, {len(expected_events)} Events")
    for event in expected_events:
        print(f"  Event: Change {event['change_id']} by {event['author']} at {event['timestamp']}")
