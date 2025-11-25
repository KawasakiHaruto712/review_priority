# Features モジュール

コードレビューの優先順位付けに使用する特徴量の計算機能を提供するモジュールです。各種メトリクスを計算し、機械学習モデルの入力として使用できる形式で出力します。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `bug_metrics.py` | バグ修正関連の特徴量計算 |
| `change_metrics.py` | コード変更関連の特徴量計算 |
| `developer_metrics.py` | 開発者関連の特徴量計算 |
| `project_metrics.py` | プロジェクト関連の特徴量計算 |
| `refactoring_metrics.py` | リファクタリング関連の特徴量計算 |
| `review_metrics.py` | レビュー関連の特徴量計算 |

## 🔧 特徴量カテゴリ

### 🐛 バグメトリクス (`bug_metrics.py`)
- **`calculate_bug_fix_confidence`**: バグ修正の確信度（0-2スコア）
  - タイトル・説明文からバグ修正パターンを検出
  - バグトラッキングシステムのID参照を検出

### 📊 変更メトリクス (`change_metrics.py`)
- **`calculate_lines_added`**: 追加行数
- **`calculate_lines_deleted`**: 削除行数
- **`calculate_files_changed`**: 変更ファイル数
- **`calculate_elapsed_time`**: 作成からの経過時間（時間）
- **`calculate_revision_count`**: リビジョン数
- **`check_test_code_presence`**: テストコードの存在確認

### 👥 開発者メトリクス (`developer_metrics.py`)
- **`calculate_past_report_count`**: 過去のレポート数
- **`calculate_recent_report_count`**: 最近のレポート数
- **`calculate_merge_rate`**: 全体マージ率
- **`calculate_recent_merge_rate`**: 最近のマージ率

### 🏗️ プロジェクトメトリクス (`project_metrics.py`)
- **`calculate_days_to_major_release`**: メジャーリリースまでの日数
- **`calculate_predictive_target_ticket_count`**: 予測対象チケット数
- **`calculate_reviewed_lines_in_period`**: 期間内レビュー行数
- **`add_lines_info_to_dataframe`**: 行数情報の追加

### 🔄 リファクタリングメトリクス (`refactoring_metrics.py`)
- **`calculate_refactoring_confidence`**: リファクタリング確信度
  - コード整理・構造改善パターンの検出

### 📝 レビューメトリクス (`review_metrics.py`)
- **`calculate_uncompleted_requests`**: 未完了リクエスト数
  - レビューコメントの未対応数を計算

## 📊 特徴量ベクトル

全ての特徴量を組み合わせて、16次元の特徴量ベクトルを生成：

```python
feature_vector = [
    bug_fix_confidence,      # バグ修正確信度 (0-2)
    lines_added,             # 追加行数
    lines_deleted,           # 削除行数  
    files_changed,           # 変更ファイル数
    elapsed_time,            # 経過時間（時間）
    revision_count,          # リビジョン数
    test_code_presence,      # テストコード存在 (0/1)
    past_report_count,       # 過去レポート数
    recent_report_count,     # 最近レポート数
    merge_rate,              # マージ率 (0-1)
    recent_merge_rate,       # 最近マージ率 (0-1)
    days_to_major_release,   # リリースまで日数
    open_ticket_count,       # オープンチケット数
    reviewed_lines_in_period, # 期間内レビュー行数
    refactoring_confidence,  # リファクタリング確信度 (0-1)
    uncompleted_requests     # 未完了リクエスト数
]
```

## 🚀 使用方法

### 単一特徴量の計算

```python
from src.features.bug_metrics import calculate_bug_fix_confidence
from src.features.change_metrics import calculate_lines_added

# バグ修正確信度の計算
title = "Fix memory leak in compute service"
description = "Closes-Bug: #1234567"
bug_confidence = calculate_bug_fix_confidence(title, description)

# 追加行数の計算
lines_added = calculate_lines_added(change_data)
```

### 全特徴量の一括計算

```python
from src.learning.irl_models import ReviewPriorityDataProcessor

processor = ReviewPriorityDataProcessor()
changes_df = processor.load_openstack_data()[0]

# 全特徴量を含むDataFrameを生成
features_df = processor.extract_features(
    changes_df, 
    analysis_time=datetime.now()
)
```

## 📈 特徴量の意味と解釈

### 高優先度を示す特徴量
- **高いバグ修正確信度**: セキュリティ・安定性に影響
- **多い削除行数**: 技術的負債の解消
- **テストコード存在**: 品質保証済み
- **高いマージ率**: 承認されやすい変更
- **リリース直前**: 緊急性が高い

### 低優先度を示す特徴量
- **多いオープンチケット**: リソースが分散
- **高いリファクタリング確信度**: 機能追加より優先度低
- **多い未完了リクエスト**: 対応が困難

## 🎯 特徴量エンジニアリング

### 正規化
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
```

### 欠損値処理
```python
# 数値特徴量のデフォルト値
default_values = {
    'lines_added': 0,
    'lines_deleted': 0,
    'files_changed': 1,
    'elapsed_time': 0.0,
    'merge_rate': 0.5
}
```

## 📊 統計情報の例

```
特徴量統計:
├── bug_fix_confidence: 平均=0.3, 標準偏差=0.6
├── lines_added: 平均=45.2, 標準偏差=89.1  
├── lines_deleted: 平均=23.1, 標準偏差=67.3
├── files_changed: 平均=3.2, 標準偏差=5.1
├── merge_rate: 平均=0.78, 標準偏差=0.25
└── days_to_major_release: 平均=120.5, 標準偏差=98.7
```

## ⚡ パフォーマンス最適化

1. **ベクトル化**: NumPy/Pandasを使用した高速計算
2. **キャッシュ**: 重い計算結果のメモリ保存
3. **並列処理**: 複数特徴量の同時計算
4. **増分更新**: 新規データのみの特徴量計算

## ⚠️ 注意事項

1. **データ品質**: 欠損値・異常値の適切な処理
2. **スケール**: 特徴量間のスケール差に注意
3. **相関**: 高相関特徴量の冗長性
4. **時系列**: 時間依存する特徴量の考慮

## 🔧 拡張方法

新しい特徴量の追加：

```python
def calculate_new_metric(change_data):
    """新しいメトリクスの計算"""
    # 計算ロジックを実装
    return metric_value

# 特徴量ベクトルに追加
feature_columns.append('new_metric')
```
