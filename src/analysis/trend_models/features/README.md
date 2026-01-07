# Features - 特徴量抽出・前処理モジュール

Changeデータから特徴量を抽出し、前処理を行う機能を提供します。

## モジュール構成

| ファイル | 説明 |
|---------|------|
| `extractor.py` | 16種類のメトリクスを抽出 |
| `preprocessor.py` | ラベル付け・正規化・欠損値処理 |

## extractor.py

### FeatureExtractor

`src/features`モジュールの関数を使用して、Changeデータから16種類のメトリクスを計算します。

#### 使用例

```python
from src.analysis.trend_models.features.extractor import FeatureExtractor, extract_features_from_changes

# 単一のChangeから特徴量を抽出
extractor = FeatureExtractor(
    all_prs_df=all_prs_df,
    releases_df=releases_df,
    project_name='nova'
)
features = extractor.extract(change_data, period_start)

# 複数のChangeから特徴量を抽出してDataFrameを作成
features_df = extract_features_from_changes(
    changes=changes,
    all_prs_df=all_prs_df,
    releases_df=releases_df,
    project_name='nova',
    period_start=period_start
)
```

#### 抽出する特徴量

| カテゴリ | 特徴量 | 計算元 |
|---------|--------|-------|
| Bug Metrics | `bug_fix_confidence` | `src.features.bug_metrics` |
| Change Metrics | `lines_added`, `lines_deleted`, `files_changed`, `elapsed_time`, `revision_count`, `test_code_presence` | `src.features.change_metrics` |
| Developer Metrics | `past_report_count`, `recent_report_count`, `merge_rate`, `recent_merge_rate` | `src.features.developer_metrics` |
| Project Metrics | `days_to_major_release`, `open_ticket_count`, `reviewed_lines_in_period` | `src.features.project_metrics` |
| Refactoring Metrics | `refactoring_confidence` | `src.features.refactoring_metrics` |
| Review Metrics | `uncompleted_requests` | `src.features.review_metrics` |

## preprocessor.py

### Preprocessor

特徴量データの前処理を行うクラス。

#### 使用例

```python
from src.analysis.trend_models.features.preprocessor import Preprocessor

preprocessor = Preprocessor(
    feature_names=FEATURE_NAMES,
    bot_names=bot_names,
    core_developers=core_developers
)

# ラベル付け（レビュー済み/未レビュー）
features_df = preprocessor.add_labels(features_df, changes, period_start, period_end)

# 開発者タイプを追加
features_df = preprocessor.add_developer_type(features_df, changes, 'nova', period_start, period_end)

# 欠損値処理
features_df = preprocessor.handle_missing_values(features_df, strategy='zero')

# 正規化
features_df = preprocessor.normalize(features_df, fit=True)

# 特徴量行列とラベル配列を取得
X, y = preprocessor.get_feature_matrix(features_df)
```

#### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `add_labels(features_df, changes, period_start, period_end)` | レビュー済み/未レビューのラベルを付与 |
| `add_developer_type(features_df, changes, project_name, period_start, period_end)` | レビューアの開発者タイプ（Core/Non-Core）を追加 |
| `handle_missing_values(features_df, strategy)` | 欠損値を処理（zero/mean/median/drop） |
| `normalize(features_df, fit)` | 特徴量を標準化 |
| `get_feature_matrix(features_df)` | 特徴量行列Xとラベル配列yを取得 |
| `split_by_developer_type(features_df)` | 開発者タイプでデータを分割 |

#### ラベル付けのロジック

1. 期間内のメッセージを走査
2. ボット・SERVICE_USER・Change作成者・自動生成メッセージを除外
3. 残ったメッセージの著者をレビューアとしてカウント
4. レビューアが1人以上いれば `reviewed=1`、いなければ `reviewed=0`

#### 開発者タイプの判定

1. コア開発者のメールアドレスリストを取得
2. 期間中のレビューアにコア開発者が含まれれば `developer_type='core'`
3. 含まれなければ `developer_type='non-core'`
