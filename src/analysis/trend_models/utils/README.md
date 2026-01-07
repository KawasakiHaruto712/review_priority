# Utils - ユーティリティモジュール

定数定義とデータ読み込み機能を提供します。

## モジュール構成

| ファイル | 説明 |
|---------|------|
| `constants.py` | 定数定義（モデルパラメータ、特徴量名、期間設定等） |
| `data_loader.py` | データ読み込み・期間フィルタリング |

## constants.py

### 主要な定数

#### TREND_MODEL_CONFIG
分析対象プロジェクトとリリースバージョンの設定。

```python
TREND_MODEL_CONFIG = {
    'project': {
        'nova': ['2015.1.0', '12.0.0', '13.0.0', ...]
    }
}
```

#### ANALYSIS_PERIODS
分析期間の定義。

| 期間タイプ | 基準日 | 開始オフセット | 終了オフセット |
|-----------|--------|--------------|--------------|
| `early` | 現リリース日 | 0日 | +30日 |
| `late` | 次リリース日 | -30日 | 0日 |
| `all` | 現リリース日 | 0日 | 次リリース日 |

#### MODEL_PARAMS
各モデルのデフォルトハイパーパラメータ。

```python
MODEL_PARAMS = {
    'random_forest': {'n_estimators': 100, 'max_depth': 10, ...},
    'gradient_boosting': {...},
    'logistic_regression': {...},
    'svm': {...},
    'tabnet': {...},
    'ft_transformer': {...}
}
```

#### FEATURE_NAMES
使用する16種類の特徴量名リスト。

#### MODEL_TYPES
使用するモデルタイプのリスト。コメントアウトで無効化可能。

```python
MODEL_TYPES = [
    'random_forest',
    'gradient_boosting',
    'logistic_regression',
    'svm',
    'tabnet',
    'ft_transformer',
]
```

## data_loader.py

### 主要な関数

#### load_major_releases_summary
メジャーリリースサマリーCSVを読み込む。

```python
df = load_major_releases_summary(data_dir)
# Returns: DataFrame with columns [component, version, release_date]
```

#### load_all_changes
指定プロジェクトの全Changeデータを読み込む。

```python
changes = load_all_changes('nova', data_dir, use_collected=True)
# Returns: List[Dict] - Changeデータのリスト
```

#### load_core_developers
コア開発者情報を読み込む。

```python
core_devs = load_core_developers('nova', data_dir)
# Returns: Dict[project_name, List[member_info]]
```

#### load_bot_names_from_config
ボット名のリストを読み込む。

```python
bot_names = load_bot_names_from_config()
# Returns: List[str]
```

#### get_period_dates
期間タイプに応じた開始日・終了日を計算。

```python
start, end = get_period_dates(current_date, next_date, 'early')
# Returns: Tuple[datetime, datetime]
```

#### filter_changes_by_period
期間内にオープンだったChangeをフィルタリング。

```python
filtered = filter_changes_by_period(changes, period_start, period_end, next_release_date)
# Returns: List[Dict]
```

#### get_reviewers_in_period
期間内にレビューしたレビューアのリストを取得。

```python
reviewers = get_reviewers_in_period(change, period_start, period_end, bot_names)
# Returns: List[str] - レビューアのメールアドレスまたはユーザー名
```

#### changes_to_dataframe
ChangeリストをDataFrameに変換。

```python
df = changes_to_dataframe(changes)
# Returns: DataFrame
```

#### get_release_pairs
連続するリリースペアを取得。

```python
pairs = get_release_pairs(releases_df, 'nova', target_releases)
# Returns: List[Tuple[Dict, Dict]] - (現在のリリース, 次のリリース)
```
