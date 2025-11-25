# Utils モジュール

## 概要

`utils`モジュールは、trend_metrics分析全体で使用される共通のユーティリティ機能を提供します。データ読み込み、コアレビューア判定、定数定義などの基盤機能を含みます。

---

## モジュール構成

### 1. `data_loader.py` - データ読み込み

外部データソース（CSV、JSON、設定ファイル）からデータを読み込みます。

### 2. `core_reviewer_checker.py` - コアレビューア判定

レビューアがコアレビューアかどうかを判定します。

### 3. `constants.py` - 定数定義

分析設定、メトリクス定義、パス設定などの定数を管理します。

---

## data_loader.py

### 主要な関数

#### `load_major_releases_summary()`

メジャーリリース情報をCSVから読み込みます。

```python
releases_df = load_major_releases_summary()
# または
releases_df = load_major_releases_summary(csv_path=Path('custom/path/releases.csv'))
```

**パラメータ:**
- `csv_path` (Optional[Path]): CSVファイルのパス。省略時はデフォルトパス使用

**戻り値:**
- `pd.DataFrame`: メジャーリリース情報
  - `component` (または `project`): プロジェクト名
  - `version`: バージョン番号
  - `release_date`: リリース日（pd.Timestamp）

**データ例:**
```
component  version   release_date
nova       20.0.0    2019-10-15
nova       21.0.0    2020-05-11
neutron    15.0.0    2019-10-15
```

**デフォルトパス:** `data/openstack/major_releases_summary.csv`

---

#### `get_release_date()`

指定プロジェクト・バージョンのリリース日を取得します。

```python
release_date = get_release_date(
    releases_df=releases_df,
    project='nova',
    version='20.0.0'
)
# => pd.Timestamp('2019-10-15 00:00:00')
```

**パラメータ:**
- `releases_df`: リリース情報のDataFrame
- `project`: プロジェクト名（例: `'nova'`）
- `version`: バージョン番号（例: `'20.0.0'`）

**戻り値:**
- `pd.Timestamp`: リリース日

**例外:**
- `ValueError`: 該当するリリースが見つからない場合

**互換性:**
- CSVの列名が`component`または`project`のどちらでも対応

---

#### `load_core_developers()`

コアレビューア情報をJSONから読み込みます。

```python
core_data = load_core_developers()
# または
core_data = load_core_developers(json_path=Path('custom/path/core_devs.json'))
```

**パラメータ:**
- `json_path` (Optional[Path]): JSONファイルのパス。省略時はデフォルトパス使用

**戻り値:**
- `Dict`: プロジェクトごとのコアレビューア情報

**データ構造:**
```json
{
  "project": {
    "nova": {
      "members": [
        {
          "name": "Dan Smith",
          "email": "dms@danplanet.com"
        },
        ...
      ]
    },
    "neutron": {
      "members": [...]
    }
  }
}
```

**重要な仕様:**
- 実データの構造をそのまま返す（`'project'`キーを含む完全な構造）
- `core_reviewer_checker.py`はこの構造を前提に動作

**デフォルトパス:** `data/openstack_collected/core_developers.json`

---

#### `load_all_changes()`

指定ディレクトリまたはプロジェクトから全Changeデータを読み込みます。

```python
# プロジェクト名から読み込み
changes = load_all_changes(project='nova')

# ディレクトリパスから読み込み（優先）
changes = load_all_changes(changes_dir=Path('data/custom/changes'))
```

**パラメータ:**
- `project` (Optional[str]): プロジェクト名。`changes_dir`未指定時に使用
- `changes_dir` (Optional[Path]): Changeデータディレクトリ。指定時は優先

**戻り値:**
- `List[Dict]`: Changeデータのリスト

**動作:**
1. ディレクトリ内の全`*.json`ファイルを探索
2. 各JSONファイルを読み込み
   - リスト形式の場合: 展開して追加
   - 辞書形式の場合: そのまま追加
3. 読み込みエラーは警告ログを出力して継続

**デフォルトパス:** `data/openstack_collected/{project}/changes`

**例外:**
- `ValueError`: `project`と`changes_dir`の両方が未指定
- `FileNotFoundError`: 指定ディレクトリが存在しない

---

#### `load_bot_names_from_config()`

設定ファイルからbot名のリストを読み込みます。

```python
bot_names = load_bot_names_from_config()
# => ['jenkins', 'zuul', 'elastic-recheck', ...]
```

**パラメータ:**
- `config_path` (Optional[Path]): 設定ファイルのパス。省略時はデフォルトパス使用

**戻り値:**
- `List[str]`: bot名のリスト（小文字変換済み）

**設定ファイル形式:**
```ini
[organization]
bots = Jenkins, Zuul, Elastic Recheck, OpenStack Proposal Bot, ...
```

**重要な仕様:**
- `[organization]`セクションの`bots`キーから読み込み
- カンマ区切り形式をパース
- 各bot名は小文字に変換（`is_bot()`関数での判定用）
- ファイルが存在しない場合やエラー時は空リストを返す（例外を投げない）

**デフォルトパス:** `src/config/gerrymanderconfig.ini`

---

#### `is_bot()`

著者がbotかどうかを判定します。

```python
is_bot_result = is_bot(
    author_name='Jenkins',
    bot_names=['jenkins', 'zuul']
)
# => True
```

**パラメータ:**
- `author_name`: 著者名
- `bot_names`: bot名のリスト（小文字）

**戻り値:**
- `bool`: botの場合`True`

**判定ロジック:**
- `author_name`を小文字に変換
- `bot_names`のいずれかが`author_name`に含まれるかチェック（部分一致）
- 例: `'OpenStack Jenkins'` は `'jenkins'` にマッチ

**特殊ケース:**
- `author_name`が空の場合: `False`
- `bot_names`が空の場合: `False`

---

## core_reviewer_checker.py

### 主要な関数

#### `is_core_reviewer()`

レビューアがコアレビューアかどうかを判定します。

```python
is_core = is_core_reviewer(
    reviewer_email='dms@danplanet.com',
    project_name='nova',
    core_reviewers_data=core_data
)
# => True
```

**パラメータ:**
- `reviewer_email`: レビューアのメールアドレス
- `project_name`: 分析対象プロジェクト名（例: `'nova'`）
- `core_reviewers_data`: `load_core_developers()`から取得した全データ

**戻り値:**
- `bool`: コアレビューアの場合`True`

**判定ロジック:**
1. `core_reviewers_data['project'][project_name]['members']`を取得
2. 各メンバーの`email`フィールドを抽出
3. `reviewer_email`がリストに含まれるかチェック（完全一致）

**重要な仕様:**
- **プロジェクト固有**: 分析対象プロジェクトのコアレビューアリストのみ参照
- **メールアドレス判定**: 名前ではなくメールアドレスで判定
- **他プロジェクト除外**: 他プロジェクトのコアレビューアは`False`

**データ構造:**
```python
core_reviewers_data = {
    'project': {
        'nova': {
            'members': [
                {'name': 'Dan Smith', 'email': 'dms@danplanet.com'},
                ...
            ]
        }
    }
}
```

**特殊ケース:**
- `reviewer_email`が空の場合: `False`
- プロジェクトが存在しない場合: `False`

---

#### `get_project_core_reviewers()`

指定プロジェクトのコアレビューアメールアドレスリストを取得します。

```python
core_emails = get_project_core_reviewers(
    core_reviewers_data=core_data,
    project_name='nova'
)
# => ['dms@danplanet.com', 'sean.mcginnis@gmail.com', ...]
```

**パラメータ:**
- `core_reviewers_data`: `load_core_developers()`から取得した全データ
- `project_name`: プロジェクト名

**戻り値:**
- `List[str]`: コアレビューアメールアドレスのリスト

**動作:**
1. `core_reviewers_data['project'][project_name]['members']`を取得
2. 各メンバーの`email`フィールドを抽出
3. メールアドレスのリストを返す

**ログ出力:**
```
INFO - novaのコアレビューア: 8 名
```

---

## constants.py

### 主要な定数

#### `TREND_ANALYSIS_CONFIG`

分析対象プロジェクトとリリースの設定。

```python
TREND_ANALYSIS_CONFIG = {
    'project': 'nova',
    'target_releases': ['20.0.0', '21.0.0'],  # [current_release, next_release]
}
```

---

#### `ANALYSIS_PERIODS`

分析期間の定義（前期・後期）。

```python
ANALYSIS_PERIODS = {
    'early': {
        'base_date': 'current_release',  # 基準日：現在のリリース日
        'offset_start': 0,                # リリース日当日から
        'offset_end': 30,                 # 30日後まで
        'description': 'リリース直後30日間'
    },
    'late': {
        'base_date': 'next_release',      # 基準日：次のリリース日
        'offset_start': -30,              # 30日前から
        'offset_end': 0,                  # リリース日当日まで
        'description': '次リリース直前30日間'
    }
}
```

**計算例:**
- Current Release: 2019-10-15
- Next Release: 2020-05-11
- Early Period: 2019-10-15 ~ 2019-11-14（30日間）
- Late Period: 2020-04-11 ~ 2020-05-11（30日間）

---

#### `REVIEWER_TYPES`

レビューアタイプの定義。

```python
REVIEWER_TYPES = {
    'core_reviewed': {
        'label': 'core_reviewed',
        'description': 'コアレビューアがレビューした',
        'condition': 'has_core_review == True'
    },
    'non_core_reviewed': {
        'label': 'non_core_reviewed',
        'description': '非コアレビューアのみがレビューした',
        'condition': 'has_core_review == False and has_non_core_review == True'
    },
    # ...
}
```

---

#### `ANALYSIS_GROUPS`

分析グループのリスト（期間 × レビューアタイプ）。

```python
ANALYSIS_GROUPS = [
    'early_core_reviewed',
    'early_core_not_reviewed',
    'early_non_core_reviewed',
    'early_non_core_not_reviewed',
    'late_core_reviewed',
    'late_core_not_reviewed',
    'late_non_core_reviewed',
    'late_non_core_not_reviewed'
]
```

**注意:** 現在の実装では上記8グループすべて（期間 × core/non-core × reviewed/not_reviewed）を使用。

---

#### `METRIC_COLUMNS`

分析対象メトリクスのリスト。

```python
METRIC_COLUMNS = [
    'bug_fix_confidence',
    'lines_added',
    'lines_deleted',
    'files_changed',
    'elapsed_time',
    'revision_count',
    'test_code_presence',
    'past_report_count',
    'recent_report_count',
    'merge_rate',
    'recent_merge_rate',
    'days_to_major_release',
    'open_ticket_count',
    'reviewed_lines_in_period',
    'refactoring_confidence',
    'uncompleted_requests'
]
```

---

#### `METRIC_DATA_SCOPE`

メトリクスごとのデータ範囲設定。

```python
METRIC_DATA_SCOPE = {
    'bug_fix_confidence': 'period_only',      # 期間内のみ
    'past_report_count': 'all_data',          # 全期間（累積実績）
    'recent_report_count': 'recent_data',     # 直近90日
    # ...
}
```

**データ範囲オプション:**
- `'period_only'`: 分析期間内のChangeデータのみ
- `'all_data'`: 収集した全Changeデータ（開発者の累積実績評価用）
- `'recent_data'`: 期間開始前90日 + 期間内のデータ

---

#### `RECENT_DATA_PERIOD_DAYS`

`recent_data`で使用する期間（日数）。

```python
RECENT_DATA_PERIOD_DAYS = 90  # 3ヶ月
```

---

#### `METRIC_DISPLAY_NAMES`

メトリクスの表示名。

```python
METRIC_DISPLAY_NAMES = {
    'bug_fix_confidence': 'Bug Fix Confidence',
    'lines_added': 'Lines Added',
    # ...
}
```

---

#### `OUTPUT_DIR_BASE`

出力ディレクトリのベースパス。

```python
OUTPUT_DIR_BASE = 'data/analysis/trend_metrics'
```

---

#### `STATISTICAL_TEST_CONFIG`

統計検定の設定。

```python
STATISTICAL_TEST_CONFIG = {
    'alpha': 0.05,                    # 有意水準
    'use_bonferroni': True,           # Bonferroni補正を使用するか
    'test_method': 'mann_whitney_u',  # 使用する検定手法
    'effect_size_method': 'cohen_d'   # 効果量の計算方法
}
```

---

#### `DATA_LOAD_CONFIG`

データ読み込み設定（各種ファイルパス）。

```python
DATA_LOAD_CONFIG = {
    'major_releases_file': 'data/openstack/major_releases_summary.csv',
    'core_developers_file': 'data/openstack_collected/core_developers.json',
    'changes_dir_template': 'data/openstack_collected/{project}/changes',
    'bot_config_file': 'src/config/gerrymanderconfig.ini'
}
```

---

#### `VISUALIZATION_CONFIG`

可視化設定。

```python
VISUALIZATION_CONFIG = {
    'figure_size': (16, 12),
    'dpi': 300,
    'color_palette': 'Set2',
    'boxplot_grid_size': (2, 4),  # 8グループを2行×4列で表示
    'save_format': 'pdf'
}
```

---

## 使用例

### 基本的な使用方法

```python
from src.analysis.trend_metrics.utils.data_loader import (
    load_major_releases_summary,
    get_release_date,
    load_core_developers,
    load_all_changes,
    load_bot_names_from_config,
    is_bot
)
from src.analysis.trend_metrics.utils.core_reviewer_checker import (
    is_core_reviewer,
    get_project_core_reviewers
)
from src.analysis.trend_metrics.utils.constants import (
    TREND_ANALYSIS_CONFIG,
    ANALYSIS_PERIODS,
    DATA_LOAD_CONFIG
)

# 1. リリース情報を読み込み
releases_df = load_major_releases_summary()

# 2. リリース日を取得
current_release_date = get_release_date(
    releases_df, 
    project='nova', 
    version='20.0.0'
)
next_release_date = get_release_date(
    releases_df, 
    project='nova', 
    version='21.0.0'
)

# 3. コアレビューア情報を読み込み
core_data = load_core_developers()

# 4. bot名を読み込み
bot_names = load_bot_names_from_config()

# 5. 全Changeデータを読み込み
all_changes = load_all_changes(project='nova')

# 6. コアレビューアリストを取得
core_emails = get_project_core_reviewers(core_data, 'nova')

# 7. レビューアがコアかチェック
is_core = is_core_reviewer(
    reviewer_email='dms@danplanet.com',
    project_name='nova',
    core_reviewers_data=core_data
)

# 8. bot判定
if is_bot(author_name='Jenkins', bot_names=bot_names):
    print("This is a bot")
```

---

## データフロー

```
1. constants.py
   ├─ TREND_ANALYSIS_CONFIG: プロジェクト・リリース指定
   ├─ DATA_LOAD_CONFIG: ファイルパス設定
   └─ ANALYSIS_PERIODS: 期間定義
        ↓
2. data_loader.py
   ├─ load_major_releases_summary() → releases_df
   ├─ get_release_date() → release_date
   ├─ load_core_developers() → core_data
   ├─ load_all_changes() → all_changes
   └─ load_bot_names_from_config() → bot_names
        ↓
3. core_reviewer_checker.py
   ├─ get_project_core_reviewers() → core_emails
   └─ is_core_reviewer() → bool判定
        ↓
4. 他モジュール（metrics_extraction, etc）
```

---

## ファイル構造

```
utils/
├── __init__.py
├── constants.py              # 定数定義
├── data_loader.py            # データ読み込み
├── core_reviewer_checker.py  # コアレビューア判定
└── README.md                 # このファイル
```

---

## テスト

テストコードは`tests/release_impact/`にあります。

```bash
# data_loaderのテスト
pytest tests/release_impact/test_data_loader.py -v

# core_reviewer_checkerのテスト
pytest tests/release_impact/test_core_reviewer_checker.py -v
```

---

## 注意事項

### データ構造の互換性

#### リリース情報CSV

- 列名が`component`または`project`のどちらでも対応
- `get_release_date()`が自動判別

#### コアレビューア情報JSON

- 完全な構造を返す（`'project'`キーを含む）
- `core_reviewer_checker.py`はこの構造を前提
- 構造変更時は両モジュールを同時に更新する必要あり

### bot判定

- 部分一致で判定（例: `'OpenStack Jenkins'` → `'jenkins'`）
- bot名は小文字で管理
- 設定ファイルが存在しない場合もエラーを投げない（空リスト返却）

### コアレビューア判定

- **プロジェクト固有**: 他プロジェクトのコアレビューアは対象外
- **メールアドレス完全一致**: 名前ではなくメールアドレスで判定
- 大文字小文字は区別される（データソースに依存）

### エラーハンドリング

- `load_bot_names_from_config()`: エラー時も例外を投げず空リストを返す
- `load_all_changes()`: 個別ファイルの読み込みエラーは警告ログを出力して継続
- `get_release_date()`: リリースが見つからない場合は`ValueError`を投げる

---

## ログ出力

このモジュールは`logging`を使用してログを出力します。

**出力例:**
```
INFO - メジャーリリース情報を読み込みました: 101 件
INFO - リリース日を取得: nova 20.0.0 -> 2019-10-15 00:00:00
INFO - コアレビューア情報を読み込みました: 6 プロジェクト
INFO - Changeファイルを読み込み中: 26276 件
INFO - Changeデータを読み込みました: 26276 件
INFO - Bot名を読み込みました: 60 件
INFO - novaのコアレビューア: 8 名
```

---

## まとめ

`utils`モジュールは以下を提供します：

1. **データ読み込み**: CSV、JSON、設定ファイルからのデータ取得
2. **コアレビューア判定**: プロジェクト固有のコアレビューア判定ロジック
3. **定数管理**: 分析設定、メトリクス定義、パス設定の一元管理
4. **bot除外**: bot名設定と判定機能

これらの機能により、trend_metrics分析全体で一貫したデータアクセスと判定ロジックを提供します。
