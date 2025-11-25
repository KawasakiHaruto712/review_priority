# Metrics Extraction モジュール

## 概要

`metrics_extraction`モジュールは、OpenStackプロジェクトのリリース期間におけるChangeデータを抽出し、レビューアタイプ別に分類するための機能を提供します。

## 主な機能

### 1. 期間別データ抽出 (`period_extractor.py`)

リリース期間（前期・後期)におけるChangeデータを抽出し、レビューア情報を付与します。

#### 主要な関数

##### `calculate_periods()`

リリース日と期間設定から前期・後期の期間を計算します。

```python
early_period, late_period = calculate_periods(
    current_release_date=pd.Timestamp('2019-10-15'),
    next_release_date=pd.Timestamp('2020-05-11'),
    period_config=ANALYSIS_PERIODS
)
# 前期: 2019-10-15 ~ 2019-11-14 (リリース後30日)
# 後期: 2020-04-11 ~ 2020-05-11 (次リリース前30日)
```

**パラメータ:**
- `current_release_date`: 現在のリリース日
- `next_release_date`: 次のリリース日
- `period_config`: 期間設定（例: `{'early': {'offset_start': 0, 'offset_end': 30}, ...}`）

**戻り値:**
- `Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]`: 前期と後期の期間タプル

---

##### `extract_changes_in_period()`

指定期間内に**openだったChange**を抽出します。

```python
early_changes = extract_changes_in_period(
    all_changes=all_changes,
    period=(early_start, early_end),
    next_release_date=next_release_date
)
```

**抽出ロジック:**

期間内に一瞬でもopenだったChangeを対象とします：

1. **作成日フィルタ**: `created < period_end` (期間終了前に作成)
2. **履歴フィルタ**: `created >= next_release_date - 1 year` (次リリースから1年以内)
3. **オープン判定**: 以下のいずれか
   - `status == 'NEW'` (まだオープン)
   - `close_date > period_start` (期間開始後にクローズ)

**重要な仕様:**

- **期間判定**: "期間内に作成された"ではなく、**"期間内にopenだった"**
  - 期間前に作成され期間中にクローズされたChangeも含む
  - 期間中に作成され期間後にクローズされたChangeも含む

- **履歴フィルタ**: 次のリリースから1年以内に作成されたChangeのみを対象
  - 例: 2020-05-11リリースの分析では、2019-05-11以降に作成されたChangeのみ
  - 古い放置されたChange（例: 2013年作成、2021年放棄）を除外

**パラメータ:**
- `all_changes`: 全Changeデータのリスト
- `period`: 期間タプル `(start_datetime, end_datetime)`
- `next_release_date`: 次のリリース日（履歴フィルタ用）

**戻り値:**
- `List[Dict]`: 期間内にopenだったChangeのリスト

---

##### `extract_reviewers_from_messages()`

Changeのメッセージからレビューアのメールアドレスを抽出します（bot除外）。

```python
reviewer_emails = extract_reviewers_from_messages(
    messages=change['messages'],
    bot_names=bot_names,
    period=(early_start, early_end)
)
# 例: ['reviewer1@example.com', 'reviewer2@example.com']
```

**抽出ロジック:**

1. 各メッセージの`author.email`を取得
2. `author.name`がbotリストに含まれていないかチェック
3. 期間指定がある場合、期間内のメッセージのみ対象
4. 重複を除外してメールアドレスのリストを返す

**パラメータ:**
- `messages`: Changeのメッセージリスト
- `bot_names`: 除外するbot名のリスト
- `period`: 期間タプル（オプション。指定時は期間内のレビューのみ抽出）

**戻り値:**
- `List[str]`: レビューアのメールアドレスリスト（重複なし）

---

##### `add_reviewer_info_to_changes()`

各Changeにレビューア情報（`reviewers`キー）を追加します。

```python
changes_with_reviewers = add_reviewer_info_to_changes(
    changes=early_changes,
    bot_names=bot_names,
    period=(early_start, early_end)
)
# 各changeに 'reviewers': [...] が追加される
```

**パラメータ:**
- `changes`: Changeリスト
- `bot_names`: bot名のリスト
- `period`: 期間（この期間内のレビューのみを抽出）

**戻り値:**
- `List[Dict]`: レビューア情報付きChangeリスト

---

##### `get_changes_for_metric_calculation()`

メトリクス計算用のChangeデータを取得します（データ範囲設定に応じて）。

```python
metric_changes = get_changes_for_metric_calculation(
    all_changes=all_changes,
    period_changes=early_changes,
    metric_name='complexity',
    metric_data_scope=METRIC_DATA_SCOPE,
    period_start=early_start,
    recent_period_days=90
)
```

**データ範囲オプション:**
- `'period_only'`: 期間内のChangeのみ
- `'all_data'`: 全Changeデータ
- `'recent_data'`: 期間開始前90日 + 期間内のChange

**パラメータ:**
- `all_changes`: 全Changeデータ
- `period_changes`: 期間内のChangeデータ
- `metric_name`: メトリクス名
- `metric_data_scope`: メトリクスのデータ範囲設定
- `period_start`: 期間の開始日時
- `recent_period_days`: recent_dataで使用する日数（デフォルト: 90）

**戻り値:**
- `List[Dict]`: メトリクス計算用のChangeデータ

---

### 2. レビューアタイプ分類 (`reviewer_classifier.py`)

Changeをレビューアタイプ（コア/非コア/未レビュー）で分類します。

#### 主要な関数

##### `classify_by_reviewer_type()`

1つのChangeをレビューアタイプで分類します。

```python
reviewer_types = classify_by_reviewer_type(
    change=change,
    core_reviewers_data=core_data,
    project_name='nova'
)
# 例: ['core_reviewed', 'non_core_reviewed']
```

**分類ロジック:**

1. Changeの`reviewers`リスト（メールアドレス）を取得
2. 各レビューアがコアレビューアかチェック
3. 結果を返す:
   - コアレビューアあり → `'core_reviewed'`を含む
   - 非コアレビューアあり → `'non_core_reviewed'`を含む
   - レビューアなし → `['not_reviewed']`

**重要な仕様:**

- **複数タイプ対応**: コアと非コア両方がレビューした場合、両方のタイプを返す
  - 例: `['core_reviewed', 'non_core_reviewed']`
  - これにより、同じChangeが複数のグループに属することができる

- **メールアドレス判定**: レビューアの判定はメールアドレスで行う

- **bot除外済み**: `reviewers`リストは既にbot除外済み

**パラメータ:**
- `change`: Changeデータ（`reviewers`キーを持つ）
- `core_reviewers_data`: コアレビューア情報（全プロジェクト分）
- `project_name`: 分析対象プロジェクト名

**戻り値:**
- `List[str]`: レビューアタイプのリスト
  - `['core_reviewed']`: コアレビューアのみ
  - `['non_core_reviewed']`: 非コアレビューアのみ
  - `['core_reviewed', 'non_core_reviewed']`: 両方
  - `['not_reviewed']`: レビューなし

---

##### `classify_changes_into_groups()`

前期・後期のChangeを8グループに分類します。

```python
groups = classify_changes_into_groups(
    early_changes=early_changes,
    late_changes=late_changes,
    core_reviewers_data=core_data,
    project_name='nova'
)
```

**8つのグループ:**

| グループ名 | 説明 |
|-----------|------|
| `early_core_reviewed` | 前期、コアレビューアがレビュー |
| `early_core_not_reviewed` | 前期、コアレビューアがレビューしていない |
| `early_non_core_reviewed` | 前期、非コアレビューアがレビュー |
| `early_non_core_not_reviewed` | 前期、非コアレビューアがレビューしていない |
| `late_core_reviewed` | 後期、コアレビューアがレビュー |
| `late_core_not_reviewed` | 後期、コアレビューアがレビューしていない |
| `late_non_core_reviewed` | 後期、非コアレビューアがレビュー |
| `late_non_core_not_reviewed` | 後期、非コアレビューアがレビューしていない |

**重要な仕様:**

- **重複カウント**: 1つのChangeが複数のグループに属する可能性あり
  - 例: コアと非コアの両方がレビュー → `early_core_reviewed`と`early_non_core_reviewed`の両方
  - 合計件数が期間内のChange数を超えることがある

**パラメータ:**
- `early_changes`: 前期のChangeリスト（レビューア情報付き）
- `late_changes`: 後期のChangeリスト（レビューア情報付き）
- `core_reviewers_data`: コアレビューア情報
- `project_name`: プロジェクト名

**戻り値:**
- `Dict[str, List[Dict]]`: 8グループに分類されたChange

**出力例:**
```
==========================================================
Changeの分類結果:
  early_core_reviewed: 105 件
  early_core_not_reviewed: 180 件
  early_non_core_reviewed: 180 件
  early_non_core_not_reviewed: 105 件
  late_core_reviewed: 92 件
  late_core_not_reviewed: 132 件
  late_non_core_reviewed: 132 件
  late_non_core_not_reviewed: 92 件
==========================================================
```

---

## データフロー

```
1. リリース情報読み込み
   ↓
2. 期間計算 (calculate_periods)
   ↓
3. 全Changeデータ読み込み
   ↓
4. 期間内Change抽出 (extract_changes_in_period)
   - 前期Changeリスト
   - 後期Changeリスト
   ↓
5. レビューア情報付与 (add_reviewer_info_to_changes)
   - bot除外
   - 期間内のレビューのみ
   ↓
6. レビューアタイプ分類 (classify_changes_into_groups)
    - 8グループ（core/non-core × reviewed/not_reviewed）に分類
    - 重複カウントあり
   ↓
7. メトリクス計算（別モジュール）
```

---

## 使用例

### 基本的な使用方法

```python
from src.analysis.trend_metrics.metrics_extraction.period_extractor import (
    calculate_periods,
    extract_changes_in_period,
    add_reviewer_info_to_changes
)
from src.analysis.trend_metrics.metrics_extraction.reviewer_classifier import (
    classify_changes_into_groups
)

# 1. 期間計算
early_period, late_period = calculate_periods(
    current_release_date=pd.Timestamp('2019-10-15'),
    next_release_date=pd.Timestamp('2020-05-11'),
    period_config=ANALYSIS_PERIODS
)

# 2. 期間内Change抽出
early_changes = extract_changes_in_period(
    all_changes, 
    early_period, 
    next_release_date=pd.Timestamp('2020-05-11')
)
late_changes = extract_changes_in_period(
    all_changes, 
    late_period, 
    next_release_date=pd.Timestamp('2020-05-11')
)

# 3. レビューア情報付与
early_changes = add_reviewer_info_to_changes(early_changes, bot_names, early_period)
late_changes = add_reviewer_info_to_changes(late_changes, bot_names, late_period)

# 4. レビューアタイプ分類
groups = classify_changes_into_groups(
    early_changes, 
    late_changes, 
    core_reviewers_data, 
    'nova'
)

# 5. グループごとに処理
for group_name, changes in groups.items():
    print(f"{group_name}: {len(changes)} 件")
    # メトリクス計算など...
```

---

## テスト

テストコードは`tests/release_impact/test_period_extractor.py`と`tests/release_impact/test_reviewer_classifier.py`にあります。

```bash
# テスト実行
pytest tests/release_impact/test_period_extractor.py -v
pytest tests/release_impact/test_reviewer_classifier.py -v
```

---

## 注意事項

### 期間判定の仕様

- **"期間内にopenだった"を採用**: 期間内に作成されたChangeだけでなく、期間中にopenだったすべてのChangeが対象
- **理由**: リリース期間全体の開発活動を正確に捉えるため

### 履歴フィルタの必要性

- **1年フィルタ**: 次のリリースから1年以内に作成されたChangeのみを対象
- **理由**: 古い放置されたChange（数年前に作成され、分析期間に放棄されたもの）を除外
- **例**: 2013年作成のChangeが2021年に放棄された場合、2019-2020の分析には含めない

### 重複カウント

- **仕様**: 1つのChangeが複数のグループに属する可能性あり
- **理由**: コアレビューアと非コアレビューアの両方の活動を正確に捉えるため
- **影響**: 各グループの合計が期間内のChange総数を超える場合がある

### bot除外

- `extract_reviewers_from_messages()`でbot名のチェックを実施
- bot判定は`src/analysis/trend_metrics/utils/data_loader.py`の`is_bot()`関数を使用
- 設定ファイル: `gerrymanderconfig.ini`の`[organization]`セクション、`bots`キー

### コアレビューア判定

- メールアドレスベースで判定
- プロジェクト固有のコアレビューアリストを使用
- データソース: `data/openstack_collected/core_developers.json`

---

## 依存モジュール

- `src.analysis.trend_metrics.utils.data_loader`: データ読み込み、bot判定
- `src.analysis.trend_metrics.utils.core_reviewer_checker`: コアレビューア判定

---

## 設定

### 期間設定例

```python
ANALYSIS_PERIODS = {
    'early': {
        'offset_start': 0,    # リリース日から0日後
        'offset_end': 30      # リリース日から30日後
    },
    'late': {
        'offset_start': -30,  # 次リリース日から30日前
        'offset_end': 0       # 次リリース日
    }
}
```

### データ範囲設定例

```python
METRIC_DATA_SCOPE = {
    'complexity': 'period_only',    # 期間内のみ
    'review_speed': 'recent_data',  # 直近90日 + 期間内
    'developer_count': 'all_data'   # 全データ
}
```

---

## ログ出力

このモジュールは`logging`を使用してログを出力します。

**出力例:**
```
INFO - 前期: 2019-10-15 00:00:00 ~ 2019-11-14 00:00:00
INFO - 後期: 2020-04-11 00:00:00 ~ 2020-05-11 00:00:00
INFO - 期間内にopenだったChange: 285 件 (2019-10-15 00:00:00 ~ 2019-11-14 00:00:00)
INFO - 285 件のChangeにレビューア情報を追加しました
INFO - ==========================================================
INFO - Changeの分類結果:
INFO -   early_core_reviewed: 105 件
INFO -   early_core_not_reviewed: 180 件
INFO -   early_non_core_reviewed: 180 件
INFO -   early_non_core_not_reviewed: 105 件
INFO -   late_core_reviewed: 92 件
INFO -   late_core_not_reviewed: 132 件
INFO -   late_non_core_reviewed: 132 件
INFO -   late_non_core_not_reviewed: 92 件
INFO - ==========================================================
```

---

## まとめ

`metrics_extraction`モジュールは以下を提供します：

1. **期間別データ抽出**: リリース期間内にopenだったChangeの抽出
2. **レビューア情報付与**: bot除外、期間内レビューの抽出
3. **レビューアタイプ分類**: コア/非コア × reviewed/not_reviewed の8グループ分類
4. **柔軟なデータ範囲設定**: メトリクスごとに適切なデータ範囲を選択

これらの機能により、OpenStackプロジェクトのリリース期間におけるレビュー活動を詳細に分析できます。
