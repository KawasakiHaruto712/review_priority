# Analysis モジュール

## 概要

`analysis`モジュールは、グループ分類されたChangeデータに対して統計分析とトレンド比較を行うための機能を提供します。記述統計量の計算、統計検定の実行、前期/後期の変化分析などを行います。

---

## モジュール構成

### 1. `statistical_analyzer.py` - 統計分析

グループ別のメトリクスに対して記述統計量を計算し、統計検定を実行します。

### 2. `trend_comparator.py` - トレンド比較

前期/後期の変化を分析し、レビューアタイプ別の傾向を比較します。

---

## statistical_analyzer.py

### 主要な関数

#### `calculate_summary_statistics()`

各グループのメトリクスについて記述統計量を計算します。

```python
from src.analysis.trend_metrics.analysis.statistical_analyzer import (
    calculate_summary_statistics
)

summary_stats = calculate_summary_statistics(
    groups=classified_groups,
    metrics=['lines_added', 'bug_fix_confidence']
)
```

**パラメータ:**
- `groups`: グループ別のChangeデータ
  - 例: `{'early_core_reviewed': [...], 'late_core_reviewed': [...]}`
- `metrics`: 計算対象メトリクスのリスト（省略時は`METRIC_COLUMNS`使用）

**戻り値:**
- `Dict[str, Dict[str, Dict[str, float]]]`: グループ別・メトリクス別の統計量

**出力形式:**
```python
{
    'early_core_reviewed': {
        'lines_added': {
            'count': 105,
            'mean': 145.2,
            'std': 52.1,
            'min': 10,
            '25%': 80,
            '50%': 130,
            '75%': 190,
            'max': 500
        },
        'bug_fix_confidence': {...}
    },
    'late_core_reviewed': {...}
}
```

**計算される統計量:**
- `count`: サンプル数
- `mean`: 平均値
- `std`: 標準偏差
- `min`: 最小値
- `25%`: 第1四分位数
- `50%`: 中央値
- `75%`: 第3四分位数
- `max`: 最大値

---

#### `mann_whitney_u_test()`

Mann-Whitney U検定（ノンパラメトリック検定）を実行します。

```python
test_result = mann_whitney_u_test(
    group1_values=[100, 120, 150, 180],
    group2_values=[80, 90, 110, 130],
    alpha=0.05
)
```

**パラメータ:**
- `group1_values`: グループ1の値リスト
- `group2_values`: グループ2の値リスト
- `alpha`: 有意水準（デフォルト: 0.05）

**戻り値:**
```python
{
    'statistic': 8.0,           # U統計量
    'p_value': 0.032,           # p値
    'significant': True,        # 有意かどうか (p < alpha)
    'test_method': 'mann_whitney_u'
}
```

**使用場面:**
- 2グループ間の分布の違いを検定
- 正規分布を仮定しないノンパラメトリック検定
- サンプルサイズが小さい場合にも使用可能

---

#### `calculate_effect_size_cohens_d()`

Cohen's d（効果量）を計算します。

```python
cohens_d = calculate_effect_size_cohens_d(
    group1_values=[100, 120, 150, 180],
    group2_values=[80, 90, 110, 130]
)
# => 0.45
```

**パラメータ:**
- `group1_values`: グループ1の値リスト
- `group2_values`: グループ2の値リスト

**戻り値:**
- `float`: Cohen's d値

**効果量の解釈:**
- `|d| < 0.2`: negligible（無視できる）
- `0.2 ≤ |d| < 0.5`: small（小）
- `0.5 ≤ |d| < 0.8`: medium（中）
- `0.8 ≤ |d| < 1.2`: large（大）
- `1.2 ≤ |d|`: very large（非常に大）

---

#### `perform_pairwise_tests()`

全グループ間でペアワイズ検定を実行します。

```python
pairwise_results = perform_pairwise_tests(
    groups=classified_groups,
    metric='lines_added',
    test_config=STATISTICAL_TEST_CONFIG
)
```

**パラメータ:**
- `groups`: グループ別のChangeデータ
- `metric`: 検定対象メトリクス
- `test_config`: 検定設定（省略時は`STATISTICAL_TEST_CONFIG`使用）

**戻り値:**
```python
{
    'early_core_reviewed_vs_late_core_reviewed': {
        'p_value': 0.032,
        'significant': True,
        'effect_size': 0.45,
        'alpha': 0.00238,          # Bonferroni補正後
        'bonferroni_corrected': True,
        'group1_name': 'early_core_reviewed',
        'group2_name': 'late_core_reviewed',
        'group1_n': 105,
        'group2_n': 92
    },
    ...
}
```

**Bonferroni補正:**
- 多重比較を調整するための補正
- 調整後の有意水準: `α_adjusted = α / 比較回数`
- 例: 8グループで28回比較の場合、`α_adjusted = 0.05 / 28 ≈ 0.00179`

---

#### `perform_all_statistical_tests()`

全メトリクスについて統計検定を実行します。

```python
all_test_results = perform_all_statistical_tests(
    groups=classified_groups,
    metrics=METRIC_COLUMNS,
    test_config=STATISTICAL_TEST_CONFIG
)
```

**パラメータ:**
- `groups`: グループ別のChangeデータ
- `metrics`: 検定対象メトリクスのリスト（省略時は`METRIC_COLUMNS`使用）
- `test_config`: 検定設定

**戻り値:**
```python
{
    'lines_added': {
        'early_core_reviewed_vs_late_core_reviewed': {...},
        'early_core_reviewed_vs_early_non_core_reviewed': {...},
        ...
    },
    'bug_fix_confidence': {...},
    ...
}
```

---

#### `summarize_significant_results()`

有意な結果をサマリーします。

```python
significant_summary = summarize_significant_results(
    test_results=all_test_results,
    alpha=0.05
)
```

**パラメータ:**
- `test_results`: `perform_all_statistical_tests()`の結果
- `alpha`: 有意水準

**戻り値:**
```python
{
    'lines_added': [
        {
            'pair': 'early_core_reviewed_vs_late_core_reviewed',
            'p_value': 0.001,
            'effect_size': 0.65,
            'group1_name': 'early_core_reviewed',
            'group2_name': 'late_core_reviewed'
        },
        ...
    ],
    'bug_fix_confidence': [...]
}
```

**特徴:**
- 有意な結果のみを抽出
- p値でソート（小さい順）
- メトリクスごとにまとめ

---

#### `interpret_effect_size()`

Cohen's dの効果量を解釈します。

```python
interpretation = interpret_effect_size(0.65)
# => 'medium'
```

**パラメータ:**
- `cohens_d`: Cohen's d値

**戻り値:**
- `str`: 効果量の解釈
  - `'negligible'`: |d| < 0.2
  - `'small'`: 0.2 ≤ |d| < 0.5
  - `'medium'`: 0.5 ≤ |d| < 0.8
  - `'large'`: 0.8 ≤ |d| < 1.2
  - `'very large'`: 1.2 ≤ |d|

---

## trend_comparator.py

### 主要な関数

#### `compare_early_vs_late_by_reviewer_type()`

レビューアタイプ別に前期/後期を比較します。

```python
from src.analysis.trend_metrics.analysis.trend_comparator import (
    compare_early_vs_late_by_reviewer_type
)

comparison = compare_early_vs_late_by_reviewer_type(
    groups=classified_groups,
    metrics=['lines_added', 'bug_fix_confidence']
)
```

**パラメータ:**
- `groups`: グループ別のChangeデータ
- `metrics`: 比較対象メトリクスのリスト（省略時は`METRIC_COLUMNS`使用）

**戻り値:**
```python
{
    'core_reviewed': {
        'lines_added': {
            'early_mean': 145.2,
            'late_mean': 132.5,
            'change_absolute': -12.7,
            'change_percentage': -8.7,
            'direction': 'decrease',
            'early_n': 105,
            'late_n': 92
        },
        'bug_fix_confidence': {...}
    },
    'non_core_reviewed': {
        'lines_added': {
            'early_mean': 98.5,
            'late_mean': 115.3,
            'change_absolute': 16.8,
            'change_percentage': 17.0,
            'direction': 'increase',
            'early_n': 180,
            'late_n': 132
        },
        ...
    }
}
```

**変化の方向:**
- `'increase'`: 後期が前期より大きい
- `'decrease'`: 後期が前期より小さい
- `'no_change'`: ほぼ変化なし（差が1e-6未満）

---

#### `calculate_metric_change()`

前期→後期のメトリクス変化を計算します。

```python
change_info = calculate_metric_change(
    early_values=[100, 120, 150, 180],
    late_values=[80, 90, 110, 130]
)
```

**パラメータ:**
- `early_values`: 前期の値リスト
- `late_values`: 後期の値リスト

**戻り値:**
```python
{
    'early_mean': 137.5,
    'late_mean': 102.5,
    'change_absolute': -35.0,
    'change_percentage': -25.5,
    'direction': 'decrease',
    'early_n': 4,
    'late_n': 4
}
```

---

#### `identify_diverging_trends()`

レビューアタイプ間で傾向が異なるメトリクスを特定します。

```python
diverging = identify_diverging_trends(
    comparison_results=comparison,
    threshold_percentage=10.0
)
```

**パラメータ:**
- `comparison_results`: `compare_early_vs_late_by_reviewer_type()`の結果
- `threshold_percentage`: 変化率の閾値（%）

**戻り値:**
```python
{
    'lines_added': {
        'core_reviewed': {
            'change_percentage': -8.7,
            'direction': 'decrease'
        },
        'non_core_reviewed': {
            'change_percentage': 17.0,
            'direction': 'increase'
        },
        'divergence': 25.7,
        'interpretation': 'core_reviewedはdecrease（-8.7%）、non_core_reviewedはincrease（+17.0%）'
    },
    ...
}
```

**使用場面:**
- レビューアタイプ間で逆の傾向を示すメトリクスを発見
- 例: コアレビューアは減少、非コアレビューアは増加

---

#### `calculate_group_size_changes()`

グループサイズの変化を計算します（前期→後期）。

```python
size_changes = calculate_group_size_changes(groups=classified_groups)
```

**パラメータ:**
- `groups`: グループ別のChangeデータ

**戻り値:**
```python
{
    'core_reviewed': {
        'early_count': 105,
        'late_count': 92,
        'change_absolute': -13,
        'change_percentage': -12.4
    },
    'non_core_reviewed': {
        'early_count': 180,
        'late_count': 132,
        'change_absolute': -48,
        'change_percentage': -26.7
    }
}
```

**使用場面:**
- レビュー活動の変化を把握
- 例: 後期にコアレビューアの関与が減少したか

---

#### `generate_trend_summary()`

トレンド分析の総合サマリーを生成します。

```python
summary = generate_trend_summary(
    comparison_results=comparison,
    test_results=all_test_results,
    size_changes=size_changes
)
```

**パラメータ:**
- `comparison_results`: `compare_early_vs_late_by_reviewer_type()`の結果
- `test_results`: `perform_all_statistical_tests()`の結果（オプション）
- `size_changes`: `calculate_group_size_changes()`の結果（オプション）

**戻り値:**
```python
{
    'reviewer_type_comparison': {...},
    'group_size_changes': {...},
    'key_findings': [
        {
            'type': 'group_size_change',
            'reviewer_type': 'non_core_reviewed',
            'change_percentage': -26.7,
            'description': 'non_core_reviewedのグループサイズが-26.7%変化'
        },
        {
            'type': 'significant_change',
            'metric': 'lines_added',
            'significant_pairs': 3,
            'description': 'lines_addedで3ペアに有意な差'
        }
    ]
}
```

---

## 使用例

### 基本的なワークフロー

```python
from src.analysis.trend_metrics.analysis.statistical_analyzer import (
    calculate_summary_statistics,
    perform_all_statistical_tests,
    summarize_significant_results
)
from src.analysis.trend_metrics.analysis.trend_comparator import (
    compare_early_vs_late_by_reviewer_type,
    identify_diverging_trends,
    calculate_group_size_changes,
    generate_trend_summary
)

# 1. 記述統計量を計算
summary_stats = calculate_summary_statistics(
    groups=classified_groups,
    metrics=METRIC_COLUMNS
)

# 2. 統計検定を実行
test_results = perform_all_statistical_tests(
    groups=classified_groups,
    metrics=METRIC_COLUMNS,
    test_config=STATISTICAL_TEST_CONFIG
)

# 3. 有意な結果をサマリー
significant_results = summarize_significant_results(
    test_results=test_results,
    alpha=0.05
)

# 4. 前期/後期を比較
comparison = compare_early_vs_late_by_reviewer_type(
    groups=classified_groups,
    metrics=METRIC_COLUMNS
)

# 5. 乖離する傾向を特定
diverging = identify_diverging_trends(
    comparison_results=comparison,
    threshold_percentage=10.0
)

# 6. グループサイズの変化を計算
size_changes = calculate_group_size_changes(groups=classified_groups)

# 7. 総合サマリーを生成
summary = generate_trend_summary(
    comparison_results=comparison,
    test_results=test_results,
    size_changes=size_changes
)

# 8. 結果を保存
import json
with open('analysis_results.json', 'w') as f:
    json.dump({
        'summary_statistics': summary_stats,
        'test_results': test_results,
        'comparison': comparison,
        'summary': summary
    }, f, indent=2)
```

---

## データフロー

```
1. グループ分類されたChangeデータ
   ↓
2. statistical_analyzer.py
   ├─ calculate_summary_statistics() → 記述統計量
   ├─ perform_all_statistical_tests() → 検定結果
   └─ summarize_significant_results() → 有意な結果
   ↓
3. trend_comparator.py
   ├─ compare_early_vs_late_by_reviewer_type() → 前期/後期比較
   ├─ identify_diverging_trends() → 乖離する傾向
   ├─ calculate_group_size_changes() → サイズ変化
   └─ generate_trend_summary() → 総合サマリー
   ↓
4. 分析結果の可視化・レポート生成
```

---

## 統計検定の設定

`constants.py`の`STATISTICAL_TEST_CONFIG`で設定：

```python
STATISTICAL_TEST_CONFIG = {
    'alpha': 0.05,                    # 有意水準
    'use_bonferroni': True,           # Bonferroni補正を使用するか
    'test_method': 'mann_whitney_u',  # 使用する検定手法
    'effect_size_method': 'cohen_d'   # 効果量の計算方法
}
```

---

## 注意事項

### サンプルサイズ

- 統計検定は各グループに少なくとも2サンプル必要
- サンプルサイズが小さい場合、検定力が低下
- 効果量で実質的な差を評価することが重要

### 多重比較

- Bonferroni補正を使用して多重比較を調整
- 補正により有意水準が厳しくなる（偽陽性を減らす）
- 比較回数が多い場合、有意な差を検出しにくくなる

### 欠損値

- メトリクス値が`None`の場合は除外して計算
- 除外後のサンプルサイズを確認すること

### 変化率の計算

- 前期の平均がゼロの場合、変化率は`None`
- ゼロ除算を回避する処理を実装済み

---

## ログ出力

このモジュールは`logging`を使用してログを出力します。

**出力例:**
```
INFO - グループ early_core_reviewed の統計量を計算しました（105 件）
INFO - メトリクス lines_added の検定を開始...
INFO - メトリクス lines_added について 15 ペアの検定を完了しました (adjusted α=0.0033)
INFO - 全メトリクス (16 種類) の検定が完了しました
INFO - 5 メトリクスで有意な差が検出されました
INFO - レビューアタイプ core_reviewed の前期/後期比較を完了 (前期: 105 件, 後期: 92 件)
INFO - 3 メトリクスで乖離する傾向が検出されました
INFO - グループサイズの変化を計算しました
INFO - トレンド分析サマリーを生成しました（8 件の主要な発見）
```

---

## まとめ

`analysis`モジュールは以下を提供します：

1. **記述統計量**: グループ別・メトリクス別の基本統計
2. **統計検定**: Mann-Whitney U検定によるグループ間比較
3. **効果量**: Cohen's dによる実質的な差の評価
4. **トレンド比較**: 前期/後期の変化分析
5. **乖離検出**: レビューアタイプ間で異なる傾向の特定
6. **総合サマリー**: 主要な発見のまとめ

これらの機能により、OpenStackプロジェクトのリリース期間におけるメトリクスの変化を統計的に分析できます。
