# Visualization Module

このモジュールは、トレンド分析の結果を可視化するための機能を提供します。
8つのグループ（Early/Late × Core/Non-Core × Reviewed/Not Reviewed）に基づいた詳細な分析結果をグラフ化します。

## 機能

### 1. トレンドプロッター (`trend_plotter.py`)

データの分布や推移を可視化します。

- **`plot_boxplots_8groups`**: 8つのグループごとのメトリクス分布を箱ひげ図で表示します。
- **`plot_trend_lines`**: 期間（Early/Late）ごとの平均値の推移を折れ線グラフで表示します。
- **`plot_metric_changes`**: 期間間の変化率（%）を棒グラフで表示します。

### 2. ヒートマップジェネレーター (`heatmap_generator.py`)

統計検定の結果を可視化します。

- **`generate_heatmap`**: 統計的検定のp値および効果量（Cohen's d）をヒートマップとして出力します。有意な差がある箇所を視覚的に特定するのに役立ちます。

## 使用方法

```python
from src.analysis.trend_metrics.visualization import (
    plot_boxplots_8groups,
    plot_trend_lines,
    plot_metric_changes,
    generate_heatmap
)

# データフレームと出力ディレクトリを準備
# df: 'group'カラム（例: 'early_core_reviewed'）を持つデータフレーム
# output_dir: Pathオブジェクト

# 1. 分布の可視化
plot_boxplots_8groups(df, output_dir)

# 2. トレンドの可視化
plot_trend_lines(df, output_dir)

# 3. 変化率の可視化
# change_df: calculate_changes() の結果
plot_metric_changes(change_df, output_dir)

# 4. 統計検定結果のヒートマップ
# test_results: perform_all_statistical_tests() の結果
generate_heatmap(test_results, output_dir)
```

## 出力ファイル

指定された出力ディレクトリに以下のファイルが生成されます。

- `boxplot_{metric}.pdf`: グループごとの箱ひげ図
- `trend_line_{metric}.pdf`: 平均値の推移グラフ
- `metric_changes.pdf`: 変化率の棒グラフ
- `heatmap_pvalues.pdf`: p値のヒートマップ
- `heatmap_effect_sizes.pdf`: 効果量のヒートマップ

## 依存ライブラリ

- matplotlib
- seaborn
- pandas
- numpy
