# Trend Metrics Analysis

前期/後期でメトリクスの値がどのように変化するのかを分析するモジュールです。

## 概要

OpenStackプロジェクトにおいて、連続する2つのリリース期間を比較し、16種類の特徴量（メトリクス）がコアレビューアおよび非コアレビューアのレビュー行動によってどのように変化するかを分析します。

## セットアップ

### 必要なデータ

1. **Changeデータ**: `data/openstack_collected/{project}/changes/*.json`
2. **メジャーリリース情報**: `data/openstack/major_releases_summary.csv`
3. **コアレビューア情報**: `data/openstack_collected/core_developers.json`

### 設定

`src/analysis/trend_metrics/utils/constants.py`で分析対象を設定：

```python
TREND_ANALYSIS_CONFIG = {
    'project': 'nova',  # 分析対象プロジェクト
    'target_releases': ['20.0.0', '21.0.0'],  # [現在のリリース, 次のリリース]
}
```

## 使用方法

### コマンドライン実行

```bash
# プロジェクトルートから実行
python -m src.analysis.trend_metrics.main

# または
cd src/analysis/trend_metrics
python main.py
```

### Pythonコードから実行

```python
from src.analysis.trend_metrics import TrendMetricsAnalyzer

# デフォルト設定で実行
analyzer = TrendMetricsAnalyzer()
summary = analyzer.run_analysis()

# カスタム設定で実行
analyzer = TrendMetricsAnalyzer(
    project_name='neutron',
    target_releases=['18.0.0', '19.0.0']
)
summary = analyzer.run_analysis()
```

## 出力

分析結果は `data/analysis/trend_metrics/{project}_{current_release}/` に保存されます：

- `classified_changes.csv`: 分類済みChangeデータ
- `group_statistics.json`: グループ別統計
- `analysis_summary.json`: 分析サマリー

## ディレクトリ構造

```
src/analysis/trend_metrics/
├── design.md                    # 設計書
├── README.md                    # 本ファイル
├── main.py                      # エントリーポイント
├── utils/                       # ユーティリティ
│   ├── constants.py            # 定数定義
│   ├── data_loader.py          # データ読み込み
│   └── core_reviewer_checker.py # コアレビューア判定
├── metrics_extraction/          # データ抽出・分類
│   ├── period_extractor.py     # 期間別抽出
│   └── reviewer_classifier.py  # レビューア分類
├── analysis/                    # 統計分析
│   ├── statistical_analyzer.py
│   └── trend_comparator.py
└── visualization/               # 可視化（TODO）
    ├── trend_plotter.py
    └── heatmap_generator.py
```

## 開発状況

### ✅ 実装済み

- データ読み込み機能
- 期間別データ抽出
- レビューアタイプ分類（8グループ）
- 基本データの保存

### 🚧 TODO

- メトリクス計算機能
- 統計分析機能
- 可視化機能