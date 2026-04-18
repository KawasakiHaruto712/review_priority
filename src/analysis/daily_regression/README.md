# Daily Regression Analysis

日次単位で Open な Change を対象に、レビュー優先順位またはレビュー待ち時間を目的変数とした重回帰分析（OLS）を実行するモジュールです。

## 概要

本モジュールは以下を実行します。

1. 指定バージョン期間を日次で走査
2. 各日で分析対象 Change を抽出
3. 16特徴量を計算
4. 目的変数（順位または時間）を用いて OLS を実行
5. 回帰係数・統計値・可視化画像を保存

デフォルトの目的変数は順位（`review_priority_rank`）です。

## 必要データ

- Change データ: `data/openstack_collected/{project}/changes/*.json`
- メジャーリリース情報: `data/openstack/major_releases_summary.csv`
- Bot 設定: `src/config/gerrymanderconfig.ini`
- Review メトリクス設定:
  - `data/processed/review_keywords.json`
  - `data/processed/review_label.json`

## 設定

`src/analysis/daily_regression/utils/constants.py` で分析対象を設定します。

- `DAILY_REGRESSION_CONFIG['project']`: 対象プロジェクトとバージョン
- `MIN_SAMPLES`: 日次回帰を実行する最小サンプル数
- `MAX_CENSORING_SECONDS`: 目的変数の打ち切り秒数
- `OUTPUT_DIR_BASE`: 出力先ディレクトリ
- `DEFAULT_TARGET_MODE`: 目的変数モード（`rank` / `time`）

## 実行方法

### コマンドライン

```bash
# プロジェクトルートで実行
python -m src.analysis.daily_regression.main

# 目的変数をレビュー待ち時間(秒)に切り替える
python -m src.analysis.daily_regression.main --target-mode time
```

### Python から実行

```python
from src.analysis.daily_regression import DailyRegressionAnalyzer

# デフォルト（constants.py の設定）
analyzer = DailyRegressionAnalyzer()
summary = analyzer.run_analysis()

# プロジェクトとバージョンを明示
analyzer = DailyRegressionAnalyzer(
    project_name="nova",
    versions=["14.0.0", "15.0.0"],
    target_mode="rank",  # "rank" または "time"
)
summary = analyzer.run_analysis()
```

## 出力

出力先: `data/analysis/daily_regression/`

各バージョンごとに `data/analysis/daily_regression/{project}_{version}/` が作成されます。

- `daily_coefficients.csv`
  - 日次の回帰係数と p 値
- `daily_regression_stats.csv`
  - 日次の統計値（`r_squared`, `adj_r_squared`, `f_statistic`, `f_pvalue` など）
- `daily_regression_detail.json`
  - 日次結果の詳細（係数、標準誤差、t値、p値、スキップ日など）
- `plots/coef_*.png`
  - 各特徴量の標準化回帰係数の時系列
- `plots/r_squared.png`
  - R² と Adjusted R² の時系列

全体サマリーは `data/analysis/daily_regression/summary/analysis_summary.json` に保存されます。

## ディレクトリ構成

```text
src/analysis/daily_regression/
├── __init__.py
├── README.md
├── desigin.md
├── main.py
├── regression/
│   ├── sample_extractor.py
│   ├── metrics_calculator.py
│   └── ols_executor.py
├── utils/
│   ├── constants.py
│   └── data_loader.py
└── visualization/
    └── coefficient_plotter.py
```

## 実装メモ

- OLS 実行時は `standardize=True` を指定し、標準化回帰係数（β）を出力します。
- 目的変数モード:
  - `rank`: `review_priority_rank`（1が最優先）
  - `time`: `time_to_review_seconds`（連続値）
- 日次で有効サンプルが `MIN_SAMPLES` 未満の場合、その日はスキップします。
