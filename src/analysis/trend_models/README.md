# Trend Models Analysis

OpenStackプロジェクトにおけるレビュー優先度予測モデルを構築・評価するためのモジュールです。

## 概要

本モジュールは、Gerritのコードレビューデータを分析し、各Changeがレビューされるかどうかを予測する機械学習モデルを構築します。リリースサイクルにおける異なる期間（リリース直後・直前）での予測性能を評価し、トレンド分析を行います。

## ディレクトリ構成

```
trend_models/
├── README.md                 # 本ファイル
├── __init__.py
├── main.py                   # メインエントリーポイント
├── evaluation/               # モデル評価・可視化
│   ├── README.md
│   ├── __init__.py
│   ├── evaluator.py          # 評価指標の計算・交差検証
│   └── visualizer.py         # 結果の可視化・レポート生成
├── features/                 # 特徴量抽出・前処理
│   ├── README.md
│   ├── __init__.py
│   ├── extractor.py          # 16種類のメトリクス抽出
│   └── preprocessor.py       # ラベル付け・正規化・欠損値処理
├── models/                   # 予測モデル
│   ├── README.md
│   ├── __init__.py
│   ├── base_model.py         # 基底クラス
│   └── trend_predictor.py    # 各種機械学習モデルの実装
└── utils/                    # ユーティリティ
    ├── README.md
    ├── __init__.py
    ├── constants.py          # 定数定義
    └── data_loader.py        # データ読み込み・期間フィルタリング
```

## 使用方法

### コマンドライン実行

```bash
# デフォルト設定で実行（constants.pyのMODEL_TYPESに定義された全モデルを使用）
python3 src/analysis/trend_models/main.py

# 特定のモデルを指定
python3 src/analysis/trend_models/main.py --model random_forest

# プロジェクトを指定
python3 src/analysis/trend_models/main.py --project nova

# 詳細ログを出力
python3 src/analysis/trend_models/main.py --verbose

# 開発者タイプ（Core/Non-Core）で結果を分割
python3 src/analysis/trend_models/main.py --split-by-developer
```

### オプション

| オプション | 短縮形 | 説明 | デフォルト |
|-----------|--------|------|-----------|
| `--project` | `-p` | 分析対象プロジェクト名 | `nova` |
| `--model` | `-m` | 使用するモデルタイプ | `constants.py`のMODEL_TYPES |
| `--split-by-developer` | `-d` | 開発者タイプで結果を分割 | `False` |
| `--data-dir` | - | データディレクトリのパス | `data/` |
| `--output-dir` | - | 出力ディレクトリのパス | `data/analysis/trend_models/{project}` |
| `--verbose` | `-v` | 詳細ログを出力 | `False` |

### Pythonコードからの使用

```python
from src.analysis.trend_models.main import TrendModelsAnalyzer

# アナライザを作成
analyzer = TrendModelsAnalyzer(
    project_name='nova',
    model_type='random_forest'
)

# 分析を実行
result = analyzer.run()

# 複数モデルで評価する場合
analyzer = TrendModelsAnalyzer(project_name='nova')
analyzer.load_data()
analyzer.prepare_release_data()

for model_type in ['random_forest', 'gradient_boosting', 'svm']:
    result = analyzer.run_with_model(model_type)
    print(f"{model_type}: F1={result['summary'][0]['f1_mean']:.4f}")
```

## サポートするモデル

| モデル名 | 識別子 | 説明 | 依存パッケージ |
|---------|--------|------|---------------|
| Random Forest | `random_forest` | ランダムフォレスト分類器 | scikit-learn |
| Gradient Boosting | `gradient_boosting` | 勾配ブースティング分類器 | scikit-learn |
| Logistic Regression | `logistic_regression` | ロジスティック回帰 | scikit-learn |
| SVM | `svm` | サポートベクターマシン | scikit-learn |
| TabNet | `tabnet` | Attentionベースの深層学習モデル | pytorch-tabnet |
| FT-Transformer | `ft_transformer` | MLP実装 | torch |

## 特徴量（16種類）

| カテゴリ | 特徴量名 | 説明 |
|---------|---------|------|
| Bug Metrics | `bug_fix_confidence` | バグ修正の確信度 |
| Change Metrics | `lines_added` | 追加行数 |
| | `lines_deleted` | 削除行数 |
| | `files_changed` | 変更ファイル数 |
| | `elapsed_time` | 経過時間 |
| | `revision_count` | リビジョン数 |
| | `test_code_presence` | テストコードの有無 |
| Developer Metrics | `past_report_count` | 過去の投稿数 |
| | `recent_report_count` | 直近の投稿数 |
| | `merge_rate` | マージ率 |
| | `recent_merge_rate` | 直近のマージ率 |
| Project Metrics | `days_to_major_release` | 次リリースまでの日数 |
| | `open_ticket_count` | オープンチケット数 |
| | `reviewed_lines_in_period` | 期間内のレビュー行数 |
| Refactoring Metrics | `refactoring_confidence` | リファクタリングの確信度 |
| Review Metrics | `uncompleted_requests` | 未完了リクエスト数 |

## 分析期間

| 期間タイプ | 説明 |
|-----------|------|
| `early` | リリース直後30日間 |
| `late` | 次リリース直前30日間 |
| `all` | 当該リリースの全期間 |

## 出力ファイル

実行後、以下のファイルが`data/analysis/trend_models/{project}/{model}/`に出力されます：

| ファイル名 | 形式 | 説明 |
|-----------|------|------|
| `cv_detail_*.csv` | CSV | 交差検証の詳細結果 |
| `cv_summary_*.csv` | CSV | 交差検証のサマリー |
| `results_*.json` | JSON | 全体の結果（メタデータ含む） |
| `figures/heatmap_f1_*.png` | PNG | 期間交差評価のヒートマップ |
| `figures/cv_results_*.png` | PNG | 評価指標の棒グラフ |
| `figures/feature_importance_*.png` | PNG | 特徴量重要度グラフ |

## 依存パッケージ

### 必須

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### オプション（深層学習モデル用）

- torch（FT-Transformer用）
- pytorch-tabnet（TabNet用）

```bash
pip install pytorch-tabnet torch
```
