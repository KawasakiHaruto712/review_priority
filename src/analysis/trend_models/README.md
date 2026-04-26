# Trend Models Analysis

OpenStackプロジェクトにおけるレビュー優先度予測モデルを構築・評価するためのモジュールです。

## 概要

本モジュールは、Gerritのコードレビューデータを分析し、以下の2方式でモデルを構築・評価します。

- 分類モード: 各Changeがレビューされるかどうかを予測
- ランキングモード: 日次の候補集合を順位付けし、優先度の妥当性を評価

リリースサイクルにおける異なる期間（リリース直後・直前）で性能を評価し、トレンド分析を行います。

## ディレクトリ構成

```
trend_models/
├── README.md                 # 本ファイル
├── __init__.py
├── designs.md                # 設計メモ
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
├── classification_models/    # 分類モデル
│   ├── README.md
│   ├── __init__.py
│   ├── base_model.py         # 基底クラス
│   └── trend_predictor.py    # 各種機械学習モデルの実装
├── ranking/                  # ランキング学習・推論
│   ├── __init__.py
│   ├── daily_rank_builder.py # 日次ランキング学習データ生成
│   ├── ranking_dataset.py    # ranking用行列/目的変数生成
│   └── ranking_predictor.py  # pointwise rankingモデル
└── utils/                    # ユーティリティ
    ├── README.md
    ├── __init__.py
    ├── constants.py          # 定数定義
    └── data_loader.py        # データ読み込み・期間フィルタリング
```

## 使用方法

### コマンドライン実行

```bash
# デフォルト設定で実行
# （デフォルト task-mode=ranking、constants.py の RANKING_MODEL_TYPES を順次実行）
python3 src/analysis/trend_models/main.py

# 分類モードで特定モデルを指定
python3 src/analysis/trend_models/main.py --task-mode classification --model random_forest

# プロジェクトを指定
python3 src/analysis/trend_models/main.py --project nova

# 詳細ログを出力
python3 src/analysis/trend_models/main.py --verbose

# 開発者タイプ（Core/Non-Core）で結果を分割
python3 src/analysis/trend_models/main.py --split-by-developer

# ランキングモードで実行（日次ランキングを学習）
python3 src/analysis/trend_models/main.py --task-mode ranking --ranking-model random_forest_regressor

# SVR でランキングを実行
python3 src/analysis/trend_models/main.py --task-mode ranking --ranking-model svr

# 分類とランキングを同時実行して比較
python3 src/analysis/trend_models/main.py --task-mode both --model random_forest --ranking-model gradient_boosting_regressor

# ランキングの目的変数を変更（rank / rank_pct / time）
python3 src/analysis/trend_models/main.py --task-mode ranking --ranking-label rank_pct --k-values 5,10,20
```

### 実行時のモデル選択ルール

- `task-mode=classification`: `--model` 未指定時は `constants.py` の `MODEL_TYPES` を順次実行
- `task-mode=ranking`: `--ranking-model` 未指定時は `constants.py` の `RANKING_MODEL_TYPES` を順次実行
- `task-mode=both`: 分類は `--model`（未指定なら `DEFAULT_MODEL_TYPE`）を1つ実行し、ランキングは `--ranking-model` 未指定時に `RANKING_MODEL_TYPES` を順次実行

### オプション

| オプション | 短縮形 | 説明 | デフォルト |
|-----------|--------|------|-----------|
| `--task-mode` | - | 実行モード（`classification` / `ranking` / `both`） | `ranking` |
| `--project` | `-p` | 分析対象プロジェクト名 | `nova` |
| `--model` | `-m` | 分類モデルタイプ（classification時は未指定で複数モデル実行） | `constants.py`のMODEL_TYPES |
| `--ranking-model` | - | ranking モデルタイプ（未指定時は複数モデルを順次実行） | `constants.py`のRANKING_MODEL_TYPES |
| `--ranking-label` | - | ranking 目的変数（`rank` / `rank_pct` / `time`） | `rank_pct` |
| `--k-values` | - | ranking 指標の K 値（カンマ区切り） | `5,10,20` |
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
    task_mode='classification',
    model_type='random_forest',
)

# 分析を実行
result = analyzer.run()

# 複数モデルで評価する場合
analyzer = TrendModelsAnalyzer(project_name='nova', task_mode='classification')
analyzer.load_data()
analyzer.prepare_release_data()

for model_type in ['random_forest', 'gradient_boosting', 'svm']:
    result = analyzer.run_with_model(model_type)
    print(f"{model_type}: F1={result['summary'][0]['f1_mean']:.4f}")

# ranking + classification を同時実行
analyzer = TrendModelsAnalyzer(
    project_name='nova',
    task_mode='both',
    model_type='random_forest',
    ranking_model_type='random_forest_regressor',
    ranking_label_mode='rank_pct',
)
result = analyzer.run()
print(result['ranking']['summary'][0]['ndcg_at_10_mean'])
```

## サポートするモデル

### 分類モデル

| モデル名 | 識別子 | 説明 | 依存パッケージ |
|---------|--------|------|---------------|
| Random Forest | `random_forest` | ランダムフォレスト分類器 | scikit-learn |
| Gradient Boosting | `gradient_boosting` | 勾配ブースティング分類器 | scikit-learn |
| Logistic Regression | `logistic_regression` | ロジスティック回帰 | scikit-learn |
| SVM | `svm` | サポートベクターマシン | scikit-learn |
| TabNet | `tabnet` | Attentionベースの深層学習モデル | pytorch-tabnet |
| FT-Transformer | `ft_transformer` | MLP実装 | torch |

### ランキングモデル

| モデル名 | 識別子 | 説明 | 依存パッケージ |
|---------|--------|------|---------------|
| Random Forest Regressor | `random_forest_regressor` | ランダムフォレスト回帰 | scikit-learn |
| Gradient Boosting Regressor | `gradient_boosting_regressor` | 勾配ブースティング回帰 | scikit-learn |
| SVR | `svr` | サポートベクター回帰 | scikit-learn |
| Linear Regression | `linear_regression` | 線形回帰 | scikit-learn |

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

## ランキング目的変数

| モード | カラム | 説明 |
|-------|--------|------|
| `rank` | `review_priority_rank` | 日次query内の dense rank（1が最優先） |
| `rank_pct` | `review_priority_rank_pct` | queryサイズ補正済み順位（0が最優先） |
| `time` | `time_to_review_seconds` | 次レビューまでの秒数 |

ランキングデータは日次 query 単位（`query_id`）で生成され、`NDCG@K`, `MRR`, `Spearman`, `Pairwise Accuracy`, `MAE`, `RMSE` で評価されます。

## 評価指標

### 分類モード

| 指標 | 説明 |
|------|------|
| `Precision` | 予測をレビュー対象としたChangeのうち、実際にレビューされた割合 |
| `Recall` | 実際にレビューされたChangeのうち、モデルが検出できた割合 |
| `F1` | PrecisionとRecallの調和平均 |
| `Accuracy` | 全予測のうち正解した割合 |

分類モードでは Leave-One-Release-Out 交差検証を行い、期間の組み合わせごとに平均値・標準偏差を集計します。

### ランキングモード

| 指標 | 説明 |
|------|------|
| `NDCG@5 / @10 / @20` | 上位k件の順位品質を評価（上位ほど重みを高く評価） |
| `MRR` | 最初に正解を提示できた順位の逆数平均 |
| `Spearman` | 予測順位と真の順位の順位相関 |
| `Pairwise Accuracy` | アイテム対の優先関係を正しく並べられた割合 |
| `MAE / RMSE` | 目的変数（rank / rank_pct / time）に対する回帰誤差 |
| `Precision / Recall / F1` | 上位提案を正例とみなした補助的な二値評価 |

ランキングモードでは日次queryごとに指標を計算し、期間・リリースをまたいで平均化してレポート化します。

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
| `rank_detail_*.csv` | CSV | ランキング評価の詳細結果 |
| `rank_summary_*.csv` | CSV | ランキング評価のサマリー |
| `ranking_results_*.json` | JSON | ランキング評価の全体結果 |
| `figures/rank_heatmap_ndcg_at_10_*.png` | PNG | ランキング評価ヒートマップ |
| `figures/rank_ndcg_*.png` | PNG | NDCG@K 比較グラフ |

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
