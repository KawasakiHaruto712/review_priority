# Evaluation - モデル評価・可視化モジュール

予測モデルの評価と結果の可視化機能を提供します。

## モジュール構成

| ファイル | 説明 |
|---------|------|
| `evaluator.py` | 評価指標の計算・交差検証の実行 |
| `visualizer.py` | 結果の可視化・レポート生成 |

## evaluator.py

### データクラス

#### EvaluationResult
単一の評価結果を格納。

| フィールド | 型 | 説明 |
|-----------|---|------|
| `precision` | float | 適合率 |
| `recall` | float | 再現率 |
| `f1` | float | F1スコア |
| `accuracy` | float | 正解率 |
| `confusion_matrix` | np.ndarray | 混同行列 |
| `n_train` | int | 学習データ数 |
| `n_eval` | int | 評価データ数 |
| `n_positive` | int | 正例数 |
| `n_negative` | int | 負例数 |

#### CVResult
交差検証結果を格納。

| フィールド | 型 | 説明 |
|-----------|---|------|
| `eval_release` | str | 評価リリース |
| `eval_period` | str | 評価期間 |
| `train_period` | str | 学習期間 |
| `developer_type` | str | 開発者タイプ |
| `precision` | float | 適合率 |
| `recall` | float | 再現率 |
| `f1` | float | F1スコア |
| `n_train` | int | 学習データ数 |
| `n_eval` | int | 評価データ数 |
| `feature_importances` | Dict | 特徴量重要度 |

### Evaluator

モデル評価を行うクラス。

#### 使用例

```python
from src.analysis.trend_models.evaluation.evaluator import Evaluator

evaluator = Evaluator(model_type='random_forest', feature_names=FEATURE_NAMES)

# 学習と評価を同時に行う
result, model = evaluator.train_and_evaluate(X_train, y_train, X_eval, y_eval)

# 予測結果を評価
result = evaluator.evaluate(y_true, y_pred, n_train=len(X_train))
```

### 主要な関数

#### leave_one_out_cv
Leave-One-Out交差検証を実行。各リリースを順番に評価データとして使用し、残りのリリースで学習。

```python
from src.analysis.trend_models.evaluation.evaluator import leave_one_out_cv

results = leave_one_out_cv(
    release_data,          # Dict[release, Dict[period, DataFrame]]
    feature_names=FEATURE_NAMES,
    model_type='random_forest',
    split_by_developer_type=False
)
```

#### summarize_cv_results
交差検証結果を集計。

```python
from src.analysis.trend_models.evaluation.evaluator import summarize_cv_results

summary_df = summarize_cv_results(results)
# Returns: DataFrame with mean/std of precision, recall, f1 by period combination
```

## visualizer.py

### Visualizer

評価結果の可視化を行うクラス。

#### 使用例

```python
from src.analysis.trend_models.evaluation.visualizer import Visualizer

visualizer = Visualizer(
    output_dir=output_dir,
    project_name='nova',
    filename_suffix='random_forest'
)

# 詳細結果をCSVで保存
visualizer.save_cv_detail(results)

# サマリーをCSVで保存
visualizer.save_cv_summary(results)

# JSONで保存
visualizer.save_results_json(results, metadata)

# ヒートマップを作成
visualizer.plot_cross_period_heatmap(results, metric='f1')

# 棒グラフを作成
visualizer.plot_cv_results(results)

# 特徴量重要度グラフを作成
visualizer.plot_feature_importances(importance_dict, top_n=10)
```

### generate_evaluation_report

全ての評価レポートを生成する便利関数。

```python
from src.analysis.trend_models.evaluation.visualizer import generate_evaluation_report

outputs = generate_evaluation_report(
    results,
    project_name='nova',
    metadata=metadata,
    output_dir=output_dir,
    filename_suffix='random_forest'
)
# Returns: Dict[str, Path] - 出力ファイル名とパスの辞書
```

### 出力ファイル

| ファイル | 形式 | 内容 |
|---------|------|------|
| `cv_detail_*.csv` | CSV | 各交差検証の詳細結果 |
| `cv_summary_*.csv` | CSV | 期間・開発者タイプ別の集計結果 |
| `results_*.json` | JSON | 全体の結果（メタデータ含む） |
| `figures/heatmap_f1_*.png` | PNG | F1スコアのヒートマップ |
| `figures/cv_results_*.png` | PNG | Precision/Recall/F1の棒グラフ |
| `figures/feature_importance_*.png` | PNG | 特徴量重要度の横棒グラフ |
