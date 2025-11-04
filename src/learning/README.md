# Learning モジュール

逆強化学習（Inverse Reinforcement Learning, IRL）を用いたコードレビュー優先順位付けモデルを提供するモジュールです。レビューアーの行動から優先順位の判断基準を学習し、新しいレビューの優先度を予測します。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `irl_models.py` | IRLモデルの実装とデータ処理クラス |
| `temporal_weight_analysis.py` | 時系列重み分析（スライディングウィンドウでIRL重み変動を追跡） |
| `temporal_model_evaluation.py` | 時系列モデル評価（Balanced Random Forestで正負例を分類） |

## 🧠 アルゴリズム

### 最大エントロピー逆強化学習 (MaxEnt IRL)
- **専門家の行動模倣**: レビューアーの優先順位付け行動を学習
- **特徴量重み推定**: 各メトリクスの重要度を自動計算
- **確率的モデル**: 不確実性を考慮した優先度予測

## 🔧 主要クラス

### `MaxEntIRLModel`
逆強化学習モデルの実装

```python
class MaxEntIRLModel:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Args:
            learning_rate: 学習率
            max_iterations: 最大反復回数  
            tolerance: 収束判定閾値
        """
```

**主要メソッド:**
- `fit(features, priorities)`: モデル学習
- `predict_priority_scores(features)`: 優先度予測
- `save_model(path)`, `load_model(path)`: モデル保存・読み込み

### `ReviewPriorityDataProcessor`
データ前処理とパイプライン管理

```python
class ReviewPriorityDataProcessor:
    def load_openstack_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """OpenStackデータの読み込み"""
        
    def extract_features(self, changes_df, analysis_time) -> pd.DataFrame:
        """特徴量抽出"""
```

## 📊 逆強化学習モデル (`irl_models.py`)

### 概要
最大エントロピー逆強化学習（MaxEnt IRL）を用いて、レビューアーの優先順位付け行動から特徴量の重要度を学習します。学習用データの抽出、特徴量ベクトル化、優先度ラベルの計算、そしてIRLモデルの学習と保存を行います。

### 主な機能

#### データ抽出関数
- **`extract_learning_events(changes_df, bot_names)`**
  - ボットの操作を除外して、レビュー（コメントやステータス変更）に相当するイベントを時系列で抽出します。

- **`get_open_changes_at_time(changes_df, target_time)`**
  - 指定時刻にオープンであったChangeの一覧を返します（created <= target_time < closed などの判定）。

- **`calculate_review_priorities(open_changes, current_time, bot_names)`**
  - 各Changeについて、次にレビューされるまでの相対順位から優先度スコアを計算します。順位に基づき (総数 - 順位) / 総数 のような正規化された重みを付与します。

#### IRLモデルクラス
- **`MaxEntIRLModel`**
  - L-BFGS による最大エントロピーIRLの学習を実装しています。
  - 主要メソッド:
    - `fit(X, y)` : 特徴量行列Xと優先度重みyを受け取り重みベクトルを学習します。
    - `predict_priority_scores(X)` : 学習済み重みを用いてスコアを計算します。
    - `save_model(path)` / `load_model(path)` : モデルの永続化（pickle）。

#### データ処理クラス
- **`ReviewPriorityDataProcessor`**
  - OpenStackのJSONデータを読み込み、16次元の特徴量を抽出するユーティリティを備えています。
  - `extract_features` は以下のような特徴量を生成します: 行追加/削除数、ファイル数、過去の報告数、マージ率、リリースまでの日数、レビュー検出に関する統計など。

### 出力ファイル
`data/results/`ディレクトリに保存：

- `irl_model_{project}_{start}_{end}.pkl` : 学習済みモデル（pickle形式）
- `irl_analysis_{start}_{end}.json` : 学習統計と特徴量重みのサマリ

### 使用例

```python
from src.learning.irl_models import run_temporal_irl_analysis

# プロジェクト別IRL学習の実行
run_temporal_irl_analysis(projects=['nova'])
```

### 注意事項

- 特徴量抽出関数は `src/features/` 以下の補助モジュールに依存します。環境に応じてこれらの関数が利用可能であることを確認してください。
- 最適化は収束性に依存するため、データのスケーリングや初期値の調整が必要になる場合があります。

---

## 📊 時系列重み分析 (`temporal_weight_analysis.py`)

### 概要
スライディングウィンドウでIRLを実行し、特徴量の重み（feature weights）の時系列変化を可視化・保存します。プロジェクトごとに、各ウィンドウで独立にIRLを学習し、得られた重みを蓄積してCSV/JSON/PDFで出力します。

### 主な機能

#### TemporalWeightAnalyzerクラス
- **`generate_time_windows(start_date, end_date)`** : 指定期間をウィンドウに分割
- **`load_bot_names()`** : `src/config/gerrymanderconfig.ini` からボット名を読み込み
- **`analyze_window(changes_df, window_start, window_end, bot_names, project, releases_df)`** : ウィンドウごとに学習イベント抽出→IRL学習→重み・統計を返す
- **`run_temporal_analysis(projects=None, start_date=None, end_date=None)`** : 全ウィンドウ・全プロジェクトを処理して `temporal_results` を整形
- **`save_results(output_dir=None)`** : CSV / JSON / PDF を生成して保存
- **`create_weight_visualization(project, output_dir)`** : 16特徴量を4x4グリッドでプロットしPDFに保存

### 出力ファイル
`data/temporal_analysis/`ディレクトリに保存：

- `temporal_weights_{project}.csv` : ウィンドウ毎の重みテーブル
- `temporal_weights_{project}.json` : 各ウィンドウの詳細と重みの履歴
- `temporal_weights_{project}.pdf` : 16個の特徴量の時系列図（4x4グリッド）

### 使用例

```python
from src.learning.temporal_weight_analysis import run_temporal_weight_analysis

# 重みの時系列分析（PDF/CSV/JSON出力）
run_temporal_weight_analysis(projects=['nova'])
```

### 注意事項

- 各ウィンドウで独立に学習を行うため、ウィンドウ数が増えると計算時間が線形に増加します。必要に応じて `sliding_step` を増やしてウィンドウ数を減らしてください。
- 可視化には `matplotlib` / `seaborn` が必要です。未インストールの場合はPDF出力をスキップするか、事前にパッケージをインストールしてください。

---
```

## 📊 学習プロセス

### 1. データ準備
```python
# レビューデータの読み込み
processor = ReviewPriorityDataProcessor()
changes_df, releases_df = processor.load_openstack_data()

# 学習イベントの抽出
events = extract_learning_events(changes_df, bot_names)
```

### 2. 特徴量抽出
```python
# 16次元特徴量ベクトルの生成
features_df = processor.extract_features(changes_df, analysis_time)
feature_columns = [
    'bug_fix_confidence', 'lines_added', 'lines_deleted', 'files_changed',
    'elapsed_time', 'revision_count', 'test_code_presence',
    'past_report_count', 'recent_report_count', 'merge_rate', 'recent_merge_rate',
    'days_to_major_release', 'open_ticket_count', 'reviewed_lines_in_period',
    'refactoring_confidence', 'uncompleted_requests'
]
```

### 3. 優先度ラベル生成
```python
# レビュー時刻に基づく優先度重み計算
priorities = calculate_review_priorities(open_changes, current_time, bot_names)
```

### 4. モデル学習
```python
# IRLモデルの学習
model = MaxEntIRLModel(learning_rate=0.01, max_iterations=100)
model.feature_names = feature_columns
training_stats = model.fit(X, y)
```

## 📈 結果出力

### 学習結果JSON
```json
{
  "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
  "timestamp": "20241001_123000",
  "data_summary": {
    "total_learning_events": 1500,
    "successful_learning_data": 1350,
    "total_training_samples": 8000,
    "feature_dimensions": 16,
    "projects_analyzed": 6,
    "projects_failed": 0
  },
  "project_results": {
    "nova": {
      "training_stats": {
        "converged": true,
        "final_objective": 1.23,
        "iterations": 45
      },
      "model_path": "/path/to/irl_model_nova_20241001_20241231.pkl",
      "data_summary": {
        "training_samples": 2000,
        "learning_events": 350
      },
      "feature_weights": {
        "bug_fix_confidence": 0.15,
        "lines_added": 0.08,
        "recent_merge_rate": 0.42,
        "days_to_major_release": -0.25,
        ...
      }
    }
  }
}
```

### 重み解釈
- **正の重み**: 高優先度要因（例: `recent_merge_rate: 0.42`）
- **負の重み**: 低優先度要因（例: `days_to_major_release: -0.25`）

## 🚀 使用方法

### 基本的な学習・予測

```python
from src.learning.irl_models import run_temporal_irl_analysis

# プロジェクト別IRL学習の実行
projects = ['nova', 'neutron', 'swift', 'cinder', 'keystone', 'glance']
results = run_temporal_irl_analysis(projects)

# 結果の確認
for project, result in results['project_results'].items():
    print(f"{project}: 収束={result['training_stats']['converged']}")
    print(f"重要特徴量: {max(result['feature_weights'].items(), key=lambda x: x[1])}")
```

### 学習済みモデルによる予測

```python
# モデルの読み込み
model = MaxEntIRLModel()
model.load_model("irl_model_nova_20241001_20241231.pkl")

# 新しいレビューの優先度予測
new_features = extract_features(new_review_data)
priority_scores = model.predict_priority_scores(new_features)
```

## 📊 評価メトリクス

### 学習品質の指標

| メトリクス | 説明 | 良好な値 |
|-----------|------|----------|
| `converged` | 学習の収束 | `True` |
| `final_objective` | 目的関数の最終値 | 0.0-1.5 |
| `iterations` | 反復回数 | 10-100 |

### モデル性能
- **予測精度**: 専門家の判断との一致率
- **特徴量重要度**: 解釈可能な重み分布
- **汎化性能**: 未見データでの性能

## ⚡ パフォーマンス

### 学習時間
- **小規模データ** (100サンプル): ~1秒
- **中規模データ** (1,000サンプル): ~10秒  
- **大規模データ** (10,000サンプル): ~60秒

### メモリ使用量
- **基本モデル**: ~2MB
- **特徴量データ**: ~10MB (1,000サンプル)
- **学習済みモデル**: ~1.4KB (保存時)

## 🔧 ハイパーパラメータ調整

```python
# 学習率の調整
model = MaxEntIRLModel(
    learning_rate=0.001,     # 小さくすると安定、遅い
    max_iterations=500,      # 多くすると精度向上、時間増
    tolerance=1e-8           # 小さくすると厳密、時間増
)
```

### 推奨設定
- **高精度**: `learning_rate=0.001, max_iterations=1000, tolerance=1e-8`
- **高速**: `learning_rate=0.1, max_iterations=100, tolerance=1e-4`
- **バランス**: `learning_rate=0.01, max_iterations=500, tolerance=1e-6`

## 🎯 実用的な応用

### 1. リアルタイム優先順位付け
```python
# 新着レビューの優先度計算
def prioritize_new_reviews(new_reviews):
    features = extract_features(new_reviews)
    scores = model.predict_priority_scores(features)
    return sorted(zip(new_reviews, scores), key=lambda x: x[1], reverse=True)
```

### 2. レビュアー推薦
```python
# 高優先度レビューのレビュアー推薦
def recommend_reviewers(high_priority_reviews, reviewers_expertise):
    # 特徴量に基づいてマッチング
    return match_reviewers_to_reviews(high_priority_reviews, reviewers_expertise)
```

### 3. リソース配分
```python
# 開発リソースの配分決定
def allocate_resources(review_queue, available_resources):
    priorities = [model.predict_priority_scores(r.features)[0] for r in review_queue]
    return optimize_resource_allocation(review_queue, priorities, available_resources)
```

## ⚠️ 注意事項

1. **データ品質**: 学習データの質が結果に大きく影響
2. **ドメイン依存**: プロジェクト特性に応じた調整が必要
3. **継続学習**: 新しいデータでの定期的な再学習
4. **解釈性**: 重みの意味を適切に理解して活用

## 🔬 研究・拡張の方向性

- **深層学習**: ニューラルネットワークベースのIRL
- **多目的最適化**: 複数の評価軸の同時考慮
- **強化学習**: 動的な優先順位調整

## 📊 時系列モデル評価 (`temporal_model_evaluation.py`)

### 概要
Balanced Random Forestを用いてウィンドウごとに**過去データで学習したモデル**を作成し、**現在ウィンドウのデータを予測**することでレビュー優先順位付けモデルを評価します。

### 学習方式
各ウィンドウについて、時系列的に正しい訓練・評価分離を行います：

- **学習データ**: ウィンドウ開始の14日前からウィンドウ開始まで（過去14日間）
- **評価データ**: ウィンドウ開始からウィンドウ終了まで（現在ウィンドウ）

この方式により、**未来のデータで過去を予測する**というデータリークを防ぎ、実際の運用環境に近い評価が可能になります。学習期間と評価期間は同じ長さ（14日間）ですが、時系列的に完全に分離されています。

### 正負例の定義
- **正例（ラベル=1）**: ウィンドウ期間中に1回以上レビューされたChange
- **負例（ラベル=0）**: ウィンドウ期間中に1度もレビューされなかったChange

### 学習と評価のフロー

```
時系列: [ウィンドウ開始-14日] ← [ウィンドウ開始] ← [ウィンドウ終了]
              ↓                        ↓                ↓
         学習期間開始              学習期間終了      評価期間終了
                                   評価期間開始
              
              [────── 学習データ ──────][────── 評価データ ──────]
                    (過去14日間)            (現在ウィンドウ)
                         ↓                         ↓
                    モデル学習                予測・評価
                                           (Precision/Recall/F1を算出)
```

### 評価指標
各ウィンドウについて、以下の指標を算出します：

- **Precision（適合率）**: 予測した正例のうち実際に正例だった割合
- **Recall（再現率）**: 実際の正例のうち正しく予測できた割合
- **F1 Score（F値）**: PrecisionとRecallの調和平均
- **訓練サンプル数**: 過去14日間から収集されたサンプル数
- **評価サンプル数**: 現在ウィンドウのサンプル数

### 出力ファイル
`data/temporal_evaluation/`ディレクトリに保存：

- `temporal_evaluation_{project}.csv`: ウィンドウごとの評価指標
- `temporal_evaluation_{project}.json`: 全体のサマリー統計と詳細結果
- `temporal_evaluation_{project}.pdf`: 評価結果の可視化レポート（8つのグラフ）

### 可視化レポートの内容
PDFには以下の8つのグラフが含まれます：

1. **F1スコアの時系列変化**: 各ウィンドウのF1スコアの推移
2. **Precisionの時系列変化**: 各ウィンドウのPrecisionの推移
3. **Recallの時系列変化**: 各ウィンドウのRecallの推移
4. **全評価指標の比較**: F1/Precision/Recallの同時表示
5. **総サンプル数の推移**: 各ウィンドウの総Change数
6. **正負例サンプル数の推移**: レビューされた/されなかったChangeの数
7. **学習/評価データ数の推移**: 学習データと評価データのサンプル数比較
8. **正例比率の推移**: レビューされたChangeの割合

### データ不足時の動作
- 学習期間に十分なデータがない場合、そのウィンドウは評価されず `evaluation_status: 'failure'` として記録されます
- 訓練データが10サンプル未満、または正負例のどちらかしか存在しない場合も学習をスキップします
- 評価データが5サンプル未満の場合も評価をスキップします
- 結果JSONには不足理由（`error_message`）が記録されます

### 使用例

```python
from src.learning.temporal_model_evaluation import run_temporal_model_evaluation

# 全プロジェクトの評価（デフォルトの期間で実行）
results = run_temporal_model_evaluation()

# 特定プロジェクトの評価
results = run_temporal_model_evaluation(projects=['nova', 'neutron'])

# 結果の確認
for project, evaluations in results['evaluation_results'].items():
    successful = [r for r in evaluations if r.get('evaluation_status') == 'success']
    print(f"{project}: {len(successful)}/{len(evaluations)} ウィンドウが成功")
    
    if successful:
        avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
        avg_precision = sum(r['precision'] for r in successful) / len(successful)
        avg_recall = sum(r['recall'] for r in successful) / len(successful)
        print(f"  平均 F1={avg_f1:.3f}, P={avg_precision:.3f}, R={avg_recall:.3f}")
```

### 出力JSON形式

```json
{
  "project": "nova",
  "evaluation_config": {
    "window_size": 14,
    "sliding_step": 1,
    "random_state": 42
  },
  "summary_statistics": {
    "mean_f1": 0.72,
    "std_f1": 0.08,
    "mean_precision": 0.75,
    "std_precision": 0.09,
    "mean_recall": 0.70,
    "std_recall": 0.10,
    "successful_windows": 180,
    "total_windows": 200
  },
  "results": [
    {
      "window_start": "2024-01-15T00:00:00",
      "window_end": "2024-01-29T00:00:00",
      "train_window_start": "2024-01-01T00:00:00",
      "train_window_end": "2024-01-15T00:00:00",
      "total_samples": 150,
      "train_samples": 450,
      "test_samples": 150,
      "positive_samples": 80,
      "negative_samples": 70,
      "precision": 0.78,
      "recall": 0.72,
      "f1_score": 0.75,
      "evaluation_status": "success"
    },
    {
      "window_start": "2024-01-16T00:00:00",
      "window_end": "2024-01-30T00:00:00",
      "train_window_start": "2024-01-02T00:00:00",
      "train_window_end": "2024-01-16T00:00:00",
      "evaluation_status": "failure",
      "error_message": "Insufficient training data: 8 samples (minimum 10 required)"
    }
  ]
}
```

### パフォーマンスと推奨設定

#### 処理時間の目安
- **小規模** (1プロジェクト、3ヶ月分): ~10-20分
- **中規模** (3プロジェクト、6ヶ月分): ~1-2時間
- **大規模** (6プロジェクト、1年分): ~3-6時間

#### メモリ使用量
- **基本**: ~500MB
- **大規模データ**: ~2-4GB（多数のウィンドウと履歴データの場合）

#### 推奨設定
```python
# 高速テスト実行
evaluator = TemporalModelEvaluator(
    window_size=14,      # 2週間
    sliding_step=7       # 1週間ずつずらす（ウィンドウ数を削減）
)

# 詳細分析
evaluator = TemporalModelEvaluator(
    window_size=14,      # 2週間
    sliding_step=1       # 1日ずつずらす（詳細な時系列分析）
)
```

### 注意事項

1. **初期ウィンドウ**: 分析開始直後の14日間は学習データが不足するため、評価が行えません
2. **時系列的正しさ**: 学習データと評価データが時系列的に分離されているため、データリークがありません
3. **ボットフィルタリング**: `gerrymanderconfig.ini`で定義されたボットのレビューは正負例判定から除外されます
4. **欠損データ**: 特徴量に欠損値がある場合は0で補完されます
5. **期間の定義**: ウィンドウ期間中に「一度でもオープンだった」Changeを対象とします（常にオープンである必要はありません）

### 解釈のポイント

- **時系列的な学習**: 過去14日間のデータで現在を予測するため、実際の運用環境に近い評価が可能です
- **F1スコアの推移**: 時系列でF1が向上している場合、モデルが時間経過とともに改善されていることを示唆します
- **訓練サンプル数**: 過去14日間のデータ量により変動します。多いほど安定したモデルが期待できます
- **正例比率**: 極端に偏っている（0.1未満や0.9超）場合、クラス不均衡の影響に注意が必要です
- **学習期間と評価期間**: 同じ長さ（14日間）ですが、完全に分離されているため、未来を使って過去を予測することはありません

## 🔬 研究・拡張の方向性

- **深層学習**: ニューラルネットワークベースのIRL
- **多目的最適化**: 複数の評価軸の同時考慮
- **強化学習**: 動的な優先順位調整
- **説明可能AI**: より詳細な判断根拠の提供
- **適応的ウィンドウ**: データ量や変動に応じたウィンドウサイズの自動調整
- **オンライン学習**: リアルタイムでの継続的なモデル更新
