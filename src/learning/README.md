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
Balanced Random Forestを用いてウィンドウごとに**履歴データで学習したモデル**を作成し、**現在ウィンドウのデータを予測**することでレビュー優先順位付けモデルを評価します。

### 履歴モデルの学習方式
各ウィンドウについて、以下の3種類の履歴期間を用いてモデルを学習します：

1. **1ヶ月モデル（1m）**: ウィンドウ開始の1ヶ月前まで（30日間）のデータで学習
2. **3ヶ月モデル（3m）**: ウィンドウ開始の3ヶ月前まで（90日間）のデータで学習
3. **6ヶ月モデル（6m）**: ウィンドウ開始の6ヶ月前まで（180日間）のデータで学習

各履歴モデルは、指定された期間内の**過去ウィンドウ**からデータを収集し、Balanced Random Forestで学習した後、**現在ウィンドウのデータに対して予測**を行います。

### 正負例の定義
- **正例（ラベル=1）**: ウィンドウ期間中に1回以上レビューされたChange
- **負例（ラベル=0）**: ウィンドウ期間中に1度もレビューされなかったChange

### 学習と評価のフロー

```
時系列: [過去6ヶ月] ← [過去3ヶ月] ← [過去1ヶ月] ← [現在ウィンドウ]
              ↓              ↓              ↓              ↓
           訓練データ     訓練データ     訓練データ      評価データ
           (6mモデル)     (3mモデル)     (1mモデル)
                ↓              ↓              ↓
                └──────────────┴──────────────┘
                              ↓
                        現在ウィンドウを予測
                     (Precision/Recall/F1を算出)
```

### 評価指標（履歴モデルごと）
各履歴モデル（1m/3m/6m）について、以下の指標を算出します：

- **Precision（適合率）**: 予測した正例のうち実際に正例だった割合
- **Recall（再現率）**: 実際の正例のうち正しく予測できた割合
- **F1 Score（F値）**: PrecisionとRecallの調和平均
- **訓練サンプル数**: 履歴期間から収集されたサンプル数
- **評価サンプル数**: 現在ウィンドウのサンプル数

### 出力ファイル
`data/temporal_evaluation/`ディレクトリに保存：

- `temporal_evaluation_{project}.csv`: ウィンドウごとの評価指標（全履歴モデル含む）
- `temporal_evaluation_{project}.json`: 全体のサマリー統計と履歴モデル別の詳細結果
- `temporal_evaluation_{project}.pdf`: 評価結果の可視化レポート（8つのグラフ）

### 可視化レポートの内容
PDFには以下の8つのグラフが含まれます：

1. **F1スコアの時系列変化（履歴モデル別）**: 1m/3m/6mモデルのF1スコアの推移
2. **Precisionの時系列変化（履歴モデル別）**: 1m/3m/6mモデルのPrecisionの推移
3. **Recallの時系列変化（履歴モデル別）**: 1m/3m/6mモデルのRecallの推移
4. **訓練サンプル数の推移（履歴モデル別）**: 各履歴期間から収集されたサンプル数
5. **総サンプル数の推移**: 各ウィンドウの総Change数
6. **正負例サンプル数の推移**: レビューされた/されなかったChangeの数
7. **訓練データと現在データの比較**: 履歴データの合計と現在ウィンドウのサンプル数
8. **正例比率の推移**: レビューされたChangeの割合

### 履歴データ不足時の動作
- 履歴期間に十分なデータがない場合、そのモデルは作成されず `trained: False` として記録されます
- 訓練データが10サンプル未満、または正負例のどちらかしか存在しない場合も学習をスキップします
- 結果JSONには不足理由（`no_training_data`, `insufficient_samples_or_classes`等）が記録されます

### 使用例

```python
from src.learning.temporal_model_evaluation import run_temporal_model_evaluation

# 全プロジェクトの評価（デフォルトの期間で実行）
results = run_temporal_model_evaluation()

# 特定プロジェクトの評価
results = run_temporal_model_evaluation(projects=['nova', 'neutron'])

# 結果の確認
for project, evaluations in results['evaluation_results'].items():
    for window_result in evaluations:
        print(f"ウィンドウ: {window_result['window_start']}")
        for lookback in ['1m', '3m', '6m']:
            model_info = window_result['models'].get(lookback, {})
            if model_info.get('trained'):
                print(f"  {lookback}: F1={model_info['f1_score']:.3f}")
            else:
                print(f"  {lookback}: 訓練されませんでした ({model_info.get('reason', 'unknown')})")
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
      "total_samples": 150,
      "positive_samples": 80,
      "negative_samples": 70,
      "models": {
        "1m": {
          "trained": true,
          "train_samples": 450,
          "current_samples": 150,
          "precision": 0.78,
          "recall": 0.72,
          "f1_score": 0.75
        },
        "3m": {
          "trained": true,
          "train_samples": 1200,
          "current_samples": 150,
          "precision": 0.82,
          "recall": 0.74,
          "f1_score": 0.78
        },
        "6m": {
          "trained": false,
          "reason": "no_training_data"
        }
      },
      "evaluation_status": "success"
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

1. **初期ウィンドウ**: 分析開始直後のウィンドウは履歴データが不足するため、6mモデルや3mモデルが作成されないことがあります
2. **計算リソース**: 各ウィンドウで3つのモデルを学習するため、単純なウィンドウ評価より処理時間が長くなります
3. **ボットフィルタリング**: `gerrymanderconfig.ini`で定義されたボットのレビューは正負例判定から除外されます
4. **欠損データ**: 特徴量に欠損値がある場合は0で補完されます

### 解釈のポイント

- **1mモデル vs 6mモデル**: 1mモデルは直近のトレンドを反映し、6mモデルは長期的なパターンを学習します
- **F1スコアの推移**: 時系列でF1が向上している場合、履歴データからの学習が効果的であることを示唆します
- **訓練サンプル数**: 多いほど安定したモデルが期待できますが、古いデータの影響も大きくなります
- **正例比率**: 極端に偏っている（0.1未満や0.9超）場合、クラス不均衡の影響に注意が必要です

## 🔬 研究・拡張の方向性

- **深層学習**: ニューラルネットワークベースのIRL
- **多目的最適化**: 複数の評価軸の同時考慮
- **強化学習**: 動的な優先順位調整
- **説明可能AI**: より詳細な判断根拠の提供
- **適応的ウィンドウ**: データ量や変動に応じたウィンドウサイズの自動調整
- **アンサンブル学習**: 複数の履歴モデルを組み合わせた予測
