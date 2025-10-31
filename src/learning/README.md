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
Balanced Random Forestを用いてウィンドウごとに正負例を定義し、レビュー優先順位付けモデルを評価します。

### 正負例の定義
- **正例（ラベル=1）**: ウィンドウ期間中に1回以上レビューされたPR
- **負例（ラベル=0）**: ウィンドウ期間中に1度もレビューされなかったPR

### データ分割
- PR単位で学習:評価 = 8:2に分割
- `random_state=42`で再現性を確保
- 同じPRが学習と評価の両方に含まれないよう保証

### 評価指標
- **Precision（適合率）**: 予測した正例のうち実際に正例だった割合
- **Recall（再現率）**: 実際の正例のうち正しく予測できた割合  
- **F1 Score（F値）**: PrecisionとRecallの調和平均

### 出力ファイル
`data/temporal_evaluation/`ディレクトリに保存：
- `temporal_evaluation_{project}.csv`: ウィンドウごとの評価指標
- `temporal_evaluation_{project}.json`: 全体のサマリー統計
- `temporal_evaluation_{project}.pdf`: 評価結果の可視化レポート

### 使用例
```python
from src.learning.temporal_model_evaluation import run_temporal_model_evaluation

# 全プロジェクトの評価
results = run_temporal_model_evaluation()

# 特定プロジェクトの評価
results = run_temporal_model_evaluation(projects=['nova', 'neutron'])
```
- **説明可能AI**: より詳細な判断根拠の提供
