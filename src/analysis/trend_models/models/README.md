# Models - 予測モデルモジュール

レビュー優先度予測のための機械学習モデルを提供します。

## モジュール構成

| ファイル | 説明 |
|---------|------|
| `base_model.py` | 予測モデルの基底抽象クラス |
| `trend_predictor.py` | 各種機械学習モデルの実装 |

## base_model.py

### BaseModel（抽象基底クラス）

全ての予測モデルが継承する基底クラス。

#### 抽象メソッド

| メソッド | 説明 |
|---------|------|
| `fit(X, y)` | モデルを学習 |
| `predict(X)` | 予測を行う |
| `predict_proba(X)` | 確率予測を行う |
| `get_feature_importances()` | 特徴量重要度を取得 |

#### プロパティ

| プロパティ | 説明 |
|-----------|------|
| `is_fitted` | モデルが学習済みかどうか |

## trend_predictor.py

### TrendPredictor

レビュー優先度を予測する分類モデル。

#### サポートするモデル

| モデル識別子 | アルゴリズム | 特徴量重要度 |
|-------------|-------------|-------------|
| `random_forest` | RandomForestClassifier | ✅ |
| `gradient_boosting` | GradientBoostingClassifier | ✅ |
| `logistic_regression` | LogisticRegression | ✅（係数の絶対値） |
| `svm` | SVC | ❌ |
| `tabnet` | TabNetClassifier | ✅ |
| `ft_transformer` | MLP (PyTorch) | ❌ |

#### 使用例

```python
from src.analysis.trend_models.models.trend_predictor import TrendPredictor
import numpy as np

# モデルを作成
predictor = TrendPredictor(model_type='random_forest')

# 学習
X_train = np.random.randn(100, 16)
y_train = np.random.randint(0, 2, 100)
predictor.fit(X_train, y_train)

# 予測
X_test = np.random.randn(10, 16)
predictions = predictor.predict(X_test)
probabilities = predictor.predict_proba(X_test)

# 特徴量重要度を取得
importances = predictor.get_feature_importances()
importance_dict = predictor.get_feature_importance_dict()
```

#### コンストラクタ引数

| 引数 | 型 | 説明 | デフォルト |
|------|---|------|-----------|
| `model_type` | str | モデルタイプ | `DEFAULT_MODEL_TYPE` |
| `model_params` | Dict | ハイパーパラメータ | `MODEL_PARAMS`から取得 |
| `feature_names` | List[str] | 特徴量名リスト | `FEATURE_NAMES` |

#### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `fit(X, y)` | モデルを学習する |
| `predict(X)` | 予測ラベルを返す |
| `predict_proba(X)` | クラスごとの予測確率を返す |
| `get_feature_importances()` | 特徴量重要度の配列を返す |
| `get_feature_importance_dict()` | 特徴量名と重要度のペアを辞書で返す |
| `reset()` | モデルをリセットする |

### 深層学習モデルの注意点

#### TabNet
- `pytorch-tabnet`パッケージが必要
- 学習時に検証データを20%で分割
- Early stoppingを使用（patience=15）
- クラス重み付けによる不均衡対策

#### FT-Transformer
- `torch`パッケージが必要
- 3層MLPとして実装（128-64-32）
- BatchNorm、Dropout、Early stoppingを使用
- CUDA/MPS/CPUを自動選択
