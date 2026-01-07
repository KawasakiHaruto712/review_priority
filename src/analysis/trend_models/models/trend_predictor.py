"""
Trend Models Analysis - 予測モデル

本モジュールでは、レビュー優先度予測のための機械学習モデルを提供します。
"""

import logging
from typing import Dict, Any, Optional, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# TabNetのインポート（オプション）
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

# FT-Transformer用のPyTorchインポート（オプション）
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from src.analysis.trend_models.models.base_model import BaseModel
from src.analysis.trend_models.utils.constants import (
    MODEL_PARAMS,
    DEFAULT_MODEL_TYPE,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# ロギング設定
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TrendPredictor(BaseModel):
    """
    トレンド予測モデル
    
    レビュー優先度（レビューされる/されない）を予測するための分類モデルです。
    RandomForest, GradientBoosting, LogisticRegression, SVM, TabNet, FT-Transformerをサポートします。
    """
    
    SUPPORTED_MODELS = [
        'random_forest', 
        'gradient_boosting', 
        'logistic_regression', 
        'svm',
        'tabnet',
        'ft_transformer'
    ]
    
    def __init__(
        self,
        model_type: str = None,
        model_params: Dict[str, Any] = None,
        feature_names: List[str] = None
    ):
        """
        Args:
            model_type: モデルタイプ ('random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'tabnet', 'ft_transformer')
            model_params: モデルのハイパーパラメータ（Noneの場合はデフォルト値を使用）
            feature_names: 特徴量名のリスト（特徴量重要度の取得に使用）
        """
        self.model_type = model_type or DEFAULT_MODEL_TYPE
        
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"サポートされていないモデルタイプ: {self.model_type}. "
                           f"サポート: {self.SUPPORTED_MODELS}")
        
        # デフォルトパラメータを取得
        default_params = MODEL_PARAMS.get(self.model_type, {})
        
        # ユーザー指定パラメータでオーバーライド
        if model_params:
            params = {**default_params, **model_params}
        else:
            params = default_params
        
        super().__init__(params)
        
        self.feature_names = feature_names or FEATURE_NAMES
        self._create_model()
    
    def _create_model(self):
        """内部モデルを作成する"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**self.model_params)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**self.model_params)
        elif self.model_type == 'svm':
            self.model = SVC(**self.model_params)
        elif self.model_type == 'tabnet':
            if not TABNET_AVAILABLE:
                raise ImportError(
                    "TabNetを使用するにはpytorch-tabnetをインストールしてください: "
                    "pip install pytorch-tabnet"
                )
            # TabNetはfitで初期化するためNoneに設定
            self.model = None
        elif self.model_type == 'ft_transformer':
            if not PYTORCH_AVAILABLE:
                raise ImportError(
                    "FT-Transformerを使用するにはPyTorchをインストールしてください: "
                    "pip install torch"
                )
            # FT-Transformerはfitで初期化するためNoneに設定
            self.model = None
        else:
            raise ValueError(f"不明なモデルタイプ: {self.model_type}")
        
        logger.debug(f"モデルを作成: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TrendPredictor':
        """
        モデルを学習する
        
        Args:
            X: 特徴量行列 (n_samples, n_features)
            y: ラベル配列 (n_samples,)
            
        Returns:
            self: 学習済みモデル
        """
        if len(X) == 0:
            logger.warning("学習データが空です")
            return self
        
        # ラベルのユニーク値をチェック
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            logger.warning(f"ラベルのユニーク値が2未満です: {unique_labels}")
            return self
        
        # 深層学習モデルの場合は専用のfit処理
        if self.model_type == 'tabnet':
            self._fit_tabnet(X, y)
        elif self.model_type == 'ft_transformer':
            self._fit_ft_transformer(X, y)
        else:
            self.model.fit(X, y)
        
        self._is_fitted = True
        
        logger.debug(f"モデル学習完了: n_samples={len(X)}, n_features={X.shape[1]}")
        return self
    
    def _fit_tabnet(self, X: np.ndarray, y: np.ndarray):
        """TabNetを学習する"""
        # 検証データを分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TabNetの初期化
        params = self.model_params.copy()
        max_epochs = params.pop('max_epochs', 100)
        patience = params.pop('patience', 15)
        batch_size = params.pop('batch_size', 256)
        virtual_batch_size = params.pop('virtual_batch_size', 128)
        
        self.model = TabNetClassifier(**params)
        
        # クラス重みを計算（不均衡対策）
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        weights = {0: n_pos / len(y_train), 1: n_neg / len(y_train)}
        
        # 学習
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=weights
        )
    
    def _fit_ft_transformer(self, X: np.ndarray, y: np.ndarray):
        """FT-Transformerを学習する（簡易MLP実装）"""
        # 検証データを分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # パラメータ取得
        params = self.model_params.copy()
        hidden_dims = params.get('hidden_dims', [128, 64, 32])
        dropout_rate = params.get('dropout_rate', 0.3)
        lr = params.get('lr', 1e-3)
        weight_decay = params.get('weight_decay', 1e-4)
        batch_size = params.get('batch_size', 256)
        epochs = params.get('epochs', 100)
        patience = params.get('patience', 15)
        
        # デバイス設定
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        # MLPモデルを構築
        layers = []
        prev_dim = X.shape[1]
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers).to(device)
        self._device = device
        
        # クラス重みを計算（float32を明示的に指定）
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        
        # 損失関数とオプティマイザ
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # データをTensorに変換
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        # 学習ループ
        best_loss = float('inf')
        patience_counter = 0
        
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # 検証損失
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            self.model.train()
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.model.eval()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を行う
        
        Args:
            X: 特徴量行列 (n_samples, n_features)
            
        Returns:
            np.ndarray: 予測ラベル配列 (n_samples,)
        """
        if not self._is_fitted:
            logger.warning("モデルが学習されていません")
            return np.zeros(len(X))
        
        if self.model_type == 'tabnet':
            return self.model.predict(X)
        elif self.model_type == 'ft_transformer':
            return self._predict_ft_transformer(X)
        else:
            return self.model.predict(X)
    
    def _predict_ft_transformer(self, X: np.ndarray) -> np.ndarray:
        """FT-Transformerの予測"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self._device)
            outputs = self.model(X_t).cpu().numpy().flatten()
        return (outputs >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う
        
        Args:
            X: 特徴量行列 (n_samples, n_features)
            
        Returns:
            np.ndarray: クラスごとの予測確率 (n_samples, n_classes)
        """
        if not self._is_fitted:
            logger.warning("モデルが学習されていません")
            return np.zeros((len(X), 2))
        
        if self.model_type == 'tabnet':
            return self.model.predict_proba(X)
        elif self.model_type == 'ft_transformer':
            return self._predict_proba_ft_transformer(X)
        else:
            return self.model.predict_proba(X)
    
    def _predict_proba_ft_transformer(self, X: np.ndarray) -> np.ndarray:
        """FT-Transformerの確率予測"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self._device)
            outputs = self.model(X_t).cpu().numpy().flatten()
        # クラス0と1の確率
        proba = np.column_stack([1 - outputs, outputs])
        return proba
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        特徴量重要度を取得する
        
        Returns:
            np.ndarray: 特徴量重要度 (n_features,) または None
        """
        if not self._is_fitted:
            logger.warning("モデルが学習されていません")
            return None
        
        # TabNetの場合
        if self.model_type == 'tabnet' and TABNET_AVAILABLE:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            return None
        
        # FT-Transformerの場合は特徴量重要度を取得できない
        if self.model_type == 'ft_transformer':
            logger.debug("FT-Transformerでは特徴量重要度を直接取得できません")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # ロジスティック回帰の場合は係数の絶対値を返す
            return np.abs(self.model.coef_[0])
        else:
            return None
    
    def get_feature_importance_dict(self) -> Optional[Dict[str, float]]:
        """
        特徴量名と重要度のペアを辞書として取得する
        
        Returns:
            Dict[str, float]: 特徴量名をキー、重要度を値とする辞書
        """
        importances = self.get_feature_importances()
        if importances is None:
            return None
        
        # 特徴量名と重要度をペアにして辞書化
        importance_dict = {}
        for i, importance in enumerate(importances):
            if i < len(self.feature_names):
                name = self.feature_names[i]
            else:
                name = f'feature_{i}'
            importance_dict[name] = float(importance)
        
        # 重要度でソート
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def reset(self):
        """モデルをリセットする"""
        self._is_fitted = False
        self._create_model()
        logger.debug("モデルをリセットしました")
    
    def __repr__(self) -> str:
        return f"TrendPredictor(model_type={self.model_type}, is_fitted={self._is_fitted})"
