"""
Trend Models Analysis - 基底モデルクラス

本モジュールでは、予測モデルの基底クラスを定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ロギング設定
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseModel(ABC):
    """
    予測モデルの基底クラス
    
    全ての予測モデルはこのクラスを継承し、fit, predict, predict_proba メソッドを実装する必要があります。
    """
    
    def __init__(self, model_params: Dict[str, Any] = None):
        """
        Args:
            model_params: モデルのハイパーパラメータ
        """
        self.model_params = model_params or {}
        self.model = None
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        モデルを学習する
        
        Args:
            X: 特徴量行列 (n_samples, n_features)
            y: ラベル配列 (n_samples,)
            
        Returns:
            self: 学習済みモデル
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を行う
        
        Args:
            X: 特徴量行列 (n_samples, n_features)
            
        Returns:
            np.ndarray: 予測ラベル配列 (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う
        
        Args:
            X: 特徴量行列 (n_samples, n_features)
            
        Returns:
            np.ndarray: クラスごとの予測確率 (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        特徴量重要度を取得する
        
        Returns:
            np.ndarray: 特徴量重要度 (n_features,) または None
        """
        pass
    
    @property
    def is_fitted(self) -> bool:
        """モデルが学習済みかどうか"""
        return self._is_fitted
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(is_fitted={self._is_fitted})"
