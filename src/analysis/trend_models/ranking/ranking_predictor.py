"""
Trend Models Analysis - Ranking predictor
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from src.analysis.trend_models.utils.constants import (
    DEFAULT_RANKING_MODEL_TYPE,
    FEATURE_NAMES,
    RANKING_MODEL_PARAMS,
)

logger = logging.getLogger(__name__)


class RankingPredictor:
    """Pointwise 回帰によりランキングスコアを推定するモデル。"""

    SUPPORTED_MODELS = [
        'random_forest_regressor',
        'gradient_boosting_regressor',
        'svr',
        'linear_regression',
    ]

    def __init__(
        self,
        model_type: str = None,
        model_params: Dict[str, Any] = None,
        feature_names: List[str] = None,
    ):
        self.model_type = model_type or DEFAULT_RANKING_MODEL_TYPE
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"サポートされていない ranking model type: {self.model_type}. "
                f"support={self.SUPPORTED_MODELS}"
            )

        default_params = RANKING_MODEL_PARAMS.get(self.model_type, {})
        if model_params:
            self.model_params = {**default_params, **model_params}
        else:
            self.model_params = default_params

        self.feature_names = feature_names or FEATURE_NAMES
        self.model = None
        self._is_fitted = False
        self._create_model()

    def _create_model(self):
        if self.model_type == 'random_forest_regressor':
            self.model = RandomForestRegressor(**self.model_params)
        elif self.model_type == 'gradient_boosting_regressor':
            self.model = GradientBoostingRegressor(**self.model_params)
        elif self.model_type == 'svr':
            self.model = SVR(**self.model_params)
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression(**self.model_params)
        else:
            raise ValueError(f"不明なモデルタイプ: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, group: Optional[np.ndarray] = None) -> 'RankingPredictor':
        """モデルを学習する。group は将来拡張用で現在は未使用。"""
        if len(X) == 0:
            logger.warning("学習データが空です")
            return self

        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            logger.warning("モデルが学習されていません")
            return np.zeros(len(X), dtype=float)
        return self.model.predict(X)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """ランキングスコア（大きいほど優先）を返す。"""
        pred_target = self.predict(X)
        return -pred_target

    def get_feature_importances(self) -> Optional[np.ndarray]:
        if not self._is_fitted:
            return None

        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        if hasattr(self.model, 'coef_'):
            coef = np.asarray(self.model.coef_)
            if coef.ndim == 1:
                return np.abs(coef)
            return np.abs(coef[0])
        return None

    def get_feature_importance_dict(self) -> Optional[Dict[str, float]]:
        importances = self.get_feature_importances()
        if importances is None:
            return None

        output = {}
        for i, value in enumerate(importances):
            name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            output[name] = float(value)

        return dict(sorted(output.items(), key=lambda item: item[1], reverse=True))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
