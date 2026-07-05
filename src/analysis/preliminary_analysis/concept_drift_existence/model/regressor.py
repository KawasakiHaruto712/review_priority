"""pointwise 回帰の学習/予測（registry。§5.3）。

モデルは指標に依存しないので、1 反復あたり 1 回学習し、その予測から 3 指標をまとめて算出する。
予測は time_to_next_review の推定時間（小さいほど高優先）。
"""
from __future__ import annotations

import numpy as np

from src.analysis.preliminary_analysis.concept_drift_existence.utils import constants


def _make_lightgbm(seed: int):
    import lightgbm as lgb
    return lgb.LGBMRegressor(random_state=seed, n_estimators=200, n_jobs=-1, verbose=-1)


def _make_random_forest(seed: int):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=seed, n_estimators=200, n_jobs=-1)


MODEL_REGISTRY = {
    "lightgbm": _make_lightgbm,
    "random_forest": _make_random_forest,
}


def train(x_train: np.ndarray, y_train: np.ndarray, model_name: str, seed: int,
          sample_weight: np.ndarray | None = None):
    """指定モデルを学習して返す。

    sample_weight: 各学習行の重み（None なら等価）。LightGBM/RandomForest とも fit が標準対応。
    """
    factory = MODEL_REGISTRY.get(model_name)
    if factory is None:
        raise ValueError(f"未知のモデル: {model_name}（registry: {list(MODEL_REGISTRY)}）")
    model = factory(seed)
    if constants.OBJECTIVE != "regression":
        raise NotImplementedError(f"OBJECTIVE={constants.OBJECTIVE} は未対応（regression のみ）")
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def predict(model, x_eval: np.ndarray) -> np.ndarray:
    """予測時間（小さいほど高優先）を返す。"""
    return np.asarray(model.predict(x_eval), dtype=float)
