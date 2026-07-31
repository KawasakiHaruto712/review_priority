"""2値分類の学習/予測（registry。§4）。

目的変数は「Δ以内にレビューされたか」の2値。predict は正例(=1)の確率を返す。
LightGBM / RandomForest とも SHAP（TreeExplainer）対応、sample_weight 対応。
"""
from __future__ import annotations

import numpy as np


def _make_lightgbm(seed: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(random_state=seed, n_estimators=200, n_jobs=-1, verbose=-1)


def _make_random_forest(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=seed, n_estimators=200, n_jobs=-1)


MODEL_REGISTRY = {
    "lightgbm": _make_lightgbm,
    "random_forest": _make_random_forest,
}


def train(x_train: np.ndarray, y_train: np.ndarray, model_name: str, seed: int,
          sample_weight: np.ndarray | None = None):
    """指定モデル（2値分類器）を学習して返す。sample_weight は各学習行の重み（None なら等価）。"""
    factory = MODEL_REGISTRY.get(model_name)
    if factory is None:
        raise ValueError(f"未知のモデル: {model_name}（registry: {list(MODEL_REGISTRY)}）")
    model = factory(seed)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def predict_proba(model, x_eval: np.ndarray) -> np.ndarray:
    """正例(=1)の確率を返す。学習時に1クラスしか無かった場合にも耐える。"""
    proba = model.predict_proba(x_eval)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
    return np.asarray(proba[:, idx], dtype=float)
