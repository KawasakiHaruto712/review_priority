"""pointwise の学習/予測（registry。§5.3）。

目的変数の objective により回帰器 / 分類器を切り替える。
- regression: time_to_next_review 等。predict は連続値（予測時間）。
- classification: decision_result（merge=1/reject=0）等。predict は正例(=1)の確率。

モデルは指標に依存しないので、1 反復あたり 1 回学習し、その予測から各指標をまとめて算出する。
"""
from __future__ import annotations

import numpy as np


def _make_lightgbm(seed: int, objective: str):
    import lightgbm as lgb
    cls = lgb.LGBMClassifier if objective == "classification" else lgb.LGBMRegressor
    return cls(random_state=seed, n_estimators=200, n_jobs=-1, verbose=-1)


def _make_random_forest(seed: int, objective: str):
    if objective == "classification":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_estimators=200, n_jobs=-1)
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=seed, n_estimators=200, n_jobs=-1)


MODEL_REGISTRY = {
    "lightgbm": _make_lightgbm,
    "random_forest": _make_random_forest,
}


def train(x_train: np.ndarray, y_train: np.ndarray, model_name: str, seed: int,
          sample_weight: np.ndarray | None = None, objective: str = "regression"):
    """指定モデルを学習して返す。

    objective: "regression"（回帰器）/ "classification"（分類器）。
    sample_weight: 各学習行の重み（None なら等価）。LightGBM/sklearn とも fit が標準対応。
    """
    factory = MODEL_REGISTRY.get(model_name)
    if factory is None:
        raise ValueError(f"未知のモデル: {model_name}（registry: {list(MODEL_REGISTRY)}）")
    if objective not in ("regression", "classification"):
        raise ValueError(f"未対応の objective: {objective}")
    model = factory(seed, objective)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def predict(model, x_eval: np.ndarray, objective: str = "regression") -> np.ndarray:
    """予測を返す。regression=予測値（小さいほど高優先）、classification=正例(1)の確率。"""
    if objective == "classification":
        proba = model.predict_proba(x_eval)
        # 正例(ラベル=1)の列の確率を返す。クラスが1種類しか無い学習時にも耐える。
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
        return np.asarray(proba[:, idx], dtype=float)
    return np.asarray(model.predict(x_eval), dtype=float)
