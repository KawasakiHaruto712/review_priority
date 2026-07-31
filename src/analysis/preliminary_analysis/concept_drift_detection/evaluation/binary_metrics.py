"""二値分類の評価指標（§2.8）。目的変数＝「Δ以内にレビューされたか」（正例=1）。

入力 y_true は 0/1、y_pred は正例(1)の確率。しきい値でクラス化してから算出する。
評価ビン内の全レコードをプールして 1 値にする。sample_weight で Change 均等重み等にも対応。
"""
from __future__ import annotations

import numpy as np

BINARY_METRICS = {"mcc", "f1", "precision", "recall", "accuracy"}


def compute(y_true: np.ndarray, y_pred_proba: np.ndarray, metric_names: list[str],
            threshold: float = 0.5, sample_weight: np.ndarray | None = None) -> dict:
    """指定の二値指標をまとめて算出（正例＝1）。不定義は NaN。

    sample_weight: 各レコードの重み（None なら等価）。EVAL_CHANGE_BALANCED_WEIGHT 用。
    """
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)

    yt = np.asarray(y_true, dtype=int)
    yp = (np.asarray(y_pred_proba, dtype=float) >= threshold).astype(int)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    out = {}
    for m in metric_names:
        if m == "accuracy":
            out[m] = float(accuracy_score(yt, yp, sample_weight=sw))
        elif m == "precision":
            out[m] = float(precision_score(yt, yp, zero_division=0, sample_weight=sw))
        elif m == "recall":
            out[m] = float(recall_score(yt, yp, zero_division=0, sample_weight=sw))
        elif m == "f1":
            out[m] = float(f1_score(yt, yp, zero_division=0, sample_weight=sw))
        elif m == "mcc":
            if len(np.unique(yt)) < 2:
                out[m] = float("nan")
            else:
                out[m] = float(matthews_corrcoef(yt, yp, sample_weight=sw))
        else:
            raise ValueError(f"未知の二値指標: {m}（{sorted(BINARY_METRICS)}）")
    return out
