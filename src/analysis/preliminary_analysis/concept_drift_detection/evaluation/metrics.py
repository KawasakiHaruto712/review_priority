"""評価指標の計算（design.md §8.1）。

正例＝Δ以内にレビュー。評価レコードをプールして算出する。
- auc  : ROC-AUC（しきい値フリー・不均衡に頑健。主指標）
- ap   : Average Precision（PR曲線下面積。不均衡向けの補助）
- f1 / precision / recall : しきい値（既定 0.5）で 2 値化して算出（補助）
片クラスしか無い等で計算不能なときは NaN を返す。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def compute(y_true, y_pred_proba, metric_names, threshold: float = 0.5) -> dict:
    """プールした (y_true, y_pred確率) から指標を計算して {指標名: 値} を返す。"""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_proba, dtype=float)
    out: dict[str, float] = {}
    both_classes = len(np.unique(yt)) >= 2
    yhat = (yp >= threshold).astype(int)

    for m in metric_names:
        try:
            if m == "auc":
                out[m] = float(roc_auc_score(yt, yp)) if both_classes else float("nan")
            elif m == "ap":
                out[m] = float(average_precision_score(yt, yp)) if both_classes else float("nan")
            elif m == "f1":
                out[m] = float(f1_score(yt, yhat, zero_division=0))
            elif m == "precision":
                out[m] = float(precision_score(yt, yhat, zero_division=0))
            elif m == "recall":
                out[m] = float(recall_score(yt, yhat, zero_division=0))
            elif m == "accuracy":
                out[m] = float((yhat == yt).mean())
            else:
                out[m] = float("nan")
        except Exception:
            out[m] = float("nan")
    return out
