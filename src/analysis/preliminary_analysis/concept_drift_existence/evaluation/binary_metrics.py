"""二値分類の評価指標（§2.8.5。目的変数 decision_result = merge(1)/reject(0) 用）。

入力 y_true は 0/1、y_pred は正例(1)の確率。しきい値 0.5 でクラス化してから算出する。
評価ビン内の全レコードをプールして 1 値にする（§2.8.4 と同じ集約）。正例＝merge=1。
"""
from __future__ import annotations

import numpy as np

BINARY_METRICS = {"mcc", "f1", "precision", "recall", "accuracy"}


def compute(y_true: np.ndarray, y_pred_proba: np.ndarray, metric_names: list[str],
            threshold: float = 0.5) -> dict:
    """指定の二値指標をまとめて算出（正例＝1）。全て 1 件からでも計算可（不定義は NaN）。"""
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)

    yt = np.asarray(y_true, dtype=int)
    yp = (np.asarray(y_pred_proba, dtype=float) >= threshold).astype(int)
    out = {}
    for m in metric_names:
        if m == "accuracy":
            out[m] = float(accuracy_score(yt, yp))
        elif m == "precision":
            out[m] = float(precision_score(yt, yp, zero_division=0))
        elif m == "recall":
            out[m] = float(recall_score(yt, yp, zero_division=0))
        elif m == "f1":
            out[m] = float(f1_score(yt, yp, zero_division=0))
        elif m == "mcc":
            # 真値が1クラスのみ等で分母が 0 になると sklearn は 0 を返す。定義不能は NaN に寄せる。
            if len(np.unique(yt)) < 2:
                out[m] = float("nan")
            else:
                out[m] = float(matthews_corrcoef(yt, yp))
        else:
            raise ValueError(f"未知の二値指標: {m}（{sorted(BINARY_METRICS)}）")
    return out
