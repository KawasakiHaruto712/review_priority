"""評価指標（design.md §8.1）。lookback_window と同一の一式（B2＝本ディレクトリに複製）。

- compute_day_metrics : 1 評価日（1 スナップショット）の全指標
- metric_columns      : 指標カラム名の一覧

セル値は「評価日ごとに compute_day_metrics → 位置（列）の日数で平均」で作る（design.md §8.1）。
その集約は drift_matrix 側で行うため、本モジュールは日次指標の算出までを担う。

指標：AUC / precision・recall・f1(0.5) / precision@k・recall@k・f1@k / MAP / 正規化順位 / MRR。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

NAN = float("nan")


def _topk(y_sorted: np.ndarray, k: int, n_pos: int) -> tuple[float, float, float]:
    """スコア降順に並べた y_sorted の上位 k での precision/recall/f1。"""
    n = len(y_sorted)
    if k is None or k <= 0 or n_pos <= 0:
        return NAN, NAN, NAN
    kk = min(k, n)
    hits = float(y_sorted[:kk].sum())
    prec = hits / kk
    rec = hits / n_pos
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_day_metrics(y_true, score, k_list, threshold: float = 0.5) -> dict:
    """1 評価日の全指標を {指標名: 値} で返す。計算不能は NaN。"""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(score, dtype=float)
    n = len(yt)
    n_pos = int(yt.sum())
    n_neg = n - n_pos
    both = n_pos > 0 and n_neg > 0
    yhat = (yp >= threshold).astype(int)

    out: dict[str, float] = {}
    out["auc"] = float(roc_auc_score(yt, yp)) if both else NAN
    out["map"] = float(average_precision_score(yt, yp)) if n_pos > 0 else NAN
    out["precision"] = float(precision_score(yt, yhat, zero_division=0))
    out["recall"] = float(recall_score(yt, yhat, zero_division=0))
    out["f1"] = float(f1_score(yt, yhat, zero_division=0))

    order = np.argsort(-yp, kind="stable")
    y_sorted = yt[order]
    for k, label in [(k, str(k)) for k in k_list] + [(n_pos, "pos")]:
        prec, rec, f1 = _topk(y_sorted, k, n_pos)
        out[f"precision@{label}"] = prec
        out[f"recall@{label}"] = rec
        out[f"f1@{label}"] = f1

    if n_pos > 0 and n > 1:
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)          # 0-indexed rank（0=最上位）
        pos_ranks = ranks[yt == 1]
        out["norm_rank"] = float(np.mean(pos_ranks / (n - 1)))
        out["mrr"] = float(np.max(1.0 / (pos_ranks + 1)))
    else:
        out["norm_rank"] = NAN
        out["mrr"] = NAN
    return out


def metric_columns(k_list) -> list[str]:
    """指標カラム名の一覧（固定順）。"""
    cols = ["auc", "map", "precision", "recall", "f1"]
    for label in [str(k) for k in k_list] + ["pos"]:
        cols += [f"precision@{label}", f"recall@{label}", f"f1@{label}"]
    cols += ["norm_rank", "mrr"]
    return cols
