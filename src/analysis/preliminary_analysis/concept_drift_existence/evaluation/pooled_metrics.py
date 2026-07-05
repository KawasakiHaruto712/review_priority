"""プールして算出する評価指標（§2.8.2 回帰誤差 / §2.8.3 分類）。

順位指標（ranking_metrics）と違い、これらは **1 レコード単体で計算できる**ので、
評価ビン内の全レコードをまとめて（プールして）一度に算出する（§2.8.4。案1＝レコード重み）。
入力 y_true / y_pred はともに hours 単位（学習が log でも expm1 済み。§8）。

- 回帰誤差（log10 時間上）: mae_log / rmse_log / r2_log
- 分類（順序付きバケツ）: macro_f1 / micro_f1 / qwk
  バケツ境界は hours の昇順リスト（既定 [24, 168, 720] = 1日/1週/1ヶ月。constants.CLASS_BUCKETS_HOURS）。

誤差の底は **常用対数(底10)** log10(1+hours)。差 1 = 10倍、差 2 = 100倍 と直感的に読めるため。
（学習側の log1p=自然対数とは底が違うが、予測 hours から測り直すので影響しない。r2_log は比なので底不変。）
"""
from __future__ import annotations

import numpy as np

REGRESSION_METRICS = {"mae_log", "rmse_log", "r2_log"}
CLASSIFICATION_METRICS = {"macro_f1", "micro_f1", "qwk"}
POOLED_METRICS = REGRESSION_METRICS | CLASSIFICATION_METRICS


# ── 回帰誤差（log10 時間） ──────────────────────────────
def _log(values: np.ndarray) -> np.ndarray:
    """常用対数 log10(1+hours)。差 1 が 10倍、差 2 が 100倍に対応し解釈しやすい。"""
    return np.log10(1.0 + np.asarray(values, dtype=float))


def mae_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """log10 時間上の平均絶対誤差（小さいほど良い）。値 1 ≒ 典型的に 10倍ズレ。"""
    return float(np.mean(np.abs(_log(y_pred) - _log(y_true))))


def rmse_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """log10 時間上の二乗平均平方根誤差（小さいほど良い、大外しに敏感）。"""
    d = _log(y_pred) - _log(y_true)
    return float(np.sqrt(np.mean(d ** 2)))


def r2_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """log 時間上の決定係数（1=完璧 / 0=平均並み / 負=それ以下）。底に依らず同値。分散ゼロなら NaN。"""
    yt, yp = _log(y_true), _log(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")  # 真値の log がすべて同じ＝R² 定義不能
    return 1.0 - ss_res / ss_tot


# ── 分類（順序付きバケツ） ────────────────────────────
def bucketize(values: np.ndarray, buckets: list[float]) -> np.ndarray:
    """hours を昇順境界 buckets で順序付きクラス（0..len(buckets)）に離散化。

    例 buckets=[24,168,720] → 0:≤1日 / 1:≤1週 / 2:≤1ヶ月 / 3:>1ヶ月。
    """
    return np.digitize(np.asarray(values, dtype=float), bins=list(buckets), right=False)


def _classify(y_true, y_pred, buckets):
    n_classes = len(buckets) + 1
    labels = list(range(n_classes))
    yt = bucketize(y_true, buckets)
    yp = bucketize(y_pred, buckets)
    return yt, yp, labels


def macro_f1(y_true, y_pred, buckets) -> float:
    """クラスごと F1 の単純平均（少数クラスも等価。大きいほど良い）。"""
    from sklearn.metrics import f1_score
    yt, yp, labels = _classify(y_true, y_pred, buckets)
    return float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))


def micro_f1(y_true, y_pred, buckets) -> float:
    """全クラス合算 F1（単一ラベル多クラスでは正解率に一致。大きいほど良い）。"""
    from sklearn.metrics import f1_score
    yt, yp, labels = _classify(y_true, y_pred, buckets)
    return float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0))


def qwk(y_true, y_pred, buckets) -> float:
    """二次重み付きカッパ（順序を踏まえた一致度。−1〜1）。真値が1クラスのみなら NaN。"""
    from sklearn.metrics import cohen_kappa_score
    yt, yp, labels = _classify(y_true, y_pred, buckets)
    if len(np.unique(yt)) < 2:
        return float("nan")  # 真値が全部同じクラス＝順序一致を測れない
    return float(cohen_kappa_score(yt, yp, labels=labels, weights="quadratic"))


_REG_FUNCS = {"mae_log": mae_log, "rmse_log": rmse_log, "r2_log": r2_log}
_CLS_FUNCS = {"macro_f1": macro_f1, "micro_f1": micro_f1, "qwk": qwk}


def compute(y_true: np.ndarray, y_pred: np.ndarray, metric_names: list[str],
            buckets: list[float]) -> dict:
    """指定のプール指標をまとめて算出（評価ビン内の全レコードに対して）。"""
    out = {}
    for m in metric_names:
        if m in _REG_FUNCS:
            out[m] = _REG_FUNCS[m](y_true, y_pred)
        elif m in _CLS_FUNCS:
            out[m] = _CLS_FUNCS[m](y_true, y_pred, buckets)
        else:
            raise ValueError(f"未知のプール指標: {m}（{sorted(POOLED_METRICS)}）")
    return out
