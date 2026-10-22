"""1 版分の計算（design.md §3.1）：評価日 X × 行 r ごとに、評価日の集合と学習の日の集合を特徴量ごと・群ごとに比べる。

pairing は「評価日と行から学習の日を返す部品」（RQ1 は pairing.time_distance、RQ2 は pairing.feature_distance。design.md §3.2）。
必要な属性：rows（行の数）・positions（{評価日 -> 評価時期 p}）・train_days(x, r, available)。
あれば使う属性：pretrain_days（事前学習の表にある日の集合。class_counts の n_days_train_pretrain。2026-10-22）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.preliminary_analysis.data_drift.stats.compare import compare
from src.analysis.preliminary_analysis.data_drift.utils import constants

_STAT_KEYS = ["n_eval", "n_train", "u", "p", "delta", "median_eval", "median_train"]


def _group_mask(y: np.ndarray, group: str) -> np.ndarray:
    if group == "pos":
        return y == 1
    if group == "neg":
        return y == 0
    return np.ones(len(y), dtype=bool)


def run_version(day_sets: dict, pairing, feature_names: list[str], groups=constants.GROUPS):
    """(daily, class_counts) の DataFrame を返す。

    daily        : day・row・position・feature・group・n_eval・n_train・u・p・delta・median_eval・median_train
    class_counts : day・row・position・正例／負例の件数と割合（評価側・学習側）
    """
    available = sorted(day_sets)
    eval_days = [d for d in available if d in pairing.positions]
    n_feat = len(feature_names)
    cols: dict[str, list] = {k: [] for k in ["day", "row", "position", "feature", "group"] + _STAT_KEYS}
    counts: list[dict] = []
    for x in eval_days:
        es = day_sets[x]
        ex = np.asarray(es.feats, dtype=np.float64)
        ey = np.asarray(es.labels, dtype=float)
        p = pairing.positions[x]
        for r in range(pairing.rows):
            tdays = pairing.train_days(x, r, available)
            if tdays:
                tx = np.vstack([np.asarray(day_sets[d].feats, dtype=np.float64) for d in tdays])
                ty = np.concatenate([np.asarray(day_sets[d].labels, dtype=float) for d in tdays])
            else:
                tx, ty = np.zeros((0, n_feat)), np.zeros(0)
            counts.append(_count_row(x, r, p, ey, ty, len(tdays))
                          | _time_cols(x, tdays, getattr(pairing, "pretrain_days", None)))
            if len(ty) == 0:
                continue                          # 学習の日がない（データの始まりより前）。比べない
            for g in groups:
                me, mt = _group_mask(ey, g), _group_mask(ty, g)
                st = compare(ex[me], tx[mt])
                cols["day"] += [x.isoformat()] * n_feat
                cols["row"] += [r] * n_feat
                cols["position"] += [p] * n_feat
                cols["feature"] += list(feature_names)
                cols["group"] += [g] * n_feat
                for k in _STAT_KEYS:
                    cols[k].extend(st[k].tolist())
    return pd.DataFrame(cols), pd.DataFrame(counts)


def _time_cols(x, tdays, pretrain_days) -> dict:
    """学習の日が時間的にどこか（design.md §4。2026-10-22）：事前学習の区間の日数・評価日の何日前かの中央値。
    事前学習の区間の日の一覧を持たない部品（RQ1）では事前学習の区間の日数は NaN。学習の日がなければ両方 NaN。"""
    if not tdays:
        return {"n_days_train_pretrain": np.nan, "median_days_before": np.nan}
    n_pre = np.nan if pretrain_days is None else int(sum(d in pretrain_days for d in tdays))
    return {"n_days_train_pretrain": n_pre, "median_days_before": float(np.median([(x - d).days for d in tdays]))}


def _count_row(x, r, p, ey, ty, n_days_train) -> dict:
    def rate(a, b):
        return a / (a + b) if (a + b) else np.nan
    pe, ne = int((ey == 1).sum()), int((ey == 0).sum())
    pt, nt = int((ty == 1).sum()), int((ty == 0).sum())
    return {"day": x.isoformat(), "row": r, "position": p, "n_days_train": n_days_train,
            "n_pos_eval": pe, "n_neg_eval": ne, "n_pos_train": pt, "n_neg_train": nt,
            "pos_rate_eval": rate(pe, ne), "neg_rate_eval": rate(ne, pe),
            "pos_rate_train": rate(pt, nt), "neg_rate_train": rate(nt, pt)}
