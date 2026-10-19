"""表から日ごとの集合（TSet）を作る（design.md §2.9）。

範囲 [pool_start, cycle_end] で切り出すと、`build_records(pool_start, cycle_end)` → `build_sets` と同じ集合になる：
行の `t` の日付が「pool_start 以上 cycle_end 以下の計測点の日」に入り、かつ Change の作成が cycle_end 以下の行
（record_builder は作成が cycle_end より後の Change を飛ばす）。LOOKBACK とラベルの観測の終わりは範囲によらない。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from src.analysis.preliminary_analysis.pretrained_encoders.dataset.record_builder import daily_grid
from src.analysis.preliminary_analysis.pretrained_encoders.dataset.set_builder import TSet
from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_set import MODEL_FEATURES
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre


def _select_columns(tab: dict, features: list[str]) -> np.ndarray:
    names = [str(n) for n in tab["feature_names"]]
    pos = {n: i for i, n in enumerate(names)}
    missing = [n for n in features if n not in pos]
    if missing:
        raise KeyError(f"表にない特徴量: {missing}（MODEL_FEATURES は表の 67 個の名前だけにする）")
    return tab["features"][:, [pos[n] for n in features]]


def range_mask(tab: dict, pool_start: datetime | None, cycle_end: datetime | None) -> np.ndarray:
    """範囲で切り出す行（上の決まり）。どちらも None なら全行。"""
    n = len(tab["t"])
    if pool_start is None and cycle_end is None:
        return np.ones(n, dtype=bool)
    grid = daily_grid(pool_start, cycle_end, pre.MEASUREMENT_STEP_DAYS)
    grid_days = np.array([np.datetime64(g, "D") for g in grid], dtype="datetime64[D]")
    days = tab["t"].astype("datetime64[D]")
    return np.isin(days, grid_days) & (tab["created"] <= np.datetime64(cycle_end, "us"))


def day_sets_from_table(tab: dict, pool_start: datetime | None = None, cycle_end: datetime | None = None,
                        features: list[str] | None = None, max_set_size: int | None = None) -> list[TSet]:
    """表の行を t の日付でまとめた集合（t 昇順）。集合の中は (t, 番号) の順（表の並びのまま）。特徴量は numpy の配列。"""
    features = MODEL_FEATURES if features is None else features
    mask = range_mask(tab, pool_start, cycle_end)
    idx = np.flatnonzero(mask)
    feats = _select_columns(tab, features)[idx]
    t = tab["t"][idx]
    days = t.astype("datetime64[D]")
    labels = tab["label"][idx]
    ids = tab["change_id"][idx]
    open_until = tab["open_until"][idx]
    sets: list[TSet] = []
    if len(idx) == 0:
        return sets
    uniq, starts = np.unique(days, return_index=True)      # 表は (t, 番号) の順なので、同じ日は続いている
    bounds = list(starts) + [len(idx)]
    for k, day in enumerate(uniq):
        lo, hi = bounds[k], bounds[k + 1]
        if max_set_size is not None and hi - lo > max_set_size:
            hi = lo + max_set_size
        sets.append(TSet(
            t=day.astype("datetime64[us]").tolist(),
            bin=None,
            feats=feats[lo:hi],
            labels=[float(x) for x in labels[lo:hi]],
            ids=[int(x) for x in ids[lo:hi]],
            times=t[lo:hi].tolist(),
            open_until=open_until[lo:hi].tolist(),          # NaT は None になる（閉じていない）
        ))
    return sets
