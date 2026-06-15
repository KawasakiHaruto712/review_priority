"""
外れ値・放置 Change の除外（§5.4）。

判定はすべて各 Change の screening_duration（per-Change の代表所要時間）に対して行う。
「ずっと放置されている Change」が分析の邪魔にならないよう、
① 未決/放置（screening 未定義）の除外、② screening 値の両側外れ値除外、の 2 段で除く。
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 外れ値除去を行う最小レコード数。
# これ未満（リリース内の有効 Change が極端に少ない）では分位点（Q1/Q3）が不安定で
# 過剰除去・誤除去を招くため、外れ値除去をスキップする（全件を残す）。
# 注意: この件数しきい値による「スキップ」も分析パイプラインの一部であり、
#       論文等で外れ値処理を記述する際はこの条件も併記すること（§5.4 / README 参照）。
MIN_RECORDS_FOR_OUTLIER = 4


def drop_unfinished(records: list) -> list:
    """decision_time または screening_duration が未定義の Change を除外する。

    - decision_time is None … 未決（まだ Open のまま）
    - screening_duration is None … 放置（例: 一度も人間レビューが無い）
    """
    kept = [
        r
        for r in records
        if r.decision_time is not None and r.screening_duration is not None
    ]
    logger.info(f"未決/放置を除外: {len(records)} → {len(kept)} 件")
    return kept


def drop_outliers(
    records: list,
    method: str = "iqr",
    iqr_k: float = 1.5,
    percentile: float = 99.0,
) -> tuple[list, list]:
    """screening_duration に対し両側で外れ値を除外する。

    Returns:
        (kept, dropped) のタプル。
    """
    if method == "none":
        return list(records), []
    if len(records) < MIN_RECORDS_FOR_OUTLIER:
        # データが少なすぎると分位点が不安定なので除外しない（全件を残す）。
        logger.warning(
            f"レコードが {len(records)} 件（< {MIN_RECORDS_FOR_OUTLIER}）のため "
            f"外れ値除去をスキップしました（全件保持）。"
        )
        return list(records), []

    values = np.array([r.screening_duration.total_seconds() for r in records])

    if method == "iqr":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low = q1 - iqr_k * iqr
        high = q3 + iqr_k * iqr
    elif method == "percentile":
        low = np.percentile(values, 100.0 - percentile)
        high = np.percentile(values, percentile)
    else:
        raise ValueError(f"未対応の外れ値除去 method です: {method}")

    kept, dropped = [], []
    for r, v in zip(records, values):
        if low <= v <= high:
            kept.append(r)
        else:
            dropped.append(r)
    logger.info(
        f"外れ値除去({method}, 両側): {len(records)} → {len(kept)} 件"
        f"（除外 {len(dropped)} 件, 範囲=[{low:.1f}, {high:.1f}] 秒）"
    )
    return kept, dropped
