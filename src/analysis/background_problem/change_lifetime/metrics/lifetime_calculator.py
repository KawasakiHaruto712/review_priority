"""
Change の生存期間（lifetime）の母集団抽出・計算と、ヒストグラム/要約統計の算出（§5.1）。

lifetime = decision_time − created（decision_time は MERGED/ABANDONED の `updated`）。
各 Change は 1 回だけ数える（計測点に依存しない）。
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from src.analysis.background_problem.common.time_utils import parse_dt, to_unit

# 「終了済み」とみなす status（これ以外は未決＝対象外）
DECISION_STATUSES = {"MERGED", "ABANDONED"}


def extract_lifetimes(
    changes: list,
    cycle_start,
    cycle_end,
    statuses: set,
    lookback_days: float,
    unit: str = "days",
) -> tuple[list, list, int]:
    """§2.2 の母集団を抽出し、各 Change の生存期間（指定単位）を返す。

    母集団: `cycle_start - lookback_days <= created <= cycle_end` かつ status が `statuses`。

    Returns:
        (values, change_numbers, n_excluded_unfinished)
        - values: lifetime のリスト（unit 単位、lifetime>0 のみ）
        - change_numbers: 各 value に対応する change_number（混合時の重複排除用）
        - n_excluded_unfinished: 期間内だが未決（NEW 等）で除外した件数
    """
    lower = cycle_start - timedelta(days=lookback_days)

    values: list = []
    change_numbers: list = []
    n_excluded_unfinished = 0

    for ch in changes:
        created = parse_dt(ch.get("created"))
        if created is None or not (lower <= created <= cycle_end):
            continue

        status = ch.get("status")
        if status not in DECISION_STATUSES:
            # 期間内だが未決（lifetime を確定できない）
            n_excluded_unfinished += 1
            continue
        if status not in statuses:
            # このグループの対象外（例: merged グループの ABANDONED）。除外件数には数えない
            continue

        decision = parse_dt(ch.get("updated"))
        if decision is None:
            continue
        delta = decision - created
        if delta.total_seconds() <= 0:
            continue  # updated < created 等の異常

        values.append(to_unit(delta, unit))
        change_numbers.append(ch.get("change_number", ch.get("_number")))

    return values, change_numbers, n_excluded_unfinished


def make_histogram(
    values: list,
    bins: int = 50,
    log_bins: bool = True,
    min_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """頻度ヒストグラムの (counts, edges) を返す。

    log_bins=True のとき、ビン境界を対数等間隔（np.logspace）にする。
    値が無い場合は空配列を返す。
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([])

    if log_bins:
        pos = arr[arr > 0]
        if pos.size == 0:
            return np.array([]), np.array([])
        lo = min_value if (min_value is not None and min_value > 0) else float(pos.min())
        hi = float(pos.max())
        if hi <= lo:
            hi = lo * 10.0  # 退化回避
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        counts, _ = np.histogram(pos, bins=edges)
    else:
        lo = float(arr.min()) if min_value is None else float(min_value)
        hi = float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(arr, bins=edges)

    return counts, edges


def summarize(values: list, percentiles: list) -> dict:
    """n / mean / median / min / max / 各分位点 を返す。"""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "mean": None, "median": None,
            "min": None, "max": None, "percentiles": {},
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "percentiles": {int(q): float(np.percentile(arr, q)) for q in percentiles},
    }
