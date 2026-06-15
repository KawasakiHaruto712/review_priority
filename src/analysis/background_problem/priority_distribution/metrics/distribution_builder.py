"""
計測点ごとの分布の時系列を構築する（§5.5）。

考え方:
- 「新しい Change が投稿された瞬間」を計測点 T とする。
- 各 T で、その時点で Open な Change 集合（T - lookback <= created <= T < decision_time）を取り出す。
- Open な各 Change の縦軸値（metric.value_fn(change, T)）を集め、計測点ごとに
  (A) 平均±標準偏差 と (B) 分位点（percentiles）の 2 種類の要約を計算する。
- 横軸はリリースサイクル内の相対位置（relative_x）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from src.analysis.background_problem.priority_distribution.utils.time_utils import (
    relative_x,
    to_unit,
)

logger = logging.getLogger(__name__)

# 既定の分位点（constants.PERCENTILES が渡されなかったときのフォールバック）
DEFAULT_PERCENTILES = [10, 30, 50, 70, 90]


@dataclass
class DistributionPoint:
    """分布図の 1 点（1 計測点）。"""

    x: float
    timestamp: datetime
    n_active: int
    mean: float
    std: float
    lower: float
    upper: float
    # {分位点(int): 値(float)} 例: {10: 2.0, 30: 5.5, 50: 9.0, 70: 14.0, 90: 30.0}
    percentiles: dict = field(default_factory=dict)
    release_version: Optional[str] = None


def build_distribution(
    records: list,
    metric,
    bot_names: set,
    cycle_start: datetime,
    cycle_end: datetime,
    x_mode: str = "normalized",
    duration_unit: str = "hours",
    min_active: int = 1,
    band_std_factor: float = 0.5,
    lookback_days: Optional[float] = 365,
    percentiles: Optional[list] = None,
    release_version: Optional[str] = None,
) -> list[DistributionPoint]:
    """計測点ごとの DistributionPoint 列を返す。

    records は drop_unfinished 済み（decision_time が確定している）前提。

    - 計測点 T = サイクル `[cycle_start, cycle_end]` 内に投稿された Change の `created`。
    - アクティブ集合 = `T - lookback <= created <= T < decision_time`
      （T から lookback_days 以内に投稿され、かつ T 時点で Open な Change）。
      `lookback_days=None` なら下限なし（投稿済みかつ Open のすべて）。
    - 各計測点で平均±std と分位点（percentiles）を算出する。
    - 寄与 Change 数が `max(min_active, len(percentiles))` 未満の計測点は出力しない
      （分位線の本数分のデータが無いと各分位線を定義できないため。§2.5）。
    """
    lookback = timedelta(days=lookback_days) if lookback_days is not None else None
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES
    # 計測点を出力する実効最小件数（分位線の本数を下回らせない）
    effective_min = max(min_active, len(percentiles))

    # 計測点 = サイクル内に投稿された Change の created（重複は 1 点に集約）
    measurement_points = sorted(
        {
            r.created
            for r in records
            if r.created is not None and cycle_start <= r.created <= cycle_end
        }
    )

    points: list[DistributionPoint] = []
    for t in measurement_points:
        # T から lookback 以内に投稿され、かつ T 時点で Open な Change 集合（§2.4）
        lower_bound = (t - lookback) if lookback is not None else None
        active = [
            r
            for r in records
            if r.created is not None
            and r.decision_time is not None
            and r.created <= t < r.decision_time
            and (lower_bound is None or r.created >= lower_bound)
        ]

        # Open な各 Change の縦軸値を集める（寄与しないものは None なので除く）
        values: list[float] = []
        for r in active:
            delta = metric.value_fn(r.change, t, bot_names)
            if delta is not None:
                values.append(to_unit(delta, duration_unit))

        if len(values) < effective_min:
            continue

        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))  # n=1 のとき 0 になる（母標準偏差）
        half = band_std_factor * std

        # 分位点（10/30/50/70/90 など）。キーは int、値は float。
        pct_values = np.percentile(arr, percentiles)
        pct = {int(q): float(v) for q, v in zip(percentiles, pct_values)}

        points.append(
            DistributionPoint(
                x=relative_x(t, cycle_start, cycle_end, x_mode),
                timestamp=t,
                n_active=len(values),
                mean=mean,
                std=std,
                lower=mean - half,
                upper=mean + half,
                percentiles=pct,
                release_version=release_version,
            )
        )

    # プロット用に横軸でソートしておく
    points.sort(key=lambda p: p.x)
    logger.info(
        f"分布構築: 計測点 {len(measurement_points)} 件 → 出力点 {len(points)} 件"
        + (f"（{release_version}）" if release_version else "")
    )
    return points


def bin_points(points: list[DistributionPoint], n_bins: int) -> list[DistributionPoint]:
    """全リリース混合図向け: x を等幅ビンに区切り、各ビンの平均で代表点にまとめる（任意）。"""
    if not points or n_bins <= 0:
        return list(points)

    xs = np.array([p.x for p in points])
    edges = np.linspace(xs.min(), xs.max(), n_bins + 1)
    binned: list[DistributionPoint] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # 最終ビンは右端も含める
        if i == n_bins - 1:
            members = [p for p in points if lo <= p.x <= hi]
        else:
            members = [p for p in points if lo <= p.x < hi]
        if not members:
            continue
        means = np.array([p.mean for p in members])
        # 各分位点をビン内で平均する（キーは最初のメンバーの分位点集合に揃える）
        pct_keys = list(members[0].percentiles.keys())
        pct = {
            q: float(np.mean([p.percentiles[q] for p in members if q in p.percentiles]))
            for q in pct_keys
        }
        binned.append(
            DistributionPoint(
                x=float((lo + hi) / 2),
                timestamp=members[len(members) // 2].timestamp,
                n_active=sum(p.n_active for p in members),
                mean=float(means.mean()),
                std=float(means.std(ddof=0)),
                lower=float(np.mean([p.lower for p in members])),
                upper=float(np.mean([p.upper for p in members])),
                percentiles=pct,
                release_version="mixed",
            )
        )
    return binned
