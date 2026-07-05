"""レコード生成（毎日 0 時グリッド × アクティブ集合。§2.2, §5.4）。

計測点 T = 毎日 0:00:00 の定点グリッド（MEASUREMENT_STEP_DAYS 刻み）。
各 T で Open な Change を 1 レコード (change_id, T, features, label) にする。
同一 Change は複数の日次 T に現れる（推移）。打ち切り（label=None）は除外。
"""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from src.analysis.background_problem.common.time_utils import parse_dt
from src.analysis.preliminary_analysis.concept_drift_existence.features import feature_builder
from src.analysis.preliminary_analysis.concept_drift_existence.labeling import label_builder
from src.analysis.preliminary_analysis.concept_drift_existence.utils import constants, review_utils

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """1 件の (Change, T) レコード。"""
    change_id: object
    t: datetime
    created: datetime
    decision_time: datetime | None  # None なら未決（Open のまま）
    features: list[float]
    label: float
    bin: int | None = field(default=None)  # binning で付与（§5.5）


def decision_time(change: dict) -> datetime | None:
    """マージ/放棄の判断時刻（MERGED/ABANDONED は updated を採用、それ以外は None）。§10。"""
    if change.get("status") in ("MERGED", "ABANDONED"):
        return parse_dt(change.get("updated"))
    return None


def daily_grid(pool_start: datetime, cycle_end: datetime, step_days: int) -> list[datetime]:
    """[pool_start, cycle_end] を覆う毎日 0 時の計測点グリッド（step_days 刻み）。"""
    t0 = datetime(pool_start.year, pool_start.month, pool_start.day)  # 0 時に正規化
    grid: list[datetime] = []
    t = t0
    while t <= cycle_end:
        if t >= pool_start:
            grid.append(t)
        t += timedelta(days=step_days)
    return grid


def _is_active(created: datetime, decision: datetime | None, t: datetime,
               lookback: timedelta) -> bool:
    """T 時点で Open か: T-LOOKBACK <= created <= T < decision_time。"""
    if not (created <= t):
        return False
    if t - created > lookback:
        return False
    if decision is not None and not (t < decision):
        return False
    return True


def build_records(changes: list[dict], project: str, pool_start: datetime, cycle_end: datetime,
                  bot_names: set[str], all_prs_df: pd.DataFrame,
                  releases_df: pd.DataFrame) -> list[Record]:
    """毎日 0 時グリッド × アクティブ集合から (Change, T) レコードを作る。"""
    grid = daily_grid(pool_start, cycle_end, constants.MEASUREMENT_STEP_DAYS)
    if not grid:
        return []
    index = feature_builder.build_index(all_prs_df)
    comp = feature_builder.build_releases_df(releases_df, project)
    lookback = timedelta(days=constants.LOOKBACK_DAYS)

    label_name = constants.LABEL_NAME
    use_ttnr = label_name == "time_to_next_review"  # 既定ラベルは高速経路（人間レビュー時刻をキャッシュ）
    unit_div = label_builder._UNIT_DIV.get(constants.DURATION_UNIT, 3600.0)

    records: list[Record] = []
    for idx, change in enumerate(changes):
        created = parse_dt(change.get("created"))
        if created is None:
            continue
        dec = decision_time(change)
        # 人間レビュー時刻は Change ごとに 1 回だけ集計（各 T で再計算しない）
        review_times = review_utils.human_comment_times(change, bot_names) if use_ttnr else None
        # この Change がアクティブになりうる T の範囲を grid から切り出す（高速化）
        lo = bisect.bisect_left(grid, created)
        cid = change.get("change_number", idx)
        for t in grid[lo:]:
            if t - created > lookback:
                break  # これ以降は LOOKBACK 超過（grid 昇順なので打ち切ってよい）
            if dec is not None and t >= dec:
                break  # 決着以降は Open でない
            # ここまで来れば created<=t かつ LOOKBACK 内かつ Open
            if use_ttnr:
                # T より後の最初の人間レビュー（二分探索）。無ければ打ち切り。
                j = bisect.bisect_right(review_times, t)
                if j >= len(review_times):
                    continue
                label = (review_times[j] - t).total_seconds() / unit_div
            else:
                label = label_builder.build_label(change, t, bot_names, label_name)
                if label is None:
                    continue
            feats = feature_builder.build_features(change, t, index, comp, project)
            records.append(Record(cid, t, created, dec, feats, label))
    logger.info(f"[{project}] レコード数: {len(records)}（計測点 {len(grid)}）")
    return records
