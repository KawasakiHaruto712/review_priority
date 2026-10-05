"""レコード生成（毎日 0 時グリッド × アクティブ集合。design.md §2）。

計測点 T = 毎日 0:00:00 の定点グリッド（MEASUREMENT_STEP_DAYS 刻み）。
各 T で Open な Change を 1 レコード (change_id, T, features, label) にする。
Open かどうかは review_utils.open_periods（作成・放棄・復活・マージの記録から作る Open の期間）で判定する（design.md §11.1）。
同一 Change は複数の日次 T に現れる（複数レコード＝実運用忠実）。
ラベル = Δ以内レビューの2値。未確定（窓が観測末尾超）のレコードは作らない。
"""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from src.analysis.background_problem.common.time_utils import parse_dt
from src.analysis.preliminary_analysis.pretrained_encoders.features import feature_builder
from src.analysis.preliminary_analysis.pretrained_encoders.labeling import label_builder
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants, review_utils

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """1 件の (Change, T) レコード。

    labels: 目的変数名 -> ラベル値（本設計は {constants.TARGET: 0.0/1.0} の単一目的変数）。
    """
    change_id: object
    t: datetime
    created: datetime
    decision_time: datetime | None  # 最後に閉じた時刻（submitted か最後の放棄）。None なら未決（Open のまま）
    features: list[float]
    labels: dict
    bin: int | None = field(default=None)


def decision_time(change: dict) -> datetime | None:
    """最後に閉じた時刻（MERGED は submitted、ABANDONED は最後の放棄の記録）。未決なら None。

    旧版は updated（最後の更新）を使っていた。design.md §11.1。
    """
    periods = review_utils.open_periods(change)
    return periods[-1][1] if periods else None


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


def build_records(changes: list[dict], project: str, pool_start: datetime, cycle_end: datetime,
                  bot_names: set[str], all_prs_df: pd.DataFrame,
                  releases_df: pd.DataFrame) -> list[Record]:
    """毎日 0 時グリッド × アクティブ集合から (Change, T) レコードを作る。

    ラベル = 「T から Δ 以内に人間レビューが付くか」の2値。窓が観測末尾を超えて未確定なものは除外。
    """
    grid = daily_grid(pool_start, cycle_end, constants.MEASUREMENT_STEP_DAYS)
    if not grid:
        return []
    index = feature_builder.build_index(all_prs_df)
    comp = feature_builder.build_releases_df(releases_df, project)
    lookback = timedelta(days=constants.LOOKBACK_DAYS)
    delta = timedelta(days=constants.REVIEW_HORIZON_DAYS)

    # 観測末尾（グローバルな収集ホライズン）。ここを超える先読みは未確定＝ラベル None。
    data_end = all_prs_df["updated"].max()
    data_end = cycle_end if pd.isna(data_end) else pd.Timestamp(data_end).to_pydatetime()

    records: list[Record] = []
    for idx, change in enumerate(changes):
        created = parse_dt(change.get("created"))
        if created is None or created > cycle_end:
            continue
        # 最後の更新が期間の始まりより前なら、期間中は一度も Open でない（updated は閉じた時刻以降なので安全）。
        # 記録をたどる前に飛ばして速くする（結果は変わらない）
        updated = parse_dt(change.get("updated"))
        if updated is not None and updated < grid[0] and change.get("status") in ("MERGED", "ABANDONED"):
            continue
        periods = review_utils.open_periods(change)
        dec = periods[-1][1] if periods else None
        review_times = review_utils.human_comment_times(change, bot_names)
        lo = bisect.bisect_left(grid, created)
        cid = change.get("change_number", idx)
        for t in grid[lo:]:
            if t - created > lookback:
                break  # LOOKBACK 超過（grid 昇順なので打ち切ってよい）
            if dec is not None and t >= dec:
                break  # 最後に閉じた時刻以降は Open でない
            if not review_utils.is_open_at(periods, t):
                continue  # 放棄されてから復活するまでの間（design.md §11.1）
            label = label_builder.reviewed_within_delta(review_times, t, delta, data_end)
            if label is None:
                continue  # 窓が観測末尾を超え未確定 → このレコードは作らない
            feats = feature_builder.build_features(change, t, index, comp, project)
            records.append(Record(cid, t, created, dec, feats, {constants.TARGET: label}))
    review_utils.log_fallbacks()
    pos = sum(1 for r in records if r.labels[constants.TARGET] == 1.0)
    logger.info(f"[{project}] レコード数: {len(records)}（計測点 {len(grid)}, 正例 {pos} = "
                f"{(pos / len(records) * 100 if records else 0):.1f}%, Δ={constants.REVIEW_HORIZON_DAYS}日）")
    return records
