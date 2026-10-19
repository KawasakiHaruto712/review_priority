"""レコード生成（毎日 0 時グリッド × アクティブ集合。design.md §2）。

計測点 T = 毎日 0:00:00 の定点グリッド（MEASUREMENT_STEP_DAYS 刻み）。
各 T で Open な Change を 1 レコード (change_id, T, features, label) にする。
Open かどうかは review_utils.open_periods（作成・放棄・復活・マージの記録から作る Open の期間）で判定する（design.md §11.1）。
同一 Change は複数の日次 T に現れる（複数レコード＝実運用忠実）。
加えて、日の途中で Open になった Change（作成・復活）も、その時刻を予測の時点としてその日のレコードにする（design.md §12.4）。
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
from src.utils.bot_detection import BotDetector

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """1 件の (Change, T) レコード。

    labels: 目的変数名 -> ラベル値（本設計は {constants.TARGET: 0.0/1.0} の単一目的変数）。
    t: 予測の時点。0 時のレコードは日の 0 時、日の途中で Open になったレコードはその時刻（design.md §12.4）。
       どの日の集合に入るかは t の日付（0 時に切り捨て）で決まる。
    open_until: t を含む Open の期間の終わり（閉じていなければ None）。文脈のマスクに使う（§12.4）。
    filled: 既存の 15 個のうち、値が出せずに代わりの値を入れた特徴量の名前（§13.2。feature_table の印。モデルには入れない）。
    """
    change_id: object
    t: datetime
    created: datetime
    decision_time: datetime | None  # 最後に閉じた時刻（submitted か最後の放棄）。None なら未決（Open のまま）
    features: list[float]
    labels: dict
    bin: int | None = field(default=None)
    open_until: datetime | None = field(default=None)
    filled: frozenset = field(default=frozenset())


def decision_time(change: dict) -> datetime | None:
    """最後に閉じた時刻（MERGED は submitted、ABANDONED は最後の放棄の記録）。未決なら None。

    旧版は updated（最後の更新）を使っていた。design.md §11.1。
    """
    periods = review_utils.open_periods(change)
    return periods[-1][1] if periods else None


def day_of(t: datetime) -> datetime:
    """t を含む日の 0 時（UTC。集合はこの日ごとにまとめる。design.md §12.4）。"""
    return datetime(t.year, t.month, t.day)


def period_end_at(periods: list[tuple[datetime, datetime | None]], t: datetime) -> datetime | None:
    """t を含む Open の期間の終わり（閉じていなければ None）。t がどの期間にも入らなければ None。"""
    for s, e in periods:
        if s <= t and (e is None or t < e):
            return e
    return None


def first_revision_time(change: dict) -> datetime | None:
    """最初の版（パッチセット）の作成時刻。版の記録がなければ None。"""
    revs = change.get("revisions")
    if not isinstance(revs, dict):
        return None
    times = [parse_dt(r.get("created")) for r in revs.values() if isinstance(r, dict)]
    times = [x for x in times if x is not None]
    return min(times) if times else None


def midday_open_points(change: dict, periods: list[tuple[datetime, datetime | None]]
                       ) -> list[tuple[datetime, datetime | None]]:
    """日の途中で Open になった時点の列 [(予測の時点, その Open の期間の終わり)]（design.md §12.4）。

    - 作成（最初の期間）：予測の時点は「作成時刻」と「最初の版の作成時刻」の遅いほう（版の記録がなければ作成時刻）
    - 復活（2 つ目以降の期間）：復活の時刻のまま
    - ちょうど 0 時に Open になったものは 0 時のレコードとして数えるので、ここには入れない
    - 予測の時点が日をまたぐもの、予測の時点までに閉じたもの（Open の期間の長さが 0 のものを含む）は入れない
    """
    out: list[tuple[datetime, datetime | None]] = []
    for k, (s, e) in enumerate(periods):
        day = day_of(s)
        if s == day:
            continue                                  # 0 時ちょうど → 0 時のレコード
        t = s
        if k == 0:
            first_rev = first_revision_time(change)
            if first_rev is not None and first_rev > t:
                t = first_rev                         # 中身（最初の版）が見えるようになった時点
        if t >= day + timedelta(days=1):
            continue                                  # 日をまたいだ → その日の途中のレコードは作らない
        if e is not None and t >= e:
            continue                                  # 予測の時点までに閉じている（長さ 0 を含む）
        out.append((t, e))
    return out


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
                  detector: BotDetector, all_prs_df: pd.DataFrame,
                  releases_df: pd.DataFrame) -> list[Record]:
    """毎日 0 時グリッド × アクティブ集合から (Change, T) レコードを作る。

    ラベル = 「T から Δ 以内に人間レビューが付くか」の2値。窓が観測末尾を超えて未確定なものは除外。
    中身は iter_records と同じ（それをリストにしたもの。2026-10-18 に分けた）。
    """
    records = [rec for _idx, rec in iter_records(changes, project, pool_start, cycle_end,
                                                  detector, all_prs_df, releases_df)]
    pos = sum(1 for r in records if r.labels[constants.TARGET] == 1.0)
    logger.info(f"[{project}] レコード数: {len(records)}（正例 {pos} = "
                f"{(pos / len(records) * 100 if records else 0):.1f}%, Δ={constants.REVIEW_HORIZON_DAYS}日）")
    return records


def iter_records(changes: list[dict], project: str, pool_start: datetime, cycle_end: datetime,
                 detector: BotDetector, all_prs_df: pd.DataFrame, releases_df: pd.DataFrame):
    """build_records と同じレコードを 1 件ずつ (changes の中の位置, Record) で返す（全件をメモリに溜めないため。
    feature_table が使う。feature_table/design.md §2.3）。並びは Change の順（Change の中では 0 時のレコード、日の途中のレコードの順）。
    """
    grid = daily_grid(pool_start, cycle_end, constants.MEASUREMENT_STEP_DAYS)
    if not grid:
        return
    index = feature_builder.build_index(all_prs_df)
    comp = feature_builder.build_releases_df(releases_df, project)
    lookback = timedelta(days=constants.LOOKBACK_DAYS)
    delta = timedelta(days=constants.REVIEW_HORIZON_DAYS)

    # 観測末尾（グローバルな収集ホライズン）。ここを超える先読みは未確定＝ラベル None。
    data_end = all_prs_df["updated"].max()
    data_end = cycle_end if pd.isna(data_end) else pd.Timestamp(data_end).to_pydatetime()

    grid_days = set(grid)
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
        review_times = review_utils.human_comment_times(change, detector)
        lo = bisect.bisect_left(grid, created)
        cid = change.get("change_number", idx)
        first_rev = first_revision_time(change)
        for t in grid[lo:]:
            if t - created > lookback:
                break  # LOOKBACK 超過（grid 昇順なので打ち切ってよい）
            if dec is not None and t >= dec:
                break  # 最後に閉じた時刻以降は Open でない
            if first_rev is not None and t < first_rev:
                continue  # 最初の版がまだない（中身のない Change）。0 時のレコードも作らない（design.md §12.4。2026-10-18）
            if not review_utils.is_open_at(periods, t):
                continue  # 放棄されてから復活するまでの間（design.md §11.1）
            label = label_builder.reviewed_within_delta(review_times, t, delta, data_end)
            if label is None:
                continue  # 窓が観測末尾を超え未確定 → このレコードは作らない
            filled: set = set()
            feats = feature_builder.build_features(change, t, index, comp, project,
                                                   observation_end=data_end, filled=filled)
            yield idx, Record(cid, t, created, dec, feats, {constants.TARGET: label},
                              open_until=period_end_at(periods, t), filled=frozenset(filled))
        # 日の途中で Open になった Change（作成・復活。design.md §12.4）。
        # ラベルの窓は (予測の時点, その日の終わり]。0 時のレコードの (T, T+Δ] と同じ終わり
        for t_i, end in midday_open_points(change, periods):
            day = day_of(t_i)
            if day not in grid_days or t_i - created > lookback:
                continue
            label = label_builder.reviewed_within_delta(review_times, t_i, day + delta - t_i, data_end)
            if label is None:
                continue
            filled = set()
            feats = feature_builder.build_features(change, t_i, index, comp, project,
                                                   observation_end=data_end, filled=filled)
            yield idx, Record(cid, t_i, created, dec, feats, {constants.TARGET: label},
                              open_until=end, filled=frozenset(filled))
    review_utils.log_fallbacks()
    logger.info(f"[{project}] 計測点 {len(grid)}")
