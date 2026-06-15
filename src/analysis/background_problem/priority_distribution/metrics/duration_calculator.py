"""
縦軸メトリクスの定義（このパッケージの「縦軸の柔軟さ」の中心）。

各メトリクスは 3 つの関数の組（MetricDefinition）として定義する:
- decision_fn  : アクティブ判定に使う「Change が閉じる時刻」(= decision_time)。両指標共通。
- screening_fn : 外れ値・放置除外に使う「Change 1 件あたりの代表所要時間」。
- value_fn     : 計測点 T における縦軸値（寄与しないなら None）。

新しい縦軸変数を増やしたいときは、関数を足して METRIC_REGISTRY に登録するだけでよい。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from src.analysis.background_problem.priority_distribution.utils.data_loader import (
    human_comment_times,
)
from src.analysis.background_problem.priority_distribution.utils.time_utils import parse_dt

# Change が「判断済み」とみなすステータス（§2.3）
DECISION_STATUSES = {"MERGED", "ABANDONED"}


@dataclass
class MetricDefinition:
    """1 つの縦軸メトリクスの定義。"""

    name: str
    # アクティブ判定に使う decision_time（未決なら None）。
    decision_fn: Callable[[dict], Optional[datetime]]
    # 外れ値・放置除外用の per-Change 代表所要時間（未定義なら None → 除外）。
    screening_fn: Callable[[dict, set], Optional[timedelta]]
    # 計測点 T における縦軸値（寄与しないなら None）。
    value_fn: Callable[[dict, datetime, set], Optional[timedelta]]


@dataclass
class ChangeRecord:
    """1 Change の中間表現（計測に必要な確定値だけを持つ）。"""

    change: dict
    change_number: object
    created: Optional[datetime]
    decision_time: Optional[datetime]
    screening_duration: Optional[timedelta]


# ── 基本ヘルパー ───────────────────────────────────────
def created(change: dict) -> Optional[datetime]:
    """Change の投稿時刻。"""
    return parse_dt(change.get("created"))


def decision_time(change: dict) -> Optional[datetime]:
    """マージ/リジェクト判断時刻（§2.3）。

    確定済み（MERGED / ABANDONED）なら updated を判断時刻として採用する。
    それ以外（未決）は None。
    """
    if change.get("status") not in DECISION_STATUSES:
        return None
    return parse_dt(change.get("updated"))


def first_human_review(change: dict, bot_names: set) -> Optional[datetime]:
    """最初の人間レビューコメントの時刻（無ければ None）。"""
    times = human_comment_times(change, bot_names)
    return times[0] if times else None


def next_human_review_after(
    change: dict, t: datetime, bot_names: set
) -> Optional[datetime]:
    """計測点 t より後（date > t）の、最初の人間レビューコメントの時刻（無ければ None）。"""
    for dt in human_comment_times(change, bot_names):
        if dt > t:
            return dt
    return None


# ── 各メトリクスの screening / value ───────────────────────
def _next_review_screening(change: dict, bot_names: set) -> Optional[timedelta]:
    """time_to_next_review の放置判定用: 最初の人間レビューまでの待ち時間。"""
    first = first_human_review(change, bot_names)
    start = created(change)
    if first is None or start is None:
        return None
    return first - start


def _next_review_value(change: dict, t: datetime, bot_names: set) -> Optional[timedelta]:
    """time_to_next_review の縦軸値: t 以降の最初の人間レビュー - t。"""
    nxt = next_human_review_after(change, t, bot_names)
    if nxt is None:
        return None
    return nxt - t


def _decision_screening(change: dict, bot_names: set) -> Optional[timedelta]:
    """time_to_decision の放置判定用: 投稿起点の全所要時間（T 非依存）。"""
    dt = decision_time(change)
    start = created(change)
    if dt is None or start is None:
        return None
    return dt - start


def _decision_value(change: dict, t: datetime, bot_names: set) -> Optional[timedelta]:
    """time_to_decision の縦軸値: 計測点 t から判断までの残り時間。"""
    dt = decision_time(change)
    if dt is None:
        return None
    return dt - t


# ── レジストリ ────────────────────────────────────────
METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "time_to_next_review": MetricDefinition(
        name="time_to_next_review",
        decision_fn=decision_time,
        screening_fn=_next_review_screening,
        value_fn=_next_review_value,
    ),
    "time_to_decision": MetricDefinition(
        name="time_to_decision",
        decision_fn=decision_time,
        screening_fn=_decision_screening,
        value_fn=_decision_value,
    ),
}


def get_metric(name: str) -> MetricDefinition:
    """名前から MetricDefinition を取得する。"""
    if name not in METRIC_REGISTRY:
        raise KeyError(
            f"未知のメトリクスです: {name}（利用可能: {list(METRIC_REGISTRY)}）"
        )
    return METRIC_REGISTRY[name]


def compute_change_records(
    changes: list[dict], metric: MetricDefinition, bot_names: set
) -> list[ChangeRecord]:
    """各 Change について created / decision_time / screening_duration を確定させる。

    縦軸値そのもの（value_fn）は計測点 T に依存するため、ここでは計算しない
    （distribution_builder 側で T ごとに計算する）。
    """
    records: list[ChangeRecord] = []
    for change in changes:
        records.append(
            ChangeRecord(
                change=change,
                change_number=change.get("change_number", change.get("_number")),
                created=created(change),
                decision_time=metric.decision_fn(change),
                screening_duration=metric.screening_fn(change, bot_names),
            )
        )
    return records
