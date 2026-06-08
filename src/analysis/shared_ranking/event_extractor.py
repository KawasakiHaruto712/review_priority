"""
計測イベント抽出。

design.md §5.1 に基づき、各 change から
  - created イベント（revision_number=1）
  - revision_update イベント（_number > 1 の各 revision）
を抽出する。draft / WIP リビジョンも含める（要確認事項 §12.1: 含める方針）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeasurementEvent:
    """1 つの計測イベント = 1 つのスナップショットを生む。"""

    event_type: str  # 'created' | 'revision_update'
    measurement_time: datetime
    trigger_change_number: int
    trigger_revision_number: int


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Gerrit 形式（'YYYY-MM-DD HH:MM:SS.nnnnnnnnn' 等）含む日時を datetime 化。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None

    # 9 桁ナノ秒を 6 桁に丸める
    if "." in text:
        head, frac = text.split(".", 1)
        frac = frac[:6]
        text = f"{head}.{frac}"
    text = text.replace(" ", "T")

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None


def _get_change_number(change: Dict[str, Any]) -> Optional[int]:
    number = change.get("change_number")
    if number is None:
        number = change.get("_number")
    if number is None:
        return None
    try:
        return int(number)
    except (TypeError, ValueError):
        return None


def extract_events_from_change(change: Dict[str, Any]) -> List[MeasurementEvent]:
    """1 つの change から計測イベント列を抽出する。"""
    change_number = _get_change_number(change)
    if change_number is None:
        return []

    created_dt = _parse_datetime(change.get("created"))
    if created_dt is None:
        return []

    events: List[MeasurementEvent] = [
        MeasurementEvent(
            event_type="created",
            measurement_time=created_dt,
            trigger_change_number=change_number,
            trigger_revision_number=1,
        )
    ]

    revisions = change.get("revisions", {}) or {}
    if isinstance(revisions, dict):
        revision_iter: Iterable[Dict[str, Any]] = revisions.values()
    elif isinstance(revisions, list):
        revision_iter = revisions
    else:
        revision_iter = []

    for rev in revision_iter:
        if not isinstance(rev, dict):
            continue
        rev_number = rev.get("_number") or rev.get("revision_number")
        try:
            rev_number = int(rev_number) if rev_number is not None else None
        except (TypeError, ValueError):
            rev_number = None
        if rev_number is None or rev_number <= 1:
            # _number == 1 は created と重複するので除外
            continue

        rev_created = _parse_datetime(rev.get("created"))
        if rev_created is None:
            continue
        # 安全側: created より前のリビジョン時刻は無視
        if rev_created < created_dt:
            continue

        events.append(
            MeasurementEvent(
                event_type="revision_update",
                measurement_time=rev_created,
                trigger_change_number=change_number,
                trigger_revision_number=rev_number,
            )
        )

    return events


def extract_events_in_period(
    all_changes: List[Dict[str, Any]],
    period_start: datetime,
    period_end: datetime,
) -> List[MeasurementEvent]:
    """
    全 change から、measurement_time が [period_start, period_end) の計測イベントを
    時系列昇順で返す。
    """
    events: List[MeasurementEvent] = []
    for change in all_changes:
        for ev in extract_events_from_change(change):
            if ev.measurement_time < period_start:
                continue
            if ev.measurement_time >= period_end:
                continue
            events.append(ev)

    events.sort(key=lambda e: (e.measurement_time, e.trigger_change_number, e.trigger_revision_number))
    logger.info(
        "計測イベント抽出: period=[%s, %s), events=%d",
        period_start,
        period_end,
        len(events),
    )
    return events
