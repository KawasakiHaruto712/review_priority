"""「次の人間レビュー」の抽出（design.md §2）と Open の期間。

ボット・作成者本人の判定は src/utils/bot_detection.py にまとめてある（設計書 src/utils/bot_detection.md）。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

from src.analysis.background_problem.common.time_utils import parse_dt
from src.utils.bot_detection import BotDetector, is_human_review_message

logger = logging.getLogger(__name__)


# ── 人間のレビューの時刻 ─────────────────────────────────
def human_comment_times(change: dict, detector: BotDetector) -> list[datetime]:
    """Change の人間レビューコメント時刻の一覧（昇順）。判定は bot_detection.md §3。"""
    times: list[datetime] = []
    for message in change.get("messages", []) or []:
        if isinstance(message, dict) and is_human_review_message(message, change, detector):
            dt = parse_dt(message.get("date"))
            if dt is not None:
                times.append(dt)
    times.sort()
    return times


def next_human_review_after(change: dict, t: datetime, detector: BotDetector) -> datetime | None:
    """計測時点 t より後の「次の人間レビューコメント」時刻。無ければ None。"""
    for dt in human_comment_times(change, detector):
        if dt > t:
            return dt
    return None


# ── Open の期間（design.md §11.1） ───────────────────────────
# 放棄・復活の記録の 1 行目。6 件の収集データでは次の 2 つの書き方だけだった（2026-10 確認）。
#   "Abandoned" / "Patch Set 3: Abandoned"、"Restored" / "Patch Set 3: Restored"
_ABANDON_RE = re.compile(r"^(Patch Set \d+:\s*)?Abandoned")
_RESTORE_RE = re.compile(r"^(Patch Set \d+:\s*)?Restored")

# updated で代用した Change（取れなかった理由 -> change_number の集合）。実行の最後にログへ出す。
# open_periods は同じ Change について何度も呼ばれるので、件数ではなく Change の集合で持つ
fallback_changes: dict[str, set] = {}


def _note_fallback(reason: str, change: dict) -> None:
    fallback_changes.setdefault(reason, set()).add(change.get("change_number", change.get("_number")))


def _first_line(message: dict) -> str:
    return str(message.get("message", "")).split("\n", 1)[0].strip()


def merged_time(change: dict) -> datetime | None:
    """マージされた時刻（`submitted`）。MERGED 以外は None。

    `submitted` が無いときだけ `updated` で代用する（nova では 31,463 件中 2 件）。
    """
    if change.get("status") != "MERGED":
        return None
    dt = parse_dt(change.get("submitted"))
    if dt is None:
        _note_fallback("MERGED に submitted なし", change)
        dt = parse_dt(change.get("updated"))
    return dt


def open_periods(change: dict) -> list[tuple[datetime, datetime | None]]:
    """Change が Open だった期間の列 [(開いた時刻, 閉じた時刻 or None), ...]（時刻順）。

    作成で開き、放棄の記録で閉じ、復活の記録で開き、マージ（submitted）で閉じる。
    期間は [開いた時刻, 閉じた時刻) で、閉じた時刻が None なら最後まで Open（状態 NEW）。
    旧版は「作成 〜 updated」の 1 期間だった。updated はマージ後の通知や CI の結果で遅れ、
    放棄から復活までの閉じている間も Open とみなしていた。
    """
    created = parse_dt(change.get("created"))
    if created is None:
        return []

    events: list[tuple[datetime, str]] = []
    for m in change.get("messages", []) or []:
        if not isinstance(m, dict):
            continue
        line = _first_line(m)
        kind = "close" if _ABANDON_RE.match(line) else "open" if _RESTORE_RE.match(line) else None
        if kind is None:
            continue
        dt = parse_dt(m.get("date"))
        if dt is not None:
            events.append((dt, kind))
    events.sort(key=lambda e: e[0])

    status = change.get("status")
    if status == "MERGED":
        events.append((merged_time(change), "close"))
    elif status == "ABANDONED" and not any(k == "close" for _, k in events):
        # 放棄の記録が見つからない（6 件の収集データでは 0 件）
        _note_fallback("ABANDONED に放棄の記録なし", change)
        events.append((parse_dt(change.get("updated")) or created, "close"))

    periods: list[tuple[datetime, datetime | None]] = []
    start: datetime | None = created
    for dt, kind in events:
        if kind == "close" and start is not None:
            periods.append((start, max(dt, start)))
            start = None
        elif kind == "open" and start is None:
            start = dt
    if start is not None:
        if status in ("MERGED", "ABANDONED"):
            # 最後が復活で終わっているのに状態は閉じている（記録の矛盾）。updated で閉じる
            _note_fallback("閉じた Change が復活で終わっている", change)
            periods.append((start, max(parse_dt(change.get("updated")) or start, start)))
        else:
            periods.append((start, None))
    return periods


def is_open_at(periods: list[tuple[datetime, datetime | None]], t: datetime) -> bool:
    """t が Open の期間 [開いた時刻, 閉じた時刻) のどれかに入るか。"""
    return any(s <= t and (e is None or t < e) for s, e in periods)


def log_fallbacks() -> None:
    """updated で代用した件数をログに出す（design.md §11.1）。"""
    if fallback_changes:
        logger.warning("Open の期間で updated を代用: " + ", ".join(f"{k} {len(v)} 件" for k, v in fallback_changes.items()))
