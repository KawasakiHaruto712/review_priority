"""ラベルの計算（§2.3, §5.2）。

既定ラベルは time_to_next_review（計測時点 T から次の人間レビューまでの時間）。
将来 time_to_decision などを足せるよう registry 形式にしておく。
"""
from __future__ import annotations

from datetime import datetime

from src.analysis.preliminary_analysis.concept_drift_existence.utils import constants
from src.analysis.preliminary_analysis.concept_drift_existence.utils import review_utils

# 時間単位への換算係数（timedelta.total_seconds() に掛ける）
_UNIT_DIV = {"hours": 3600.0, "days": 86400.0, "minutes": 60.0, "seconds": 1.0}


def _to_unit(seconds: float, unit: str) -> float:
    return seconds / _UNIT_DIV.get(unit, 3600.0)


def time_to_next_review(change: dict, t: datetime, bot_names: set[str]) -> float | None:
    """T から次の人間レビューまでの時間（DURATION_UNIT）。次が無ければ None（打ち切り）。

    観測窓は呼び出し側で渡す change の messages 全体（収集末尾まで）に依存する（§2.3）。
    """
    nxt = review_utils.next_human_review_after(change, t, bot_names)
    if nxt is None:
        return None
    return _to_unit((nxt - t).total_seconds(), constants.DURATION_UNIT)


# ラベル registry（名前 -> 計算関数）。
LABEL_REGISTRY = {
    "time_to_next_review": time_to_next_review,
}


def build_label(change: dict, t: datetime, bot_names: set[str],
                label_name: str | None = None) -> float | None:
    """指定ラベルを計算する。打ち切り（該当イベント無し）は None。

    同一 Change でも計測点 T ごとに値が変わる（推移）。レコード (Change, T) 単位で呼ぶ。
    """
    name = label_name or constants.LABEL_NAME
    fn = LABEL_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"未知のラベル: {name}（registry: {list(LABEL_REGISTRY)}）")
    return fn(change, t, bot_names)
