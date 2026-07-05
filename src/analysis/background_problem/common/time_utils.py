"""
日時のパースと、所要時間・横軸位置の計算ヘルパー（分析モジュール共通）。

Gerrit の日時は "YYYY-MM-DD HH:MM:SS.fffffffff"（ナノ秒, 9 桁）形式で、
標準の datetime ではそのままパースできないため pandas を使う。
タイムゾーンは持たない（naive）。比較は naive どうしで統一する。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def parse_dt(value) -> datetime | None:
    """Gerrit の日時文字列（ナノ秒付き）を naive な datetime に変換する。

    パースできない / 欠損の場合は None を返す。
    """
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    # tz 情報があれば落として naive に統一する
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.to_pydatetime()


def to_unit(delta: timedelta, unit: str = "hours") -> float:
    """timedelta を指定単位の数値（float）に変換する。"""
    seconds = delta.total_seconds()
    if unit == "hours":
        return seconds / 3600.0
    if unit == "days":
        return seconds / 86400.0
    if unit == "seconds":
        return seconds
    raise ValueError(f"未対応の単位です: {unit}（hours / days / seconds のいずれか）")


def daily_grid(start: datetime, end: datetime, step_days: int = 1) -> list[datetime]:
    """[start, end] を覆う毎日 0:00 の計測点グリッド（step_days 刻み）を返す。

    計測点は「Change が投稿された時刻」ではなく毎日 0 時の定点にする
    （preliminary_analysis/concept_drift_existence と同じ方式）。
    """
    t0 = datetime(start.year, start.month, start.day)  # 0 時に正規化
    grid: list[datetime] = []
    t = t0
    while t <= end:
        if t >= start:
            grid.append(t)
        t += timedelta(days=step_days)
    return grid


def relative_x(
    t: datetime,
    cycle_start: datetime,
    cycle_end: datetime,
    mode: str = "normalized",
) -> float:
    """計測点 t の横軸位置を計算する。

    - "normalized": サイクル内の正規化進捗 [0,1]（0=サイクル開始, 1=リリース日）。
    - "days_until_release": リリースまでの残り日数（cycle_end - t）。
    """
    if mode == "normalized":
        span = (cycle_end - cycle_start).total_seconds()
        if span <= 0:
            return 0.0
        return (t - cycle_start).total_seconds() / span
    if mode == "days_until_release":
        return (cycle_end - t).total_seconds() / 86400.0
    raise ValueError(f"未対応の x_mode です: {mode}")
