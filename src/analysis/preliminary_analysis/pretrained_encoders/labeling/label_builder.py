"""ラベルの計算（design.md §2）。

目的変数は「計測点 T から Δ 以内に人間レビューが付くか」の2値。
  正例(1.0) = [T, T+Δ] に人間レビューあり
  負例(0.0) = レビュー無し かつ 窓 [T, T+Δ] を完全に観測できている（T+Δ <= data_end）
  None      = レビュー無し かつ 窓が観測末尾を超える（T+Δ > data_end）→ 未確定なので除外

`review_times`（その Change の人間レビュー時刻・昇順）は record_builder が Change ごとに 1 回だけ
集計して渡す（毎 T 再計算を避ける高速経路）。
"""
from __future__ import annotations

import bisect
from datetime import datetime, timedelta


def reviewed_within_delta(review_times: list[datetime], t: datetime,
                          delta: timedelta, data_end: datetime) -> float | None:
    """[T, T+Δ] に人間レビューが付くかの2値ラベル。未確定は None（design.md §2）。

    review_times: その Change の人間レビュー時刻（昇順）。
    """
    j = bisect.bisect_right(review_times, t)  # T より後の最初の人間レビュー
    if j < len(review_times) and review_times[j] <= t + delta:
        return 1.0
    if t + delta <= data_end:
        return 0.0
    return None
