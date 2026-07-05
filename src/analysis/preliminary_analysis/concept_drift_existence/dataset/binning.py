"""計測点を時系列ビンに割り当てる（§2.5, §5.5）。

ビン幅 = 当該リリースを BIN_COUNT 等分（bin_width = (cycle_end - cycle_start) / BIN_COUNT）。
同じ幅を前リリース側へも延長する。各レコードは自分の T が入るビンに属する。
当該リリース内ビン index は 0..BIN_COUNT-1（＝位置 p の候補）、前リリース側は負の index。
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from src.analysis.preliminary_analysis.concept_drift_existence.dataset.record_builder import Record


def bin_width_seconds(cycle_start: datetime, cycle_end: datetime, bin_count: int) -> float:
    return (cycle_end - cycle_start).total_seconds() / bin_count


def bin_index_of(t: datetime, cycle_start: datetime, bw_sec: float) -> int:
    """T が属するビン index（0.. が当該リリース、負が前リリース側）。"""
    return math.floor((t - cycle_start).total_seconds() / bw_sec)


def make_bins(records: list[Record], bin_count: int, mode: str,
              cycle_start: datetime, cycle_end: datetime) -> dict[int, list[Record]]:
    """各レコードに bin index を付与し、bin index -> レコード列 の辞書を返す。

    既定 mode="equal_time"（当該リリースの時間等分→同幅で前へ延長）。
    """
    if mode != "equal_time":
        raise NotImplementedError(f"BINNING={mode} は未対応（equal_time のみ）")
    bw = bin_width_seconds(cycle_start, cycle_end, bin_count)
    bins: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        r.bin = bin_index_of(r.t, cycle_start, bw)
        bins[r.bin].append(r)
    return dict(bins)
