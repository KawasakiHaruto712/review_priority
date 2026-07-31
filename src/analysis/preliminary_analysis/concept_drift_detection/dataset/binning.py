"""計測点を時系列ビンに割り当てる（§2.5, §5.5）。

ビン幅 = 当該リリースを BIN_COUNT 等分（bin_width = (cycle_end - cycle_start) / BIN_COUNT）。
同じ幅を前リリース側へも延長する。各レコードは自分の T が入るビンに属する。
当該リリース内ビン index は 0..BIN_COUNT-1（＝位置 p の候補）、前リリース側は負の index。

BIN_DAY_ALIGNED=True のときは、基準開始を 0 時に丸め、ビン幅を整数日に切り上げて、
全ビン境界を毎日 0 時の計測点グリッドと一致させる（Δ=MEASUREMENT_STEP 日なら除外が厳密に 0）。
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from src.analysis.preliminary_analysis.concept_drift_detection.dataset.record_builder import Record
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants


def bin_width_seconds(cycle_start: datetime, cycle_end: datetime, bin_count: int) -> float:
    return (cycle_end - cycle_start).total_seconds() / bin_count


def resolve_bin_params(cycle_start: datetime, cycle_end: datetime,
                       bin_count: int) -> tuple[datetime, float]:
    """ビンの基準開始時刻とビン幅(秒)を返す（make_bins と行列側で共通に使う）。

    BIN_DAY_ALIGNED=True: 開始を 0 時に丸め、ビン幅を整数日に切り上げる。全境界が 0 時に揃い、
      毎日 0 時の計測点と一致 → Δ=MEASUREMENT_STEP 日なら学習末尾マージンの除外が 0 に。
    False: 従来どおり秒単位の等時間分割。
    """
    if not getattr(constants, "BIN_DAY_ALIGNED", False):
        return cycle_start, bin_width_seconds(cycle_start, cycle_end, bin_count)
    cs_day = datetime(cycle_start.year, cycle_start.month, cycle_start.day)  # 0 時へ
    span_days = (cycle_end - cs_day).total_seconds() / 86400.0
    bw_days = max(1, math.ceil(span_days / bin_count))  # 末尾がはみ出ないよう切り上げ
    return cs_day, float(bw_days * 86400)


def bin_index_of(t: datetime, cycle_start: datetime, bw_sec: float) -> int:
    """T が属するビン index（0.. が当該リリース、負が前リリース側）。"""
    return math.floor((t - cycle_start).total_seconds() / bw_sec)


def make_bins(records: list[Record], bin_count: int, mode: str,
              cycle_start: datetime, cycle_end: datetime) -> dict[int, list[Record]]:
    """各レコードに bin index を付与し、bin index -> レコード列 の辞書を返す。

    既定 mode="equal_time"（当該リリースの時間等分→同幅で前へ延長）。
    日丸めでビン幅を切り上げた結果、末尾が index==bin_count にはみ出た分は最終ビンへ寄せる
    （データ喪失防止。負の index はそのまま前リリース側）。
    """
    if mode != "equal_time":
        raise NotImplementedError(f"BINNING={mode} は未対応（equal_time のみ）")
    cs, bw = resolve_bin_params(cycle_start, cycle_end, bin_count)
    bins: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        idx = bin_index_of(r.t, cs, bw)
        if idx >= bin_count:          # 日丸めの端数で末尾を超えた分は最終ビンへ
            idx = bin_count - 1
        r.bin = idx
        bins[idx].append(r)
    return dict(bins)
