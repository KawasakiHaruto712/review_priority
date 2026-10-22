"""RQ1（時期のずれ）：評価日 X と時期のずれ d から、学習に使う日を返す部品（design.md §3.2）。

時期のずれ×評価時期の精度表（concept_drift_detection）と同じ設定と関数をそのまま使う：
学習の窓 ＝ [X − 1 − 刻み × d − W + 1, X − 1 − 刻み × d]、評価日と評価時期 p は binning.position_of_days、段数は版ごとの N_v。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from src.analysis.preliminary_analysis.concept_drift_detection.dataset import binning
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as det

KIND = "time_distance"
ROW_LABEL = "時期のずれ"


def accuracy_path(project: str, version: str, metric: str):
    """同じセルの精度表（RQ1）。図の題の recall@pos・AUC に使う（2026-10-22：main.py から移した。RQ2 と同じ形にするため）。"""
    return det.OUTPUT_ROOT / project / det.MODEL_NAME[0] / version / "probe" / metric / "drift_matrix.csv"


@dataclass
class TimeDistancePairing:
    """1 版分の設定。rows ＝ 段数（時期のずれ d は 0..rows−1）、positions ＝ {評価日 -> 評価時期 p}。"""
    window: int
    step: int
    rows: int
    ce_R: datetime
    record_start: datetime
    positions: dict = field(default_factory=dict)
    kind: str = KIND
    row_label: str = ROW_LABEL

    def train_days(self, x: date, r: int, available: list) -> list[date]:
        """評価日 x・時期のずれ r の学習の窓に入る日（available＝集合がある日の昇順の列。そこにある日だけ）。"""
        end = x - timedelta(days=1 + self.step * r)
        start = end - timedelta(days=self.window - 1)
        return available[bisect.bisect_left(available, start):bisect.bisect_right(available, end)]


def for_version(rel_df: pd.DataFrame, project: str, version: str, window: int | None = None,
                step: int | None = None) -> TimeDistancePairing:
    """その版の部品を作る（窓長・刻みは既定で concept_drift_detection の設定）。"""
    window = det.window_for(project) if window is None else int(window)
    step = (det.step_for(project) if step is None else int(step))
    cs_R, ce_R, _prev = binning.target_cycles(rel_df, project, version)
    cycle_len = binning.cycle_days(cs_R, ce_R, det.BIN_DAY_ALIGNED)
    rows = max(1, -(-cycle_len // step))                       # ⌈サイクル長 ÷ 刻み⌉（det.grid_for と同じ）
    start = binning.record_start(ce_R, window, step, rows, det.RECORD_MARGIN_DAYS, det.BIN_DAY_ALIGNED)
    positions = binning.position_of_days(ce_R, step, rows, det.BIN_DAY_ALIGNED)
    return TimeDistancePairing(window, step, rows, ce_R, start, positions)
