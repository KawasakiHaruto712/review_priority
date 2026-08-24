"""計測点を「リリースごとの時系列ビン」に割り当てる（design.md §4）。

新方式（per_release）：**各リリースを、そのリリース自身の期間で `BIN_COUNT` 等分**する。
- 対象リリース R の 6 分割 → ローカルビン index 0..5（＝位置 p の候補）。
- 直前リリースの 6 分割 → ローカルビン index -6..-1（-6 が直前リリースの最初のビン、-1 が最後のビン）。
- こうすると **距離 d=6 がちょうど 1 リリース前の同じフェーズ**に対応する（design.md §4）。

境界は 0 時に揃える（`BIN_DAY_ALIGNED`）。リリース日程は `_drop_release_anomalies` で異常エントリを除外。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

from src.analysis.background_problem.common.data_loader import _drop_release_anomalies
from src.analysis.preliminary_analysis.pretrained_encoders.dataset.record_builder import Record


def _midnight(dt: datetime) -> datetime:
    """最も近い 0 時に丸める（12 時以降は翌日 0 時）。"""
    base = datetime(dt.year, dt.month, dt.day)
    return base + timedelta(days=1) if dt.hour >= 12 else base


def _cleaned_release_dates(rel_df: pd.DataFrame, project: str) -> list[pd.Timestamp]:
    """プロジェクトの（異常除外済み）リリース日を昇順で返す。"""
    pdf = _drop_release_anomalies(rel_df[rel_df["project"] == project])
    return sorted(pd.to_datetime(pdf["release_date"]).tolist())


def target_cycles(rel_df: pd.DataFrame, project: str, version: str):
    """対象リリース R と直前リリースの開発サイクル境界を返す。

    返り値: (cs_R, ce_R, cs_prev)
      - cs_R  = 直前リリース日（R のサイクル開始）
      - ce_R  = R のリリース日（R のサイクル終了）
      - cs_prev = さらに 1 つ前のリリース日（直前リリースのサイクル開始）／無ければ None
    """
    dates = _cleaned_release_dates(rel_df, project)
    target = rel_df[(rel_df["project"] == project) & (rel_df["version"] == version)]
    if target.empty:
        raise ValueError(f"リリースが見つかりません: {project} {version}")
    ce_R = pd.to_datetime(target["release_date"].iloc[0])
    earlier = [d for d in dates if d < ce_R]
    if not earlier:
        raise ValueError(f"直前リリースが見つかりません: {project} {version}")
    cs_R = earlier[-1]
    before_cs_R = [d for d in earlier if d < cs_R]
    cs_prev = before_cs_R[-1] if before_cs_R else None
    return cs_R.to_pydatetime(), ce_R.to_pydatetime(), (cs_prev.to_pydatetime() if cs_prev is not None else None)


def bin_edges(cs: datetime, ce: datetime, bin_count: int, day_aligned: bool) -> list[datetime]:
    """[cs, ce] を bin_count 等分する境界（bin_count+1 個）。day_aligned なら各境界を 0 時に丸める。

    丸めで境界が潰れないよう、単調増加（前より最低 1 日後）を保証する。
    """
    span = (ce - cs).total_seconds()
    edges = [cs + timedelta(seconds=span * j / bin_count) for j in range(bin_count + 1)]
    if not day_aligned:
        return edges
    snapped: list[datetime] = []
    for e in edges:
        m = _midnight(e)
        if snapped and m <= snapped[-1]:
            m = snapped[-1] + timedelta(days=1)  # 単調増加を保証
        snapped.append(m)
    return snapped


def _which_bin(t: datetime, edges: list[datetime]) -> int | None:
    """t が edges のどの区間 [edges[j], edges[j+1]) に入るか（0..len-2）。範囲外は None。"""
    if t < edges[0] or t >= edges[-1]:
        return None
    for j in range(len(edges) - 1):
        if edges[j] <= t < edges[j + 1]:
            return j
    return None


def make_local_bins(records: list[Record], rel_df: pd.DataFrame, project: str, version: str,
                    bin_count: int, day_aligned: bool) -> dict[int, list[Record]]:
    """対象リリース R とその直前リリースを各々 6 分割し、各レコードをローカルビンに割り当てる。

    返り値: {ローカルビン index -> レコード列}
      - 0..bin_count-1 : 対象リリース R のビン（位置 p の候補）
      - -bin_count..-1 : 直前リリースのビン（-bin_count が直前リリース最初のビン）
    """
    cs_R, ce_R, cs_prev = target_cycles(rel_df, project, version)
    edges_R = bin_edges(cs_R, ce_R, bin_count, day_aligned)
    edges_prev = bin_edges(cs_prev, cs_R, bin_count, day_aligned) if cs_prev is not None else None

    bins: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        j = _which_bin(r.t, edges_R)
        if j is not None:
            r.bin = j                      # 0..bin_count-1（対象リリース）
            bins[j].append(r)
            continue
        if edges_prev is not None:
            jp = _which_bin(r.t, edges_prev)
            if jp is not None:
                idx = jp - bin_count       # -bin_count..-1（直前リリース）
                r.bin = idx
                bins[idx].append(r)
    return dict(bins)


def pool_start(rel_df: pd.DataFrame, project: str, version: str) -> datetime:
    """レコード生成に使う遡り開始点＝直前リリースのサイクル開始（無ければ対象サイクル開始）。

    d×p 行列で参照する最古の学習ビンは「直前リリースの最初のビン」なので、そこまで遡れば十分。
    """
    cs_R, ce_R, cs_prev = target_cycles(rel_df, project, version)
    return cs_prev if cs_prev is not None else cs_R
