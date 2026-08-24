"""レコードを「計測点 T の集合」にまとめる（design.md §2, §3）。

集合Transformer の 1 入力単位は「ある T のアクティブ集合（その T の全 Change）」。
record_builder が作った (Change, T) レコードを **同じ T ごとにまとめて 1 集合**にする。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from src.analysis.preliminary_analysis.pretrained_encoders.dataset.record_builder import Record
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants


@dataclass
class TSet:
    """計測点 T の集合（1 スナップショット）。モデルの 1 入力単位。

    feats:   [集合内 Change 数, 15] の特徴（list[list[float]]）
    labels:  [集合内 Change 数] の 0/1 ラベル
    ids:     [集合内 Change 数] の change_id（予測を紐づけ直す用）
    """
    t: datetime
    bin: int
    feats: list[list[float]]
    labels: list[float]
    ids: list[object]

    def __len__(self) -> int:
        return len(self.labels)


def build_sets(records: list[Record], max_set_size: int | None = None) -> list[TSet]:
    """レコードを T ごとにまとめて TSet のリストにする（t 昇順）。

    max_set_size を超える集合は、先頭 max_set_size 件に打ち切る（メモリ対策）。
    """
    by_t: dict[datetime, list[Record]] = defaultdict(list)
    for r in records:
        by_t[r.t].append(r)

    sets: list[TSet] = []
    for t in sorted(by_t):
        recs = by_t[t]
        if max_set_size is not None and len(recs) > max_set_size:
            recs = recs[:max_set_size]
        sets.append(TSet(
            t=t,
            bin=recs[0].bin,
            feats=[r.features for r in recs],
            labels=[float(r.labels[constants.TARGET]) for r in recs],
            ids=[r.change_id for r in recs],
        ))
    return sets


def sets_in_bins(bins: dict[int, list[Record]], bin_indices, max_set_size: int | None = None) -> list[TSet]:
    """指定ビン（複数可）に属するレコードから TSet のリストを作る（下流の d×p 行列で使用）。"""
    if isinstance(bin_indices, int):
        bin_indices = [bin_indices]
    recs: list[Record] = []
    for i in bin_indices:
        recs.extend(bins.get(i, []))
    return build_sets(recs, max_set_size)


def count_records(sets: list[TSet]) -> int:
    """集合群に含まれる (Change, T) レコードの総数。"""
    return sum(len(s) for s in sets)
