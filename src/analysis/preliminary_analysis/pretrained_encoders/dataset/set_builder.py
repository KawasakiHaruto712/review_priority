"""レコードを「計測点 T の集合」にまとめる（design.md §2, §3）。

集合Transformer の 1 入力単位は「ある T のアクティブ集合（その T の全 Change）」。
record_builder が作った (Change, T) レコードを **同じ日ごとにまとめて 1 集合**にする
（日の途中で Open になったレコードも、その日の集合に入る。design.md §12.4）。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from src.analysis.preliminary_analysis.pretrained_encoders.dataset.record_builder import Record, day_of
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants


@dataclass
class TSet:
    """計測点 T の集合（1 スナップショット）。モデルの 1 入力単位。

    t:          集合の日の 0 時（集合は日ごとにまとめる。design.md §12.4）
    feats:      [集合内 Change 数, 特徴量の数] の特徴（list[list[float]]）
    labels:     [集合内 Change 数] の 0/1 ラベル
    ids:        [集合内 Change 数] の change_id（予測を紐づけ直す用）
    times:      [集合内 Change 数] の予測の時点（文脈のマスク用。None なら全員が互いに見える旧来の扱い）
    open_until: [集合内 Change 数] の、予測の時点を含む Open の期間の終わり（閉じていなければ None）
    """
    t: datetime
    bin: int
    feats: list[list[float]]
    labels: list[float]
    ids: list[object]
    times: list[datetime] | None = None
    open_until: list[datetime | None] | None = None

    def __len__(self) -> int:
        return len(self.labels)


def build_sets(records: list[Record], max_set_size: int | None = None) -> list[TSet]:
    """レコードを T ごとにまとめて TSet のリストにする（t 昇順）。

    max_set_size を超える集合は、先頭 max_set_size 件に打ち切る（メモリ対策）。
    """
    by_day: dict[datetime, list[Record]] = defaultdict(list)
    for r in records:
        by_day[day_of(r.t)].append(r)    # 予測の時点ではなく日ごとにまとめる（design.md §12.4）

    sets: list[TSet] = []
    for day in sorted(by_day):
        # 0 時のレコードが先、途中で Open になったものは時刻順。同じ時刻は番号の順（2026-10-18 改めた：表から作る集合と並びを
        # そろえるため。feature_table/design.md §2.9。並びは自己注意の結果に効かず、浮動小数点の足し順だけが変わる）
        recs = sorted(by_day[day], key=lambda r: (r.t, r.change_id))
        if max_set_size is not None and len(recs) > max_set_size:
            recs = recs[:max_set_size]
        sets.append(TSet(
            t=day,
            bin=recs[0].bin,
            feats=[r.features for r in recs],
            labels=[float(r.labels[constants.TARGET]) for r in recs],
            ids=[r.change_id for r in recs],
            times=[r.t for r in recs],
            open_until=[r.open_until for r in recs],
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
