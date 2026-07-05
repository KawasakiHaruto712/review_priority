"""漏れない学習/評価分け（層1/2/3・またぎ禁止・固定 Change 数 8:2。§2.6, §5.6）。

セル (i, j)（学習期間=ビン i, 評価期間=ビン j）について:
- 対象 Change を層1/2/3 に分類（ビン i / ビン j にレコードを持つか）。
- 層2（両方に持つ）を Change 単位で排他にランダム振り分け（またぎ禁止）。
- 学習・評価に使う Change 数は固定（n_train : n_eval = 8:2）。各側の内部比は母集団比のまま、
  層2 の振り分け数は固定数から従属的に決める。
- 学習は割り当て側（ビン i）の T レコードのみ、評価はビン j の T レコードのみを使う（反対側は捨てる）。
- 供給不足のセルは None（→ NaN セル）。

数えるのは「Change 数」。同一 Change は複数の T レコードを持つので、実際のレコード数は可変（許容。§2.6）。
"""
from __future__ import annotations

import random

from src.analysis.preliminary_analysis.concept_drift_existence.dataset.record_builder import Record


def _change_ids(records: list[Record]) -> set:
    return {r.change_id for r in records}


def split_train_eval(bins: dict[int, list[Record]], i: int, j: int,
                     n_train: int, n_eval: int, seed: int):
    """セル (i, j) の学習/評価レコードを返す。供給不足なら None。

    Returns: (train_records, eval_records) または None
    """
    recs_i = bins.get(i, [])
    recs_j = bins.get(j, [])
    changes_i = _change_ids(recs_i)
    changes_j = _change_ids(recs_j)

    layer1 = changes_i - changes_j  # 学習期間のみ
    layer2 = changes_i & changes_j  # 両方（またぐ）
    layer3 = changes_j - changes_i  # 評価期間のみ
    n1, n2, n3 = len(layer1), len(layer2), len(layer3)

    # 内部比（母集団比）から各層の取り分を決める（層2 は従属的）
    if n1 + n2 == 0 or n3 + n2 == 0:
        return None
    train_from_l1 = round(n_train * n1 / (n1 + n2))
    train_from_l2 = n_train - train_from_l1
    eval_from_l3 = round(n_eval * n3 / (n3 + n2))
    eval_from_l2 = n_eval - eval_from_l3

    # 供給チェック（足りなければ NaN セル）
    if train_from_l1 > n1 or eval_from_l3 > n3:
        return None
    if train_from_l2 < 0 or eval_from_l2 < 0:
        return None
    if train_from_l2 + eval_from_l2 > n2:
        return None

    rng = random.Random(seed)
    l1 = sorted(layer1, key=str); l2 = sorted(layer2, key=str); l3 = sorted(layer3, key=str)
    rng.shuffle(l1); rng.shuffle(l2); rng.shuffle(l3)

    train_changes = set(l1[:train_from_l1]) | set(l2[:train_from_l2])
    eval_changes = set(l3[:eval_from_l3]) | set(l2[train_from_l2:train_from_l2 + eval_from_l2])

    # 学習はビン i の T レコードのみ、評価はビン j の T レコードのみ（反対側期間の T は捨てる）
    train_records = [r for r in recs_i if r.change_id in train_changes]
    eval_records = [r for r in recs_j if r.change_id in eval_changes]
    return train_records, eval_records
