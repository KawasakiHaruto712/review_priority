"""学習/評価データの抽出（§2.6, §2.7）。

旧設計の層1/2/3・排他振り分けは廃止。同一 Change が学習・評価に跨る（混ぜ）ことを許容する。
- 学習: 学習ビンの Change から `n_train` 個を固定数サンプリング（比較可能性のため）。
- 評価: 評価ビンの Change を全件（n_eval="all"）、または `n_eval` 個をサンプリング。
数えるのは Change 数。選んだ Change の（その側のビンの）全レコードを入れる。
Δ マージンによる末尾除外は呼び出し側（drift_matrix, ビン境界を持つ）で適用済みのレコードを受け取る。
"""
from __future__ import annotations

import random


def _distinct_changes(records) -> list:
    seen, out = set(), []
    for r in records:
        if r.change_id not in seen:
            seen.add(r.change_id)
            out.append(r.change_id)
    return out


def sample_train_eval(train_recs, eval_recs, n_train, n_eval,
                      min_train: int, min_eval: int, seed: int):
    """セルの学習/評価レコードを返す。供給不足なら None。

    Returns: (train_records, eval_records) または None
    - 学習: n_train 個の Change を無作為抽出（供給 < min_train なら None）。
    - 評価: n_eval=="all" で全件、数値ならその数の Change を抽出（供給 < min_eval なら None）。
    - 混ぜ許容: 学習と評価で同じ Change が出てもよい（排他にしない）。
    """
    rng = random.Random(seed)

    tr_changes = _distinct_changes(train_recs)
    if len(tr_changes) < min_train:
        return None
    ev_changes = _distinct_changes(eval_recs)
    if len(ev_changes) < min_eval:
        return None

    tr_changes = sorted(tr_changes, key=str)
    rng.shuffle(tr_changes)
    sel_tr = set(tr_changes[:n_train])
    train_out = [r for r in train_recs if r.change_id in sel_tr]

    if n_eval == "all" or n_eval is None:
        eval_out = list(eval_recs)
    else:
        ev = sorted(ev_changes, key=str)
        rng.shuffle(ev)
        sel_ev = set(ev[:n_eval])
        eval_out = [r for r in eval_recs if r.change_id in sel_ev]

    return train_out, eval_out


def sample_train(train_recs, n_train, min_train: int, seed: int):
    """学習ビンから n_train 個の Change を抽出したレコード列（供給 < min_train なら None）。

    `sample_train_eval` の学習側と**同一結果**（同 seed で決定的、eval と独立）。
    層(1/2/3)廃止で学習済みモデルは学習ビンだけで決まるため、ここで 1 回サンプル→学習し、
    距離 d 方向のセル間でモデルを使い回せる（drift_matrix 側）。
    """
    tr_changes = _distinct_changes(train_recs)
    if len(tr_changes) < min_train:
        return None
    rng = random.Random(seed)
    tr_changes = sorted(tr_changes, key=str)
    rng.shuffle(tr_changes)
    sel = set(tr_changes[:n_train])
    return [r for r in train_recs if r.change_id in sel]


def sample_eval(eval_recs, n_eval, min_eval: int, seed: int):
    """評価ビンのレコード列。n_eval=="all" で全件、数値ならその数の Change を抽出（供給 < min_eval なら None）。

    n_eval=="all"（既定）では seed 非依存＝全件。数値時は seed で決定的に抽出。
    """
    ev_changes = _distinct_changes(eval_recs)
    if len(ev_changes) < min_eval:
        return None
    if n_eval == "all" or n_eval is None:
        return list(eval_recs)
    rng = random.Random(seed)
    ev = sorted(ev_changes, key=str)
    rng.shuffle(ev)
    sel = set(ev[:n_eval])
    return [r for r in eval_recs if r.change_id in sel]
