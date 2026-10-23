"""IG を計算する Change の標本（design.md §11.7。2026-10-23。§6.4 の「全件」の例外）。

- 評価日ごとに、正例は全件、負例は単純無作為抽出（非復元）。正例・負例で分けて抜く層別抽出の形。
- 負例の件数 n ＝ ⌈n₀ ÷（1 ＋（n₀ − 1）÷ N）⌉（n₀ ＝ 384 で信頼度 95%・許容誤差 ±5%。有限母集団修正つき）。N ≤ n ならその日は全件。
- 乱数の種は project と評価日から CRC32 で決める（Python の hash は実行ごとに変わるので使わない）。
  同じ日なら、どのモデル・行・ヘッドでも、RQ1 と RQ2 でも同じ Change が選ばれる。
"""
from __future__ import annotations

import math
import zlib
from datetime import date

import numpy as np


def neg_sample_size(n_neg: int, n0: int) -> int:
    """その日の負例 N 件から抜く件数（有限母集団修正つき。N 以下）。"""
    if n_neg <= 0:
        return 0
    return min(n_neg, math.ceil(n0 / (1 + (n0 - 1) / n_neg)))


def day_seed(project: str, day: date) -> int:
    """project と評価日から決まる乱数の種。"""
    return zlib.crc32(f"{project}|{day.isoformat()}".encode("utf-8"))


def select_items(labels, project: str, day: date, n0: int) -> list[int]:
    """IG を計算する Change の位置（集合の中の 0 始まりの番号。昇順）。正例は全件、負例は標本。"""
    y = np.asarray(labels, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n = neg_sample_size(len(neg), n0)
    if n < len(neg):
        neg = np.random.default_rng(day_seed(project, day)).choice(neg, size=n, replace=False)
    return sorted(int(i) for i in np.concatenate([pos, neg]))
