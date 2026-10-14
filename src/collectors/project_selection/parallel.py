"""
手元のファイルを読む処理の並列化（design.md §4.4）

第 4 段階（E1〜E3 の算出）と検算は、収集した Change のファイルを全部読む。
時間のほとんどは JSON の解読（CPU の仕事）なので、ファイルの一覧をかたまりに分けて
別々の処理に同時に読ませ、数えた結果をあとで足し合わせる。件数は分けて数えて足しても
同じ値になるので、結果は 1 本で読んだ場合と同じになる。
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 進み具合をログに出す間隔（ファイル数。design.md §4.4）
PROGRESS_EVERY_FILES = 50_000

# 1 回に 1 つの処理へ渡すファイルの数。小さすぎると受け渡しの手間が増え、
# 大きすぎると最後のかたまりを待つ時間が延びる
CHUNK_FILES = 2_000


def _cgroup_cpu_quota() -> Optional[int]:
    """コンテナなどでかけられた CPU の上限（cgroup v2 の cpu.max）。無ければ None。"""
    try:
        fields = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except OSError:
        return None
    if len(fields) < 2 or fields[0] == "max":
        return None
    try:
        quota, period = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    if period <= 0:
        return None
    return max(1, quota // period)


def os_core_ceiling() -> Tuple[int, str]:
    """OS から読める、この処理が実際に使ってよいコアの数と、その出どころ（design.md §4.4 の 1）。

    Linux では割り当てられたコア（sched_getaffinity）と cgroup の上限を読む。
    読めない環境（macOS など）ではコアの総数（os.cpu_count）を使う。
    """
    if hasattr(os, "sched_getaffinity"):
        cores, source = len(os.sched_getaffinity(0)), "sched_getaffinity"
    else:
        cores, source = os.cpu_count() or 1, "os.cpu_count"
    quota = _cgroup_cpu_quota()
    if quota is not None and quota < cores:
        cores, source = quota, "cgroup cpu.max"
    return max(1, cores), source


def decide_workers(requested: Optional[int], reserved: int, max_workers: int) -> Tuple[int, str]:
    """並列に読む本数と、その根拠を返す（design.md §4.4）。

    1. OS から読めるコアの数を天井とし、どの指定もこれを超えない。
    2. requested（--workers）があればそれを使う。
    3. 無ければ「天井 − reserved」と max_workers の小さいほう。最低 1 本。
    """
    ceiling, source = os_core_ceiling()
    if requested is not None:
        workers = max(1, min(requested, ceiling))
        reason = f"--workers {requested}（天井 {ceiling}：{source}）"
    else:
        workers = max(1, min(ceiling - reserved, max_workers))
        reason = f"天井 {ceiling}（{source}）− 残すコア {reserved} と上限 {max_workers} の小さいほう"
    return workers, reason


def map_files(label: str, paths: Sequence[Path], worker: Callable[..., Any],
              extra: tuple, workers: int) -> Iterator[Any]:
    """paths をかたまりに分けて worker(かたまり, *extra) に渡し、各かたまりの結果を返す。

    workers が 1 なら同じ処理の中で順に読む。結果の順番はかたまりの順とは限らない
    （足し合わせる前提なので順番は問わない）。進み具合を PROGRESS_EVERY_FILES ごとにログに出す。
    worker と extra は別の処理へ渡すため、モジュールの直下で定義した関数と、受け渡しできる値にする。
    """
    total = len(paths)
    chunks = [list(paths[i:i + CHUNK_FILES]) for i in range(0, total, CHUNK_FILES)]
    started = time.time()
    done = 0
    next_log = PROGRESS_EVERY_FILES

    def progress(n: int) -> None:
        nonlocal done, next_log
        done += n
        if done >= next_log or done == total:
            elapsed = time.time() - started
            remaining = elapsed / done * (total - done) if done else 0.0
            logger.info(f"[{label}] {done:,} / {total:,} ファイル（{done / total:.0%}）"
                        f"・経過 {elapsed / 60:.1f} 分・残り約 {remaining / 60:.1f} 分")
            while next_log <= done:
                next_log += PROGRESS_EVERY_FILES

    if workers <= 1:
        for chunk in chunks:
            result = worker(chunk, *extra)
            progress(len(chunk))
            yield result
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, chunk, *extra): len(chunk) for chunk in chunks}
        for future in as_completed(futures):
            result = future.result()
            progress(futures[future])
            yield result
