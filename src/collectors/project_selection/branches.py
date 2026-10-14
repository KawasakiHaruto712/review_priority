"""
ブランチの一覧：収集済みデータから、リポジトリごとのブランチと、その Change の数・作成日の範囲を出す

リリースの区切りの根拠（版ごとのブランチがどう切られるか、どのブランチの中身がどの版として出るか）を
調べる前に、手元のデータでどんなブランチがあるかを確かめるために使う。
「同じ版のブランチが、複数のリポジトリに同じ時期に現れるか」は、branch と first_created を並べれば見られる。

収集済みデータだけを使うので、通信はしない。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.collectors.project_selection import parallel
from src.config.path import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

COLLECTED_DIR = DEFAULT_DATA_DIR / "openstack_collected"

# ブランチごとの集計：[全件の数, 選定の期間の数, 最初の作成日時, 最後の作成日時]
BranchStats = Dict[str, list]


def _merge_stats(total: BranchStats, part: BranchStats) -> None:
    """ブランチごとの集計を足し合わせる。作成日時は早いほう・遅いほうを残す。"""
    for branch, (n_all, n_period, first, last) in part.items():
        if branch not in total:
            total[branch] = [n_all, n_period, first, last]
            continue
        stats = total[branch]
        stats[0] += n_all
        stats[1] += n_period
        stats[2] = min(stats[2], first)
        stats[3] = max(stats[3], last)


def _branch_chunk(paths: List[Path], period_start: str, period_end: str) -> Tuple[BranchStats, int, int]:
    """ファイルのかたまりを読み、(ブランチごとの集計, 読めなかった数, 作成日時のない数) を返す。"""
    stats: BranchStats = {}
    broken = 0
    no_created = 0
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            broken += 1
            continue
        # 作成日時は "YYYY-MM-DD HH:MM:SS"（UTC）まで残す
        created = (data.get("created") or "")[:19]
        if not created:
            no_created += 1
            continue
        in_period = int(period_start <= created[:10] < period_end)
        branch = data.get("branch") or "(不明)"
        _merge_stats(stats, {branch: [1, in_period, created, created]})
    return stats, broken, no_created


def collect(directory_name: str, repository: str, period_start: str, period_end: str,
            workers: int = 1) -> List[Dict[str, object]]:
    """収集済みディレクトリの全件を読み、ブランチごとの行を Change の多い順に返す。

    workers が 2 以上なら、ファイルをかたまりに分けて並列に読む（design.md §4.4）。結果は同じ。
    """
    changes_dir = COLLECTED_DIR / directory_name / "changes"
    if not changes_dir.is_dir():
        raise FileNotFoundError(f"収集済みデータが見つかりません: {changes_dir}")

    total: BranchStats = {}
    broken = 0
    no_created = 0
    paths = sorted(changes_dir.glob("change_*.json"))
    for part, part_broken, part_no_created in parallel.map_files(
            repository, paths, _branch_chunk, (period_start, period_end), workers):
        _merge_stats(total, part)
        broken += part_broken
        no_created += part_no_created

    if broken:
        logger.warning(f"{repository}: 読めなかったファイルが {broken} 件ありました")
    if no_created:
        logger.warning(f"{repository}: 作成日時のない Change が {no_created} 件ありました（数えていない）")
    logger.info(f"{repository}: {sum(s[0] for s in total.values()):,} 件・ブランチ {len(total):,} 本")

    rows = [{
        "repository": repository,
        "branch": branch,
        "changes_all": n_all,
        "changes_period": n_period,
        "first_created": first,
        "last_created": last,
    } for branch, (n_all, n_period, first, last) in total.items()]
    # Change の多い順。同数のときはブランチ名の順（並列に読んでも並びが変わらないように）
    rows.sort(key=lambda r: (-r["changes_all"], r["branch"]))
    return rows
