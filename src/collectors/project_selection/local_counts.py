"""
走査から外したリポジトリを、ローカルに収集済みのデータから数える（design.md §3.3.3）

Qt の qtbase / qtcreator、LibreOffice の core は全期間を収集済みなので、
API に問い合わせずに作成日で数えられる。API で数えるのと同じ定義のまま、
正確な件数が得られる。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from src.config.path import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

COLLECTED_DIR = DEFAULT_DATA_DIR / "openstack_collected"


def load(directory_name: str) -> Tuple[List[str], Counter]:
    """収集済みディレクトリから (作成日の一覧, ブランチの分布) を返す。

    作成日は "YYYY-MM-DD" の文字列。期間での絞り込みは呼び出し側が行う。
    """
    changes_dir = COLLECTED_DIR / directory_name / "changes"
    if not changes_dir.is_dir():
        raise FileNotFoundError(f"収集済みデータが見つかりません: {changes_dir}")

    created_dates: List[str] = []
    branches: Counter = Counter()
    broken = 0

    for path in changes_dir.glob("change_*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            broken += 1
            continue
        created = (data.get("created") or "")[:10]
        if created:
            created_dates.append(created)
            branches[data.get("branch") or "(不明)"] += 1

    if broken:
        logger.warning(f"{directory_name}: 読めなかったファイルが {broken} 件ありました")
    logger.info(f"{directory_name}: ローカルから {len(created_dates):,} 件を読みました")
    return created_dates, branches
