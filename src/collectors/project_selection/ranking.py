"""
第 1 段階：順位づけと E4 の判定（design.md §3, §4.1）

対象期間に**作成された** Change の件数をリポジトリごとに数える。
Gerrit の after: / before: は最終更新日で絞るため、上限を付けずに走査して
（終わりは constants.END_DATE ＝ 2027-04-01）、created が期間内のものだけを数える（design.md §3.3）。
同じ Change は番号で 1 回だけ数え（§3.3.2）、走査した Change は 1 件ずつ保存する（§3.4）。

Change の branch は一覧取得の既定の応答に含まれるので、
同じ走査からブランチの分布を得て E4 を判定する（design.md §4.1）。
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

import requests

from src.collectors.project_selection import hosts as hosts_module

logger = logging.getLogger(__name__)

# 1 リクエストが失敗したときの待ち時間（秒）。頻度制限は解けるまで時間がかかる
RETRY_WAITS = (10.0, 30.0, 60.0, 120.0)

# Gerrit が 1 クエリで返せる件数の上限。これに達した区間はさらに細かく切り直す
QUERY_LIMIT = 10000

# 1 日でも上限に達したときの割り直しの段階（design.md §3.3.2）。
# 1 時間 → 10 分 → 1 分 → 1 秒。1 秒でも上限に達したら記録して先へ進む
SUBDAY_STEPS = (timedelta(hours=1), timedelta(minutes=10), timedelta(minutes=1), timedelta(seconds=1))


def _slice_dates(start: date, end: date, unit: str) -> Iterator[Tuple[date, date]]:
    """[start, end) を unit ごとの区間に切る。区間は隙間なく連続する。"""
    step = {"day": timedelta(days=1), "week": timedelta(days=7)}.get(unit)
    cursor = start
    while cursor < end:
        if step is not None:
            nxt = min(cursor + step, end)
        else:  # month
            nxt = date(cursor.year + (cursor.month == 12),
                       cursor.month % 12 + 1, 1)
            nxt = min(nxt, end)
        yield cursor, nxt
        cursor = nxt


def _fetch_page(session: requests.Session, base: str, query: str,
                page_size: int, skip: int, options: List[str],
                delay: float) -> List[Dict[str, Any]]:
    """1 ページ取得する。失敗したら間隔を空けて投げ直す。"""
    url = (f"{base}/changes/?q={quote(query)}&n={page_size}&S={skip}"
           + "".join(f"&o={o}" for o in options))
    last_error: Optional[Exception] = None
    for wait in (0.0,) + RETRY_WAITS:
        if wait:
            time.sleep(wait)
        try:
            response = session.get(url, timeout=180)
            response.raise_for_status()
            body = response.text
            # Gerrit は XSSI 対策の )]}' を先頭に付ける
            return json.loads(body[body.index("\n") + 1:] if body.startswith(")]}'") else body)
        except Exception as exc:  # 頻度制限・一時的な 5xx・切断
            last_error = exc
            logger.warning(f"取得に失敗（{type(exc).__name__}）。投げ直します: {query}")
    raise RuntimeError(f"{base} から取得できませんでした: {query}") from last_error


def _fetch_all(session: requests.Session, base: str, query: str, page_size: int,
               options: List[str], delay: float) -> List[Dict[str, Any]]:
    """1 つの問い合わせを最後のページまで取る。QUERY_LIMIT に達したらそこで止める。"""
    buffered: List[Dict[str, Any]] = []
    skip = 0
    while True:
        page = _fetch_page(session, base, query, page_size, skip, options, delay)
        buffered += page
        if delay:
            time.sleep(delay)
        if not page or not page[-1].get("_more_changes"):
            break
        skip += len(page)
        if len(buffered) >= QUERY_LIMIT:
            break
    return buffered


def _gerrit_time(t: datetime) -> str:
    """時刻付きの after: / before: に書く値（UTC。引用符で囲む。design.md §3.3.2）。"""
    return f'"{t:%Y-%m-%d %H:%M:%S}"'


def _scan_times(session: requests.Session, base: str, query_base: str,
                start: datetime, end: datetime, steps: Tuple[timedelta, ...],
                page_size: int, options: List[str], delay: float,
                on_change, truncated: List[str] | None = None) -> int:
    """[start, end) を steps[0] ごとに時刻で区切って走査する（1 日でも上限に達したとき。design.md §3.3.2）。

    1 区間が QUERY_LIMIT に達したら、その区間だけ steps の次の段階（10 分 → 1 分 → 1 秒）で割り直す。
    最後の段階（1 秒）でも達したら、記録して先へ進む。
    """
    total = 0
    step = steps[0]
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        query = f"{query_base} after:{_gerrit_time(cursor)} before:{_gerrit_time(nxt)}".strip()
        # 上限に達したら割り直すので、確定するまで on_change に渡さず溜めておく
        buffered = _fetch_all(session, base, query, page_size, options, delay)
        if len(buffered) >= QUERY_LIMIT:
            if len(steps) > 1:
                logger.info(f"{query} が上限（{QUERY_LIMIT:,}）に達したため、{steps[1]} ごとに割り直します")
                total += _scan_times(session, base, query_base, cursor, nxt, steps[1:],
                                     page_size, options, delay, on_change, truncated)
                cursor = nxt
                continue
            logger.error(f"{query} が上限に達しましたが、1 秒より細かく切れません。"
                         f"**この区間は欠損している可能性があります**")
            if truncated is not None:
                truncated.append(query)
        for change in buffered:
            on_change(change)
        total += len(buffered)
        cursor = nxt
    return total


def _scan_range(session: requests.Session, base: str, query_base: str,
                start: date, end: date, unit: str, page_size: int,
                options: List[str], delay: float,
                on_change, truncated: List[str] | None = None) -> int:
    """[start, end) を unit ごとに走査し、Change 1 件ずつを on_change に渡す。

    1 区間が QUERY_LIMIT に達した場合は、その区間だけ 1 段細かく切り直す。

    **上限に達したかの判定は「取得件数が QUERY_LIMIT 以上か」だけで行う。**
    Gerrit は上限に達すると `_more_changes` を落として「終端」として返すため、
    「ちょうど上限ちょうどで取り終わった」と「上限で打ち切られた」を区別できない。
    前者を切り直すのは無駄なだけだが、後者を見逃すと**静かに欠損する**ので、
    疑わしきは切り直す。

    また、**切り直す区間の Change は数えない**。
    数えてから切り直すと、同じ Change を 2 回数えることになる。
    """
    total = 0
    finer = {"month": "week", "week": "day"}.get(unit)
    for chunk_start, chunk_end in _slice_dates(start, end, unit):
        query = (f"{query_base} after:{chunk_start.isoformat()} "
                 f"before:{chunk_end.isoformat()}").strip()
        # 上限に達したら切り直すので、確定するまで on_change に渡さず溜めておく
        buffered = _fetch_all(session, base, query, page_size, options, delay)

        if len(buffered) >= QUERY_LIMIT:
            if finer and (chunk_end - chunk_start).days > 1:
                logger.info(f"{query} が上限（{QUERY_LIMIT:,}）に達したため "
                            f"{finer} 単位で切り直します")
                total += _scan_range(session, base, query_base, chunk_start, chunk_end,
                                     finer, page_size, options, delay, on_change, truncated)
                continue
            # 1 日でも上限に達した。時刻で割り直す（design.md §3.3.2。2026-10-04 追加）。
            # 以前はここで「これ以上細かく切れない」として 1 万件だけを数えて先へ進んでいた
            logger.info(f"{query} が 1 日でも上限（{QUERY_LIMIT:,}）に達したため、時刻で割り直します")
            day_start = datetime(chunk_start.year, chunk_start.month, chunk_start.day)
            day_end = datetime(chunk_end.year, chunk_end.month, chunk_end.day)
            total += _scan_times(session, base, query_base, day_start, day_end, SUBDAY_STEPS,
                                 page_size, options, delay, on_change, truncated)
            continue

        for change in buffered:
            on_change(change)
        total += len(buffered)
    return total


class HostTally:
    """1 インスタンス分の集計結果。"""

    def __init__(self, name: str, scan_writer: Optional[Any] = None):
        self.name = name
        # 同じ Change を 2 回数えないための番号の集合（design.md §3.3.2）。
        # 区切りの境目（after:/before: は境目の日を両方に含む）と、走査中の更新で同じ Change が再び現れる
        self.seen: set = set()
        self.duplicates = 0
        # 走査した Change を 1 件ずつ書き出す先（design.md §3.4。csv.writer。None なら書かない）
        self.scan_writer = scan_writer
        # 上限に達したのに、これ以上細かく切れなかった区間（欠損の疑い）
        self.truncated: List[str] = []
        self.created: Counter = Counter()                    # repository -> 作成数
        self.created_by_year: Dict[int, Counter] = defaultdict(Counter)
        self.branches: Dict[str, Counter] = defaultdict(Counter)  # repository -> branch -> 件数

    def add(self, change: Dict[str, Any], period_start: str, period_end: str) -> None:
        number = change.get("_number")
        if number is not None:
            if number in self.seen:
                self.duplicates += 1
                return
            self.seen.add(number)
        if self.scan_writer is not None:
            self.scan_writer.writerow([number, change.get("project") or "", change.get("branch") or "",
                                       change.get("created") or "", change.get("updated") or ""])
        created = (change.get("created") or "")[:10]
        if not created or not (period_start <= created < period_end):
            return
        repository = change.get("project") or "(不明)"
        self.created[repository] += 1
        self.created_by_year[int(created[:4])][repository] += 1
        self.branches[repository][change.get("branch") or "(不明)"] += 1

    def add_local(self, repository: str, created_dates: List[str],
                  branches: Counter, period_start: str, period_end: str) -> None:
        """走査から外したリポジトリを、ローカルのデータから数える（design.md §3.3.3）。"""
        for created in created_dates:
            if period_start <= created < period_end:
                self.created[repository] += 1
                self.created_by_year[int(created[:4])][repository] += 1
        self.branches[repository] = branches

    def top_branch_share(self, repository: str) -> float:
        """最大ブランチの占有率。E4 の判定に使う（design.md §5.2）。"""
        counts = self.branches.get(repository)
        if not counts:
            return 0.0
        return counts.most_common(1)[0][1] / sum(counts.values())

    def branch_count(self, repository: str) -> int:
        return len(self.branches.get(repository, ()))


SCAN_COLUMNS = ["number", "repository", "branch", "created", "updated"]


def scan_host(name: str, host: Dict[str, Any],
              period_start: str, period_end: str, scan_end: str,
              scan_path: Optional[Path] = None) -> HostTally:
    """1 インスタンスを走査して集計する。

    最終更新日が period_start 以降の Change を scan_end まで走査し、作成日が
    [period_start, period_end) のものを数える。最終更新日は必ず作成日以降なので、
    走査の終わりを数える期間より先（2027-04-01）に置けば、対象期間に作成された Change を
    漏れなく捉えられる（design.md §3.3）。以前は走査の終わりも period_end（2025-01-01）にしており、
    期間内に作成され 2025 年以降にも更新された Change を数えていなかった（nova で 11.9%）。

    scan_path を与えると、走査した Change を 1 件ずつ gzip の CSV に書き出す（design.md §3.4）。
    途中で失敗したときに古いファイルを壊さないよう、`.partial` に書いてから最後に置き換える。
    """
    partial = None
    handle = None
    writer = None
    if scan_path is not None:
        scan_path.parent.mkdir(parents=True, exist_ok=True)
        partial = scan_path.with_name(scan_path.name + ".partial")
        handle = gzip.open(partial, "wt", encoding="utf-8", newline="")
        writer = csv.writer(handle)
        writer.writerow(SCAN_COLUMNS)
    try:
        tally = _scan_host(name, host, period_start, period_end, scan_end, writer)
    finally:
        if handle is not None:
            handle.close()
    if partial is not None:
        partial.replace(scan_path)
        logger.info(f"[{name}] 走査した Change を保存しました: {scan_path}（{len(tally.seen):,} 件）")
    return tally


def _scan_host(name: str, host: Dict[str, Any], period_start: str, period_end: str,
               scan_end: str, writer: Optional[Any]) -> HostTally:
    tally = HostTally(name, writer)
    session = hosts_module.create_session(host)
    options = hosts_module.query_options(host)
    delay = float(host.get("delay", hosts_module.DEFAULT_REQUEST_DELAY))

    excluded = hosts_module.excluded_repositories(host)
    query_base = " ".join(f"-project:{r}" for r in excluded)

    start = date.fromisoformat(period_start)
    end = date.fromisoformat(str(scan_end)[:10])  # 走査の終わり（数える期間の終わりではない）

    logger.info(f"[{name}] 走査を開始します（区切り {host.get('slice', 'month')}"
                + (f"・{len(excluded)} リポジトリを除外" if excluded else "") + "）")
    started = time.time()
    seen = _scan_range(session, host["base"], query_base, start, end,
                       host.get("slice", "month"), int(host.get("page_size", 500)),
                       options, delay,
                       lambda c: tally.add(c, period_start, period_end),
                       tally.truncated)
    elapsed = time.time() - started
    logger.info(f"[{name}] 走査 {seen:,} 件 / 重複を除いて {len(tally.seen):,} 件"
                f"（重複 {tally.duplicates:,} 件）/ 対象期間に作成 {sum(tally.created.values()):,} 件"
                f"（{elapsed / 60:.1f} 分）")
    if tally.truncated:
        logger.error(f"[{name}] **{len(tally.truncated)} 区間で欠損の可能性があります**")
    return tally


def refill_host(name: str, host: Dict[str, Any], period_start: str, period_end: str,
                scan_end: str, scan_path: Path, days: List[date]
                ) -> Tuple[HostTally, Dict[date, int], int]:
    """欠けた日だけを取り直し、保存してある走査結果に足して数え直す（--refill。design.md §3.4）。

    取り直すのは次の 2 つ。
      ① days（1 日でも上限に達した日）を時刻で区切って（1 時間 → 10 分 → 1 分 → 1 秒）
      ② 走査を始めた時刻以降に更新された分。走査のあとにまた更新された Change は最終更新日が
         変わっていて、①の日を問い合わせても出てこないため。走査を始めた時刻は、保存してある
         走査結果のファイルの作成時刻から取る。日ごとに区切り、1 日でも上限に達したら時刻で割る

    書き換える前の走査結果は `<インスタンス>.before_refill.csv.gz` として残す（消さない）。
    返り値は (数え直した集計, 日ごとに新しく加わった Change の数, ②で新しく加わった Change の数)。
    """
    if not scan_path.exists():
        raise FileNotFoundError(f"走査結果が見つかりません: {scan_path}。先に --stage 1 で走査してください")
    stat = scan_path.stat()
    scan_started = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime), timezone.utc)

    partial = scan_path.with_name(scan_path.name + ".partial")
    backup = scan_path.with_name(scan_path.name.replace(".csv.gz", ".before_refill.csv.gz"))
    handle = gzip.open(partial, "wt", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(SCAN_COLUMNS)
    try:
        tally = HostTally(name, writer)
        # 保存してある走査結果を読み込み、走査したときと同じ数え方で数え直す
        with gzip.open(scan_path, "rt", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tally.add({"_number": int(row["number"]), "project": row["repository"],
                           "branch": row["branch"], "created": row["created"],
                           "updated": row["updated"]}, period_start, period_end)
        logger.info(f"[{name}] 保存してある走査結果 {len(tally.seen):,} 件を読み込みました"
                    f"（走査を始めた時刻 {scan_started:%Y-%m-%d %H:%M} UTC）")

        session = hosts_module.create_session(host)
        options = hosts_module.query_options(host)
        delay = float(host.get("delay", hosts_module.DEFAULT_REQUEST_DELAY))
        query_base = " ".join(f"-project:{r}" for r in hosts_module.excluded_repositories(host))
        page_size = int(host.get("page_size", 500))

        def add(change: Dict[str, Any]) -> None:
            tally.add(change, period_start, period_end)

        added: Dict[date, int] = {}
        for day in days:
            before = len(tally.seen)
            _scan_times(session, host["base"], query_base,
                        datetime(day.year, day.month, day.day),
                        datetime(day.year, day.month, day.day) + timedelta(days=1),
                        SUBDAY_STEPS, page_size, options, delay, add, tally.truncated)
            added[day] = len(tally.seen) - before
            logger.info(f"[{name}] {day} を取り直しました：新しく加わった Change {added[day]:,} 件")

        before = len(tally.seen)
        _scan_range(session, host["base"], query_base, scan_started.date(),
                    date.fromisoformat(str(scan_end)[:10]), "day", page_size, options, delay,
                    add, tally.truncated)
        added_since = len(tally.seen) - before
        logger.info(f"[{name}] 走査を始めた時刻以降に更新された分：新しく加わった Change {added_since:,} 件")
    finally:
        handle.close()

    if backup.exists():
        logger.info(f"[{name}] {backup.name} は既にあるので、上書きせずに残します")
    else:
        scan_path.rename(backup)
        logger.info(f"[{name}] 書き換える前の走査結果を {backup.name} として残しました")
    partial.replace(scan_path)
    logger.info(f"[{name}] 走査結果を書き出しました: {scan_path}（{len(tally.seen):,} 件）/ "
                f"対象期間に作成 {sum(tally.created.values()):,} 件")
    if tally.truncated:
        logger.error(f"[{name}] **{len(tally.truncated)} 区間で、1 秒に割っても上限に達しました**")
    return tally, added, added_since
