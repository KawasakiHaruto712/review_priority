"""
分析対象リポジトリの選定（design.md §4）

  --stage 1       順位づけと E4 の判定。全インスタンスを走査する
                  --refill を付けると、欠けた日だけを取り直して数え直す（design.md §3.4）
                  --recount を付けると、走査せず、保存してある一覧だけから数え直す（通信なし。design.md §3.5）
  --stage repo-scan  収集済みリポジトリ（qtbase・qtcreator・core）を 1 つずつ走査し、一覧を保存する
                  （--host で指定したインスタンス。順位のファイルには触れない。design.md §3.3.3）
  --stage fill    第 2 段階の後。第 1 段階で走査した Change の番号と照らし合わせ、
                  収集から抜けた Change を 1 件ずつ取る（design.md §4.3）
  --stage verify  第 2 段階の検算。収集した件数をランキングの件数と突き合わせる
  --stage bots    ボット一覧の作成を支援する（アカウントの活動量を多い順に出す）
  --stage metrics 収集済みデータから E1・E2・E3 の指標を出す
  --stage branches 収集済みデータから、リポジトリごとのブランチと Change の数・作成日の範囲を出す
                  （リリースの区切りの根拠を調べる前の確認。通信なし）

第 2 段階の本収集は既存の change_collector を使う（design.md §7）。
"""

from __future__ import annotations

import argparse
from datetime import date
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.collectors.project_selection import branches as branches_module
from src.collectors.project_selection import hosts as hosts_module
from src.collectors.project_selection import local_counts, metrics as metrics_module
from src.collectors.project_selection import parallel as parallel_module
from src.collectors.project_selection import ranking as ranking_module
from src.config.path import DEFAULT_DATA_DIR
from src.utils.bot_detection import BotDetector, host_of
from src.utils.constants import END_DATE

logger = logging.getLogger(__name__)

OUTPUT_DIR = DEFAULT_DATA_DIR / "project_selection"

# 選定に使う期間（design.md §3.5）。作成日がこの範囲の Change を数える。
# 2026-10-12 に 2022-01-01〜2024-12-31 から 2023-01-01〜2025-12-31 に変えた（評価する約 5 版が収まるように）
PERIOD_START = "2023-01-01"
PERIOD_END = "2026-01-01"

# 走査の始まり（design.md §3.3・§3.5）。最終更新がこの日以降の Change を走査する。
# 数える期間とは別に持つ。期間を変えても走査と収集の範囲は変えない
SCAN_START = "2022-01-01"

# 走査の終わり（design.md §3.3）。数える期間（PERIOD_END）とは別に持つ。
# 期間内に作成され、その後にも更新された Change を漏らさないよう、終わりは付けない
# （収集と同じ constants.END_DATE ＝ 2027-04-01。走査が日をまたいでも漏れない）。
SCAN_END = END_DATE

# 第 1 段階で走査した Change を 1 件ずつ保存する場所（design.md §3.4）
SCAN_DIR = OUTPUT_DIR / "scan"

# 第 2 段階で収集するリポジトリの数（design.md §4。2026-10-13 に 12 件から 15 件、2026-10-14 に 30 件に広げた）
COLLECT_TOP = 30

# 手元のファイルを並列に読むときの本数の決め方（第 4 段階と検算。design.md §4.4）。
# OS から読めるコアの数を天井とし、--workers の指定が無ければ
# 「天井 − RESERVED_CORES」と MAX_WORKERS の小さいほうを使う（最低 1 本）。
# MAX_WORKERS は共用のサーバ（40 コア）でほかの人の処理を妨げないための上限
RESERVED_CORES = 2
MAX_WORKERS = 8


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"書き出しました: {path}（{len(rows)} 行）")


def _years(period_start: str, period_end: str) -> List[int]:
    """年ごとの順位を作る年（期間から求める。PERIOD_END は翌年の 1 月 1 日）。"""
    return list(range(int(period_start[:4]), int(period_end[:4])))


def _repo_scan_path(name: str, host: Dict[str, Any], repository: str) -> Path:
    """収集済みリポジトリを個別に走査した一覧の保存先（scan/<インスタンス>.<キー>.csv.gz。design.md §3.4）。

    キーは手元のディレクトリ名（hosts.py の local）。
    """
    key = hosts_module.local_directory(host, repository) or repository.replace("/", "_")
    return SCAN_DIR / f"{name}.{key}.csv.gz"


def _file_date(path: Path) -> str:
    """ファイルの更新日（一覧を取った日として scanned_at に入れる。design.md §3.5）。"""
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


def _pending_truncated() -> Dict[str, List[date]]:
    """ranking_truncated.csv のうち、まだ取り直していない行の日を、インスタンスごとに返す（design.md §3.4）。"""
    path = OUTPUT_DIR / "ranking_truncated.csv"
    pending: Dict[str, List[date]] = {}
    if not path.exists():
        return pending
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("refilled_at"):
                continue
            m = re.search(r'after:"?(\d{4}-\d{2}-\d{2})', r["query"])
            if m:
                day = date.fromisoformat(m.group(1))
                if day not in pending.setdefault(r["host"], []):
                    pending[r["host"]].append(day)
    return pending


def _write_rankings(tallies: List[ranking_module.HostTally], scanned_at_of: Callable[[str, str], str],
                    period_start: str, period_end: str, out_dir: Path) -> None:
    """順位表（期間全体と年ごと）を書き出す。数えたインスタンスの行だけを差し替え、他の行は残す（design.md §3.4）。

    scanned_at_of(インスタンス, リポジトリ) は、その行の scanned_at（走査した日）を返す。
    """
    scanned = {t.name for t in tallies}
    rows = []
    for tally in tallies:
        for repository, count in tally.created.items():
            rows.append({
                "host": tally.name,
                "repository": repository,
                "created_changes": count,
                "branches": tally.branch_count(repository),
                "top_branch_share": round(tally.top_branch_share(repository), 4),
                "scanned_at": scanned_at_of(tally.name, repository),
            })
    _merge_write(out_dir / "ranking_period.csv", rows, scanned,
                 ["host", "repository", "created_changes", "branches",
                  "top_branch_share", "scanned_at"], sort_key="created_changes")

    for year in _years(period_start, period_end):
        year_rows = []
        for tally in tallies:
            for repository, count in tally.created_by_year.get(year, {}).items():
                year_rows.append({"host": tally.name, "repository": repository,
                                  "created_changes": count,
                                  "scanned_at": scanned_at_of(tally.name, repository)})
        _merge_write(out_dir / f"ranking_{year}.csv", year_rows, scanned,
                     ["host", "repository", "created_changes", "scanned_at"],
                     sort_key="created_changes")


def stage_ranking(only: List[str] | None, refill: bool = False) -> None:
    """第 1 段階：順位づけと E4 の判定。

    refill=True のときは全期間を走査せず、ranking_truncated.csv に記録された日と、
    走査を始めた時刻以降に更新された分だけを取り直して数え直す（--refill。design.md §3.4）。
    """
    tallies: List[ranking_module.HostTally] = []
    failed: Dict[str, str] = {}
    pending = _pending_truncated() if refill else {}
    refilled: Dict[str, Dict[date, int]] = {}  # インスタンス -> 日 -> 新しく加わった Change の数

    for name, host in hosts_module.HOSTS.items():
        if only and name not in only:
            continue
        if refill and not pending.get(name):
            logger.info(f"[{name}] 取り直す日がありません（ranking_truncated.csv に未処理の行がない）")
            continue
        try:
            if refill:
                tally, added, _added_since = ranking_module.refill_host(
                    name, host, PERIOD_START, PERIOD_END, SCAN_END,
                    SCAN_DIR / f"{name}.csv.gz", pending[name])
                refilled[name] = added
            else:
                tally = ranking_module.scan_host(name, host, PERIOD_START, PERIOD_END, SCAN_END,
                                                 SCAN_DIR / f"{name}.csv.gz", scan_start=SCAN_START)
        except Exception as exc:
            # 1 インスタンスの失敗で全体を止めない。理由を記録して先へ進む（design.md §6.6）
            logger.error(f"[{name}] 走査に失敗しました: {type(exc).__name__}: {exc}")
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue

        # 走査から外したリポジトリ（収集済みのもの）も、1 つずつ走査した一覧で数える（design.md §3.3.3）。
        # 全体の走査ではここで走査し直し、--refill では保存してある一覧を読む。
        # 2026-10-12 までは手元のデータから数えていたが、欠けを確かめる相手がないのでやめた
        for repository in hosts_module.excluded_repositories(host):
            path = _repo_scan_path(name, host, repository)
            try:
                if refill:
                    sub = ranking_module.tally_from_saved(f"{name}/{repository}", path,
                                                          PERIOD_START, PERIOD_END)
                else:
                    sub = ranking_module.scan_repository(name, host, repository, PERIOD_START, PERIOD_END,
                                                         SCAN_END, path, SCAN_START)
            except Exception as exc:
                # 手元のデータで代わりに数えたりせず、失敗として記録する（黙って埋めない）
                logger.error(f"[{name}] {repository} を数えられませんでした: {type(exc).__name__}: {exc}")
                failed[f"{name}/{repository}"] = f"{type(exc).__name__}: {exc}"
                continue
            tally.merge(sub)

        tallies.append(tally)

    # 今回の走査結果を、既存のファイルに統合する。
    # **今回走査に成功したインスタンスの行だけを差し替え、他の行は残す**。
    # 以前は指定したインスタンスだけでファイルを丸ごと上書きしており、一部だけ
    # 測り直したときに他のインスタンスの結果を失った（2026-10-02）。
    # 実行日の違う結果が混ざるので、各行に走査した日（scanned_at）を残す。
    # 上限を付けていたとき（2026-10-02 まで）は後の走査ほど件数がわずかに減ったが、
    # 上限を外したのでその出入りはない（design.md §3.4）。
    today = date.today().isoformat()
    scanned = {t.name for t in tallies}
    _write_rankings(tallies, lambda _host, _repository: today, PERIOD_START, PERIOD_END, OUTPUT_DIR)

    truncated =[{"host": t.name, "query": q, "scanned_at": today, "refilled_at": "", "refilled_changes": ""}
                 for t in tallies for q in t.truncated]
    truncated_fields = ["host", "query", "scanned_at", "refilled_at", "refilled_changes"]
    if refill:
        _record_refill(OUTPUT_DIR / "ranking_truncated.csv", refilled, truncated, truncated_fields, today)
    else:
        _merge_write(OUTPUT_DIR / "ranking_truncated.csv", truncated, scanned,
                     truncated_fields, keep_empty=False)
    if truncated:
        logger.error(f"**{len(truncated)} 区間で上限に達し、これ以上細かく切れませんでした。"
                     f"ranking_truncated.csv を確認してください**")

    if failed:
        _write_csv(OUTPUT_DIR / "ranking_failures.csv",
                   [{"target": k, "reason": v, "scanned_at": today} for k, v in failed.items()],
                   ["target", "reason", "scanned_at"])
        logger.warning(f"{len(failed)} 件が失敗しました。日を改めて再試行してください"
                       f"（失敗したインスタンスの既存の行は残してあります）")


def stage_recount(period_start: str, period_end: str, out_dir: Path) -> None:
    """第 1 段階を、走査せずに保存してある一覧だけから数え直す（--stage 1 --recount。通信なし。design.md §3.5）。

    インスタンスの一覧（scan/<インスタンス>.csv.gz）と、収集済みリポジトリを個別に走査した一覧
    （scan/<インスタンス>.<キー>.csv.gz。design.md §3.3.3）を読み、作成日が [period_start, period_end) の
    Change を数える。各行の scanned_at には、もとにした一覧を取った日（ファイルの更新日）を入れる。

    period_start / period_end / out_dir を変えれば、前の期間で数え直して既存の順位表と比べる
    答え合わせに使える（design.md §3.5「数え直しの答え合わせ」）。ranking_truncated.csv には触れない。
    """
    tallies: List[ranking_module.HostTally] = []
    failed: Dict[str, str] = {}
    scanned_at: Dict[tuple, str] = {}     # (インスタンス, リポジトリ) -> 一覧を取った日
    host_date: Dict[str, str] = {}        # インスタンス -> インスタンスの一覧を取った日

    for name, host in hosts_module.HOSTS.items():
        path = SCAN_DIR / f"{name}.csv.gz"
        try:
            tally = ranking_module.tally_from_saved(name, path, period_start, period_end)
        except Exception as exc:
            logger.error(f"[{name}] 数え直せませんでした: {type(exc).__name__}: {exc}")
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        host_date[name] = _file_date(path)

        for repository in hosts_module.excluded_repositories(host):
            repo_path = _repo_scan_path(name, host, repository)
            try:
                sub = ranking_module.tally_from_saved(f"{name}/{repository}", repo_path,
                                                      period_start, period_end)
            except Exception as exc:
                # 手元のデータで代わりに数えたりせず、失敗として記録する（黙って埋めない。design.md §3.3.3）
                logger.error(f"[{name}] {repository} を数え直せませんでした: {type(exc).__name__}: {exc}。"
                             f"先に --stage repo-scan --host {name} で走査してください")
                failed[f"{name}/{repository}"] = f"{type(exc).__name__}: {exc}"
                continue
            tally.merge(sub)
            scanned_at[(name, repository)] = _file_date(repo_path)
        tallies.append(tally)

    _write_rankings(tallies, lambda h, r: scanned_at.get((h, r), host_date.get(h, "")),
                    period_start, period_end, out_dir)
    logger.info(f"数え直しました（作成日 {period_start}〜{period_end} の前日。出力先 {out_dir}）")

    if failed:
        _write_csv(out_dir / "ranking_failures.csv",
                   [{"target": k, "reason": v, "scanned_at": date.today().isoformat()}
                    for k, v in failed.items()],
                   ["target", "reason", "scanned_at"])
        logger.error(f"**{len(failed)} 件を数え直せませんでした。ranking_failures.csv を確認してください**")


def stage_repo_scan(names: List[str]) -> None:
    """収集済みリポジトリを 1 つずつ走査し、一覧を保存する（--stage repo-scan。design.md §3.3.3）。

    names で指定したインスタンスの、走査から外しているリポジトリ（hosts.py の exclude）が対象。
    走査のしかたはインスタンスの走査と同じ。順位のファイルには触れない。
    1 つでも失敗したら、理由を repo_scan_failures.csv に残して次へ進み、最後に終了コード 1 で終わる
    （&& でつないだ抜けの取得を、一覧なしで始めないため）。
    """
    if not names:
        raise SystemExit("--host を 1 つ以上指定してください")
    failed: Dict[str, str] = {}
    for name in names:
        host = hosts_module.HOSTS.get(name)
        if host is None:
            failed[name] = "hosts.py に無いインスタンス"
            logger.error(f"[{name}] hosts.py に無いインスタンスです")
            continue
        repositories = hosts_module.excluded_repositories(host)
        if not repositories:
            logger.warning(f"[{name}] 走査から外しているリポジトリがありません")
        for repository in repositories:
            path = _repo_scan_path(name, host, repository)
            try:
                tally = ranking_module.scan_repository(name, host, repository, PERIOD_START, PERIOD_END,
                                                       SCAN_END, path, SCAN_START)
            except Exception as exc:
                logger.error(f"[{name}] {repository} の走査に失敗しました: {type(exc).__name__}: {exc}")
                failed[f"{name}/{repository}"] = f"{type(exc).__name__}: {exc}"
                continue
            # 1 秒に割っても上限に達した区間は、欠けの可能性として残す（黙って抜けたままにしない）
            for i, query in enumerate(tally.truncated):
                failed[f"{name}/{repository}#truncated{i + 1}"] = f"上限に達し、これ以上切れなかった区間: {query}"
    if failed:
        _write_csv(OUTPUT_DIR / "repo_scan_failures.csv",
                   [{"target": k, "reason": v, "scanned_at": date.today().isoformat()}
                    for k, v in failed.items()],
                   ["target", "reason", "scanned_at"])
        logger.error(f"**{len(failed)} 件の走査に失敗しました。repo_scan_failures.csv を確認してください**")
        raise SystemExit(1)


def _record_refill(path: Path, refilled: Dict[str, Dict[date, int]],
                   new_rows: List[Dict[str, Any]], fields: List[str], today: str) -> None:
    """--refill の結果を ranking_truncated.csv に書き込む（design.md §3.4）。

    行は消さずに残し、取り直した行に refilled_at（取り直した日）と refilled_changes
    （新しく加わった Change の数）を書き込む。1 秒に割っても上限に達した区間は新しい行として足す。
    """
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                row = {k: r.get(k, "") or "" for k in fields}
                m = re.search(r'after:"?(\d{4}-\d{2}-\d{2})', row["query"])
                added = refilled.get(row["host"], {})
                if not row["refilled_at"] and m and date.fromisoformat(m.group(1)) in added:
                    row["refilled_at"] = today
                    row["refilled_changes"] = added[date.fromisoformat(m.group(1))]
                rows.append(row)
    _write_csv(path, rows + new_rows, fields)


def _merge_write(path: Path, new_rows: List[Dict[str, Any]], scanned_hosts: set,
                 fields: List[str], sort_key: str | None = None,
                 keep_empty: bool = True) -> None:
    """既存の CSV のうち scanned_hosts の行を new_rows で差し替えて書き出す。

    scanned_hosts に含まれないインスタンスの行は、そのまま残す。
    既存のファイルに scanned_at の列が無い場合は空欄のまま残す。
    """
    kept: List[Dict[str, Any]] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("host") in scanned_hosts:
                    continue
                kept.append({k: r.get(k, "") for k in fields})
    merged = kept + new_rows
    if not merged and not keep_empty:
        return
    if sort_key:
        merged.sort(key=lambda r: int(r[sort_key]), reverse=True)
    replaced = len(new_rows)
    _write_csv(path, merged, fields)
    logger.info(f"  統合: {path.name}（既存 {len(kept)} 行を残し、{replaced} 行を差し替え）")


# 検算で「欠損の疑い」とみなす比率（収集した件数 ÷ ランキングの件数。design.md §4.3）。
# 以前は 0.95 だったが、5% 近い欠けを見逃すので 0.99 にした（2026-10-03）。
# 上限を外したので、作成日が対象期間の Change は第 1 段階にも第 2 段階にも必ず入り、
# ずれるのは削除・非公開になったものと、ページの境目での読み飛ばしくらいしかない。
# 上限付きのときの実績：chromiumos_infra で 28,070 ÷ 28,051 = 1.0007。打ち切りが起きれば大きく下回る。
VERIFY_MIN_RATIO = 0.99

# fill で 1 件ずつ取るときに限った、読み込みの時間切れ（秒）と、時間切れのときの投げ直しの回数（design.md §4.3
# 「時間切れを長くする理由と範囲」）。第 2 段階の収集で、chromium/src の 2 件（#3941687・#3938488）が
# 1 件ずつに分けても 120 秒以内に返らず取れなかったため（2026-10-07）。
# 通常の収集（change_collector）は 120 秒のまま変えない。収集では時間切れを「区間を割る合図」に使っており、
# すべての問い合わせで長くすると、失敗の処理が遅くなり、相手のサーバにも重い処理を長くさせることになる。
FILL_READ_TIMEOUT = 300.0
FILL_TIMEOUT_RETRIES = 1     # 時間切れなら 1 回だけ投げ直す（同じ要求を合計 2 回まで）


def _retry_on_timeout(call, delay: float):
    """call() を呼び、読み込みの時間切れなら FILL_TIMEOUT_RETRIES 回まで投げ直す（design.md §4.3）。

    一時的な混雑なら投げ直しで通る。それ以上は繰り返さない（相手のサーバに重い処理を何度もさせないため）。
    投げ直す前には、収集と同じく delay 秒を空ける。時間切れ以外の失敗は、そのまま上に投げる。
    """
    import time
    import requests
    for attempt in range(FILL_TIMEOUT_RETRIES + 1):
        try:
            return call()
        except requests.exceptions.ReadTimeout:
            if attempt >= FILL_TIMEOUT_RETRIES:
                raise
            logger.warning(f"{FILL_READ_TIMEOUT:.0f} 秒で時間切れ。1 回だけ投げ直します")
            if delay:
                time.sleep(delay)


def _fetch_one(ep: Any, number: int, delay: float = 0.0) -> tuple:
    """Change を 1 件、番号で取る（収集と同じ 5 オプション）。返り値は (Change または None, 取れなかった理由)。

    重くて通らなければ、本体と版ごとのファイル一覧に分けて取る。1 つの版でもファイル一覧が
    取れなければ、取れなかったものとして扱う（中身の欠けた Change を保存しない）。
    どの問い合わせも、時間切れなら FILL_TIMEOUT_RETRIES 回まで投げ直す（時間切れの長さは呼び出し元が
    ep に設定する FILL_READ_TIMEOUT。design.md §4.3）。
    """
    import requests
    try:
        found = _retry_on_timeout(
            lambda: ep.make_request("changes/", {"q": f"change:{number}", "o": ep.FULL_OPTIONS}), delay)
    except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError, ValueError) as e:
        try:
            change = _retry_on_timeout(lambda: ep.fetch_change(number), delay)  # ファイル一覧なしの本体
            for sha, meta in (change.get("revisions") or {}).items():
                rev = meta.get("_number", sha)
                meta["files"] = _retry_on_timeout(lambda: ep.fetch_revision_files(number, rev), delay)
            return change, ""
        except Exception as e2:
            return None, f"{type(e).__name__} → 分割して取っても {type(e2).__name__}: {e2}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not found:
        return None, "見つからない（走査の後に削除・非公開になった可能性）"
    return found[0], ""


def stage_fill(keys: List[str]) -> None:
    """第 2 段階の後：収集から抜けた Change を、番号を手がかりに 1 件ずつ取る（design.md §4.3）。

    第 1 段階で走査した Change は番号つきで保存してある（scan/<インスタンス>.csv.gz）。
    収集済みリポジトリ（qtbase・qtcreator・core）は、個別に走査した一覧（scan/<インスタンス>.<キー>.csv.gz。
    design.md §3.3.3）と照らし合わせる。
    そのリポジトリの番号のうち、収集したファイル（change_<番号>.json）が無いものを、作成日に関係なく取る。

    抜けが生じる理由：収集はページを順に取るので、取っている間にその区間の Change が更新されて
    区間から外れると、後ろが 1 件ずつ前に詰まり、ページの境目で 1 件読み飛ばす。エラーにならないので
    収集の記録にも残らない。ほかに、収集で取れずに記録された Change も、ここで取り直される。

    結果はリポジトリごとに fill_<キー>.csv に残す（2026-10-12 まで fill.csv 1 つに上書きしていたが、
    同時に流すと互いの結果を消し合うため分けた）。取れなかったものは理由つきで残す（黙って抜けたままにしない）。
    """
    import gzip
    import os
    import time
    from src.collectors.change_collector import ChangeCollector
    from src.utils.constants import GERRIT_PROJECTS, SELECTION_CANDIDATES

    by_path = {spec["base"].rstrip("/").removesuffix("/a"): h for h, spec in hosts_module.HOSTS.items()}
    collector = ChangeCollector()
    rows: List[Dict[str, Any]] = []
    totals: List[str] = []
    fields = ["key", "repository", "number", "status", "reason"]

    for key in (keys or SELECTION_CANDIDATES):
        spec = GERRIT_PROJECTS[key]
        host = by_path.get(spec["host"].rstrip("/").removesuffix("/a"))
        key_rows: List[Dict[str, Any]] = []
        if host is not None and spec["path"] in hosts_module.excluded_repositories(hosts_module.HOSTS[host]):
            # 収集済みリポジトリは、個別に走査した一覧と照らし合わせる（design.md §3.3.3）
            scan_path = _repo_scan_path(host, hosts_module.HOSTS[host], spec["path"])
        else:
            scan_path = SCAN_DIR / f"{host}.csv.gz"
        if host is None or not scan_path.exists():
            logger.error(f"[{key}] 走査結果が見つかりません（{scan_path}）。照らし合わせられません")
            key_rows.append({"key": key, "repository": spec["path"], "number": "",
                             "status": "照合できず", "reason": f"走査結果なし: {scan_path}"})
            rows.extend(key_rows)
            _write_csv(OUTPUT_DIR / f"fill_{key}.csv", key_rows, fields)
            continue

        expected = set()
        with gzip.open(scan_path, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["repository"] == spec["path"]:
                    expected.add(int(r["number"]))
        changes_dir = collector.change_storage.output_dir / key / "changes"
        have = set()
        if changes_dir.is_dir():
            for entry in os.scandir(changes_dir):
                m = re.fullmatch(r"change_(\d+)\.json", entry.name)
                if m:
                    have.add(int(m.group(1)))
        missing = sorted(expected - have)
        logger.info(f"[{key}] 走査した Change {len(expected):,} 件 / 収集済み {len(have):,} 件 / "
                    f"収集から抜けている {len(missing):,} 件")
        if not missing:
            totals.append(f"{key}: 抜けなし")
            # 抜けがなかったことも、空のファイル（見出しだけ）で残す
            _write_csv(OUTPUT_DIR / f"fill_{key}.csv", key_rows, fields)
            continue

        collector._switch_to(key)
        ep = collector.endpoints["changes"]
        # この fill で使う窓口だけ、読み込みの時間切れを長くする（接続の時間切れは変えない。design.md §4.3）。
        # 窓口は _switch_to のたびに作り直されるので、通常の収集の設定（120 秒）には影響しない
        ep.timeout = (ep.timeout[0], FILL_READ_TIMEOUT)
        logger.info(f"[{key}] 1 件ずつ取ります（読み込みの時間切れ {FILL_READ_TIMEOUT:.0f} 秒・"
                    f"時間切れなら {FILL_TIMEOUT_RETRIES} 回まで投げ直す）")
        got = 0
        for number in missing:
            change, reason = _fetch_one(ep, number, delay=collector._request_delay)
            if collector._request_delay:
                time.sleep(collector._request_delay)
            if change is not None:
                detail = collector._collect_change_details(change, key)
                if detail:
                    collector.change_storage.save_changes(key, [detail["change"]])
                    got += 1
                    key_rows.append({"key": key, "repository": spec["path"], "number": number,
                                     "status": "取得", "reason": ""})
                    continue
                reason = "Change の中身の処理に失敗"
            key_rows.append({"key": key, "repository": spec["path"], "number": number,
                             "status": "取得できず", "reason": reason})
            logger.warning(f"[{key}] #{number} を取れませんでした: {reason}")
        totals.append(f"{key}: 抜け {len(missing):,} 件のうち {got:,} 件を取得、{len(missing) - got:,} 件は取れず")
        rows.extend(key_rows)
        _write_csv(OUTPUT_DIR / f"fill_{key}.csv", key_rows, fields)

    for line in totals:
        logger.info(line)
    failed = [r for r in rows if r["status"] != "取得"]
    if failed:
        logger.error(f"**{len(failed)} 件を取れませんでした。fill_<キー>.csv の reason を確認してください**")


def _workers(requested: int | None) -> int:
    """並列に読む本数を決めてログに出す（design.md §4.4）。"""
    workers, reason = parallel_module.decide_workers(requested, RESERVED_CORES, MAX_WORKERS)
    logger.info(f"並列に読む本数：{workers}（{reason}）")
    return workers


def stage_verify(keys: List[str], workers_requested: int | None = None) -> None:
    """第 2 段階の検算：収集した件数を、第 1 段階のランキングの件数と突き合わせる。

    比べる数を揃えるため、収集データも**作成日**で対象期間に絞って数える。
    収集は最終更新が 2022-01-01 以降のものを全部取るので、期間外に作成された Change も含まれている。

    Gerrit は 1 クエリ 10,000 件で**エラーを出さずに**打ち切るため、収集側のログだけでは
    欠損に気づけない。件数が大きく足りなければ、どこかで打ち切られている。
    ファイルは並列に読む（design.md §4.4）。
    """
    from src.utils.constants import GERRIT_PROJECTS, SELECTION_CANDIDATES
    workers = _workers(workers_requested)
    ranking = {}
    with open(OUTPUT_DIR / "ranking_period.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ranking[(r["host"], r["repository"])] = int(r["created_changes"])
    by_path = {}
    for h, spec in hosts_module.HOSTS.items():
        by_path[spec["base"].rstrip("/").removesuffix("/a")] = h

    rows = []
    for key in (keys or SELECTION_CANDIDATES):
        spec = GERRIT_PROJECTS[key]
        host = by_path.get(spec["host"].rstrip("/").removesuffix("/a"))
        expected = ranking.get((host, spec["path"]))
        try:
            created, _branches = local_counts.load(key, workers)
        except FileNotFoundError:
            rows.append({"key": key, "repository": spec["path"], "expected": expected,
                         "collected": 0, "ratio": 0.0, "status": "未収集"})
            continue
        got = sum(1 for c in created if PERIOD_START <= c < PERIOD_END)
        ratio = got / expected if expected else float("nan")
        status = "正常" if ratio >= VERIFY_MIN_RATIO else "★欠損の疑い"
        rows.append({"key": key, "repository": spec["path"], "expected": expected,
                     "collected": got, "ratio": round(ratio, 4), "status": status})
        logger.info(f"{key}: 収集 {got:,} / ランキング {expected:,} = {ratio:.4f}  {status}")

    _write_csv(OUTPUT_DIR / "verify.csv", rows,
               ["key", "repository", "expected", "collected", "ratio", "status"])
    bad = [r for r in rows if r["status"] != "正常"]
    if bad:
        logger.error(f"**{len(bad)} 件で欠損または未収集の疑いがあります。verify.csv を確認してください**")


def stage_bots(directories: List[str]) -> None:
    """第 3 段階：アカウントの活動量を多い順に出し、ボット一覧の作成を支援する。

    ボットの判定はディレクトリ名（GERRIT_PROJECTS のキー）からホストを決めて行う。
    """
    rows = []
    for directory in directories:
        detector = BotDetector(host_of(directory))
        for row in metrics_module.account_ranking(directory, detector):
            rows.append({"directory": directory, **row})
    _write_csv(OUTPUT_DIR / "account_ranking.csv", rows,
               ["directory", "identifier", "name", "created_changes",
                "posted_messages", "judged_bot"])
    logger.info("judged_bot が False の上位アカウントを公式資料と照合し、"
                "出どころの資料が見つかったものだけを src/config/bot_accounts.csv に追記してください"
                "（src/utils/bot_detection.md §2.3・§2.5。README の表も同時に直す）")


def stage_metrics(pairs: List[str], workers_requested: int | None = None) -> None:
    """第 4 段階：収集済みデータから E2・E3 の指標を出す。

    pairs は "ディレクトリ名=リポジトリ名" の並び。ファイルは並列に読む（design.md §4.4）。
    """
    workers = _workers(workers_requested)
    rows = []
    for pair in pairs:
        directory, _, repository = pair.partition("=")
        repository = repository or directory
        detector = BotDetector(host_of(directory))
        result = metrics_module.compute(directory, repository, detector,
                                        PERIOD_START, PERIOD_END, workers)
        rows.append(result.as_row())
    rows.sort(key=lambda r: r["changes"], reverse=True)
    _write_csv(OUTPUT_DIR / "metrics.csv", rows,
               ["repository", "changes", "bot_ratio",
                "human_comments_per_change", "revisions_per_change",
                "reviewers", "reviewed_ratio", "source_ratio", "top10_reviewer_share"])


def stage_branches(pairs: List[str], workers_requested: int | None = None) -> None:
    """収集済みデータから、リポジトリごとのブランチの一覧を出す（--stage branches。通信なし）。

    pairs は "ディレクトリ名=リポジトリ名" の並び（metrics と同じ）。全件を読み、
    changes_period は選定の期間（PERIOD_START〜PERIOD_END）に作成された数。
    """
    workers = _workers(workers_requested)
    rows = []
    for pair in pairs:
        directory, _, repository = pair.partition("=")
        repository = repository or directory
        rows.extend(branches_module.collect(directory, repository,
                                            PERIOD_START, PERIOD_END, workers))
    _write_csv(OUTPUT_DIR / "branches.csv", rows,
               ["repository", "branch", "changes_all", "changes_period",
                "first_created", "last_created"])


def main() -> None:
    parser = argparse.ArgumentParser(description="分析対象リポジトリの選定")
    parser.add_argument("--stage", required=True,
                        choices=["1", "repo-scan", "fill", "verify", "bots", "metrics", "branches"])
    parser.add_argument("--host", action="append",
                        help="第 1 段階・repo-scan で走査するインスタンスを絞る（複数指定可）")
    parser.add_argument("--refill", action="store_true",
                        help="第 1 段階で、全期間を走査せずに ranking_truncated.csv の欠けた日と、"
                             "走査を始めた時刻以降に更新された分だけを取り直す（design.md §3.4）")
    parser.add_argument("--recount", action="store_true",
                        help="第 1 段階で、走査せずに保存してある一覧だけから数え直す（通信なし。design.md §3.5）")
    parser.add_argument("--period-start", default=PERIOD_START,
                        help="--recount で数える期間の始まり（作成日。答え合わせ用。既定は PERIOD_START）")
    parser.add_argument("--period-end", default=PERIOD_END,
                        help="--recount で数える期間の終わり（その日を含まない。既定は PERIOD_END）")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="--recount の出力先（答え合わせでは作業用の場所を指定する。既定は data/project_selection）")
    parser.add_argument("--directory", action="append", default=[],
                        help="bots / metrics / branches で読む収集済みディレクトリ名")
    parser.add_argument("--workers", type=int, default=None,
                        help="verify / metrics / branches で並列に読む本数（OS から読めるコアの数を超えない。"
                             "省略時は MAX_WORKERS と RESERVED_CORES から決める。design.md §4.4）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # 期間と出力先を変えられるのは --recount だけ（走査や取り直しを別の期間・別の場所で走らせないため）
    custom = (args.period_start != PERIOD_START or args.period_end != PERIOD_END
              or args.output_dir != OUTPUT_DIR)
    if custom and not (args.stage == "1" and args.recount):
        raise SystemExit("--period-start / --period-end / --output-dir は --stage 1 --recount でだけ使えます")
    if args.workers is not None:
        if args.stage not in ("verify", "metrics", "branches"):
            raise SystemExit("--workers は --stage verify / metrics / branches でだけ使えます")
        if args.workers < 1:
            raise SystemExit("--workers は 1 以上を指定してください")

    if args.stage == "1":
        if args.recount and args.refill:
            raise SystemExit("--recount と --refill は同時に使えません")
        if args.recount:
            stage_recount(args.period_start, args.period_end, args.output_dir)
        else:
            stage_ranking(args.host, refill=args.refill)
    elif args.stage == "repo-scan":
        stage_repo_scan(args.host or [])
    elif args.stage == "fill":
        stage_fill(args.directory)
    elif args.stage == "verify":
        stage_verify(args.directory, args.workers)
    elif args.stage == "bots":
        if not args.directory:
            raise SystemExit("--directory を 1 つ以上指定してください")
        stage_bots(args.directory)
    elif args.stage == "metrics":
        if not args.directory:
            raise SystemExit("--directory を 1 つ以上指定してください")
        stage_metrics(args.directory, args.workers)
    elif args.stage == "branches":
        if not args.directory:
            raise SystemExit("--directory を 1 つ以上指定してください")
        stage_branches(args.directory, args.workers)


if __name__ == "__main__":
    main()
