"""開発状況の特徴量（22 個。design.md §13.4 と、その「表の補足」）。

リポジトリの全 Change の ChangeFacts から索引（StatusIndex）を 1 度だけ作り、各レコード（Change, 予測の時点 t）で引く。
どの値も t までの記録だけで作る（§13.2）。期間は (t − 期間, t]（t ちょうどを含む）、「3 か月」「6 か月」は 90 日・180 日。
人のコメントかどうかは MessageFact.is_human（bot_detection.is_auto_record）に任せる。
"""
from __future__ import annotations

import bisect
import posixpath
import re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from src.analysis.preliminary_analysis.pretrained_encoders.features.extra import repo_rules
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.common import (
    ChangeFacts, MessageFact, RevisionFact)
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.index import CoreIndex

MISSING = -1.0   # 値が出せないとき（§13.2）

STATUS_FEATURES = [
    "ref_other_changes", "is_in_series", "depends_on_count", "open_same_file_changes", "author_open_changes",
    "changes_created_7d", "changes_created_7d_same_dir", "closed_per_week_60d", "merge_ratio_60d",
    "core_first_response_days", "avg_days_to_merge", "avg_days_to_close", "changes_created_total",
    "merge_ratio_total", "file_recent_merges", "file_developer_count", "file_days_since_change",
    "prior_reviewer_count", "prior_message_count", "prior_first_response_days", "active_core_count",
    "non_core_change_ratio",
]
# 版がないと作れないもの（t に版がなければ −1）
_NEED_REVISION = ["ref_other_changes", "is_in_series", "depends_on_count", "open_same_file_changes",
                  "changes_created_7d_same_dir", "file_recent_merges", "file_developer_count",
                  "file_days_since_change", "prior_reviewer_count", "prior_message_count", "prior_first_response_days"]

DAYS_3M = timedelta(days=90)
DAYS_6M = timedelta(days=180)
_EPOCH = datetime(1970, 1, 1)
_DAY = 86400.0
_DEPENDS_ON_RE = re.compile(r"^Depends-On:", re.M | re.I)


def _sec(dt: datetime) -> float:
    return (dt - _EPOCH).total_seconds()


def _dirs(rev: RevisionFact) -> set[str]:
    return {posixpath.dirname(p) for p in rev.files}


def _has_vote(m: MessageFact) -> bool:
    return any(v != 0 for _lab, v in m.votes)


def _is_reviewer_event(m: MessageFact) -> bool:
    """レビュアーの動き（D26〜D28）：作者・ボット以外の、人のコメントか投票（自動の記録の中の投票も含む）。"""
    return not m.is_bot and not m.is_owner and (m.is_human or _has_vote(m))


def _is_response(m: MessageFact) -> bool:
    """反応（D18）：作者以外の、コメント・投票・マージ・放棄・復活（ボットを除く）。マージ・放棄・復活は記録の種類で判定する。

    Qt の後回し（Deferred）も放棄と同じ状態の変更として含める（2026-10-16 追加）。再開（Reopened）は is_restore に入っている。
    """
    return (not m.is_bot and not m.is_owner
            and (m.is_human or _has_vote(m) or m.is_merge or m.is_abandon or m.is_defer or m.is_restore))


def _count_window(sorted_times, lo: float, hi: float) -> int:
    """(lo, hi] に入る数。"""
    return int(np.searchsorted(sorted_times, hi, side="right") - np.searchsorted(sorted_times, lo, side="right"))


class StatusIndex:
    """開発状況の 22 個を引くための索引（リポジトリごとに 1 つ）。"""

    def __init__(self, facts: list[ChangeFacts], repo: str, core: CoreIndex | None = None):
        self.repo = repo
        self.facts = [f for f in facts if f is not None]
        self.core = core or CoreIndex(self.facts)
        self._pos = {id(f): k for k, f in enumerate(self.facts)}
        self._project_cache: dict[datetime, dict[str, float]] = {}
        self._open_cache: dict[datetime, np.ndarray] = {}
        self._build()

    # ── 索引を作る ─────────────────────────────────────
    def _build(self) -> None:
        F = self.facts
        # 作成（D13・D21・D32）。中核かどうかは作成の時刻で判定（D32）
        created = np.array([_sec(f.created) for f in F])
        order = np.argsort(created, kind="stable")
        self._created_sorted = created[order]
        noncore = np.array([0.0 if self.core.is_core(f.owner_id, f.created) else 1.0 for f in F])
        self._noncore_cum = np.concatenate([[0.0], np.cumsum(noncore[order])])

        # 閉じている状態（D20・D22）：閉じている区間 [閉じた時刻, 再び開いた時刻) を出来事に分け、累積和で引く（§13.2）
        ev = []   # (時刻, 数の増減, 日数の増減, マージの増減)
        for f in F:
            per = f.open_periods
            for i, (_s, e) in enumerate(per):
                if e is None:
                    continue
                dur = (e - f.created).total_seconds() / _DAY
                merged = 1.0 if (f.merged_at is not None and i == len(per) - 1) else 0.0
                ev.append((_sec(e), 1.0, dur, merged))
                if i + 1 < len(per):
                    ev.append((_sec(per[i + 1][0]), -1.0, -dur, -merged))
        ev.sort(key=lambda x: x[0])
        self._closed_t = np.array([e[0] for e in ev])
        self._closed_n = np.concatenate([[0.0], np.cumsum([e[1] for e in ev])])
        self._closed_dur = np.concatenate([[0.0], np.cumsum([e[2] for e in ev])])
        self._closed_merged = np.concatenate([[0.0], np.cumsum([e[3] for e in ev])])

        # マージ（D19。マージは最後の決着なので取り消されない）
        merged = sorted((_sec(f.merged_at), (f.merged_at - f.created).total_seconds() / _DAY)
                        for f in F if f.merged_at is not None)
        self._merge_t = np.array([m for m, _ in merged])
        self._merge_dur_cum = np.concatenate([[0.0], np.cumsum([d for _, d in merged])])

        # 閉じる出来事（D16・D17）：Open の期間の終わりすべて。期間内に 2 度閉じた Change は 1 件と数える
        cl = []
        for f in F:
            ends = [e for _s, e in f.open_periods if e is not None]
            for i, e in enumerate(ends):
                is_merge_ev = f.merged_at is not None and i == len(ends) - 1 and f.open_periods[-1][1] is not None
                nxt = _sec(ends[i + 1]) if i + 1 < len(ends) else np.inf
                cl.append((_sec(e), 1.0 if is_merge_ev else 0.0, nxt))
        cl.sort(key=lambda x: x[0])
        self._ev_t = np.array([c[0] for c in cl])
        self._ev_merge_cum = np.concatenate([[0.0], np.cumsum([c[1] for c in cl])])
        multi = [(c[0], c[2]) for c in cl if np.isfinite(c[2])]
        self._multi_t = np.array([m[0] for m in multi])
        self._multi_next = np.array([m[1] for m in multi])

        # Open の期間（D2・D5・D6）
        per = sorted((_sec(s), _sec(e) if e is not None else np.inf, k)
                     for k, f in enumerate(F) for s, e in f.open_periods)
        self._per_start = np.array([p[0] for p in per])
        self._per_end = np.array([p[1] for p in per])
        self._per_k = np.array([p[2] for p in per], dtype=np.int64)
        owners: dict = {}
        self._owner_code = np.array([owners.setdefault(f.owner_id, len(owners)) for f in F], dtype=np.int64)
        self._owner_bot = np.array([f.owner_is_bot for f in F], dtype=bool)

        # D1：Change-Id → 番号（手元のデータで引けるものは URL の参照と 1 件にまとめる）
        self._cid_numbers: dict[str, set[int]] = defaultdict(set)
        for f in F:
            if f.change_id:
                self._cid_numbers[f.change_id].add(f.number)

        # コミットの SHA（D2）・ファイル・ディレクトリ（D5・D14・D26〜D28）
        self._sha: dict[str, list] = defaultdict(list)          # sha -> [(k, 版の作成時刻)]
        self._file_changes: dict[str, set] = defaultdict(set)   # ファイル -> そのファイルを変えた版を持つ Change
        file_created, dir_created = defaultdict(list), defaultdict(list)
        for k, f in enumerate(F):
            files, dirs = set(), set()
            for r in f.revisions:
                self._sha[r.sha].append((k, r.created))
                files |= set(r.files)
                dirs |= _dirs(r)
            cs = _sec(f.created)
            for p in files:
                self._file_changes[p].add(k)
                file_created[p].append((cs, k))
            for d in dirs:
                dir_created[d].append((cs, k))
        self._file_created = {p: self._split_sorted(v) for p, v in file_created.items()}
        self._dir_created = {d: self._split_sorted(v) for d, v in dir_created.items()}

        # マージされた Change のファイルの履歴（D23〜D25）。ファイルはマージの時点の最新の版で見る
        file_merged, file_dev_first = defaultdict(list), defaultdict(dict)
        for k, f in enumerate(F):
            if f.merged_at is None:
                continue
            rev = f.latest_revision(f.merged_at)
            if rev is None:
                continue
            ms = _sec(f.merged_at)
            for p in rev.files:
                file_merged[p].append((ms, k))
                prev = file_dev_first[p].get(f.owner_id)
                if prev is None or ms < prev:
                    file_dev_first[p][f.owner_id] = ms
        self._file_merged = {p: self._split_sorted(v) for p, v in file_merged.items()}
        self._file_dev = {p: self._split_sorted([(t, o) for o, t in d.items()]) for p, d in file_dev_first.items()}

        # 過去の Change のレビュアー（D26〜D28）：動きの時刻と、レビュアーごとの最初の時刻
        self._rev_msg_t: list[list[float]] = []
        self._rev_first_t: list[list[float]] = []
        for f in F:
            times, first = [], {}
            for m in f.messages:
                if _is_reviewer_event(m):
                    s = _sec(m.date)
                    times.append(s)
                    first.setdefault(m.author_id, s)
            self._rev_msg_t.append(times)
            self._rev_first_t.append(sorted(first.values()))

        # 中核の開発者の最初の反応（D18）。中核かどうかは反応の時刻で判定。ボットが作った Change は除く
        resp = []
        for f in F:
            if f.owner_is_bot:
                continue
            for m in f.messages:
                if _is_response(m) and m.author_id is not None and self.core.is_core(m.author_id, m.date):
                    resp.append((_sec(m.date), (m.date - f.created).total_seconds() / _DAY))
                    break
        resp.sort()
        self._resp_t = np.array([r[0] for r in resp])
        self._resp_days = np.array([r[1] for r in resp])

        # 他人の Change への Code-Review の 0 でない投票（D31）：人ごとの時刻
        cr = defaultdict(list)
        for f in F:
            for m in f.messages:
                if m.is_bot or m.is_owner or m.author_id is None:
                    continue
                if any(lab == "Code-Review" and val != 0 for lab, val in m.votes):
                    cr[m.author_id].append(_sec(m.date))
        self._cr_votes = {a: np.array(sorted(v)) for a, v in cr.items()}

    @staticmethod
    def _split_sorted(pairs: list[tuple]) -> tuple[list, list]:
        pairs.sort(key=lambda x: x[0])
        return [p[0] for p in pairs], [p[1] for p in pairs]

    # ── 引く ─────────────────────────────────────────
    def _open_at(self, t: datetime) -> np.ndarray:
        """t に Open な Change の番号（索引の中の位置）。集合と同じ決まり（§11.1）。"""
        hit = self._open_cache.get(t)
        if hit is None:
            ts = _sec(t)
            m = int(np.searchsorted(self._per_start, ts, side="right"))
            hit = np.unique(self._per_k[:m][self._per_end[:m] > ts])
            if len(self._open_cache) > 4096:
                self._open_cache.clear()
            self._open_cache[t] = hit
        return hit

    def _project(self, t: datetime) -> dict[str, float]:
        """t だけで決まる値（どの Change でも同じ）。自分を除く補正は呼ぶ側でする。"""
        hit = self._project_cache.get(t)
        if hit is not None:
            return hit
        ts = _sec(t)
        out: dict[str, float] = {}
        c_hi = int(np.searchsorted(self._created_sorted, ts, side="right"))
        out["_created_le_t"] = float(c_hi)
        out["_created_7d"] = float(_count_window(self._created_sorted, _sec(t - timedelta(days=7)), ts))
        lo3 = int(np.searchsorted(self._created_sorted, _sec(t - DAYS_3M), side="right"))
        n3 = c_hi - lo3
        out["non_core_change_ratio"] = (self._noncore_cum[c_hi] - self._noncore_cum[lo3]) / n3 if n3 else MISSING
        # D16・D17
        lo60 = _sec(t - timedelta(days=60))
        a = int(np.searchsorted(self._ev_t, lo60, side="right"))
        b = int(np.searchsorted(self._ev_t, ts, side="right"))
        dup = int(np.sum((self._multi_t > lo60) & (self._multi_t <= ts) & (self._multi_next <= ts))) if len(self._multi_t) else 0
        closed60 = (b - a) - dup
        merged60 = self._ev_merge_cum[b] - self._ev_merge_cum[a]
        out["closed_per_week_60d"] = closed60 / (60 / 7)
        out["merge_ratio_60d"] = merged60 / closed60 if closed60 else 0.5
        # D19
        mk = int(np.searchsorted(self._merge_t, ts, side="right"))
        out["avg_days_to_merge"] = self._merge_dur_cum[mk] / mk if mk else MISSING
        # D20・D22：t の時点で閉じているもの
        ck = int(np.searchsorted(self._closed_t, ts, side="right"))
        n_closed = self._closed_n[ck]
        out["avg_days_to_close"] = self._closed_dur[ck] / n_closed if n_closed > 0 else MISSING
        out["merge_ratio_total"] = self._closed_merged[ck] / n_closed if n_closed > 0 else MISSING
        # D18
        ra = int(np.searchsorted(self._resp_t, _sec(t - DAYS_3M), side="right"))
        rb = int(np.searchsorted(self._resp_t, ts, side="right"))
        out["core_first_response_days"] = float(np.median(self._resp_days[ra:rb])) if rb > ra else MISSING
        # D31：中核かどうかは t で判定
        lo3s = _sec(t - DAYS_3M)
        out["active_core_count"] = float(sum(
            1 for a_id, first in self.core.first_pm2.items()
            if first < t and a_id in self._cr_votes and _count_window(self._cr_votes[a_id], lo3s, ts) > 0))
        if len(self._project_cache) > 4096:
            self._project_cache.clear()
        self._project_cache[t] = out
        return out

    def _ref_other_changes(self, facts: ChangeFacts, rev: RevisionFact, t: datetime) -> float:
        texts = [rev.message] + [m.text for m in facts.messages_until(t) if m.is_human]
        refs = repo_rules.referenced_changes(self.repo, texts, facts.number, facts.change_id)
        numbers = {int(r[2:]) for r in refs if r.startswith("n:")}
        count = len(numbers)
        seen_ids: set = set()
        for r in refs:
            if not r.startswith("i:"):
                continue
            nums = self._cid_numbers.get(r[2:], set()) - {facts.number}
            if nums & numbers:
                continue                       # 同じ Change を URL でも参照している
            key = frozenset(nums) if nums else r
            if key not in seen_ids:
                seen_ids.add(key)
                count += 1
        return float(count)

    def features(self, facts: ChangeFacts, t: datetime) -> dict[str, float]:
        """そのレコード（Change, t）の開発状況の 22 個。"""
        k_self = self._pos.get(id(facts))
        if k_self is None:
            raise ValueError("索引に入っていない Change です（StatusIndex を作った一覧の ChangeFacts を渡してください）")
        ts = _sec(t)
        proj = self._project(t)
        own_c = _sec(facts.created)
        out: dict[str, float] = {
            "changes_created_7d": proj["_created_7d"] - (1 if _sec(t - timedelta(days=7)) < own_c <= ts else 0),
            "changes_created_total": proj["_created_le_t"] - (1 if own_c <= ts else 0),
        }
        for key in ("closed_per_week_60d", "merge_ratio_60d", "core_first_response_days", "avg_days_to_merge",
                    "avg_days_to_close", "merge_ratio_total", "active_core_count", "non_core_change_ratio"):
            out[key] = proj[key]

        # D6：作者の Open な Change（ボットが作ったものは数えない。作者がボットなら −1）
        open_ks = self._open_at(t)
        if facts.owner_is_bot:
            out["author_open_changes"] = MISSING
        else:
            same = (self._owner_code[open_ks] == self._owner_code[k_self]) & ~self._owner_bot[open_ks] & (open_ks != k_self)
            out["author_open_changes"] = float(np.sum(same))

        rev = facts.latest_revision(t)
        if rev is None:
            out.update({k: MISSING for k in _NEED_REVISION})
            return {k: float(out[k]) for k in STATUS_FEATURES}

        files = set(rev.files)
        out["ref_other_changes"] = self._ref_other_changes(facts, rev, t)
        out["depends_on_count"] = float(len(_DEPENDS_ON_RE.findall(rev.message)))
        # D2：親のコミットが、t に Open なほかの Change の t までの版のコミットか
        open_set = set(open_ks.tolist())
        out["is_in_series"] = float(any(
            k != k_self and rc <= t and k in open_set for p in rev.parent_shas for k, rc in self._sha.get(p, ())))
        # D5：同じファイルを変える Open な Change（相手は t までの最新の版）
        cand = set()
        for p in files:
            cand |= self._file_changes.get(p, set())
        cnt = 0
        for k in cand & open_set:
            if k == k_self:
                continue
            r = self.facts[k].latest_revision(t)
            if r is not None and files & set(r.files):
                cnt += 1
        out["open_same_file_changes"] = float(cnt)
        # D14：直近 7 日に作られ、同じディレクトリを変えているほかの Change
        lo7 = _sec(t - timedelta(days=7))
        hits = set()
        for d in _dirs(rev):
            ct, ck = self._dir_created.get(d, ([], []))
            for k in ck[bisect.bisect_right(ct, lo7):bisect.bisect_right(ct, ts)]:
                if k != k_self and k not in hits:
                    r = self.facts[k].latest_revision(t)
                    if r is not None and d in _dirs(r):
                        hits.add(k)
        out["changes_created_7d_same_dir"] = float(len(hits))
        # D23〜D25：ファイルの履歴
        lo3 = _sec(t - DAYS_3M)
        recent, devs, gaps = set(), set(), []
        for p in files:
            mt, mk = self._file_merged.get(p, ([], []))
            hi = bisect.bisect_right(mt, ts)
            recent.update(k for k in mk[bisect.bisect_right(mt, lo3):hi] if k != k_self)
            if hi:
                gaps.append((ts - mt[hi - 1]) / _DAY)
            dt_, do = self._file_dev.get(p, ([], []))
            devs.update(do[:bisect.bisect_right(dt_, ts)])
        out["file_recent_merges"] = float(len(recent))
        out["file_developer_count"] = float(len(devs))
        out["file_days_since_change"] = float(np.mean(gaps)) if gaps else MISSING
        # D26〜D28：直近 180 日の、同じファイル・同じブランチの過去の Change（ボットが作ったものと自分を除く）
        lo6 = _sec(t - DAYS_6M)
        prior = set()
        for p in files:
            ct, ck = self._file_created.get(p, ([], []))
            for k in ck[bisect.bisect_right(ct, lo6):bisect.bisect_right(ct, ts)]:
                if k == k_self or k in prior:
                    continue
                f = self.facts[k]
                if f.owner_is_bot or f.branch != facts.branch:
                    continue
                r = f.latest_revision(t)
                if r is not None and files & set(r.files):
                    prior.add(k)
        if prior:
            out["prior_reviewer_count"] = float(np.median([bisect.bisect_right(self._rev_first_t[k], ts) for k in prior]))
            out["prior_message_count"] = float(np.median([bisect.bisect_right(self._rev_msg_t[k], ts) for k in prior]))
            firsts = [(self._rev_msg_t[k][0] - _sec(self.facts[k].created)) / _DAY
                      for k in prior if self._rev_msg_t[k] and self._rev_msg_t[k][0] <= ts]
            out["prior_first_response_days"] = float(np.median(firsts)) if firsts else MISSING
        else:
            out["prior_reviewer_count"] = out["prior_message_count"] = out["prior_first_response_days"] = MISSING
        return {k: float(out[k]) for k in STATUS_FEATURES}
