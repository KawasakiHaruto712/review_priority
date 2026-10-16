"""リポジトリごとの読み方（design.md §13.3〜§13.5）。

- CI の結果の読み方（`ci_*`・`days_since_last_ci`）
- 課題への参照の書き方（`has_issue_ref`）
- ほかの Change を指す URL の形（`ref_other_changes`）

どれも公式の資料か手元のデータで確かめた書き方だけを使う（独自の一覧を作らない。design.md §13.1）。
確認の記録は data/feature_survey/source_check2〜4_20261016.md・qt_ci_messages2_result.txt。
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from src.utils.constants import GERRIT_PROJECTS

# ── CI の結果（design.md §13.5 の表） ───────────────────────────
# kind="qt_integration"：Qt CI Bot のメッセージの "Continuous Integration: Failed/Passed"（Cancelled は数えない）
# kind="vote"：指定の投票の名前の ±（labels）。authors を指定すれば、その名前のアカウントの投票だけ、
#              指定しなければボットの投票だけを数える
CI_RULES: dict[str, dict] = {
    # 2026-10-16 改めた：Qt の 3 件は CI の結果を読まない（design.md §13.5。読み方が手元のデータで決めたもので、公式の根拠がないため）。
    # 以前の読み方：qtbase・qtdeclarative は Qt CI Bot の "Continuous Integration: Failed/Passed"、
    # qtcreator は Qt CI Bot・Qt CI Patch Build Bot の Verified ±1
    "qtbase":                   {"kind": "none"},
    "qtdeclarative":            {"kind": "none"},
    "qtcreator":                {"kind": "none"},
    "wikimedia_mediawiki_core": {"kind": "vote", "labels": {"Verified"}, "authors": None},
    "typo3_cms":                {"kind": "vote", "labels": {"Verified"}, "authors": None},
    "go":                       {"kind": "vote", "labels": {"TryBot-Result", "LUCI-TryBot-Result"}, "authors": None},
}

_QT_CI_RE = re.compile(r"Continuous Integration:\s*(\w+)", re.I)


def uses_ci(repo: str) -> bool:
    """そのリポジトリで CI の結果を読むか（読まないなら CI の特徴量はすべて −1。design.md §13.5）。"""
    rule = CI_RULES.get(repo)
    if rule is None:
        raise KeyError(f"CI の読み方が決まっていないリポジトリです（repo_rules.CI_RULES に足してください）: {repo}")
    return rule["kind"] != "none"


def ci_result_of(repo: str, author_name: str, author_is_bot: bool, text: str,
                 votes: list[tuple[str, int]]) -> int | None:
    """1 件のメッセージから CI の結果を読む。成功 1・失敗 0・CI の結果でなければ None。"""
    rule = CI_RULES.get(repo)
    if rule is None:
        raise KeyError(f"CI の読み方が決まっていないリポジトリです（repo_rules.CI_RULES に足してください）: {repo}")
    if rule["kind"] == "none":
        return None
    if rule["authors"] is not None and author_name not in rule["authors"]:
        return None
    if rule["kind"] == "qt_integration":
        m = _QT_CI_RE.search(text or "")
        if not m:
            return None
        word = m.group(1).lower()
        if word.startswith("fail"):
            return 0
        if word.startswith("pass"):
            return 1
        return None                                   # Cancelled など
    if rule["authors"] is None and not author_is_bot:
        return None
    for label, value in votes:
        if label in rule["labels"] and value != 0:
            return 1 if value > 0 else 0
    return None


# ── 課題への参照（design.md §13.3 has_issue_ref） ─────────────────
# Qt：.commit-template のフッター、mediawiki：Commit message guidelines の `Bug:`、
# go：Contribution Guide の "Fixes #N"・"Updates #N"（x リポジトリは "Fixes golang/go#N"）、
# TYPO3：CommitMessage の手引きの `Resolves:`・`Related:`（`Releases:` は版の一覧なので数えない）
_ISSUE_REF_RE: dict[str, re.Pattern] = {
    "qtbase":                   re.compile(r"^(Task-number|Fixes):", re.M | re.I),
    "qtdeclarative":            re.compile(r"^(Task-number|Fixes):", re.M | re.I),
    "qtcreator":                re.compile(r"^(Task-number|Fixes):", re.M | re.I),
    "wikimedia_mediawiki_core": re.compile(r"^Bug:", re.M | re.I),
    "go":                       re.compile(r"\b(Fixes|Updates)\s+(?:[\w.-]+/[\w.-]+)?#\d+", re.I),
    "typo3_cms":                re.compile(r"^(Resolves|Related):", re.M | re.I),
}


def has_issue_ref(repo: str, commit_message: str) -> int:
    rx = _ISSUE_REF_RE.get(repo)
    if rx is None:
        raise KeyError(f"課題への参照の書き方が決まっていないリポジトリです: {repo}")
    return int(bool(rx.search(commit_message or "")))


# ── ほかの Change への参照（design.md §13.4 ref_other_changes） ────────
CHANGE_ID_RE = re.compile(r"\bI[0-9a-f]{40}\b")
_REVIEWED_ON_RE = re.compile(r"^Reviewed-on:.*$", re.M | re.I)


def _public_prefix(repo: str) -> str:
    """登録表のホストから、公開の URL の接頭辞（スキームなし）を作る。Qt の認証用の /a は外す。"""
    u = urlparse(GERRIT_PROJECTS[repo]["host"])
    path = u.path.rstrip("/")
    if path == "/a":
        path = ""
    return u.netloc + path


def change_url_re(repo: str) -> re.Pattern:
    """そのホストの Change の URL（公式で確かめた 4 つの形。+ は %2B も、後ろに版などが付いてもよい）。

    /<番号>、/#/c/<番号>、/c/<リポジトリ>/+/<番号>、/#/c/<リポジトリ>/+/<番号>（Gerrit 2.15 のリリースノートほか）。
    go は go.dev/cl/<番号>・golang.org/cl/<番号> も（go.dev のブログと転送のコード）。
    """
    p = re.escape(_public_prefix(repo))
    forms = [rf"https?://{p}/(\d+)(?![\w])",
             rf"https?://{p}/#/c/(\d+)(?![\w])",
             rf"https?://{p}/(?:#/)?c/[^\s+%]+?/(?:\+|%2B)/(\d+)(?![\w])"]
    if repo == "go":
        forms.append(r"https?://(?:go\.dev|golang\.org)/cl/(\d+)(?![\w])")
    return re.compile("|".join(f"(?:{f})" for f in forms))


def referenced_changes(repo: str, texts: list[str], own_number: int, own_change_id: str) -> set:
    """テキストの中の、ほかの Change への参照（番号 'n:<番号>' と Change-Id 'i:<Id>' の集合）。

    `Reviewed-on:` の行（自分の Change を指すフッター）と、自分の番号・Change-Id は除く。
    """
    url_re = change_url_re(repo)
    found: set = set()
    for text in texts:
        if not text:
            continue
        text = _REVIEWED_ON_RE.sub("", text)
        for cid in CHANGE_ID_RE.findall(text):
            if cid != own_change_id:
                found.add("i:" + cid)
        for m in url_re.finditer(text):
            num = next(g for g in m.groups() if g)
            if int(num) != own_number:
                found.add("n:" + num)
    return found
