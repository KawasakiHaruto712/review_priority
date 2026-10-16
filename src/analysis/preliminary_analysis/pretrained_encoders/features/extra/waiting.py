"""待っている間の動きの特徴量（17 個。design.md §13.5 と、その「表の補足」）。そのチケット自身の、作成から t までの出来事から作る。

人のコメントかどうかは MessageFact.is_human（bot_detection.is_auto_record）に任せる。作者を除くかは特徴量ごと（§13.2）：
W8・W9・W18 は作者以外、W13 は作者だけ、W10 は作者を含む。
CI の結果を読まないリポジトリ（Qt の 3 件。repo_rules.uses_ci）では CI の 5 つ（W6・W15 の 3 つ・W16）をすべて −1 にする。
"""
from __future__ import annotations

from datetime import datetime

from src.analysis.preliminary_analysis.pretrained_encoders.features.extra import repo_rules
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.common import ChangeFacts
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.ticket import _MENTION_RE

MISSING = -1.0   # 値が出せないとき（§13.2）
_DAY = 86400.0

WAITING_FEATURES = [
    "created_weekday_utc", "created_hour_utc", "days_since_last_ci", "review_comment_count", "commenter_count",
    "inline_comment_count", "weighted_vote_score", "author_comment_count", "bot_event_count",
    "first_response_is_bot", "ci_last_result", "ci_fail_ratio", "ci_run_count", "recheck_count",
    "avg_days_between_revisions", "last_comment_has_mention", "was_restored",
]
CI_FEATURES = ["days_since_last_ci", "ci_last_result", "ci_fail_ratio", "ci_run_count", "recheck_count"]


def _days(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / _DAY


def weighted_vote_score(facts: ChangeFacts, t: datetime) -> float:
    """Islam 2022 の Weighted approval score。WAS(r) ＝ Σ_{j=1}^{r} j/(j+1) × S_j（§13.5 W12）。

    r は t までに作られた最新の版の番号（0 始まり）。S_j は版 j−1 の作成から版 j の作成までの投票の値の合計
    （両端を含む。投票の種類はすべて、値 0 は除く、同じ人もまとめない。自動の記録の中の投票・古い形式の投票も入る）。
    """
    revs = facts.revisions_until(t)
    msgs = facts.messages_until(t)
    total = 0.0
    for j in range(1, len(revs)):
        lo, hi = revs[j - 1].created, revs[j].created
        s = sum(v for m in msgs if lo <= m.date <= hi for _lab, v in m.votes if v != 0)
        total += j / (j + 1) * s
    return float(total)


def waiting_features(facts: ChangeFacts, t: datetime, repo: str) -> dict[str, float]:
    """そのレコード（Change, t）の待っている間の動きの 17 個。"""
    msgs = facts.messages_until(t)
    human = [m for m in msgs if m.is_human]
    reviewer_comments = [m for m in human if not m.is_owner]
    revs = facts.revisions_until(t)
    gaps = [_days(a.created, b.created) for a, b in zip(revs, revs[1:])]
    first_resp = next((m for m in msgs if not m.is_owner and (m.is_bot or m.is_human)), None)

    out = {
        "created_weekday_utc": float((facts.created.weekday() + 1) % 7),     # 日曜 0〜土曜 6
        "created_hour_utc": float(facts.created.hour),
        "review_comment_count": float(len(reviewer_comments)),
        "commenter_count": float(len({m.author_id for m in reviewer_comments})),
        "inline_comment_count": float(sum(m.inline_count for m in human)),    # 作者を含む（出典どおり）
        "weighted_vote_score": weighted_vote_score(facts, t),
        "author_comment_count": float(sum(1 for m in human if m.is_owner)),
        "bot_event_count": float(sum(1 for m in msgs if m.is_bot)),
        "first_response_is_bot": float(first_resp is not None and first_resp.is_bot),
        "avg_days_between_revisions": sum(gaps) / len(gaps) if gaps else MISSING,
        "last_comment_has_mention": float(bool(reviewer_comments)
                                          and bool(_MENTION_RE.search(reviewer_comments[-1].text))),
        "was_restored": float(any(m.is_restore for m in msgs)),
    }

    if not repo_rules.uses_ci(repo):
        out.update({k: MISSING for k in CI_FEATURES})
    else:
        ci = [m for m in msgs if m.ci_result is not None]
        recheck, last_ci = 0, None
        for m in msgs:                                   # 日時の昇順
            if m.ci_result is not None:
                last_ci = m.ci_result
            elif m.is_human and last_ci == 0 and "recheck" in m.text.lower():
                recheck += 1                             # 直前の CI の結果が失敗のときだけ
        out.update({
            "days_since_last_ci": _days(ci[-1].date, t) if ci else MISSING,
            "ci_last_result": float(ci[-1].ci_result) if ci else MISSING,
            "ci_fail_ratio": sum(1 for m in ci if m.ci_result == 0) / len(ci) if ci else MISSING,
            "ci_run_count": float(len(ci)),
            "recheck_count": float(recheck),
        })
    return {k: float(out[k]) for k in WAITING_FEATURES}
