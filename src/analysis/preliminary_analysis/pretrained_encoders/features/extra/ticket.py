"""チケット自身の特徴量（13 個。design.md §13.3）。t までの最新の版から作る。"""
from __future__ import annotations

import math
import posixpath
import re
from datetime import datetime

import pandas as pd

from src.analysis.preliminary_analysis.pretrained_encoders.features.extra import repo_rules
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.common import (
    ChangeFacts, strip_trailers, word_count)
from src.config.path import DEFAULT_DATA_DIR

MAIN_BRANCHES_CSV = DEFAULT_DATA_DIR / "openstack_collected" / "main_branches.csv"
MISSING = -1.0   # 値が出せないとき（design.md §13.2）

TICKET_FEATURES = [
    "is_main_branch", "dir_count", "subsystem_count", "change_entropy", "file_ext_count",
    "files_added", "files_deleted", "subject_word_count", "message_word_count",
    "is_documentation", "is_feature", "has_issue_ref", "has_mention",
]

# Islam 2022・Thongtanunam 2017 の語（design.md §13.3）。大文字・小文字を区別せず、語の一部としての一致も含める
_DOC_WORDS = ("doc", "copyright", "license")
_BUG_WORDS = ("bug", "fix", "defect")
# `@` に続く名前（直前が英数字・ドットならメールアドレスの一部なので除く）
_MENTION_RE = re.compile(r"(?<![\w.])@[A-Za-z0-9_][\w.-]*")


def load_main_branches(repo: str, csv_path=MAIN_BRANCHES_CSV) -> set[str]:
    """本流の表から、そのリポジトリの今と昔の本流の名前（design.md §12.2）。行がなければ止める。"""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    names = set(df.loc[df["project"] == repo, "branch"])
    if not names:
        raise ValueError(f"本流の表に {repo} の行がありません。RELEASE_TAG_TARGETS に足して表を作り直してください"
                         f"（python -m src.collectors.build_release_table）")
    return names


def _top_dir(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else ""      # 最上位に置かれたファイルは「最上位」（空文字）


def _ext(path: str) -> str:
    """拡張子。Python の os.path.splitext の決まり（先頭の `.` は拡張子とみなさない：`.gitignore` は拡張子なし）。

    拡張子のないファイルも 1 種類と数える（空文字）。2026-10-16 本人判断（独自の決まりにせず、標準の決まりに合わせる）。
    """
    return posixpath.splitext(posixpath.basename(path))[1].lstrip(".")


def _contains_any(text: str, words) -> bool:
    low = (text or "").lower()
    return any(w in low for w in words)


def ticket_features(facts: ChangeFacts, t: datetime, repo: str, main_branches: set[str]) -> dict[str, float]:
    """チケット自身の 13 個。t までに版がなければ、版から作るものは −1。"""
    out = {"is_main_branch": float(facts.branch in main_branches)}
    rev = facts.latest_revision(t)
    if rev is None:
        out.update({k: MISSING for k in TICKET_FEATURES if k != "is_main_branch"})
        return out
    paths = list(rev.files)
    churn = [ins + dele for ins, dele, _st in rev.files.values()]
    total = sum(churn)
    entropy = 0.0
    if total > 0:
        for c in churn:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
    is_doc = _contains_any(rev.subject, _DOC_WORDS)
    out.update({
        "dir_count": float(len({posixpath.dirname(p) for p in paths})),
        "subsystem_count": float(len({_top_dir(p) for p in paths})),
        "change_entropy": entropy,
        "file_ext_count": float(len({_ext(p) for p in paths})),
        "files_added": float(sum(1 for _i, _d, st in rev.files.values() if st == "A")),
        "files_deleted": float(sum(1 for _i, _d, st in rev.files.values() if st == "D")),
        "subject_word_count": float(word_count(rev.subject)),
        "message_word_count": float(word_count(strip_trailers(rev.message))),
        "is_documentation": float(is_doc),
        "is_feature": float(not is_doc and not _contains_any(rev.subject, _BUG_WORDS)),
        "has_issue_ref": float(repo_rules.has_issue_ref(repo, rev.message)),
        "has_mention": float(bool(_MENTION_RE.search(rev.message))),
    })
    return out
