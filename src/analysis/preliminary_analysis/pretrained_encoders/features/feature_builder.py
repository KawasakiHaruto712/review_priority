"""特徴量ベクトルの組み立て（design.md §2）。

既存 `src/features/*` の計算関数を呼び、計測時点 T で観測可能な情報だけから 15 次元のベクトルを作る。
developer/project 系の特徴は「全 Change の DataFrame」と「リリース日 df」を参照するため、
それらは一度だけ組み立てて各レコードの計算で使い回す（キャッシュ方針）。
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.analysis.background_problem.common.time_utils import parse_dt
from src.analysis.preliminary_analysis.pretrained_encoders.features.fast_index import FastFeatureIndex
from src.analysis.preliminary_analysis.pretrained_encoders.utils import review_utils
from src.features import bug_metrics, change_metrics, developer_metrics, project_metrics, refactoring_metrics
from src.utils.bot_detection import BotDetector

# 特徴量の並び順（固定）。uncompleted_requests は除外。
FEATURE_NAMES = [
    "bug_fix_confidence",
    "lines_added",
    "lines_deleted",
    "files_changed",
    "elapsed_time",
    "revision_count",
    "test_code_presence",
    "past_report_count",
    "recent_report_count",
    "merge_rate",
    "recent_merge_rate",
    "days_to_major_release",
    "open_ticket_count",
    "reviewed_lines_in_period",
    "refactoring_confidence",
]


def _revision_timeline(change: dict) -> tuple[list[datetime], list[int]]:
    """版（パッチセット）ごとの (作成時刻, 追加＋削除行数) を作成時刻の昇順で返す。reviewed_lines_in_period 用。"""
    revs = []
    for rev in (change.get("revisions") or {}).values():
        created = parse_dt(rev.get("created"))
        if created is None:
            continue
        files = rev.get("files") or {}
        lines = sum(int(f.get("lines_inserted", 0) or 0) + int(f.get("lines_deleted", 0) or 0)
                    for f in files.values() if isinstance(f, dict)) if isinstance(files, dict) else 0
        revs.append((created, lines))
    revs.sort(key=lambda r: r[0])
    return [r[0] for r in revs], [r[1] for r in revs]


def build_all_prs_df(changes: list[dict], detector: BotDetector) -> pd.DataFrame:
    """全 Change から developer/project 特徴に必要な DataFrame を一度だけ組み立てる。

    列: owner_email, created, merged, decision_time, updated, open_periods, review_times, rev_times, rev_lines
      merged / decision_time / open_periods  Open の期間から作る（design.md §11.1。旧版は updated）
      review_times                           人間のレビューの時刻（正解ラベルと同じ判定。§11.2）
      rev_times / rev_lines                  版ごとの作成時刻と行数（T の時点の版の行数を引くため。§11.2）
      updated                                観測末尾（data_end）を決めるためだけに使う
    """
    rows = []
    for c in changes:
        created = parse_dt(c.get("created"))
        if created is None:
            continue
        periods = review_utils.open_periods(c)
        last_close = periods[-1][1] if periods else None
        rev_times, rev_lines = _revision_timeline(c)
        rows.append({
            "owner_email": developer_metrics.get_owner_email(c),
            "created": created,
            "merged": last_close if c.get("status") == "MERGED" else None,
            "decision_time": last_close,
            "updated": parse_dt(c.get("updated")),
            "open_periods": periods,
            "review_times": review_utils.human_comment_times(c, detector),
            "rev_times": rev_times,
            "rev_lines": rev_lines,
        })
    df = pd.DataFrame(rows, columns=["owner_email", "created", "merged", "decision_time", "updated",
                                     "open_periods", "review_times", "rev_times", "rev_lines"])
    for col in ("created", "merged", "decision_time", "updated"):
        df[col] = pd.to_datetime(df[col])
    return df


def build_releases_df(releases_df: pd.DataFrame, project: str) -> pd.DataFrame:
    """common.load_release_dates の df を project_metrics 用（component 列）に整える。"""
    df = releases_df[releases_df["project"] == project].copy()
    df = df.rename(columns={"project": "component"})
    return df[["component", "version", "release_date"]]


def build_index(all_prs_df: pd.DataFrame) -> FastFeatureIndex:
    """developer/project 特徴の高速インデックスを 1 回だけ作る（design.md §2）。"""
    return FastFeatureIndex(all_prs_df)


def build_features(change: dict, t: datetime, index: FastFeatureIndex,
                   comp_releases_df: pd.DataFrame, project: str) -> list[float]:
    """1 レコード (change, T) の 15 次元特徴ベクトルを返す（T 時点で観測可能な情報のみ）。"""
    # 件名・説明文は T の時点で最新の版のもの（design.md §11.3。旧版は最新の版）
    subject, message = change_metrics.get_change_text_data(change, t)
    email = developer_metrics.get_owner_email(change)

    return [
        float(bug_metrics.calculate_bug_fix_confidence(subject, message)),
        float(change_metrics.calculate_lines_added(change, t)),
        float(change_metrics.calculate_lines_deleted(change, t)),
        float(change_metrics.calculate_files_changed(change, t)),
        float(change_metrics.calculate_elapsed_time(change, t)),
        float(change_metrics.calculate_revision_count(change, t)),
        float(change_metrics.check_test_code_presence(change, t)),
        float(index.past_report_count(email, t)),
        float(index.recent_report_count(email, t)),
        float(index.merge_rate(email, t)),
        float(index.recent_merge_rate(email, t)),
        float(project_metrics.calculate_days_to_major_release(t, project, comp_releases_df)),
        float(index.open_ticket_count(t)),
        float(index.reviewed_lines_in_period(t)),
        float(refactoring_metrics.calculate_refactoring_confidence(subject, message)),
    ]
