"""特徴量ベクトルの組み立て（§3.4, §5.4）。

既存 `src/features/*` の計算関数を呼び、計測時点 T で観測可能な情報だけから 15 次元のベクトルを作る。
`review_metrics.calculate_uncompleted_requests`（未完了リクエスト数）は除外する（§3.4）。

developer/project 系の特徴は「全 Change の DataFrame」と「リリース日 df」を参照するため、
それらは `build_context` で一度だけ組み立て、各レコードの計算で使い回す（キャッシュ方針）。
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.analysis.background_problem.common.time_utils import parse_dt
from src.analysis.preliminary_analysis.concept_drift_existence.features.fast_index import FastFeatureIndex
from src.features import bug_metrics, change_metrics, developer_metrics, project_metrics, refactoring_metrics

# 特徴量の並び順（固定）。uncompleted_requests は除外（§3.4）。
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


def _merged_time(change: dict) -> datetime | None:
    """マージ時刻（status==MERGED のとき updated を採用、それ以外は None）。merge_rate 用。"""
    if change.get("status") == "MERGED":
        return parse_dt(change.get("updated"))
    return None


def _decision_time(change: dict) -> datetime | None:
    """決着時刻（status が MERGED または ABANDONED のとき updated。未決は None）。open_ticket_count 用。"""
    if change.get("status") in ("MERGED", "ABANDONED"):
        return parse_dt(change.get("updated"))
    return None


def build_all_prs_df(changes: list[dict]) -> pd.DataFrame:
    """全 Change から developer/project 特徴に必要な DataFrame を一度だけ組み立てる。

    列: owner_email, created, merged, decision_time, updated, lines_added, lines_deleted
    - merged: マージ時刻（MERGED のみ）。merge_rate / recent_merge_rate 用。
    - decision_time: 決着時刻（MERGED または ABANDONED）。open_ticket_count 用（放棄も閉じた扱い）。
    （計算は1回きり。レコードごとの特徴計算で参照して使い回す。）
    """
    rows = []
    for c in changes:
        created = parse_dt(c.get("created"))
        if created is None:
            continue
        rows.append({
            "owner_email": developer_metrics.get_owner_email(c),
            "created": created,
            "merged": _merged_time(c),
            "decision_time": _decision_time(c),
            "updated": parse_dt(c.get("updated")),
            "lines_added": change_metrics.calculate_lines_added(c),
            "lines_deleted": change_metrics.calculate_lines_deleted(c),
        })
    df = pd.DataFrame(rows, columns=["owner_email", "created", "merged", "decision_time",
                                     "updated", "lines_added", "lines_deleted"])
    for col in ("created", "merged", "decision_time", "updated"):
        df[col] = pd.to_datetime(df[col])
    return df


def build_releases_df(releases_df: pd.DataFrame, project: str) -> pd.DataFrame:
    """common.load_release_dates の df を project_metrics 用（component 列）に整える。"""
    df = releases_df[releases_df["project"] == project].copy()
    df = df.rename(columns={"project": "component"})
    return df[["component", "version", "release_date"]]


def build_index(all_prs_df: pd.DataFrame) -> FastFeatureIndex:
    """developer/project 特徴の高速インデックスを 1 回だけ作る（§3.4）。"""
    return FastFeatureIndex(all_prs_df)


def build_features(change: dict, t: datetime, index: FastFeatureIndex,
                   comp_releases_df: pd.DataFrame, project: str) -> list[float]:
    """1 レコード (change, T) の 15 次元特徴ベクトルを返す（T 時点で観測可能な情報のみ）。

    安い特徴（per-Change）は src/features をそのまま使い、developer/project 系の重い 6 特徴は
    高速インデックス（事前集計＋二分探索）で引く。days_to_major_release は releases_df が小さいので
    src/features をそのまま使う。
    """
    subject, message = change_metrics.get_change_text_data(change)
    email = developer_metrics.get_owner_email(change)

    return [
        float(bug_metrics.calculate_bug_fix_confidence(subject, message)),
        float(change_metrics.calculate_lines_added(change, t)),
        float(change_metrics.calculate_lines_deleted(change, t)),
        float(change_metrics.calculate_files_changed(change, t)),
        float(change_metrics.calculate_elapsed_time(change, t)),
        float(change_metrics.calculate_revision_count(change, t)),
        float(change_metrics.check_test_code_presence(change)),
        float(index.past_report_count(email, t)),
        float(index.recent_report_count(email, t)),
        float(index.merge_rate(email, t)),
        float(index.recent_merge_rate(email, t)),
        float(project_metrics.calculate_days_to_major_release(t, project, comp_releases_df)),
        float(index.open_ticket_count(t)),
        float(index.reviewed_lines_in_period(t)),
        float(refactoring_metrics.calculate_refactoring_confidence(subject, message)),
    ]
