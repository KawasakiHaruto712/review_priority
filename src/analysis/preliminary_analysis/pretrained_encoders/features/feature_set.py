"""モデルに入れる特徴量の一覧（design.md §12.5。2026-10-18 追加）。

相関で残した特徴量の名前を、ここにコードとして直接書く（selection の CSV を実行時に読まない：CSV が変わったときに、
知らないうちに特徴量が変わるのを防ぐため）。名前は feature_table の表の 67 個（ALL_FEATURE_NAMES）にあるものだけ。
モデルの入力の並び（FEATURE_DIM・Scaler・config.json・寄与度分析）はすべてこの一覧を使う。
このファイルは表の値に効かないので、表の指紋の材料に入れない（feature_table/design.md §2.5）。

2026-10-18 決定：相関の決まりは割合 2/3（|ρ| ≥ 0.7 になったリポジトリが、ρ の出るリポジトリの 2/3 以上の組を相関が強いとし、
AutoSpearman の 1 段目で落とす。feature_table/design.md §3.2）。67 個のうち残った 55 個を、表の並び（ALL_FEATURE_NAMES の順）で書く。
落とした 12 個：files_changed・elapsed_time・past_report_count（既存）、change_entropy・avg_days_to_merge・changes_created_total・
file_developer_count・prior_message_count・review_comment_count・bot_event_count・ci_run_count・ci_fail_ratio。
（selection の出力に割合 2/3 の決まりを足すのは後で行う。この一覧は保存済みの相関行列からの試算による。）
"""

MODEL_FEATURES: list[str] = [
    # 既存の 15 個から残った 12 個
    "bug_fix_confidence", "lines_added", "lines_deleted", "revision_count", "test_code_presence",
    "recent_report_count", "merge_rate", "recent_merge_rate", "days_to_feature_release", "open_ticket_count",
    "reviewed_lines_in_period", "refactoring_confidence",
    # チケット自身（13 個から残った 12 個）
    "is_main_branch", "dir_count", "subsystem_count", "file_ext_count", "files_added", "files_deleted",
    "subject_word_count", "message_word_count", "is_documentation", "is_feature", "has_issue_ref", "has_mention",
    # 開発状況（22 個から残った 18 個）
    "ref_other_changes", "is_in_series", "depends_on_count", "open_same_file_changes", "author_open_changes",
    "changes_created_7d", "changes_created_7d_same_dir", "closed_per_week_60d", "merge_ratio_60d",
    "core_first_response_days", "avg_days_to_close", "merge_ratio_total", "file_recent_merges",
    "file_days_since_change", "prior_reviewer_count", "prior_first_response_days", "active_core_count",
    "non_core_change_ratio",
    # 待っている間の動き（17 個から残った 13 個）
    "created_weekday_utc", "created_hour_utc", "days_since_last_ci", "commenter_count", "inline_comment_count",
    "weighted_vote_score", "author_comment_count", "first_response_is_bot", "ci_last_result", "recheck_count",
    "avg_days_between_revisions", "last_comment_has_mention", "was_restored",
]
