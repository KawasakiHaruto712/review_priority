"""
Release Impact Analysis用の定数定義
リリースペアとその分析期間の定義
"""

# リリースペアの定義
# 各プロジェクトに対して、分析対象となるリリースバージョンを定義
# target_releaseには連続する2つのリリース（current_release -> next_release）のペアを指定
RELEASE_IMPACT_ANALYSIS = {
    'nova': {
        'target_release': [
            '20.0.0',
            '21.0.0'
        ]
    }
}

# 分析期間の定義
# 各期間グループに対して、基準日とオフセット、レビューステータスを定義
RELEASE_ANALYSIS_PERIODS = {
    'early_reviewed': {
        'base_date': 'current_release',  # 基準日：現在のリリース日
        'offset_start': 0,                # 開始オフセット：リリース日当日
        'offset_end': 30,                 # 終了オフセット：リリース日+30日
        'review_status': 'reviewed'       # レビューステータス：レビュー済み
    },
    'early_not_reviewed': {
        'base_date': 'current_release',
        'offset_start': 0,
        'offset_end': 30,
        'review_status': 'not_reviewed'   # レビューステータス：未レビュー
    },
    'late_reviewed': {
        'base_date': 'next_release',      # 基準日：次のリリース日
        'offset_start': -30,              # 開始オフセット：リリース日-30日
        'offset_end': 0,                  # 終了オフセット：リリース日当日
        'review_status': 'reviewed'
    },
    'late_not_reviewed': {
        'base_date': 'next_release',
        'offset_start': -30,
        'offset_end': 0,
        'review_status': 'not_reviewed'
    }
}

# レビューステータスの判定閾値
# review_countが0より大きい場合は「reviewed」、0の場合は「not_reviewed」
REVIEW_COUNT_THRESHOLD = 0

# 対数軸を自動適用する範囲比率の閾値
LOG_SCALE_THRESHOLD = 100

# 統計検定の有意水準
SIGNIFICANCE_LEVEL = 0.05
