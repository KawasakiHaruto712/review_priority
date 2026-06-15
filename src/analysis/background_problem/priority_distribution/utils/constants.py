"""
priority_distribution 分析の設定値。

このファイルは「設定の入口」です。分析対象や各種パラメータを変えたいときは、
原則ここだけを編集すれば済むようにしています（メトリクスの計算ロジック本体は
metrics/duration_calculator.py に置いています）。
"""
from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG

# ── 入力 ───────────────────────────────────────────────
# Change データ / リリース日の読み込みは common.data_loader に集約（既定パスもそちらが保持）。
# ここではボット判定に使う設定（priority_distribution 固有）だけを定義する。
# ボット名の定義
GERRYMANDER_CONFIG = DEFAULT_CONFIG / "gerrymanderconfig.ini"
# サードパーティ CI アカウント一覧（name,email の CSV。第三者 CI の取りこぼし補完用）
BOT_ACCOUNTS_CSV = DEFAULT_CONFIG / "third_party_ci_accounts.csv"
# 上記 2 つに載っていない追加のボット/自動アカウント一覧（zuul / jenkins 等。1 行 1 名）
EXTRA_BOTS_FILE = DEFAULT_CONFIG / "extra_bots.txt"

# ── 出力 ───────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "background_problem" / "priority_distribution"

# ── 分析対象プロジェクトと、各プロジェクトで分析するリリース ──────────
# version は major_releases_summary.csv の version と一致させる。
TARGET_PROJECTS = {
    "nova": [
        "2015.1.0",
        "12.0.0", "13.0.0", "14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0", "19.0.0", "20.0.0",
    ],
    # 必要に応じて他コンポーネントを追加可能
}

# ── 縦軸メトリクス（柔軟に切替・追加可能） ────────────────────
# ここに並べた metric だけを実行する。定義の実体は duration_calculator.METRIC_REGISTRY。
ENABLED_METRICS = ["time_to_next_review", "time_to_decision"]

# ── 外れ値除去（両側 IQR） ──────────────────────────────────
OUTLIER_METHOD = "iqr"        # "iqr" | "percentile" | "none"
OUTLIER_IQR_K = 1.5           # フェンス: [Q1 - K*IQR, Q3 + K*IQR] の外を除外（両側）
OUTLIER_PERCENTILE = 99.0     # method="percentile" のとき、下位(100-p)%/上位(100-p)%を除外
# 各指標の screening 値が未定義の Change（放置＝レビュー無し / 未決）は常に除外する。

# ── アクティブ集合（各計測点 T で対象にする Change の範囲） ──────────
# 計測点 T で「T - LOOKBACK_DAYS <= created <= T < decision_time」を満たす
# （= T から LOOKBACK_DAYS 以内に投稿され、かつ T 時点で Open な）Change を対象にする。
LOOKBACK_DAYS = 365           # 計測点からさかのぼる対象期間（日）。既定 1 年。

# ── 横軸 ───────────────────────────────────────────────
X_AXIS_MODE = "normalized"    # "normalized"(0-1) | "days_until_release"

# ── バンド幅（中心線 ± BAND_STD_FACTOR × 標準偏差） ───────────
BAND_STD_FACTOR = 0.5

# ── 分位点（percentiles 図用。本数・値とも変更可能） ───────────────
PERCENTILES = [10, 30, 50, 70, 90]

# ── プロット ───────────────────────────────────────────
Y_LOG_SCALE = True            # 縦軸（時間）を対数軸にするか
DURATION_UNIT = "hours"       # "hours" | "days" … 縦軸の所要時間の単位
# 計測点を出力する最小の寄与 Change 数。
# 実効値は max(MIN_ACTIVE_FOR_POINT, len(PERCENTILES))（分位線の本数分のデータが無い計測点は出さない）。
MIN_ACTIVE_FOR_POINT = 1
PLOT_DPI = 150

# 全リリース混合図で等幅ビン平均にする場合のビン数（None なら全点プロット）
MIXED_PLOT_BINS = None
