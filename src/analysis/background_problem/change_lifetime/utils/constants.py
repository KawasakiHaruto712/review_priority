"""
change_lifetime 分析の設定値。

分析対象や各種パラメータを変えたいときは、原則ここだけを編集すれば済むようにしている。
Change / リリース日の読み込みは background_problem.common を利用する。
"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ───────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "background_problem" / "change_lifetime"

# ── 分析対象プロジェクトとリリース ──────────────────────────
TARGET_PROJECTS = {
    "nova": [
        "2015.1.0",
        "12.0.0", "13.0.0", "14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0", "19.0.0", "20.0.0",
    ],
}

# ── 母集団 ─────────────────────────────────────────────
LOOKBACK_DAYS = 365           # cycle_start の最大何日前までに投稿された Change を含めるか
# status グループ（名前 -> 対象 status 集合）。NEW 等の未決は常に対象外。
STATUS_GROUPS = {
    "all": {"MERGED", "ABANDONED"},
    "merged": {"MERGED"},
    "abandoned": {"ABANDONED"},
}

# ── 生存期間の単位 ──────────────────────────────────────
DURATION_UNIT = "days"        # "hours" | "days"（lifetime は日単位が見やすい）

# ── ヒストグラム ────────────────────────────────────────
HIST_BINS = 50                # ビン本数
HIST_LOG_BINS = True          # True: 対数等間隔ビン（log-binning）, False: 線形等間隔
X_LOG_SCALE = True            # 横軸を対数表示にするか
HIST_MIN_VALUE = None         # 対数ビンの下限（単位は DURATION_UNIT）。None なら正の最小値を自動採用

# ── 要約統計に出す分位点 ───────────────────────────────────
SUMMARY_PERCENTILES = [10, 25, 50, 75, 90]

# ── 箱ひげ図（リリース横断の比較図） ──────────────────────────
MAKE_BOXPLOT = True           # リリースごとの生存期間を 1 枚に並べた箱ひげ図を出すか
BOXPLOT_LOG_SCALE = True      # 箱ひげ図の縦軸（生存期間）を対数にするか

# ── プロット ───────────────────────────────────────────
PLOT_DPI = 150
