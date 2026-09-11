"""concept_drift_detection の設定値（Phase1 / step2：位置×距離ヒートマップ版）。

design.md 参照。事前学習エンコーダ＋汎用ヘッド・データ構築・特徴・label・model は
`pretrained_encoders` から import して使う（本ディレクトリでは事前学習しない）。
ここに置くのは concept_drift 固有の設定（ビン分割・load 対象・指標・図・検定）。
"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_detection"

# ── 分析対象プロジェクトとリリース（§2） ──────────────────────
# project -> {"cutoff": 事前学習の締め（版ラベル）, "versions": 対象5版, "window_days": 学習窓長}。
# cutoff / versions は step1（lookback_window）と同一。window_days は **step1 の結果で選んだ最良窓**
# （判定：① Ave（要約表の右端）→ ② Ave が同一なら std（小さいほど良い））。
#   nova 1w(0.792) / neutron 1w(0.799) / cinder 2w(0.816, 1wと同値→std) /
#   glance 1w(0.818) / keystone 1M(0.740) / swift 2M(0.897, 2wと同値→std)
PROJECTS = {
    "nova":     {"cutoff": "25.0.0", "versions": ["26.0.0", "27.0.0", "28.0.0", "29.0.0", "30.0.0"], "window_days": 7},
    "neutron":  {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"], "window_days": 7},
    "cinder":   {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"], "window_days": 14},
    "glance":   {"cutoff": "24.0.0", "versions": ["25.0.0", "26.0.0", "27.0.0", "28.0.0", "29.0.0"], "window_days": 7},
    "keystone": {"cutoff": "21.0.0", "versions": ["22.0.0", "23.0.0", "24.0.0", "25.0.0", "26.0.0"], "window_days": 30},
    "swift":    {"cutoff": "2.29.0", "versions": ["2.30.0", "2.31.0", "2.32.0", "2.33.0", "2.34.0"], "window_days": 60},
}


def cutoff_for(project: str) -> str:
    """その project の事前学習 cutoff ラベル（pretrained_encoders の保存キー）。"""
    return PROJECTS[project]["cutoff"]


def versions_for(project: str) -> list[str]:
    """その project の対象5版。"""
    return list(PROJECTS[project]["versions"])


# ── 距離 d × 位置 p の格子（§4.2） ─────────────────────────
# 学習は「評価日ごとに貼り直す日次スライド」。距離 d は学習期間を d 段ぶん過去へずらす量。
# 窓長は PROJECTS の "window_days"（プロジェクト別）。下の WINDOW_DAYS はその既定値（未設定時）。
WINDOW_DAYS = 7              # 窓長の既定値（PROJECTS に window_days が無い project 用）
DISTANCE_STEP_DAYS = None    # 距離1段ぶんの日数。None なら その project の窓長と同じ（独立指定も可）
NOMINAL_CYCLE_DAYS = 182     # 基準サイクル長（約半年）。格子サイズの自動決定に使う
GRID_SIZE = None             # 距離の行数＝位置の列数。None なら 基準サイクル長 ÷ 刻み
BIN_DAY_ALIGNED = True       # 位置の境界を 0 時に揃える
MAX_SET_SIZE = 512           # 1 集合の最大 Change 数（pretrained_encoders と一致させる）
RECORD_MARGIN_DAYS = 7       # レコード生成範囲の余裕（§4.4）


def window_for(project: str) -> int:
    """その project の学習窓長（日）。PROJECTS の window_days（無ければ WINDOW_DAYS）。"""
    return int(PROJECTS[project].get("window_days", WINDOW_DAYS))


def step_for(project: str) -> int:
    """その project の距離1段ぶんの日数（未指定ならその project の窓長と同じ）。"""
    return window_for(project) if DISTANCE_STEP_DAYS is None else int(DISTANCE_STEP_DAYS)


def grid_for(project: str) -> int:
    """その project の格子サイズ N（距離の行数＝位置の列数）。未指定なら 基準サイクル長 ÷ 刻み。

    既定の窓長だと：7日→26 / 14日→13 / 30日→6 / 60日→3。
    """
    if GRID_SIZE is not None:
        return int(GRID_SIZE)
    return max(1, round(NOMINAL_CYCLE_DAYS / step_for(project)))


# ── 事前学習エンコーダの読み込み（§6.1） ─────────────────────
# pretrained_encoders の保存物（project × cutoff × seed）を load。cutoff は PROJECTS で project 別。
N_REPEATS = 10               # load するエンコーダ数（不足は pretrained_encoders.build_encoders で自動作成）
RANDOM_SEED = 42             # probe 学習の seed（k 回目 = RANDOM_SEED + k）
REPEAT_AGG = "median"        # 反復方向の集約（"median" / "mean"）

# ── データ量の床（§7） ────────────────────────────────
MIN_TRAIN = 30               # 学習期間の最小レコード数（下回る日は NaN）
MIN_EVAL = 30                # 評価日の最小レコード数

# ── 評価指標（§8.1。lookback と同一一式・主 AUC） ──────────────
K_LIST = [5, 10, 20]         # top-k の k（＋その日の正例数 '@pos' も自動で出す）
CLASSIFY_THRESHOLD = 0.5     # precision/recall/f1 用の2値しきい値
# 全指標を「評価日ごと算出→位置（列）の日数で平均→seed中央値+IQR」で統一（§8.1）。

# ── 変化区間の判定（§8.2）─────────────────────────────
PERMUTATION_N = 1000
SIGNIFICANCE = 0.05

# ── 図（§10） ────────────────────────────────────────
# N×N ヒートマップ（既定 26×26。色のみ）。数値表示は既定 False（後から切替可）。
HEATMAP_SHOW_CELL_VALUES = False
HEATMAP_VALUE_FMT = "{:.2f}"
PLOT_METRICS = ["auc"]       # ヒートマップを描く指標（既定 AUC のみ）。CSV/json は全指標を保存。

# ── モデル名・出力（§10）──────────────────────────────
MODEL_NAME = ["set_transformer"]
SAVE_PER_REPEAT = True
SAVE_DAILY_METRICS = True    # (評価日, 距離, seed) ごとの指標を保存（位置のまとめ直し用。§10）
PLOT_DPI = 150
