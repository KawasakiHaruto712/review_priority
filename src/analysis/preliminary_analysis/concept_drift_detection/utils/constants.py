"""concept_drift_detection の設定値（Phase1 / 距離×時期行列の分析：位置×距離ヒートマップ版）。

design.md 参照。事前学習エンコーダ＋汎用ヘッド・データ構築・特徴・label・model は
`pretrained_encoders` から import して使う（本ディレクトリでは事前学習しない）。
ここに置くのは concept_drift 固有の設定（ビン分割・load 対象・指標・図・検定）。
"""
import math

from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_detection"

# ── 分析対象プロジェクトとリリース（§2） ──────────────────────
# project -> {"cutoff": 事前学習の締め（版ラベル）, "versions": 対象5版, "window_days": 学習窓長}。
# cutoff / versions は窓長の調査（lookback_window）と同一。window_days は **窓長の調査の結果で選んだ最良窓**
# （判定：① Ave（要約表の右端）→ ② Ave が同一なら std（小さいほど良い））。
# 2026-10-03 のやり直し（Open の期間・特徴量の修正、リリースの表のサイクル単位化、MAX_SET_SIZE なし）の結果：
#   nova 1w(0.785) / neutron 1w(0.797) / cinder 1w(0.808) /
#   glance 2w(0.828) / keystone 1M(0.746) / swift 2M(0.893)
# 旧版（2026-09）は nova 1w / neutron 1w / cinder 2w / glance 1w / keystone 2M / swift 1M だった。
# 変わった 4 件はいずれも 1 位と 2 位の差が 0.001〜0.003 で、日次ばらつき（std 0.11〜0.24）よりずっと小さい。
# 1w〜2M の差は有意ではなく、上の規則を機械的に適用した結果を使う。
# 明確なのは 3d・1d が全プロジェクトで劣ること（1d は最良より 0.02〜0.07 低い）。
# 2026-10-15：PROJECTS を今回の論文の 6 件に切り替えた（design.md §2）。window_days は窓長の調査を 6 件で回してから入れる
# （それまでは WINDOW_DAYS = 7）。上の OpenStack の窓長の記録は OPENSTACK_PROJECTS に残す。
# 2026-10-21：window_days を入れた。窓長の調査の recall@pos の要約表の Ave が最良の窓長（本人判断。design.md §4.2）。
# 刻みは窓長と同じ（DISTANCE_STEP_DAYS = None）のままなので、qtbase・qtcreator は格子が 4×4 になる（承知のうえ。本人判断 (a)）。
PROJECTS = {
    "qtbase":                   {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"], "window_days": 60},
    "qtcreator":                {"cutoff": "13.0.0", "versions": ["14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0"], "window_days": 30},
    "wikimedia_mediawiki_core": {"cutoff": "1.40.0", "versions": ["1.41.0", "1.42.0", "1.43.0", "1.44.0", "1.45.0"], "window_days": 14},
    "qtdeclarative":            {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"], "window_days": 14},
    "go":                       {"cutoff": "go1.20", "versions": ["go1.21", "go1.22", "go1.23", "go1.24", "go1.25"], "window_days": 7},
    "typo3_cms":                {"cutoff": "13.0.0", "versions": ["13.1.0", "13.2.0", "13.3.0", "13.4.0", "14.0.0"], "window_days": 7},
}

# OpenStack 6 件（以前の対象）。今回の論文では分析しないが、設定は残す。名前を指定したときだけ引ける。
OPENSTACK_PROJECTS = {
    "nova":     {"cutoff": "25.0.0", "versions": ["26.0.0", "27.0.0", "28.0.0", "29.0.0", "30.0.0"], "window_days": 7},
    "neutron":  {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"], "window_days": 7},
    "cinder":   {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"], "window_days": 7},
    "glance":   {"cutoff": "24.0.0", "versions": ["25.0.0", "26.0.0", "27.0.0", "28.0.0", "29.0.0"], "window_days": 14},
    "keystone": {"cutoff": "21.0.0", "versions": ["22.0.0", "23.0.0", "24.0.0", "25.0.0", "26.0.0"], "window_days": 30},
    "swift":    {"cutoff": "2.29.0", "versions": ["2.30.0", "2.31.0", "2.32.0", "2.33.0", "2.34.0"], "window_days": 60},
}


def project_config(project: str) -> dict:
    """その project の設定（PROJECTS → OPENSTACK_PROJECTS の順に探す）。無ければ KeyError。"""
    cfg = PROJECTS.get(project) or OPENSTACK_PROJECTS.get(project)
    if cfg is None:
        raise KeyError(f"未知のプロジェクトです（PROJECTS か OPENSTACK_PROJECTS に定義してください）: {project}")
    return cfg


def is_known(project: str) -> bool:
    """PROJECTS か OPENSTACK_PROJECTS に定義がある project か。"""
    return project in PROJECTS or project in OPENSTACK_PROJECTS


def cutoff_for(project: str) -> str:
    """その project の事前学習 cutoff ラベル（pretrained_encoders の保存キー）。"""
    return project_config(project)["cutoff"]


def versions_for(project: str) -> list[str]:
    """その project の対象5版。"""
    return list(project_config(project)["versions"])


# ── 距離 d × 位置 p の格子（§4.2） ─────────────────────────
# 学習は「評価日ごとに貼り直す日次スライド」。距離 d は学習期間を d 段ぶん過去へずらす量。
# 窓長は PROJECTS の "window_days"（プロジェクト別）。下の WINDOW_DAYS はその既定値（未設定時）。
WINDOW_DAYS = 7              # 窓長の既定値（PROJECTS に window_days が無い project 用）
DISTANCE_STEP_DAYS = None    # 距離1段ぶんの日数。None なら その project の窓長と同じ（独立指定も可）
# 格子サイズ（距離の行数＝位置の列数）は定数を持たず、版ごとに ⌈サイクル長 ÷ 刻み⌉ で決める（grid_for）。
# 2026-10-15 に NOMINAL_CYCLE_DAYS（182）・GRID_SIZE を廃止（列の幅を刻みで固定するため。design.md §4.2）。
BIN_DAY_ALIGNED = True       # 位置の境界を 0 時に揃える
MAX_SET_SIZE = None          # 1 集合の最大 Change 数（pretrained_encoders と一致させる。旧版は 512。None で無制限）
RECORD_MARGIN_DAYS = 7       # レコード生成範囲の余裕（§4.4）


def window_for(project: str) -> int:
    """その project の学習窓長（日）。PROJECTS の window_days（無ければ WINDOW_DAYS）。"""
    return int(project_config(project).get("window_days", WINDOW_DAYS))


def step_for(project: str) -> int:
    """その project の距離1段ぶんの日数（未指定ならその project の窓長と同じ）。"""
    return window_for(project) if DISTANCE_STEP_DAYS is None else int(DISTANCE_STEP_DAYS)


def grid_for(project: str, cycle_len_days: int) -> int:
    """その版の格子サイズ N_v（距離の行数＝位置の列数）＝ ⌈サイクル長 ÷ 刻み⌉（design.md §4.2）。

    cycle_len_days は binning.cycle_days の値。列の幅は刻みで固定し、列の数が版ごとに変わる。
    例（刻み 7 日）：182 日→26 / 192 日→28 / 28 日→4 / 406 日→58。
    """
    return max(1, math.ceil(cycle_len_days / step_for(project)))


# ── 事前学習エンコーダの読み込み（§6.1） ─────────────────────
# pretrained_encoders の保存物（project × cutoff × seed）を load。cutoff は PROJECTS で project 別。
# N_REPEATS は **奇数**にする（§6.5）。奇数だと日ごとの seed 中央値が「実在する 1 個のモデルの値」に
# なり、特徴量の寄与度分析（concept_drift_cause）が「そのセル値を出したモデル」を一意に特定できる。
# 偶数（10）だと中央値が 5番目と6番目の平均になり、どのモデルの値でもなくなる。
# pretrained_encoders.N_REPEATS も同じ 5 に揃えてあるので、保存されるエンコーダは seed0〜4 の 5 個。
N_REPEATS = 5                # load するエンコーダ数（不足は pretrained_encoders.build_encoders で自動作成）
RANDOM_SEED = 42             # probe 学習の seed（k 回目 = RANDOM_SEED + k）
REPEAT_AGG = "median"        # **評価日ごと**の seed 方向の集約（"median" / "mean"）

# ── probe の保存（特徴量の寄与度分析用。§6.7） ─────────────────────
# 保存キーは (project, 窓長, seed, 学習期間の末尾日) ＝ 版に依存しない。
# ファイルは (project, 窓長, seed) ごとに 1 つ：<project>/probes/w<窓長>/seed<k>.npz
SAVE_PROBES = True

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
# N_v×N_v ヒートマップ（版ごとに大きさが違う。刻み 7 日・サイクル約半年なら約 26×26。色のみ）。数値表示は既定 False（後から切替可）。
HEATMAP_SHOW_CELL_VALUES = False
HEATMAP_VALUE_FMT = "{:.2f}"
PLOT_METRICS = ["auc"]       # ヒートマップを描く指標（既定 AUC のみ）。CSV/json は全指標を保存。

# ── モデル名・出力（§10）──────────────────────────────
MODEL_NAME = ["set_transformer"]
SAVE_PER_REPEAT = True
SAVE_DAILY_METRICS = True    # (評価日, 距離, seed) ごとの指標を保存（位置のまとめ直し用。§10）
PLOT_DPI = 150
