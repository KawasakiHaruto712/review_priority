"""lookback_window の設定値（事前分析窓長の調査：チューニング窓の長さの調査）。

design.md 参照。共有部品は pretrained_encoders から import（concept_drift_detection には依存しない）。
"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "lookback_window"

# ── 対象（design.md §2, §12） ─────────────────────────
# main の既定は「PROJECTS の全プロジェクト」を順に回す（--project で単一に絞れる）。
# 参考の代表 PROJECT = "nova" はどこからも使っていなかったので 2026-10-15 に廃止した（design.md §12）。

# プロジェクト別設定：project -> {"cutoff": 版ラベル, "versions": チューニング5版}。
# cutoff は使う事前学習エンコーダ（pretrained_encoders の保存物）の cutoff ラベル。
# pretrained_encoders/constants.py と同じ内容を、本ディレクトリも自己完結で保持する（重複は許容）。
# PROJECTS は今回の論文の 6 件（2026-10-15 に OpenStack 6 件から切り替えた。design.md §2）。
PROJECTS = {
    "qtbase":                   {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"]},
    "qtcreator":                {"cutoff": "13.0.0", "versions": ["14.0.0", "15.0.0", "16.0.0", "17.0.0", "18.0.0"]},
    "wikimedia_mediawiki_core": {"cutoff": "1.40.0", "versions": ["1.41.0", "1.42.0", "1.43.0", "1.44.0", "1.45.0"]},
    "qtdeclarative":            {"cutoff": "6.5.0", "versions": ["6.6.0", "6.7.0", "6.8.0", "6.9.0", "6.10.0"]},
    "go":                       {"cutoff": "go1.20", "versions": ["go1.21", "go1.22", "go1.23", "go1.24", "go1.25"]},
    "typo3_cms":                {"cutoff": "13.0.0", "versions": ["13.1.0", "13.2.0", "13.3.0", "13.4.0", "14.0.0"]},
}

# OpenStack 6 件（以前の対象）。今回の論文では分析しないが、設定は残す。名前を指定したときだけ引ける。
OPENSTACK_PROJECTS = {
    "nova":     {"cutoff": "25.0.0", "versions": ["26.0.0", "27.0.0", "28.0.0", "29.0.0", "30.0.0"]},
    "neutron":  {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"]},
    "cinder":   {"cutoff": "20.0.0", "versions": ["21.0.0", "22.0.0", "23.0.0", "24.0.0", "25.0.0"]},
    "glance":   {"cutoff": "24.0.0", "versions": ["25.0.0", "26.0.0", "27.0.0", "28.0.0", "29.0.0"]},
    "keystone": {"cutoff": "21.0.0", "versions": ["22.0.0", "23.0.0", "24.0.0", "25.0.0", "26.0.0"]},
    "swift":    {"cutoff": "2.29.0", "versions": ["2.30.0", "2.31.0", "2.32.0", "2.33.0", "2.34.0"]},
}


def project_config(project: str) -> dict:
    """その project の設定（PROJECTS → OPENSTACK_PROJECTS の順に探す）。無ければ KeyError。"""
    cfg = PROJECTS.get(project) or OPENSTACK_PROJECTS.get(project)
    if cfg is None:
        raise KeyError(f"未知のプロジェクトです（PROJECTS か OPENSTACK_PROJECTS に定義してください）: {project}")
    return cfg


def cutoff_for(project: str) -> str:
    """その project の事前学習 cutoff ラベルを返す。"""
    return project_config(project)["cutoff"]


def versions_for(project: str) -> list[str]:
    """その project のチューニング対象5版を返す。"""
    return list(project_config(project)["versions"])


# ── 使う事前学習エンコーダ（pretrained_encoders の保存物） ────────────
N_SEEDS = None                 # None なら保存済み全 seed

# ── 学習窓（design.md §4） ────────────────────────────
# 学習窓 = 評価日の直前 W 日（[X-W, X-1]。末尾＝前日。評価日自身は予測対象なので入れない）。
WINDOWS_DAYS = [1, 3, 7, 14, 30, 60]
WINDOW_LABELS = {1: "1d", 3: "3d", 7: "1w", 14: "2w", 30: "1M", 60: "2M"}

# ── 指標（design.md §7。全部出す → 描画は後から選ぶ） ──────────────
CLASSIFY_THRESHOLD = 0.5
K_LIST = [5, 10, 20]           # top-k（＋その日の正例数 '@pos' も自動で出す）
PLOT_METRIC_DEFAULT = "auc"    # 既定で先に描く指標
# 小さいほど良い指標（順位の集計で最小を 1 位、報告用の表で最小を最良にする。design.md §11.3・§16。2026-10-21）
LOWER_IS_BETTER = {"norm_rank"}

# ── 出力・作図 ───────────────────────────────────────
PLOT_DPI = 150
DRAW_GENERAL_DEFAULT = False   # 汎用ヘッド（チューニングなし）基準線を重ねるか（既定オフ。--no-general 不要）
# 折れ線グラフに描く窓長（WINDOWS_DAYS の部分集合。後から線を調整可）。
# 全部描くなら list(WINDOWS_DAYS)。例：1d/1w/2M だけなら [1, 7, 60]。
# `--windows` を渡すとこの既定を上書きできる。
PLOT_WINDOWS_DAYS = [1, 7, 60]   # 1d / 1w / 2M
