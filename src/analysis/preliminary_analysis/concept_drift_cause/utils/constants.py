"""concept_drift_cause の設定値（Phase1 / 特徴量の寄与度分析：IG で精度変動の原因をモデル側から説明する）。

design.md 参照。モデル・データ構築・特徴・label は `pretrained_encoders` から、
probe と日次指標は `concept_drift_detection`（距離×時期行列の分析）の保存物から読む。
ここに置くのは特徴量の寄与度分析固有の設定（対象セル・IG の刻み・出力先）。
"""
from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as detection
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_cause"
# 距離×時期行列の分析の出力（daily_metrics.csv.gz と probes/）を読む先
DETECTION_ROOT = detection.OUTPUT_ROOT

# ── 対象セル（§6.1） ──────────────────────────────────
# project -> version -> {"distances": [...] or "all", "positions": [...] or "all"}
# 距離と位置をそれぞれ指定し、**その直積**を計算する。"all" でその軸すべて。
ALL_CELLS = {"distances": "all", "positions": "all"}


def latest_version(project: str) -> str:
    """その project の最新版（距離×時期行列の分析の PROJECTS の末尾。版は昇順で並んでいる）。"""
    return detection.versions_for(project)[-1]


def older_versions(project: str) -> list[str]:
    """その project の**最新版を除く 4 版**（古い順）。最新版を回し終えたあとに足す用。"""
    return detection.versions_for(project)[:-1]


# 既定は**各プロジェクトの最新版・全セル**（全 Change で約 14.6 時間。§10）。
# 全 5 版だと約 188 時間（7.8 日）かかり、しかも古い版ほど 1 日あたりの Open Change が多くて重い。
# まず最新版で考察し、**版をまたいで傾向が一貫するか確かめたくなった段階で古い版を足す**。
#
# 版を直接書かず距離×時期行列の分析の PROJECTS から引くので、距離×時期行列の分析側の対象版を変えても自動で追従する。
# 古い版を足すときは、ここに版を書き足すか `--version` / `--all-versions` を使う（出力は追記式）。
IG_TARGETS = {p: {latest_version(p): dict(ALL_CELLS)} for p in detection.PROJECTS}

# 最新版を回し終えたあとに残りを足すための対象（**最新版を除く 4 版**・全セル）。
# 全体で約 173 時間（＝全 5 版の 188 時間 − 最新版の 14.6 時間）かかる。
# 使い方は `--older-versions`（CLI）。版のラベルは project ごとに違う（例：25.0.0 は
# neutron/cinder では最新版だが glance では古い版）ので、`--version` で並べるより取り違えが起きない。
IG_TARGETS_OLDER = {p: {v: dict(ALL_CELLS) for v in older_versions(p)} for p in detection.PROJECTS}

# ── IG（§2） ─────────────────────────────────────────
IG_STEPS = 32            # 積分の段階数。基準点から実際の値までをこの数で刻む
# 基準点は「その評価日の集合の平均」で固定（§2.3）。対象 Change だけを動かし、他は実際の値のまま。
# 対象出力はロジット（確率ではない。§2.2）。

# ── モデルの選択（§4） ───────────────────────────────
# (評価日, 距離) ごとに、この指標が中央（5 個中 3 番目）だった seed を選ぶ。
# AUC は連続値なので 5 個の順位が一意に決まる（recall@k は値が数段階しかなく決まらない）。
SELECT_METRIC = "auc"
# 同値のときは seed 番号の小さい方（再現性のため。`concept_drift_detection/design.md` §8.1 と同じ規則）。

# ── 出力（§9） ───────────────────────────────────────
IG_DAILY_NAME = "ig_daily.csv.gz"
IG_META_NAME = "ig_daily_meta.json"
EXTRACT_DIRNAME = "extract"
CSV_ENCODING = "utf-8-sig"   # Excel で開いても文字化けしないように BOM 付き

MODEL_NAME = detection.MODEL_NAME[0]
K_LIST = detection.K_LIST        # 表に載せる recall@k の k（距離×時期行列の分析と共通）

# ── 作図（§9.4） ─────────────────────────────────────
FIGURE_DIRNAME = "figures"

# 描く系列。既定は**差分（probe − 汎用ヘッド、同一 seed）**の正例・負例だけ（§9.4.1）。
#   差分にすると距離による変化が見やすく（nova の elapsed_time は 2.436→2.166 が 0.440→0.208）、
#   日ごとのばらつきも縮む（1.92→1.39。同じ日・同じ seed なら翻訳表が共通で相殺されるため）。
#   汎用ヘッド単体は距離に依存しないので、格子に並べても同じ図が並ぶだけ。
# "probe" / "general" を足すと、その系列も個別セル図＋格子一覧のペアで生成される。
PLOT_SERIES = ["diff"]
PLOT_GROUPS = ["pos", "neg"]

# 個別セル図のばらつき表示（§9.4.4）。**独立にオンオフする**。格子一覧には効かない
# （1 枚に最大 26 パネル × 15 本 ＝ 390 本入るため、線も文字も潰れる）。
PLOT_SHOW_ERROR_BARS = True   # 標準偏差を棒の先端を中心に引く。ゼロをまたぐか＝符号が定まるか
PLOT_SHOW_VALUES = True       # "+1.81 (±0.54)" を棒の右に添える

# 軸の端を決めるパーセンタイル（§9.4.5）。**「平均 ＋ 標準偏差」の分布**に対してとる。
# 平均だけで決めるとエラーバーが 2〜3 倍の位置まで伸びて枠外に出る（nova 正例で 0.45 → 1.13）。
PLOT_AXIS_PERCENTILE = 95

# 格子の折り返し方（パネル数 -> (行数, 列数)）。無い数は列数から行数を自動計算する（§9.4.6）。
PLOT_GRID_SHAPE = {26: (5, 6), 25: (5, 5), 13: (3, 5), 6: (2, 3), 3: (1, 3)}
PLOT_GRID_COLS_DEFAULT = 6

PLOT_DPI = 120


def grid_shape(n_panels: int) -> tuple[int, int]:
    """パネル数から格子の (行数, 列数) を返す。"""
    if n_panels in PLOT_GRID_SHAPE:
        return PLOT_GRID_SHAPE[n_panels]
    cols = min(PLOT_GRID_COLS_DEFAULT, max(1, n_panels))
    return (-(-n_panels // cols), cols)      # 切り上げ除算


def targets_for(project: str) -> dict:
    """その project の対象（version -> {"distances","positions"}）。"""
    return dict(IG_TARGETS.get(project, {}))


def resolve_axis(spec, grid: int) -> list[int]:
    """"all" または明示リストを 0 始まりの index 列にする。距離は 0 始まり、位置は **1 始まり**で
    指定させるので、位置側は呼び出し元で調整する。"""
    if spec == "all" or spec is None:
        return list(range(grid))
    return [int(v) for v in spec]
