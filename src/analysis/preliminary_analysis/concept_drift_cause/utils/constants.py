"""concept_drift_cause の設定値（Phase1 / 特徴量の寄与度分析：IG で精度変動の原因をモデル側から説明する）。

design.md 参照。モデル・データ構築・特徴・label は `pretrained_encoders` から、
probe と日次指標は `concept_drift_detection`（距離×時期行列の分析）の保存物から読む。
ここに置くのは特徴量の寄与度分析固有の設定（対象セル・IG の刻み・出力先）。
"""
import json

from src.analysis.preliminary_analysis.concept_drift_detection.utils import constants as detection
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ─────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "concept_drift_cause"
# 距離×時期行列の分析の出力（daily_metrics.csv.gz と probes/）を読む先
DETECTION_ROOT = detection.OUTPUT_ROOT

# ── 種類（design.md §11.5。2026-10-21） ──────────────────────
# time_distance＝RQ1（時期のずれ×評価時期の精度表。concept_drift_detection の出力を読む）
# feature_distance＝RQ2（特徴のずれ×評価時期の精度表。feature_space_window の出力ファイルを読む。コードは import しない）
KINDS = ("time_distance", "feature_distance")
ROW_LABEL = {"time_distance": "時期のずれ", "feature_distance": "特徴のずれ"}
# 2026-10-24（design.md §11.8）：図に書く行の名前は論文の言葉にし、RQ1・RQ2 とも d にそろえる（ROW_LABEL は図以外のために残す）
ROW_DISPLAY = {"time_distance": "時間的な距離 d", "feature_distance": "特徴空間上の距離 d"}
# 図の評価時期は表示の番号 p' ＝ (列の数 − 1) − 保存物の p（保存物は p0 ＝ リリース直前）。3 列なら時期の名前を添える
PHASE_NAMES = ("初期", "中期", "後期")
FSW_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "feature_space_window"


def set_variant_roots(variant_name: str) -> None:
    """読み先（RQ1・RQ2 の精度表の出力）を版の種類に合わせる（2026-10-22。design.md §11.6）。
    重みなし版は今のまま、重みあり版は各出力先の下の posw/。出力先 OUTPUT_ROOT は utils/variant.apply で切り替える。
    2026-10-24：選んだ列（selection_comparison の出力）の読み先も切り替える（§11.9）。"""
    from src.analysis.preliminary_analysis.pretrained_encoders.utils import variant
    global DETECTION_ROOT, FSW_ROOT, SELECTION_ROOT
    DETECTION_ROOT = variant.subdir(BASE_DETECTION_ROOT, variant_name)
    FSW_ROOT = variant.subdir(BASE_FSW_ROOT, variant_name)
    SELECTION_ROOT = variant.subdir(BASE_SELECTION_ROOT, variant_name)


BASE_DETECTION_ROOT = DETECTION_ROOT      # 重みなし版の読み先（set_variant_roots の元）
BASE_FSW_ROOT = FSW_ROOT
# 2026-10-24（design.md §11.9）：選んだ列（selection_comparison の selected_columns.json）の読み先
BASE_SELECTION_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "selection_comparison"
SELECTION_ROOT = BASE_SELECTION_ROOT
SELECTED_COLUMNS_NAME = "selected_columns.json"
CROSS_KIND = "cross"                      # 2 つの選び方の間の判断値の表の置き場所（output_dir の kind に使う）


def output_dir(project: str, version: str, kind: str):
    """IG の出力先（種類ごとに分ける。RQ1 の旧い出力 <project>/set_transformer/<版>/ の場所は変えない）。"""
    return OUTPUT_ROOT / project / MODEL_NAME / kind / version


def metrics_path(project: str, version: str, kind: str):
    """精度表の日ごとの指標（そのセル値を出したモデルの特定と、図の題の recall@pos・AUC に使う）。"""
    if kind == "feature_distance":
        return FSW_ROOT / project / "feature_distance" / version / "daily_metrics.csv.gz"
    return DETECTION_ROOT / project / MODEL_NAME / version / "daily_metrics.csv.gz"

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

# 2026-10-23（design.md §11.7）：重みあり版は精度表の種類ごとに対象を分ける。上の IG_TARGETS は書き換えずに残す（重みなし版の対象）。
# RQ1：F1 で d0 が負けた列の、d0 と勝った相手の行だけ（距離・位置は 0 始まり。p0 ＝ リリースの直前の 3 分の 1）
IG_TARGETS_TIME = {
    "qtcreator": {"17.0.0": {"distances": [0, 1], "positions": [0]},
                  "14.0.0": {"distances": [0, 2], "positions": [0]}},
    "qtbase": {"6.6.0": {"distances": [0, 1], "positions": [2]}},
    # 2026-10-24 追加（本人判断。design.md §11.7）：d0 が勝った列も足す。相手は d1・d2 の良い方（typo3 14.0.0 は 2 列とも d2。本人判断 (a)）
    "go": {"go1.24": {"distances": [0, 1], "positions": [1]}},
    "typo3_cms": {"14.0.0": {"distances": [0, 2], "positions": [0, 2]}},
}
# RQ2：~~今回は空（計算しない）。中身を入れれば同じ仕組みで回る~~
# ~~2026-10-23 改めた（本人判断 (b)。design.md §11.7）：g0 が上回った列と下回った列の両方（各 3 列）~~
# 2026-10-23 再び改めた（本人判断。§11.7）：g0 が下回った 3 列だけ（RQ1 と同じ）。行は g0 と、その列で比べた相手。
# 取り消した上回った列：go go1.21 p0（g0, g1）・go go1.24 p1（g0, g2）・qtdeclarative 6.7.0 p2（g0, g2）。足せば追記で計算できる
IG_TARGETS_FEATURE = {
    # 2026-10-24 三たび改めた（本人判断。design.md §11.7）：g0 が上回った列も戻す（行・列は 10/23 に取り消したものと同じ）
    "go": {"go1.21": {"distances": [0, 1], "positions": [0]},
           "go1.24": {"distances": [0, 2], "positions": [1]}},
    "qtdeclarative": {"6.7.0": {"distances": [0, 2], "positions": [2]}},
    # g0 が下回った列
    "typo3_cms": {"13.1.0": {"distances": [0, 1], "positions": [0]},
                  "13.3.0": {"distances": [0, 1], "positions": [2]}},
    "qtcreator": {"14.0.0": {"distances": [0, 1], "positions": [0]}},
}

# ── IG（§2） ─────────────────────────────────────────
IG_STEPS = 32            # 積分の段階数。基準点から実際の値までをこの数で刻む
IG_STEP_CHUNK = 8        # 一度に計算する段階の数（メモリのため。勾配を足すので値には効かない。§2.4。2026-10-20）
# 基準点は「その評価日の集合の平均」で固定（§2.3）。対象 Change だけを動かし、他は実際の値のまま。
# 対象出力はロジット（確率ではない。§2.2）。

# 2026-10-23（design.md §11.7。本人判断。§6.4 の「全件」の例外）：評価日ごとに正例は全件、負例は単純無作為抽出（非復元）。
# 負例の件数 n ＝ ⌈n₀ ÷（1 ＋（n₀ − 1）÷ N）⌉（信頼度 95%・許容誤差 ±5%。有限母集団修正つき）。N ≤ n ならその日は全件。
# その日に計算する全モデルで同じ Change を使う。乱数の種は project と評価日から決める（attribution/sampling.py）
# 2026-10-23 改めた（本人判断。design.md §11.7）：今回は全件で回す（実測で全件でも締め切りに収まるため）。標本の仕組みは True で戻せるように残す
IG_SAMPLE_NEG = False
IG_SAMPLE_N0 = 384
# 汎用ヘッドの IG を計算するか（§11.7。今回は False＝計算しない。時間をすべて probe に回す）
IG_GENERAL_HEAD = False

# ── モデルの選択（§4） ───────────────────────────────
# (評価日, 距離) ごとに、この指標が中央（5 個中 3 番目）だった seed を選ぶ。
# AUC は連続値なので 5 個の順位が一意に決まる（recall@k は値が数段階しかなく決まらない）。
SELECT_METRIC = "auc"
# 2026-10-21 改めた（design.md §4）：recall@pos → AUC → seed 番号の順に、3 つとも小さい順に並べて 3 番目（真ん中）を選ぶ。
# 精度表を recall@pos で主に読むので「recall@pos の表の値を出したモデル」を見る。recall@pos の同着は AUC で、それも同じなら番号で決める。
SELECT_METRICS = ("recall@pos", "auc")
# 同値のときは seed 番号の小さい方（再現性のため。`concept_drift_detection/design.md` §8.1 と同じ規則）。
# 2026-10-22（design.md §11.6）：版の種類で並べる指標を変える。重みあり版は F1（plug-in）→ AUC → seed 番号（窓長・精度表を F1 で見るため）。
SELECT_METRICS_BY_VARIANT = {"base": SELECT_METRICS, "posw": ("f1_plugin", "auc")}


def pre_variant() -> str:
    """今の版の種類（pretrained_encoders の ENCODER_VARIANT）。"""
    from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
    return pre.ENCODER_VARIANT


def select_metrics() -> tuple:
    """今の版の種類に合う並べる指標。"""
    return SELECT_METRICS_BY_VARIANT[pre_variant()]

# ── 出力（§9） ───────────────────────────────────────
IG_DAILY_NAME = "ig_daily.csv.gz"
IG_META_NAME = "ig_daily_meta.json"
IG_SAMPLE_NAME = "ig_sample.csv.gz"   # 対象の評価日の全 Change と、IG の計算に使ったか（§11.7。2026-10-23）
COMPARE_DIRNAME = "compare"          # d0 と相手を並べた表の置き場所（extract/<名前>/ の下。§11.7）
JUDGMENT_DIRNAME = "judgment"        # 判断値（正例 − 負例）の差で並べた表の置き場所（extract/<名前>/ の下。§11.9。2026-10-24）
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
# 2026-10-05：報告では距離×時期行列（probe の AUC）と同じ段階で比べるため、probe 単体も描く
PLOT_SERIES = ["diff", "probe"]
PLOT_GROUPS = ["pos", "neg"]

# 個別セル図のばらつき表示（§9.4.4）。**独立にオンオフする**。格子一覧には効かない
# （1 枚に最大 26 パネル × 15 本 ＝ 390 本入るため、線も文字も潰れる）。
PLOT_SHOW_ERROR_BARS = True   # 標準偏差を棒の先端を中心に引く。ゼロをまたぐか＝符号が定まるか
PLOT_SHOW_VALUES = True       # "+1.81 (±0.54)" を棒の右に添える
# 図に並べる特徴量の数（design.md §9.4.3。2026-10-18）。セルごとに、評価日を通した寄与の絶対値の平均で上位を選ぶ。
# 計算の出力と表は全特徴量のまま。None なら全特徴量を並べる
PLOT_TOP_N = 10

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


TARGETS_MODE = "manual"   # 2026-10-24（§11.9）："manual"＝手書きの対象（既定）／"selected"＝selection_comparison が選んだ列（--targets で切り替える）


def selected_columns() -> dict:
    """selection_comparison の selected_columns.json の中身（§11.9）。なければ FileNotFoundError。"""
    path = SELECTION_ROOT / SELECTED_COLUMNS_NAME
    if not path.exists():
        raise FileNotFoundError(f"選んだ列のファイルがありません（先に selection_comparison を回す）: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def selected_targets(kind: str) -> dict:
    """選んだ列から、その kind の対象 {project -> version -> {"cells": [[行, 列], ...]}}（列は保存物の番号。§11.9）。
    同じ選び方の中は d ＝ 0 と相手、2 つの選び方の間は d ＝ 0（時間・空間の両方に入れる）。重なりは 1 つにまとめる。"""
    sel = selected_columns()
    cells: dict = {}

    def add(c, d):
        cells.setdefault(c["project"], {}).setdefault(c["version"], set()).add((int(d), int(c["position"])))

    for side in sel["within"].get(kind, {}).values():
        for c in side:
            add(c, 0)
            add(c, c["opponent"])
    for side in sel.get("cross", {}).values():
        for c in side:
            add(c, 0)
    return {p: {v: {"cells": sorted(s)} for v, s in vs.items()} for p, vs in cells.items()}


def selected_cross() -> list[dict]:
    """2 つの選び方の間で選んだ列（空間が上回った側・時間が上回った側の順）。"""
    sel = selected_columns()
    return [dict(c, side=side) for side, cols in sel.get("cross", {}).items() for c in cols]


def targets(kind: str = "time_distance") -> dict:
    """今の版の種類と精度表の種類に合う対象（project -> version -> 指定）。
    2026-10-23（design.md §11.7）：重みあり版は IG_TARGETS_TIME（RQ1）／IG_TARGETS_FEATURE（RQ2）、重みなし版は今までどおり IG_TARGETS。
    2026-10-24（§11.9）：TARGETS_MODE が "selected" なら、selection_comparison が選んだ列（指定は {"cells": [[行, 列], ...]}）。"""
    if TARGETS_MODE == "selected":
        return selected_targets(kind)
    if pre_variant() == "posw":
        return IG_TARGETS_FEATURE if kind == "feature_distance" else IG_TARGETS_TIME
    return IG_TARGETS


def targets_for(project: str, kind: str = "time_distance") -> dict:
    """その project の対象（version -> {"distances","positions"}）。"""
    return dict(targets(kind).get(project, {}))


def resolve_axis(spec, grid: int) -> list[int]:
    """"all" または明示リストを 0 始まりの index 列にする。距離も位置も 0 始まりで指定する
    （d0 ＝ 直近の窓、p0 ＝ リリースの直前）。"""
    if spec == "all" or spec is None:
        return list(range(grid))
    return [int(v) for v in spec]
