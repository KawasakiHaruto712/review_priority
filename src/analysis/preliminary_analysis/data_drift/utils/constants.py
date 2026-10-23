"""data_drift（特徴量別の分布）の設定値（design.md）。"""
from src.config.path import DEFAULT_DATA_DIR

OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "preliminary_analysis" / "data_drift"

# ── 比べるもの（design.md §1） ─────────────────────────────
GROUPS = ("all", "pos", "neg")              # 全体（主）・正例どうし・負例どうし
GROUP_LABEL = {"all": "全体", "pos": "正例", "neg": "負例"}

# ── 検定と効果量（design.md §3.3） ──────────────────────────
ALPHA = 0.05                                # 有意水準（両側）。多重比較の補正はしない
EFFECT_BOUNDS = (0.147, 0.33, 0.474)        # クリフの δ の区分の境目（無視できる／小／中／大。Romano ら 2006）

# ── 図（design.md §5。寄与度の図にそろえる） ─────────────────────
TOP_N = 10                                  # セルごとに |δ の平均| の上位何件を描くか
XLIM = 1.0                                  # 横軸は −1〜1 で固定
POS_COLOR = "#4C72B0"                       # 右＝評価日の方が大きい
NEG_COLOR = "#C44E52"                       # 左＝評価日の方が小さい
SHOW_ERROR_BARS = True                      # 個別セル図のエラーバー（評価日の δ の標準偏差）
SHOW_VALUES = True                          # 個別セル図の数値 `+0.42 (±0.10)`
PLOT_DPI = 150

# ── 図の表記（design.md §10.2。2026-10-24。図の表示だけ。保存物の position は今の向き＝p0 がリリース直前） ──
# 行の名前は論文の言葉。列は表示の番号 p' ＝ (列の数 − 1) − 保存物の p で、3 列なら「p0（初期）・p1（中期）・p2（後期）」
ROW_DISPLAY = {"time_distance": "時間的な距離 d", "feature_distance": "特徴空間上の距離 d"}
PHASE_NAMES = ("初期", "中期", "後期")
METRIC_DISPLAY = {"f1_plugin": "F1-score"}
