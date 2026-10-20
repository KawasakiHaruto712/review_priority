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
