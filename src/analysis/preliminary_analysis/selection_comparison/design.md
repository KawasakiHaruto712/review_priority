# selection_comparison 設計書：学習に用いるチケットの選び方の比較（2026-10-24 作成。本人判断）

## 0. 位置づけ・要旨

- 論文のアウトライン（`paper/notes/outline.md` の「現在の版」）の **4.b 2 つの選び方の比較** と、**4.a.iv・4.b.ii の対象の列を選ぶ** ための分析。
  - 時間的な距離（コード・出力では `time_distance`。`concept_drift_detection`）と特徴空間上の距離（`feature_distance`。`feature_space_window`）を、
    同じ評価日どうしで比べる。
  - 精度が大きく変わった列（同じ選び方の中・2 つの選び方の間）を**自動で選び、ファイルに残す**。寄与度（`concept_drift_cause`）はこのファイルを読んで対象を決める。
- **ほかの分析の出力ファイルを読むだけ**（コードは import しない。ほかの分析と同じく独立させる）。モデルは動かさず、計算もし直さない。
- 対象は **重みあり版（posw）だけ**（重みなし版は本人が言うまで封印）。
- 経緯：論文の書き出しは「時間と空間を比べる」なのに、分析は時間どうし・空間どうしだけだった（本人の気づき）。10/24 に AI がその場で集計した表（`paper/notes/extra_analyses.md` の「2 つの選び方の比較」）と、
  寄与度の対象を手で選んでいた件を、コードにする。

## 1. 読むもの（重みあり版。`data/analysis/preliminary_analysis/` の下）

| 中身 | 時間的な距離 | 特徴空間上の距離 |
|---|---|---|
| 評価日 × 行 × seed の指標 | `concept_drift_detection/posw/<repo>/set_transformer/<版>/daily_metrics.csv.gz` | `feature_space_window/posw/<repo>/feature_distance/<版>/daily_metrics.csv.gz` |
| セルの値（リリースごとの精度表） | 同じ場所の `probe/f1_plugin/drift_matrix.json` | 同じ場所の `probe/f1_plugin/drift_matrix.json` |
| 分布のずれ | `data_drift/posw/time_distance/<repo>/<版>/cells.csv`・`cells_summary.csv` | `data_drift/posw/feature_distance/<repo>/<版>/` の同じファイル |

- `daily_metrics.csv.gz` の列：`day`・`distance`（行 d。汎用ヘッドは `general`）・`seed`・`position`（保存物の向き：p0 ＝ リリース直前＝後期）・`missing`・`f1_plugin` など。
- 日数 N は、それぞれの精度表を回したときの値（時間的な距離は qtcreator 60・ほか 7 日、特徴空間上の距離は 14 か 30 日）。ここでは選び直さない。
- 版の一覧は、各出力の `run_meta.json`（またはフォルダ）から取る。

## 2. 日ごとの値（精度表と同じ決まり）

- 評価日 × 行ごとに、**5 seed の F1 値（`f1_plugin`）の中央値**（欠測の seed を除く）。精度表のセルの値は、これを評価日で平均したもの。
- 比べるのは **d ＝ 0 どうし**（それぞれの選び方の本来の使い方）。同じ選び方の中では d ＝ 0 と相手（d ＝ 1・2 の良い方）。

## 3. 精度の比較（A。論文 4.b.i・5.b.i）

- 時間的な距離の d ＝ 0 と特徴空間上の距離の d ＝ 0 を、**同じ評価日** で比べる。差 ＝ 空間 − 時間（正 ＝ 空間の方が高い）。
- まとめ方は 2 通り：
  1. **評価時期ごと（5 リリースまとめ）**：リポジトリ × 評価時期（3 列）＋ 全体。
  2. **リリースごと**：リポジトリ × 版 × 評価時期（90 列）。
- 各まとまりに出すもの：時間の平均・空間の平均・差・それぞれの評価日数・両方に値のある日数・空間が勝った日の割合・時間が勝った日の割合・評価日ごとの差の検定の p 値。
  - **平均は、それぞれのモデルに値のある日すべて**で取る（精度表の値と一致させる。同じ選び方の中の表の決まりと同じ）。
  - **勝った日の割合と検定は、両方に値のある日**で取る（対にする必要があるため）。検定はウィルコクソンの符号順位検定（両側）。
- 出力（`data/analysis/preliminary_analysis/selection_comparison/posw/`）：
  - `accuracy_by_phase.csv`・`.md`（1 の表）、`accuracy_by_release.csv`（2 の表）。
  - 列の番号は、**保存物の向き（`position`）と表示の向き（`position_display` ＝ 2 − position。p0 ＝ 初期）の両方**を書く。Markdown は表示の向きで書き、「初期・中期・後期」を添える。

## 4. 列の選択（B）

- **同じ選び方の中**（時間・空間それぞれ）：リリースごとの精度表（`drift_matrix.json` の値）で、列ごとに **d ＝ 0 − 相手**（相手 ＝ d ＝ 1・2 のうち値の大きい方）を出す。
  - 差の大きい順に **上回った側の上位 K 列** と、小さい順（最も負）に **下回った側の上位 K 列**。
- **2 つの選び方の間**：A のリリースごとの表で、**空間 − 時間** の大きい順に上位 K 列（空間が上回った側）、小さい順に上位 K 列（時間が上回った側）。
- 決まり（本人判断）：**差の大きさの順だけで選ぶ**（有意かどうか・評価日の数は見ない）。同じ差のときは、リポジトリ（`PROJECTS` の順）・版（古い順）・列（保存物の番号）の順。
  K は設定（`TOP_K = 3`）。
- 出力：`selected_columns.json`
  ```
  {"top_k": 3,
   "within": {"time_distance":    {"won": [列...], "lost": [列...]},
              "feature_distance": {"won": [...],   "lost": [...]}},
   "cross":  {"space_won": [...], "time_won": [...]}}
  ```
  列 ＝ `{"project", "version", "position"（保存物）, "position_display", "diff", "d0", "opponent"（同じ選び方の中だけ。1 か 2）, "opponent_value", "n_days"}`。
  あわせて人が読む `selected_columns.md`（表示の向き）。
- 確認：今の手書きの寄与度の対象（`concept_drift_cause` の `IG_TARGETS_TIME`・`IG_TARGETS_FEATURE` の 12 列）と、ここで選ぶ同じ選び方の中の 12 列が一致することをテストで確かめる。

## 5. 分布の比較（C。論文 4.a.iv・4.b.ii のデータ側）

- B で選んだ列について、2 つの学習の日の集まりと評価日の分布のずれを並べる。**`data_drift` の保存済みの結果を読むだけ**。
  - 同じ選び方の中：その選び方の `cells_summary.csv`・`cells.csv` の、行 d ＝ 0 と相手の行。
  - 2 つの選び方の間：時間的な距離の行 d ＝ 0 と、特徴空間上の距離の行 d ＝ 0（どちらも同じ版・同じ列）。
- 出すもの（群は全体 `all`）：55 特徴量の |δ| の平均、|δ| ≥ 0.147 の特徴量の数、|δ| ≥ 0.147 の特徴量の一覧（|δ| の大きい順。相手の一覧に出てこないものに印）。
- 出力：`drift_within.csv`・`drift_cross.csv` と、それぞれの `.md`（表示の向き）。

## 6. 実行

```
python3 -m src.analysis.preliminary_analysis.selection_comparison.main --mode accuracy --variant posw
python3 -m src.analysis.preliminary_analysis.selection_comparison.main --mode select --variant posw
python3 -m src.analysis.preliminary_analysis.selection_comparison.main --mode drift --variant posw
python3 -m src.analysis.preliminary_analysis.selection_comparison.main --mode all --variant posw
```
- `select` は `accuracy` の出力を、`drift` は `select` の出力を読む。`all` は順に全部。どれも Mac で数分（ファイルを読むだけ）。
- `--variant` は省略すると重みあり版。出力先に結果があれば上書きの警告を出す（ほかの分析と同じ）。

## 7. ディレクトリ・コード構成（予定）

```
selection_comparison/
  design.md          この設計書
  main.py            --mode accuracy / select / drift / all
  utils/constants.py 読み先・出力先・TOP_K・表示の向きの名前
  io/readers.py      daily_metrics・drift_matrix.json・data_drift の cells を読む
  compare/accuracy.py  A（日ごとの値・評価時期ごと・リリースごと・検定）
  compare/select.py    B（同じ選び方の中・2 つの選び方の間の上位 K 列）
  compare/drift.py     C（分布の比較）
```
- テスト：`tests/analysis/preliminary_analysis/selection_comparison/`（gitignore されているので `git add -f`）。小さな作り物の表で、平均・勝った日の割合・検定・選ぶ順・同じ差のときの順・表示の向きを確かめる。
  実データで、10/24 にその場で集計した値（`extra_analyses.md`）と、寄与度の手書きの 12 列に一致することも確かめる。

## 8. 限界として書くもの

- 2 つの選び方で日数 N が違う（時間は 1 週か 2 ヶ月、空間は 2 週か 1 ヶ月）。それぞれの最良どうしの比較で、同じ日数どうしではない。
- 差は小さく（±0.01 前後）、90 列を検定すると偶然でも 4〜5 列は有意になる。列の選択は差の大きさだけで、有意かどうかは見ていない。
- 検定は評価日どうしの独立を仮定している。

## 9. 決定の記録

- 2026-10-24（本人判断）：この分析を作る。置き場所は AI のおすすめ（新しいディレクトリ・出力を読むだけ）。同じ選び方の中の上位 K 列も自動で選ぶ（上回った・下回った両方）。
  選ぶ決まりは差の大きさの順だけ。判断値の表（`concept_drift_cause` §11.9）は同じ選び方の中と 2 つの選び方の間の両方で作る。平均は各モデルの全日、検定は両方に値のある日。
