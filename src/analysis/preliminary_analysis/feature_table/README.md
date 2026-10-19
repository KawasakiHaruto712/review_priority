# feature_table

候補の特徴量 67 個（既存の 15 個 ＋ 新しい 52 個）を、6 件のリポジトリの事前学習の区間で計算して**表として書き出し**（`build/`）、
その表を読んで**相関の強い特徴量を落とす**（`selection/`）ためのディレクトリ。

- 表は、相関で絞ったあとも事前学習・lookback_window・concept_drift_detection・concept_drift_cause が共通に読めるように、どの段階にも属さない場所に置く。
- 特徴量やレコードを計算するコードは `pretrained_encoders` のものをそのまま呼ぶ（値は事前学習の計算と完全に同じ）。
- 詳細仕様は [design.md](./design.md)。特徴量そのものの定義と、値が出せないときの代わりの値は [pretrained_encoders/design.md](../pretrained_encoders/design.md) §11・§13。用語は **Change**（Gerrit）で統一。

## 使い方

```bash
# 1. 表を作る（データを読む。6 件で数時間。1 件で失敗しても止めずに次へ進み、最後に失敗したものを出す）
python3 -m src.analysis.preliminary_analysis.feature_table.build

# 1 件だけ作る（時間とメモリを先に確かめるときなど）
python3 -m src.analysis.preliminary_analysis.feature_table.build qtbase

# 2. 相関で絞る（表を読むだけ。数分。どれかの表がなければ止める）
python3 -m src.analysis.preliminary_analysis.feature_table.selection
```

project を指定しなければ、`pretrained_encoders/utils/constants.py` の `PROJECTS`（6 件）を回す。

## 表（`build/`。design.md §2）

- **期間**：事前学習の区間（データの始まり 〜 cutoff の版の公開日。`build_encoders` と同じ）。評価期間は今は入れない。
- **行**：事前学習と同じレコード（0 時のレコードと、日の途中で Open になったレコード）。(t, 番号) の昇順。
- **列**：

| 列 | 型 | 中身 |
|---|---|---|
| `change_id` | int64 | チケットの番号 |
| `t`・`created`・`decision_time`・`open_until` | datetime64[us] | 予測の時点・作成・最後に閉じた時刻・t を含む Open の期間の終わり（ないものは NaT） |
| `label` | float64 | 正解ラベル（t から 1 日以内に人のレビューが付くか） |
| `features` | float64（行数 × 67） | 並びは `feature_names`（既存 15 → チケット自身 13 → 開発状況 22 → 待っている間 17） |
| `filled` | bool（行数 × 20） | 並びは `filled_names`。値が出せずに代わりの値を入れたら True。**モデルには入れない**（相関で絞るときの帳簿と、論文に書く割合のため） |

- 形式は numpy の npz（圧縮なし）、精度は float64。作るときはレコードを溜めず、数値の配列のかたまりへ詰めながら作る（メモリ 16 GB で qtbase の約 337 万件を扱うため）。

### 出力

```
data/analysis/preliminary_analysis/feature_table/table/<project>/
  pretrain.npz         # 表
  pretrain_meta.json   # 作った条件（コードのコミット・定数・入力のハッシュ・行数・代わりの値の件数・かかった時間）
```

### 読み込み

```python
from src.analysis.preliminary_analysis.feature_table.build import load_table

tab = load_table("qtbase")                                          # 67 個すべて
tab = load_table("qtbase", features=["lines_added", "dir_count"])   # 列を選ぶ（並びは指定どおり）
tab["features"], tab["feature_names"], tab["filled"], tab["t"], tab["label"]
```

集合（その日に Open なチケット）は、`t` の日付で行をまとめれば事前学習と同じに作れる。

## 相関で絞る（`selection/`。design.md §3）

1. 表の全レコードの、代わりの値で埋めた後の値で、リポジトリごとに Spearman の順位相関を出す（値が一定の列は ρ が出ない → 「どれとも相関が強くない」と扱う）。
2. 6 件をまとめて「相関が強い組」（|ρ| ≥ 0.7）を決める。決まりは次の 5 通りを全部出す。**採るのは `ratio`**（2026-10-18 決定）。ほかの 4 通りは参考。
   - `ratio`：|ρ| ≥ 0.7 になったリポジトリが、ρ が出るリポジトリの 2/3 以上（6 件そろう組は 6 件中 4 件以上。CI の列のように 3 件でしか ρ が出ない組は 3 件中 2 件以上）
   - `majority3`・`majority4`・`majority5`：|ρ| ≥ 0.7 になったリポジトリが k 件以上（ρ が出ないリポジトリは「0.7 未満」に数える）
   - `mean`：6 件の |ρ| の平均が 0.7 以上（ρ が出ないリポジトリを除いた平均）
3. **AutoSpearman の 1 段目**（Jiarpakdee ら 2018）：強い組から順に、組の外の特徴量との |ρ| の平均（リポジトリごとの平均を 6 件で平均）が小さい方を残し、大きい方を落とす。VIF の 2 段目は使わない（前例 Khatoonabadi 2024）。

### 出力

```
data/analysis/preliminary_analysis/feature_table/selection/
  corr_<project>.csv      # リポジトリごとの順位相関行列（67 × 67）
  pairs.csv               # 組ごとの 6 件の ρ・|ρ| ≥ 0.7 の件数・ρ が出る件数・|ρ| の平均・それぞれの「組の外との |ρ| の平均」
  selected.csv            # 特徴量ごとに、5 通りの決まりで残るか（1）落ちるか（0）。採るのは ratio の列
  steps_<決まり>.csv      # AutoSpearman の手順の記録（組・|ρ| の平均・それぞれの「組の外との平均」・残した方と落とした方）
  feature_summary.csv     # 特徴量ごと・リポジトリごとの、代わりの値を入れた割合と 0 でない値の割合
  selection_meta.json     # 使った表の作った条件と、閾値・決まり・採った決まり（ratio）
```

## ディレクトリ構成

```
feature_table/
  design.md            # 設計書
  constants.py         # 置き場所・67 個と 20 個の並び・閾値・決まり
  build/
    table.py           # build_arrays（表の配列を作る）, save_table, load_table, build_meta, build_project
    __main__.py        # 表を作るエントリ
  selection/
    autospearman.py    # spearman_matrix, aggregate（6 件のまとめ）, auto_spearman（1 段目）
    run.py             # 表を読んで相関を出し、結果を書く
    __main__.py        # 相関で絞るエントリ
```

## 後回しにしていること（design.md §2.7）

- 事前学習（`build_encoders`）が表を読むようにする部分（相関で特徴量が絞られた後に作る）。そのときに、`pretrain_meta.json` の条件が今のコード・データと合うかを照合する処理も作る。
- 評価期間の表。

## テスト

`tests/analysis/preliminary_analysis/feature_table/`（通信なし・小さな作り物のデータ）。表の値がその場で計算した値と float64 で完全に一致すること、
AutoSpearman の手順を手計算と比べることなどを確かめる。`tests/` は git の対象外なので、新しいテストは `git add -f` が要る。

```bash
python3 -m pytest -q tests/analysis/preliminary_analysis/feature_table
```
