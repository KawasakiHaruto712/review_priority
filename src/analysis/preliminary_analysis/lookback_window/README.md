# lookback_window

事前分析 **step1：チューニング窓の長さ（＝直近どれだけのデータで調整すべきか）の調査**。

評価日の直前で終わる**学習窓の長さ（1d/3d/1w/2w/1M/2M）をスイープ**し、**日次で線形ヘッドを貼り直して**精度を測る。凍結した事前学習エンコーダの上で probe だけ学習するので日次でも軽い。

- **主目的は「適切なデータ量（窓長）の調査」**（変化は二の次）。**主成果物は「窓長 × バージョンの要約表」**、折れ線は補助。
- 共有部品（モデル・データ構築・特徴・label・Scaler・probe・embed）は **`pretrained_encoders` から import**。**concept_drift_detection には依存しない**。
- 詳細仕様は [design.md](./design.md)。用語は **Change** で統一。

## 前提
事前学習エンコーダ（`pretrained_encoders` の保存物）を load して使う。**保存済みが要求 seed 数に満たなければ、run 時に不足分を自動作成する**ので、`lookback_window` 単体を実行すれば必要数まで揃えて probe 結果まで出る。

（明示的に先に作っておくことも可能。初回の自動作成は全履歴の特徴生成で数十分かかる点に注意。）

```bash
# 任意：先に作る場合
python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders
```

## 使い方

```bash
# 実行：load → 日次×窓長 probe → 生予測・指標を保存 → 作図（表＋折れ線）
python -m src.analysis.preliminary_analysis.lookback_window.main --mode run

# 描画のみ（保存済みから・モデルを回さない）。指標を切替
python -m src.analysis.preliminary_analysis.lookback_window.main --mode plot --metric recall@10

# 生予測から指標テーブルだけ作り直す（新しい指標を足したいとき）
python -m src.analysis.preliminary_analysis.lookback_window.main --mode recompute
```

### 主なオプション
- `--metric <name>`：描画する指標（既定 `auc`。例 `recall@10`, `norm_rank`）。全指標は保存済みなので再実行不要で描き替え可。
- `--n-seeds <int>`：使う事前学習 seed 数（既定 全部）。
- `--windows <days...>`：描く窓長（例 `--windows 1 7 30`）。
- `--no-general`：汎用ヘッド（チューニングなし）基準線を描かない（既定は重ねる）。

## 出力

```
data/analysis/preliminary_analysis/lookback_window/<project>/
  <version>/predictions.csv.gz   # Change ごと (day,window,seed,change_id,y_true,score)
  <version>/metrics.csv          # (day,window,seed,カウント,missing,missing_reason,各指標...)
  <version>/lines_<metric>.png   # 補助：素の折れ線（窓長ごと・日次）
  summary/summary_<metric>.png   # 主成果物：窓長×バージョンの要約表（右端 Ave＝横断平均）
  summary/summary_<metric>.csv
```

- **要約表**：行＝窓長（2M/1M/2w/1w/3d/1d）、列＝バージョン。セル＝指標値＋小さく欠測量。**右端 Ave＝バージョン横断平均（＝どの窓長が平均的に良いか＝本分析の主結論）**。
- **折れ線**：横＝日、縦＝指標。p1〜p6 破線・月次集計は入れない素の折れ線。凡例に欠測日数を控えめ併記。汎用ヘッド基準線をデフォルトで重ねる。

## 仕組み（design.md 準拠）
- 学習窓 ＝ 評価日の直前 W 日（[X−W, X−1]。末尾＝前日 X−1。評価日自身は予測対象なので入れない）。
- 欠測ガード：probe は学習窓に **正例≥1 かつ 負例≥1**、評価は評価日に **正例≥1 かつ 負例≥1**。満たさない (評価日×窓長) は欠測。
- 埋め込みは全 Change を 1 回計算してキャッシュ → 各 (日×窓長) は線形ヘッドを貼り直すだけ。

## ディレクトリ構成

```
lookback_window/
  main.py             # モード分岐（run / plot / recompute）
  utils/              constants.py
  sweep/              window_sweep.py（窓切り出し・欠測ガード・per-(day,window) probe）
  evaluation/         metrics.py（AUC/precision/recall/F1/top-k/MAP/正規化順位/MRR）
  io/                 result_io.py（生予測・指標テーブルの保存/読込）
  visualization/      table.py（要約表）, plotter.py（折れ線）
```

## スコープ外（後続）
- **変化点検出は step2**（`concept_drift_detection` 側）。本分析では実施しない。
- feature freeze 等の節目縦線、特徴寄与（step3）、チューニング（step4）は後続。
