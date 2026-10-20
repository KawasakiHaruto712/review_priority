# feature_space_window（RQ2）

**特徴空間で近い過去の日で probe を学習し、評価日で評価する**分析です。RQ2「優先度の変化は、評価対象との特徴空間上の距離に応じて表れるか」に使います。
詳しい決まりと経緯は [design.md](design.md) にあります。

## 何をするか

各評価日 X・各 seed・各大きさ k（1・3・7・14・30・60 日）について：

1. その日に Open な Change の埋め込み（凍結エンコーダの出力 128 次元）の**平均**を、日の代表にする（ラベルは使わない）。
2. X より前のすべての日（事前学習の区間と、評価期間のうち X より前の日）を候補にし、X との近さを
   **各次元を標準化したユークリッド距離**で測る（標準偏差は X より前の全日から出し直す。1e-8 未満は 1 とみなす）。
3. 近い順に **k 日**を選び、その日の Change で線形 probe（BCE）を学習し、X の Change で評価する。

RQ1 の窓長の調査（`lookback_window`）が「直前の k 日」で学習するのに対し、ここでは「特徴空間で近い k 日」で学習します。
大きさ・欠測の決まり・指標・まとめ方（日ごとに 5 seed の中央値 → 評価日で平均）は窓長の調査と同じなので、要約表をそのまま並べて比べられます。

## 前提

- 事前学習済みのエンコーダ（`pretrained_encoders`。6 件 × 5 seed）があること。なければ止まります（ここでは作りません）。
- 特徴量の表（`feature_table` の事前学習の表と評価期間の表）。なければ作り、作った条件の指紋が違えば警告を出して続けます（警告は表の隣の `<期間>_fingerprint_warnings.jsonl` にも残ります）。
- GPU（MPS）で回すときは、事前学習と同じく GPU の上限をコマンドに付けます。

## 実行

```
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 python3 -m src.analysis.preliminary_analysis.feature_space_window.main
```

| 引数 | 意味 |
|---|---|
| `--project qtbase` | 1 件だけ（既定は 6 件を順に。`all` も同じ） |
| `--versions 6.6.0 6.7.0` | 版を絞る（既定は 5 版） |
| `--n-seeds 2` | seed の数を絞る（既定は保存済み全部） |
| `--sizes 1 7 60` | 大きさを絞る（既定は 1 3 7 14 30 60） |
| `--mode plot` | 保存済みの `metrics.csv` から要約表だけ作り直す（モデルは回さない） |
| `--metric recall@10` | 要約表の指標（既定 auc） |

時間の見込み：Mac で窓長の調査と同じくらい（6 件で 4〜6 時間）。

## 出力（`data/analysis/preliminary_analysis/feature_space_window/<project>/`）

| ファイル | 中身 |
|---|---|
| `<版>/predictions.csv.gz` | Change ごとの予測（`day, size, seed, change_id, y_true, score`） |
| `<版>/metrics.csv` | 評価日 × 大きさ × seed ごとの指標と、件数・欠測の理由 |
| `<版>/selected_days.csv.gz` | 選ばれた日（`day, seed, rank, selected_day, days_before, distance, in_pretrain`）。大きさ 60 の分だけを順位付きで持つ（近い k 日は rank ≤ k） |
| `summary/summary_auc.csv`・`summary_auc.png` | 版 × 大きさの AUC（値 ± 標準偏差、右端は全評価日の平均） |
| `summary/summary_auc_missing.csv` | 版 × 大きさの欠測の日数 |
| `run_meta.json` | 実行条件（大きさ・距離の決まり・候補の範囲・seed・表の指紋・実行日時・git commit） |

- `days_before`：評価日の何日前か（特徴空間で近い日が時間でも近いのかを見る）。`in_pretrain`：事前学習の区間の日か（エンコーダの学習に使った日）。
- **回し直すと上書きされます**（結果は使い回しません）。予測は seed ごとに書き足すので、古い結果と混ざらないよう最初の seed で作り直します。

## 構成

```
main.py                     # 入口
utils/constants.py          # 6 件・大きさ・STD_EPS・指標の k・出力先
dataset/candidates.py       # 2 つの表をつないで全日の集合を作る・版のサイクル・評価日
selection/nearest_days.py   # 日の代表・標準化・距離・近い k 日
evaluation/runner.py        # 1 seed 分のループ（埋め込み → 近い k 日 → probe → 評価・欠測）
evaluation/metrics.py       # 指標（lookback_window の複製）
io/result_writer.py         # 出力
visualization/table.py      # 要約表（lookback_window と同じ形）
```

テスト：`python3 -m pytest tests/analysis/preliminary_analysis/feature_space_window`
