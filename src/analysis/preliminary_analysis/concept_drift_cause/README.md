# concept_drift_cause

事前分析 **特徴量の寄与度分析：距離×時期行列で見えた精度変動の原因を、モデル側から説明する**。

距離×時期行列の分析（`concept_drift_detection`）は「いつ・どれだけ古い学習データだと精度が落ちるか」を示したが、「なぜ落ちたか」は答えない。ここでは **Integrated Gradients** で、**そのセルの精度を出したモデルが 15 個の特徴量をどう使っていたか**を符号付きで取り出す。

- **正解ラベルで正例群／負例群に分け**、群ごとに 15 次元のプロファイルを出す。
- **同じ seed の汎用ヘッド**も出す。汎用ヘッドは重みが更新されないので、その変化は**データが変わった分だけ**を表す。差し引けばモデルの変容が残る。
- 本ディレクトリでは**事前学習も probe 学習もしない**。エンコーダは `pretrained_encoders` から、probe は **距離×時期行列の分析が保存したもの**を load する。
- 詳細仕様は [design.md](./design.md)。用語は **Change** で統一。

## 前提

**距離×時期行列の分析を先に回し終えている必要がある。** 特徴量の寄与度分析は距離×時期行列の分析の出力を 2 つ使う。

| 使うもの | 用途 |
|---|---|
| `daily_metrics.csv.gz` | `(評価日, 距離)` ごとに AUC が中央だった seed を特定する（＝そのセル値を出したモデル） |
| 保存済み probe | その seed の probe で IG を計算する |

距離×時期行列の分析は `N_REPEATS = 5`（奇数）で回すこと。偶数だと中央値が 2 個のモデルの平均になり、モデルを一意に特定できない。

## 使い方

対象セルは `utils/constants.py` の `IG_TARGETS` で指定する（距離と位置のリストの直積。`"all"` 可）。

```bash
# 計算：IG_TARGETS のセルについて IG を計算し ig_daily.csv.gz に追記
python -m src.analysis.preliminary_analysis.concept_drift_cause.main --mode compute

# 抽出：保存済みから表を作る（何度でも／モデルを回さない）
python -m src.analysis.preliminary_analysis.concept_drift_cause.main --mode extract
```

計算と抽出は分離してある。重い計算は一度だけ回し、**どのセル・どの特徴量を考察するかは後から切り出す**。

## 目安の時間

全プロジェクトの最新版・全セルで **約 14.6 時間**。nova の 1 セルが約 2.8 分。詳細は design.md §10。
