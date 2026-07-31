# Concept Drift Detection 分析モジュール（事前分析①・再設計版）

レビュー優先順位の **「判断基準が変化する区間は存在するのか」** を検証する分析モジュールです。
**「ある時点で学習したモデルが、別の時点のレビュー優先順位をどれだけ再現できるか」** をリリースごとに測り、
**位置（時点）によって精度が落ちる ＝ 判断基準が変わった区間** があるかを統計的に判定します。

旧 `concept_drift_existence`（連続 `time_to_next_review` ／ 層1・2・3の排他振り分け ／ 順位・回帰・分類の3系統）から、
**目的変数を「Δ以内にレビューされたか」の2値**にし、評価を **precision / recall / f1** に絞った再設計版です。

> 設計の詳細・意思決定の経緯は [`design.md`](./design.md) を参照してください。
> SHAP による原因分析（事前分析②）は本書のモデル・データ設計を流用して後日（本書スコープ外）。
> merge/reject（`decision_result`）も本書スコープ外。

## 🎯 何を測るか

リリースサイクルを `BIN_COUNT`（既定 6）等分した時間ビンを使い、**「ビン i で学習 → ビン j で予測（i<j）」**
の精度を測って **四角の行列**にします。

- 計測点 `T` = **毎日 0 時の定点グリッド**（`MEASUREMENT_STEP_DAYS` 刻み）。各 `T` で Open な Change が対象。
- 1 レコード = `(Change, T)`。長く Open な Change は複数日に登場する（＝実運用で毎日並べ直す様子に忠実）。
- **目的変数**: 「`T` から **Δ（`REVIEW_HORIZON_DAYS`、既定 7 日）以内**に人間レビューが付くか」の **2 値**
  （正例=1 / 負例=0）。等価に `time_to_next_review ≤ Δ`。**2 値分類**（LightGBM / RandomForest）。
- 行列の **縦軸 = 距離 `d = j - i`（滞留期間、ビン単位）**、**横軸 = 位置 `p = j`（予測する時点）**。
  学習側ビン `i = p - d` は前リリースに及んでよい。

評価指標（正例＝Δ以内。**評価ビン内の全レコードをプール**して算出）:

| 指標 (metric) | 意味 |
|---|---|
| `precision` | Δ以内と予測したうち本当にΔ以内だった割合（大きいほど良い） |
| `recall` | 実際にΔ以内のうち当てた割合（大きいほど良い） |
| `f1` | precision と recall の調和平均（大きいほど良い） |

> 不均衡（Δ=1週で正例 ~26–30%）ゆえ **accuracy は使いません**。閾値は `CLASSIFY_THRESHOLD`（既定 0.5）。
> 補助指標（`auc` / `mcc` 等）を後で足したい場合は `ENABLED_METRICS` に追加できます。

**変化区間の判定**: 距離 `d` を固定して位置 `p` を横断比較し（距離効果を打ち消す）、
**並べ替え検定**で「偶然では説明できない位置依存の精度低下」があるかを評価します。
リリースごとに判定し、**「N 本中 k 本で変化区間あり」**と本数で集計します。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `main.py` | エントリポイント。モデルを最外ループで全リリース×全指標を実行（`run`/`recompute`/`replot`） |
| `utils/constants.py` | **設定の入口**（Δ・グリッド・学習/評価数・重み・マージン・反復・モデル・検定 など） |
| `utils/review_utils.py` | ボット/本人判定・「次の人間レビュー」抽出（3 一覧の和集合） |
| `../../background_problem/common/data_loader.py` | **共通**: Change / リリース日の読み込み・サイクル算出（再利用） |
| `labeling/label_builder.py` | Δ以内2値ラベル（`[T,T+Δ]` にレビューあり=1 / 無し=0 / 窓が観測末尾超は None＝除外） |
| `features/feature_builder.py` | 15 次元特徴ベクトルの組み立て（`src/features/*` を利用） |
| `features/fast_index.py` | developer/project 系特徴の **事前集計＋二分探索**（高速化。`src/features` と同値） |
| `dataset/record_builder.py` | 毎日 0 時グリッド × Open 集合から `(Change, T)` レコード生成＋2値ラベル付与 |
| `dataset/binning.py` | 当該リリースを `BIN_COUNT` 等分し前リリースへ延長してビン割り当て（`BIN_DAY_ALIGNED=True` で境界を 0 時・整数日幅に揃える） |
| `dataset/sampler.py` | 学習 `N_TRAIN` Change を無作為抽出・評価は全件（混ぜ許容・層なし） |
| `model/classifier.py` | 2 値分類（lightgbm / random_forest。`sample_weight` 対応。registry） |
| `evaluation/binary_metrics.py` | precision / recall / f1（プール集約・`sample_weight` 対応） |
| `evaluation/drift_matrix.py` | 四角行列の構築（学習→予測→指標。Δマージン・重み・2段集約。予測保存も） |
| `evaluation/drift_detector.py` | 距離固定の位置比較＋並べ替え検定（**再学習しない**） |
| `io/result_writer.py` | 行列・位置別スコア・検定結果・本数集計・**生予測(predictions.csv.gz)** の csv / json 出力 |
| `visualization/plotter.py` | 指標ごとのヒートマップ・位置別折れ線（日本語フォント自動選択） |

## 🚀 実行方法

プロジェクトのルートディレクトリ（`src/` がある場所）で実行します。

```bash
python -m src.analysis.preliminary_analysis.concept_drift_detection.main
```

実行すると `utils/constants.py` の `TARGET_PROJECTS` × `MODEL_NAME` を処理します。

- nova は約 4 万件の Change を読むため、レコード生成に **1 リリースあたり数分**かかります。
- 全実行（リリース × モデル × `N_REPEATS`）は **数時間規模**です（レコード生成とモデル学習が支配的）。
  まず軽く試すなら `TARGET_PROJECTS` を 1 リリース・`MODEL_NAME` を 1 つ・`N_REPEATS` を下げる、を推奨。
- **macOS で lightgbm を使う場合**は OpenMP ランタイムが必要です: `brew install libomp`。

### サブコマンド
| コマンド | 動作 |
|---|---|
| `python -m ...concept_drift_detection.main` | 全実行（学習＋表＋検定＋図）。生予測 `predictions.csv.gz` も保存 |
| `python -m ...concept_drift_detection.main recompute` | 保存済み `predictions.csv.gz` から**表・検定・図を作り直す（再学習なし）**。指標を追加した後などに使う |
| `python -m ...concept_drift_detection.main replot` | 保存済み `drift_matrix.json` から**図だけ再描画**（再計算なし） |

> `recompute` は保存済み予測（`y_true`=0/1, `y_pred`=正例確率）から同じロジックで再計算するため、
> 実行時（メモリ上）と結果が一致します（単一経路）。閾値や指標を変えたら recompute で反映できます。

## ⚙️ 設定の変え方（`utils/constants.py`）

主要なパラメータはすべてここに集約しています。

| 設定 | 既定 | 説明 |
|---|---|---|
| `TARGET_PROJECTS` | nova + versions | 分析するプロジェクトと、そのリリース version の一覧 |
| `REVIEW_HORIZON_DAYS` | `7` | **Δ（レビュー判定の期間, 日）。1週間→1日 等に変更可** |
| `MEASUREMENT_STEP_DAYS` | `1` | 計測点グリッドの刻み日数（毎日 0 時） |
| `LOOKBACK_DAYS` | `365` | アクティブ判定の遡り（`T-LOOKBACK <= created <= T < decision`） |
| `BIN_COUNT` / `BINNING` | `6` / `equal_time` | 当該リリースの等分数・ビン方式 |
| `BIN_DAY_ALIGNED` | `True` | ビン境界を 0 時に揃える（開始を 0 時に丸め・幅を整数日に切り上げ）。毎日 0 時の計測点と一致し、Δ=`MEASUREMENT_STEP_DAYS` 日なら学習末尾除外が厳密に 0。`False` で秒単位等分 |
| `N_TRAIN` | `500` | 学習 Change 数（全セル共通の固定数。比較可能性のため） |
| `N_EVAL` | `"all"` | 評価 Change 数。`"all"`=全 Change、数値なら固定数に切替 |
| `MIN_TRAIN` / `MIN_EVAL` | `30` / `30` | 供給がこの床未満のセルは除外（NaN） |
| `CHANGE_BALANCED_WEIGHT` | `True` | **学習**時、各行に `1/(その Change の学習内レコード数)` の重み（長寿 Change の水増し是正） |
| `EVAL_CHANGE_BALANCED_WEIGHT` | `False` | **評価**の Change 均等重み。既定なし（ON で長寿支配とセル間交絡を抑制） |
| `TRAIN_TAIL_MARGIN` | `True` | 学習ビン末尾 Δ を除外（学習ラベルが未来を覗くリークを防ぐ） |
| `EVAL_TAIL_MARGIN` | `False` | 評価の自主的末尾除外（絶対末尾は物理的に自動除外＝ラベル None） |
| `CLASSIFY_THRESHOLD` | `0.5` | 2 値判定しきい値 |
| `N_REPEATS` / `REPEAT_AGG` | `10` / `median` | セルの反復回数と反復方向の集約（中央値） |
| `MODEL_NAME` | `[lightgbm, random_forest]` | 使用モデル |
| `ENABLED_METRICS` | `[precision, recall, f1]` | 評価指標（`auc`/`mcc` 等を追加可） |
| `PERMUTATION_N` / `SIGNIFICANCE` | `1000` / `0.05` | 並べ替え検定の反復回数と有意水準 |
| `SAVE_PREDICTIONS` / `SAVE_PER_REPEAT` | `True` / `True` | 生予測の保存 / 各反復値を json に残すか |

### モデル・指標の追加
- モデルは `model/classifier.py` の `MODEL_REGISTRY` に factory を足し、`MODEL_NAME` に名前を追加。
- 指標は `evaluation/binary_metrics.py` に実装し、`ENABLED_METRICS` に追加（`recompute` で既存予測から反映可能）。

## 📤 出力

`data/analysis/preliminary_analysis/concept_drift_detection/` 配下に保存されます。
**モデルごと → リリースごと → 指標ごと**にフォルダを分けます（目的変数は単一なので `<target>` 階層はありません）。

```
<project>/
└── <model>/                              # lightgbm / random_forest
    ├── <version>/                        # リリースごと（例: 20.0.0）
    │   ├── predictions.csv.gz            # 生予測 (d,p,repeat,t,change_id,y_true,y_pred)。recompute の元
    │   ├── predictions_meta.json         # 列の意味（y_true=0/1, y_pred=正例確率）・Δ など
    │   ├── precision/  (指標ごと: precision / recall / f1)
    │   │   ├── drift_matrix.csv          # セル値（反復中央値）。行=距離 d、列=位置 p
    │   │   ├── drift_matrix.json         # 行列＋IQR＋各反復の生スコア＋メタ
    │   │   ├── drift_matrix.png          # 正方形ヒートマップ（固定スケール。縦=距離, 横=位置）
    │   │   ├── drift_matrix_relative.png # 相対版ヒートマップ（行列内 最良セル=1）
    │   │   ├── position_by_distance.csv  # 距離固定の位置別スコア（long 形式）
    │   │   ├── position_by_distance.png  # 距離ごとの位置別折れ線（固定スケール・生の値）
    │   │   ├── position_by_distance_relative.png  # 相対版 折れ線
    │   │   └── drift_test.json           # 検定結果（変化区間の有無・位置・p 値）
    │   ├── recall/ ...
    │   └── f1/ ...
    └── summary/
        └── precision/  (指標ごと)
            ├── drift_count.csv           # 各リリースの drift_exists / min_p
            └── drift_count.json          # N 本中 k 本で変化区間あり（本数集計）
```

- **検定の情報（p 値・有意・変化点）は `drift_test.json` のみ**。png には一切描きません。
- **png は各指標フォルダに 4 枚**：固定スケール版（`drift_matrix.png` / `position_by_distance.png`、**バージョン間比較可能**）と、
  相対版（`*_relative.png`、行列内の最良セル=1 に正規化＝コントラスト重視。**バージョン間比較は不可**）。
  色は **緑=良い・赤=悪い**（precision/recall/f1 はいずれも大きいほど良い）。

## 📝 主な定義・前提（詳細は design.md）

- **計測点 `T`**: 毎日 0 時の定点グリッド。各 `T` で Open（`T-LOOKBACK <= created <= T < decision_time`）な Change が対象。
- **decision_time**: status が MERGED / ABANDONED の Change の `updated` を採用（それ以外は未決＝Open）。
- **人間のレビュー**: 投稿者本人・自動生成メッセージ（`tag` が `autogenerated:`）・ボットを除いたコメント。
  ボット判定は **3 一覧の和集合**（`gerrymanderconfig.ini` の bots / `third_party_ci_accounts.csv` / `extra_bots.txt`）。
- **ラベル（Δ以内2値）**: `[T, T+Δ]` に人間レビューあり→1、無くて窓を完全観測できる（`T+Δ ≤ 観測末尾`）→0、
  窓が観測末尾を超えて未確定→**None（そのレコードは作らない）**。旧 `CENSORING_MODE="drop"` は不要（Δ有界）。
- **先読みマージン**:
  - 学習: 学習ビン末尾 Δ の計測点を除外（`TRAIN_TAIL_MARGIN=True`）。学習ラベルが学習期間を超えて未来を覗くリークを防ぐ。
  - 評価: 自主的な末尾除外はしない（`EVAL_TAIL_MARGIN=False`）。評価ラベルは答え合わせでモデルに入らずリークしない。
    除外されるのは**収集データの絶対末尾 Δ**（物理的に先が無い＝ラベル None）だけ。
  - `BIN_DAY_ALIGNED=True` かつ `Δ = MEASUREMENT_STEP_DAYS` 日のとき、**学習側の末尾除外は厳密に 0**（境界が計測点と一致）。残るのは評価側の絶対末尾 Δ（最後のリリースの末尾 1 日）だけ。
- **混ぜ許容（層なし）**: 同一 Change が学習と評価に跨るのを許容（実運用忠実）。旧 §2.6 の層1/2/3・排他振り分けは廃止。
- **学習/評価数**: 学習は `N_TRAIN`(=500) Change 固定、評価は全 Change（`N_EVAL="all"`）。数えるのは Change 数。
- **特徴量**: 15 次元（`src/features/*`）。`review_metrics.calculate_uncompleted_requests` は **除外**。
  developer/project 系は `fast_index`（事前集計＋二分探索）で高速化（定義は `src/features` と同一）。

## ⚠️ 結果を解釈・記述するときの注意

- **「三角」ではなく「四角」**: (学習ビン, 予測ビン) では菱形に見えますが、(距離, 位置) に取り直した正方形を扱います。
  距離・滞留期間は **ビン単位**（1 ビン = リリース長 / `BIN_COUNT`）で、月固定ではありません。
- **前リリースの混入**: 学習側ビンは当該リリース長ぶん前まで遡るため、前リリース以前のデータが学習に入りえます
  （許容。検出する位置 `p` は当該リリース内に閉じる）。**検出は複数リリースで連結・プールしない**（本数で集計）。
- **混ぜ許容の帰結**: 同一 Change が学習と評価に跨るのを許すため、精度には**楽観バイアス**が乗り、
  跨り Change を通じて**ドリフトがやや薄まりうる**（実運用忠実を優先した設計上の受容。design.md §2.6）。
- **学習は重みあり／評価は重みなし（既定）**: 学習は change 均等重みで公平化、評価は実運用の判定品質を測る（非対称）。
  評価も change 重みにしたい場合は `EVAL_CHANGE_BALANCED_WEIGHT=True`。
- **クラス不均衡**: 正例（Δ以内）は Δ=1週で ~26–30%。**accuracy ではなく precision/recall/f1** で見る。
- **並べ替え検定は再学習しない**: `PERMUTATION_N` は計算済み行列の並べ替え回数であって、モデル学習回数ではありません。
- **`drift_detector` の統計量は第 1 版**（距離固定の位置系列に対するチェンジポイント＝最大前後平均差）。差し替え可能。

> 実行ログと各 `drift_matrix.json` の `meta`、`drift_test.json`（p 値）、`summary/<metric>/drift_count.json`（本数集計）
> に処理条件が記録されるので、実際の値はそこで確認できます。
