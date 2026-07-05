# Concept Drift Existence 分析モジュール（事前分析①）

レビュー優先順位の **「判断基準が変化する区間は存在するのか」** を検証する分析モジュールです。
研究の目的（期間別モデルの **切り替えタイミング** の特定）の前提として、まず
**「ある時点で学習したモデルが、別の時点の優先順位をどれだけ再現できるか」** をリリースごとに測り、
**位置（時点）によって精度が落ちる ＝ 判断基準が変わった区間** があるかを統計的に判定します。

> 設計の詳細・意思決定の経緯は [`design.md`](./design.md) を参照してください。
> SHAP による原因分析（事前分析②）は別ディレクトリ [`../concept_drift_cause/`](../concept_drift_cause/) に予約のみ（本書のスコープ外）。

## 🎯 何を測るか

リリースサイクルを `BIN_COUNT`（既定 6）等分した時間ビンを使い、**「ビン i で学習 → ビン j で予測（i<j）」**
の精度を順位ベースで測って **四角の行列**にします。

- 計測点 `T` = **毎日 0 時の定点グリッド**（`MEASUREMENT_STEP_DAYS` 刻み）。各 `T` で Open な Change が対象。
- 1 レコード = `(Change, T)`。長く Open な Change は複数日に登場する（＝優先順位の「推移」）。
- ラベル = `time_to_next_review`（`T` から次の人間レビューまでの時間。pointwise 回帰で予測）。
- 行列の **縦軸 = 距離 `d = j - i`（滞留期間、ビン単位）**、**横軸 = 位置 `p = j`（予測する時点）**。
  これは手法図の菱形を正方形に整形したもの（`i>=j` は無し）。学習側ビン `i = p - d` は前リリースに及んでよい。

| 指標 (metric) | 意味（すべて順位ベース・0〜1・サイズ非依存） |
|---|---|
| `mae` | 正規化順位の平均絶対誤差（小さいほど良い） |
| `rmse` | 正規化順位の二乗平均平方根誤差（小さいほど良い） |
| `ndcg` | NDCG@n（上位重視のランキング良さ、大きいほど良い） |

**変化区間の判定**: 距離 `d` を固定して位置 `p` を横断比較し（距離効果を打ち消す）、
**時刻シャッフルの並べ替え検定**で「偶然では説明できない位置依存の精度低下」があるかを評価します。
リリースごとに判定し、**「N 本中 k 本で変化区間あり」**と本数で集計します。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `main.py` | エントリポイント。モデルを最外ループで全リリース×全指標を実行 |
| `utils/constants.py` | **設定の入口**（対象・グリッド・8:2固定数・反復・モデル・検定 など） |
| `utils/review_utils.py` | ボット/本人判定・「次の人間レビュー」抽出（3 一覧の和集合） |
| `../../background_problem/common/data_loader.py` | **共通**: Change / リリース日の読み込み・サイクル算出（再利用） |
| `labeling/label_builder.py` | `time_to_next_review` の計算（打ち切りは None。registry 形式） |
| `features/feature_builder.py` | 15 次元特徴ベクトルの組み立て（`src/features/*` を利用） |
| `features/fast_index.py` | developer/project 系 6 特徴の **事前集計＋二分探索**（高速化。`src/features` と同値） |
| `dataset/record_builder.py` | 毎日 0 時グリッド × Open 集合から `(Change, T)` レコード生成 |
| `dataset/binning.py` | 当該リリースを `BIN_COUNT` 等分し前リリースへ延長してビン割り当て |
| `dataset/splitter.py` | 層1/2/3 分類・またぎ禁止・固定 Change 数 8:2 の学習/評価分け |
| `model/regressor.py` | pointwise 回帰（lightgbm / random_forest。registry） |
| `evaluation/ranking_metrics.py` | 正規化順位の MAE/RMSE・NDCG@n |
| `evaluation/drift_matrix.py` | 四角行列の構築（各セル＝反復中央値。2 段中央値） |
| `evaluation/drift_detector.py` | 距離固定の位置比較＋並べ替え検定（**再学習しない**） |
| `io/result_writer.py` | 行列・位置別スコア・検定結果・本数集計の csv / json 出力 |
| `visualization/plotter.py` | 指標ごとのヒートマップ・位置別折れ線（日本語フォント自動選択） |

## 🚀 実行方法

プロジェクトのルートディレクトリ（`src/` がある場所）で実行します。

```bash
python -m src.analysis.preliminary_analysis.concept_drift_existence.main
```

> `-m` はモジュールをスクリプトとして実行するオプションです。`from src.~~~` の絶対インポートを
> 解決するため、ファイルパスではなくモジュール名で起動します。

実行すると `utils/constants.py` の `TARGET_PROJECTS` × `MODEL_NAME` × `ENABLED_METRICS` をすべて処理します。

- nova は約 4 万件の Change を読むため、初回ロードに数十秒〜数分かかります。
- 全実行（10 リリース × 2 モデル × `N_REPEATS`）は **数時間規模**です（モデル学習が支配的）。
  まず軽く試すなら `MODEL_NAME` を 1 つにする、`N_REPEATS` を下げる、などを推奨。
- **macOS で lightgbm を使う場合**は OpenMP ランタイムが必要です: `brew install libomp`。

## ⚙️ 設定の変え方（`utils/constants.py`）

主要なパラメータはすべてここに集約しています。

| 設定 | 説明 |
|---|---|
| `TARGET_PROJECTS` | 分析するプロジェクトと、そのリリース version の一覧 |
| `MEASUREMENT_STEP_DAYS` | 計測点グリッドの刻み日数（既定 1 = 毎日 0 時） |
| `LOOKBACK_DAYS` | アクティブ判定の遡り（`T-LOOKBACK <= created <= T < decision`） |
| `BIN_COUNT` | 当該リリースの等分数（＝距離の最大行数）。ビン幅・前リリースへの延長量を規定 |
| `TRAIN_EVAL_RATIO` / `N_TRAIN` / `N_EVAL` | 学習:評価の Change 数比（8:2）と全セル共通の固定 Change 数 |
| `MIN_TRAIN` / `MIN_EVAL` | 供給がこの床未満のセルは除外（NaN） |
| `N_REPEATS` / `T_AGG` / `REPEAT_AGG` | セルの 2 段集約（T 方向→反復方向、既定とも中央値） |
| `MODEL_NAME` | モデルの実行順（最外ループ。軽い lightgbm を先に） |
| `ENABLED_METRICS` / `NDCG_AT` | 出力する指標と NDCG@n の n |
| `PERMUTATION_N` / `SIGNIFICANCE` | 並べ替え検定の反復回数と有意水準 |
| `SAVE_PER_REPEAT` | 各反復の代表値も json に残すか |

### モデル・ラベルの追加
- モデルは `model/regressor.py` の `MODEL_REGISTRY` に factory を足し、`MODEL_NAME` に名前を追加。
- ラベルは `labeling/label_builder.py` の `LABEL_REGISTRY` に関数を足し、`LABEL_NAME` を切り替え。

## 📤 出力

`data/analysis/preliminary_analysis/concept_drift_existence/` 配下に保存されます。
**モデルごと → 指標ごと**にフォルダを分けます。

```
<project>/
└── <model>/                         # lightgbm / random_forest
    ├── <version>/                   # リリースごと（例: 20.0.0）
    │   ├── mae/  (rmse/, ndcg/ も同様)
    │   │   ├── drift_matrix.csv          # セル値（反復中央値）。行=距離 d、列=位置 p
    │   │   ├── drift_matrix.json         # 行列＋IQR＋各反復の生スコア＋メタ
    │   │   ├── drift_matrix.png          # 正方形ヒートマップ（固定スケール。縦=距離, 横=位置）
    │   │   ├── drift_matrix_relative.png # 相対版ヒートマップ（行列内 最良セル=1）
    │   │   ├── position_by_distance.csv  # 距離固定の位置別スコア（long 形式）
    │   │   ├── position_by_distance.png  # 距離ごとの位置別折れ線（固定スケール・生の値）
    │   │   ├── position_by_distance_relative.png  # 相対版 折れ線（行列内 最良セル=1）
    │   │   └── drift_test.json           # 検定結果（変化区間の有無・位置・p 値）
    │   └── ...
    └── summary/
        ├── mae/  (rmse/, ndcg/ も同様)
        │   ├── drift_count.csv           # 各リリースの drift_exists / min_p
        │   └── drift_count.json          # N 本中 k 本で変化区間あり（本数集計）
```

- **検定の情報（p 値・有意・変化点）は `drift_test.json` のみ**。png には一切描きません（図は素の値だけ）。
- 図には「良い向き」を **文字では** 書きません（指標名と軸の意味のみ）。ただし **ヒートマップの色は
  良い/悪いで統一**します（**緑=良い・赤=悪い**）。NDCG（大きいほど良い）と MAE/RMSE（小さいほど良い）で
  colormap を反転し、全指標で「良い端＝緑」にそろえます（増加/減少が指標で逆でも色の意味は一定）。
- **png は各指標フォルダに 4 枚**：固定スケール版（`drift_matrix.png` / `position_by_distance.png`、指標ごとの
  絶対値域で**バージョン間比較可能**）と、相対版（`*_relative.png`、「高いほど良い」に揃え**行列内の最良セル=1**
  として正規化＝行列内コントラスト重視）。相対版は各行列を自分の最良で正規化するため**バージョン間比較は不可**、
  かつ差はノイズを含むので過度に解釈しないこと（バージョン横断は固定スケール版で見る）。

## 📝 主な定義・前提（詳細は design.md）

- **計測点 `T`**: 毎日 0 時の定点グリッド。各 `T` で Open（`T-LOOKBACK <= created <= T < decision_time`）な Change が対象。
- **decision_time**: status が MERGED / ABANDONED の Change の `updated` を採用（それ以外は未決＝Open）。
- **人間のレビュー**: 投稿者本人・自動生成メッセージ（`tag` が `autogenerated:`）・ボットを除いたコメント。
  ボット判定は **3 一覧の和集合**（大文字小文字無視で `name`/`email`/`username` を照合）:
  1. `src/config/gerrymanderconfig.ini` の `[organization] bots`
  2. `src/config/third_party_ci_accounts.csv`（CI の `name` / `email`）
  3. `src/config/extra_bots.txt`（`jenkins` / `zuul` 等）
- **ラベルの観測窓**: `T` 以降、収集済みデータの末尾（およそ 2025 年）まで先読みして次レビューを探す。
  最後まで未レビューの `(Change, T)` は **打ち切りとして除外**（`CENSORING_MODE="drop"`）。
- **学習 target の log 変換**: `time_to_next_review` は指数的に広がるため、学習時は `log1p` 変換する
  （`LABEL_LOG_TRANSFORM=True`）。評価は順位ベースで log は順位不変なので指標の定義は変わらず、
  モデルの当てやすさだけが上がる。
- **特徴量**: 15 次元（`src/features/*`）。`review_metrics.calculate_uncompleted_requests` は **除外**。
  developer/project 系は `fast_index`（事前集計＋二分探索）で高速化（定義は `src/features` と同一）。

## ⚠️ 結果を解釈・記述するときの注意

データ処理には、見落としやすい **暗黙の前提** がいくつかあります。
結果を読むとき・他者に説明するとき・論文やスライドに手法を記述するときなど、**必ず併記し忘れない**よう注意してください。

- **「三角」ではなく「四角」**: (学習ビン, 予測ビン) で描くと菱形に見えますが、(距離, 位置) に取り直した
  正方形を扱っています。距離・滞留期間は **ビン単位**（1 ビン = リリース長 / `BIN_COUNT`）で、月固定ではありません。
- **前リリース（さらにその前）の混入**: 学習側ビンは当該リリース長ぶん前まで遡るので、当該リリースが
  長いと前々リリース以前のデータも学習に入りえます（許容。検出する位置 `p` は当該リリース内に閉じる）。
  ただし **検出を複数リリースで連結・プールはしない**（リリースごとに検出 → 本数で集計）。
- **8:2・固定数は「Change 数」で数える**: レコード数（行数）はセル/分割で一致しません（許容）。
  同一 Change は複数日のレコードを持つため。長期間 Open な Change が行数の大きな割合を占める場合があります。
- **またぎ禁止だが同一 Change の複数回利用は許容**: 同一 Change が学習と評価の両方には出ません（層2 排他振り分け）。
  一方、同一 Change が別の `T` で複数回 学習・評価されるのは許容（クラスタ補正はしない）。
- **評価指標は順位ベース・サイズ非依存**: 順位を [0,1] に正規化してから MAE/RMSE を算出（NDCG は元々 0〜1）。
  「MAE」とだけ書くと誤解を招くため、**正規化順位の MAE** である旨を明記する。
- **並べ替え検定は再学習しない**: `PERMUTATION_N`（既定 1000）は計算済み行列の並べ替え回数であって、
  モデル学習回数ではありません。学習回数は「有効セル数 × `N_REPEATS`」です。
- **`drift_detector` の統計量は第 1 版**（距離固定の位置系列に対するチェンジポイント＝最大前後平均差）。
  実装ノブとして差し替え可能です（design.md §5.9, §7）。

> 実行ログと各 `drift_matrix.json` の `meta`（bin/件数/反復/モデル/検定設定）、`drift_test.json`（p 値）、
> `summary/<metric>/drift_count.json`（本数集計）に処理条件が記録されるので、実際の値はそこで確認できます。
