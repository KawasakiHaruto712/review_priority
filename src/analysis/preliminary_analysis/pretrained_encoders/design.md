# pretrained_encoders 設計書（共有基盤：事前学習エンコーダ＋汎用ヘッドの作成・保存）

## 0. 位置づけ・要旨
- 集合 Transformer の**事前学習エンコーダ（表現）＋汎用ヘッド（仮予測ヘッド）を（プロジェクトごとに）一度だけ作って保存**し、下流の分析（まず `lookback_window`、将来は他分析も）が **load して使い回す**ための**共有基盤**。
- **対象は複数プロジェクト**（nova / neutron / cinder / glance / keystone / swift）。project を引数に取り、**プロジェクトごとに独立のエンコーダ群**を作る（§1, §7 の `PROJECTS`）。
- 前回（`concept_drift_detection`）は実行時にエンコーダを保存しておらず作り直しになった。その反省で**作成と利用を分離**する。
- **本ディレクトリを"基点（基盤）"とする**：モデル・データ構築・特徴・label・Scaler など必要な部品を**本ディレクトリ内に持つ**（他分析からの流用ではなく、ここを起点にして、下流はここを import／オーバーライドして使う）。`concept_drift_detection` の該当コードは**参考にしてよいが依存しない**（本ディレクトリを正とする）。
- 学習するのは**エンコーダ**と**汎用ヘッド**。下流は凍結エンコーダの上で線形ヘッドを貼り直す（＝チューニング。`lookback_window` §5）。汎用ヘッドは「チューニングしない汎用予測」の基準として下流でも使える。
- 用語：レビュー対象の単位は **Change**（Gerrit）で統一。

## 1. 事前学習データ（プロジェクトごとに 1 モデル）
- **対象プロジェクト**：nova / neutron / cinder / glance / keystone / swift（`PROJECTS`＝§7）。各プロジェクトで**独立に**エンコーダ＋汎用ヘッドを作る。
- **プロジェクト開始 〜 そのプロジェクトの cutoff リリース日まで**の全計測点を使う。cutoff は `PROJECTS` で project ごとに定義（例：nova=25.0.0 / neutron=20.0.0 / cinder=20.0.0 / glance=24.0.0 / keystone=21.0.0 / swift=2.29.0）。cutoff の**日付は `major_releases_summary.csv` から引く**（日付は二重管理しない）。
- その project の**チューニング5版のどの下流分析でも、その project の同一エンコーダ／汎用ヘッドを使う**（per-release では作らない。shared 方式）。
- リーク防止：チューニング対象期間（cutoff より後）は事前学習に**含めない**。

## 2. 計測点・集合の単位・特徴・標準化（本ディレクトリに実装）
- 計測点 T ＝ 毎日 0 時（`MEASUREMENT_STEP_DAYS=1`）。各 T で **アクティブな Change 群**（`T-LOOKBACK <= created <= T < decision_time`）を 1 つの「集合」とする。
- 特徴は **15 種類**（`FEATURE_NAMES` と算出ロジックを本ディレクトリに持つ）。
- 目的変数（事前学習の教師）：`reviewed_within_delta`、Δ=1 日（label ロジックも本ディレクトリに持つ）。
- **Scaler は本事前学習データで fit** し、統計量（mean/std）を**保存**する。下流はこの統計で transform するだけ（下流で再 fit しない）。

## 3. モデル：集合 Transformer（本ディレクトリに実装）
- `SetEncoder`（自己注意・**位置エンコなし**＝置換不変）を本ディレクトリに実装。
- 設定：`D_MODEL=128 / N_LAYERS=2 / N_HEADS=4 / FFN_DIM=256 / DROPOUT=0.1 / MAX_SET_SIZE=512`。
- 併せて `Head`（線形／MLP）・`Scaler`・`pretrain`・`embed_sets`・`train_probe` も本ディレクトリに持ち、**下流はここから import**する（＝基点）。

## 4. 事前学習手順（教師あり）＋汎用ヘッド
- **エンコーダ＋汎用ヘッド（線形の仮予測ヘッド）**で「Δ以内レビュー(0/1)」を予測し、`BCEWithLogitsLoss` で学習（`PRETRAIN_METHOD="supervised"`）。
- ハイパラ：`PRETRAIN_EPOCHS=20 / PRETRAIN_LR=1e-3 / PRETRAIN_BATCH_SETS=16`。
- 学習後、**エンコーダを凍結**。**エンコーダと汎用ヘッドの両方を保存（両方とも必須）**。
- 汎用ヘッドの役割：下流での「**チューニングしない汎用予測**」の基準線・比較（`concept_drift_detection` §8.3 の汎用ヘッド相当）、および線形ヘッドの初期値としても利用可。

## 5. 反復（seed ＝ 別エンコーダ）
- `N_REPEATS` 個を作る（seed = `RANDOM_SEED + k`）。各 seed で **エンコーダ＋汎用ヘッド**を保存。
- 既定は 10 個（ばらつき・IQR 用）。下流は保存済みの seed 群を load する。

## 6. 保存フォーマット（＝下流との「約束事」・最重要）
出力先：`data/analysis/preliminary_analysis/pretrained_encoders/`

```
<OUTPUT>/<project>/cutoff_<cutoff>/seed<k>/
  encoder.pt        # 必須：SetEncoder の state_dict（torch.save）
  general_head.pt   # 必須：事前学習の汎用（仮予測）ヘッドの state_dict
  scaler.json       # 必須：{"mean":[...15...], "std":[...15...]}
  config.json       # 必須：下記メタ（下流はこれを読んで同一構成で再構築）
```

`config.json` の内容（例）：
```json
{
  "project": "nova",
  "cutoff": "25.0.0",            // = 26.0.0 サイクル開始の直前
  "seed": 0,
  "feature_names": [...15...],
  "feature_dim": 15,
  "d_model": 128, "n_layers": 2, "n_heads": 4, "ffn_dim": 256, "dropout": 0.1,
  "max_set_size": 512,
  "head_type": "linear", "head_hidden": 64,
  "pretrain_method": "supervised",
  "pretrain_epochs": 20, "pretrain_lr": 1e-3, "pretrain_batch_sets": 16,
  "review_horizon_days": 1,
  "measurement_step_days": 1, "lookback_days": 365
}
```

**load API（下流が呼ぶ想定）**：
```python
load_pretrained(project, cutoff, seed)
    -> (encoder: SetEncoder(eval,frozen), general_head: Head, scaler: Scaler, config: dict)
list_seeds(project, cutoff) -> [seed, ...]      # 保存済み seed の列挙
```
- 下流は `config` で `SetEncoder`／`Head` を同一構成に再構築 → `encoder.pt`／`general_head.pt` を load → `eval()`／凍結、`scaler.json` から `Scaler` を復元。

## 7. パラメータ（constants に集約）
- 本ディレクトリに `constants.py` を持つ（基点）。事前学習・モデルの設定（`PRETRAIN_*`, `D_MODEL` 等, `N_REPEATS`, `RANDOM_SEED`, `MEASUREMENT_STEP_DAYS`, `LOOKBACK_DAYS`, `REVIEW_HORIZON_DAYS`）を集約。
- **`PROJECTS`（プロジェクト別設定）**：`project -> {"cutoff": <版ラベル>, "versions": [<5版>]}` の対応表を持つ。cutoff は事前学習の締め、versions は下流のチューニング対象（本ディレクトリでは cutoff のみ使用）。cutoff の**日付は `major_releases_summary.csv` から実行時に引く**。
  - 例：`{"nova": {"cutoff": "25.0.0", "versions": ["26.0.0",...,"30.0.0"]}, "swift": {"cutoff": "2.29.0", "versions": ["2.30.0",...,"2.34.0"]}, ...}`（6プロジェクト）。
  - **同じ `PROJECTS` は `lookback_window/constants.py` にも別途保持**する（各ディレクトリが自己完結するため。重複は許容）。
- 旧 `FIRST_TARGET_VERSION` / `PRETRAIN_CUTOFF`（cutoff を版から自動導出する仕組み）は**廃止**し、`PROJECTS` の cutoff ラベルに一本化する。

## 8. ディレクトリ・コード構成（自己完結・基点／機能別フォルダ）
`concept_drift_detection` と同じく機能ごとにフォルダ分けする（一貫性・流用元を 1:1 コピーしやすい）。
```
pretrained_encoders/
  design.md
  __init__.py
  utils/        constants.py, review_utils.py
  features/     feature_builder.py, fast_index.py   # 15特徴（FEATURE_NAMES と算出）
  labeling/     label_builder.py                     # reviewed_within_delta（Δ=1日）
  dataset/      record_builder.py, set_builder.py    # 計測点・アクティブ集合・record/TSet 構築
  model/        set_transformer.py                   # SetEncoder / Head / Scaler / pretrain / embed_sets / train_probe
  io/           store.py                             # 保存/読込（save_pretrained / load_pretrained / list_seeds）
  build_encoders.py                                  # エントリ：データ構築 → 学習 → seed分（encoder＋汎用ヘッド）保存
```
- **`build_encoders.build_and_save(project="nova", n=None, seed_indices=None)`**：**project を引数で受け取り**、`PROJECTS` から cutoff を引いて「プロジェクト開始 〜 cutoff リリース日」で事前学習し、`<project>/cutoff_<cutoff>/seed*` に保存する。デフォルト project は **nova**。プロジェクトごとに seed 群を作成する。
- 下流（`lookback_window` 等）は **`pretrained_encoders` を基点に import**（例：`from ...pretrained_encoders.model.set_transformer import SetEncoder, Head, Scaler, train_probe, embed_sets`）。

## 9. スコープ外・拡張
- 事前学習方式は `supervised` のみ（将来 `ssl` 等に拡張可能な形）。
- 保存フォーマットは他分析でも使えるよう汎用に保つ（キーに project/cutoff/seed）。
- チューニング方式（linear_probe/fine_tune/peft）は**下流の関心**。本ディレクトリは「凍結エンコーダ＋汎用ヘッド」を提供するところまで。

## 10. 実装方針
- **本ディレクトリを基点（基盤）**：必要部品はここに実装し、他分析はここを import／override する。`concept_drift_detection` は参考にしてよいが依存しない（移植して本ディレクトリを正とする）。
- 初学者にも読みやすく：1 ファイル 1 役割、コメントで design.md の該当節を参照。
- 作成コードは「データ構築 → 事前学習 → エンコーダ＋汎用ヘッドを保存」に集中。
