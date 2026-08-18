# pretrained_encoders

集合 Transformer の**事前学習エンコーダ＋汎用ヘッドを一度だけ作って保存**し、下流の分析（`lookback_window` 等）が **load して使い回す**ための**共有基盤**。

- 学習するのは **エンコーダ（表現）と汎用ヘッド**のみ。下流は凍結エンコーダの上で線形ヘッドを貼り直す（＝チューニング）。
- 本ディレクトリは **concept_drift_detection に依存しない自己完結の基点**。生データ読み込み・`src.features`・`src.config` のみ共有インフラを使う。
- 詳細仕様は [design.md](./design.md)。用語は **Change**（Gerrit）で統一。

## 使い方

```bash
# N_REPEATS 個（既定）のエンコーダ＋汎用ヘッドを作成・保存
python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders

# 個数を指定して作成
python -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders --n 3
```

### 事前学習データ（共有・単一モデル）
- **プロジェクト開始 〜 25.0.0 リリース日（＝26.0.0 のサイクル開始）まで**。
- 26.0.0〜30.0.0 の**どのリリースの下流分析でも同一エンコーダ**を使う（per-release では作らない）。分析対象期間（26.0.0 以降）は含めない＝リーク防止。

## 出力（保存フォーマット＝下流との約束事）

```
data/analysis/preliminary_analysis/pretrained_encoders/<project>/cutoff_<cutoff>/seed<k>/
  encoder.pt        # SetEncoder の state_dict
  general_head.pt   # 汎用（仮予測）ヘッドの state_dict
  scaler.json       # {"mean":[...15...], "std":[...15...]}
  config.json       # 再構築用メタ（d_model 等・特徴名・pretrain 設定）
```

### 下流からの読み込み

```python
from src.analysis.preliminary_analysis.pretrained_encoders.io import store

seeds = store.list_seeds("nova", "25.0.0")                 # 保存済み seed 一覧
encoder, general_head, scaler, config = store.load_pretrained("nova", "25.0.0", seeds[0])
# encoder / general_head は eval・凍結済み
```

## ディレクトリ構成

```
pretrained_encoders/
  build_encoders.py   # エントリ：データ構築 → 事前学習 → seed分保存
  utils/              constants.py（設定の集約）, review_utils.py（bot判定・人間レビュー抽出）
  features/           feature_builder.py（15特徴）, fast_index.py（高速集計）
  labeling/           label_builder.py（reviewed_within_delta, Δ=1日）
  dataset/            record_builder.py（計測点×アクティブ集合）, set_builder.py（TSet）
  model/              set_transformer.py（SetEncoder/Head/Scaler/pretrain/embed_sets/train_probe）
  io/                 store.py（save_pretrained/load_pretrained/list_seeds）
```

## 主な設定（`utils/constants.py`）
- `TARGET_PROJECT="nova"` / `FIRST_TARGET_VERSION="26.0.0"`（この直前が cutoff）
- モデル：`D_MODEL=128 / N_LAYERS=2 / N_HEADS=4 / FFN_DIM=256 / DROPOUT=0.1`
- 事前学習：`PRETRAIN_EPOCHS=20 / PRETRAIN_LR=1e-3 / PRETRAIN_BATCH_SETS=16`
- 反復：`N_REPEATS=10`（`--n` で個数指定可）、`RANDOM_SEED=42`
- 目的変数：`reviewed_within_delta`（Δ=`REVIEW_HORIZON_DAYS=1` 日）

## 備考
- `build_records`（全履歴の特徴生成）が最も重く数十分かかることがあるが、**一度保存すれば再利用**するため以降は不要。
