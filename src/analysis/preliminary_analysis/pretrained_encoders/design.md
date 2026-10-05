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
- 計測点 T ＝ 毎日 0 時（`MEASUREMENT_STEP_DAYS=1`）。各 T で **アクティブな Change 群**（`T-LOOKBACK <= created <= T` かつ **T に Open**）を 1 つの「集合」とする。Open かどうかは §11.1 の Open の期間で判定する（旧版は `T < decision_time`、decision_time＝最後の更新 `updated` だった）。
- 特徴は **15 種類**（`FEATURE_NAMES` と算出ロジックを本ディレクトリに持つ）。**どの特徴も T の時点で分かる情報だけで作る**（§11）。
- 目的変数（事前学習の教師）：`reviewed_within_delta`、Δ=1 日（label ロジックも本ディレクトリに持つ）。
- **Scaler は本事前学習データで fit** し、統計量（mean/std）を**保存**する。下流はこの統計で transform するだけ（下流で再 fit しない）。

## 3. モデル：集合 Transformer（本ディレクトリに実装）
- `SetEncoder`（自己注意・**位置エンコなし**＝置換不変）を本ディレクトリに実装。
- 設定：`D_MODEL=128 / N_LAYERS=2 / N_HEADS=4 / FFN_DIM=256 / DROPOUT=0.1 / MAX_SET_SIZE=None`。
  - `MAX_SET_SIZE` は旧版 512（集合の先頭 512 件で打ち切り。無作為抽出ではない）。**2026-10 に上限なし（None）へ変更**。
    旧版の OpenStack では 1 日の最大が 480 件（nova/27.0.0）で一度も効いておらず、上限をなくしても結果は変わらない。
    全件を見るべきという方針に合わせる。1 日に数千件になるプロジェクトを足すときは改めて扱いを決める。
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
  "max_set_size": null,           // 上限なし（§3）
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

## 11. その時点で分かる情報だけで作る規則（2026-10 改訂）

### 背景
2026-10 にコードを点検したところ、計測点 T より後に起きることを使っている箇所が 4 つ見つかった
（§11.1〜§11.4。影響の大きさは nova の収集データ 44,359 件で実測）。分析を全部やり直すのに合わせて直す。
集合・特徴は学習に使う日にも評価日にも同じように作るので、どの日でも同じ規則が効く。

各特徴の「T の時点」は、その (Change, T) の T（その日の 0 時）である。評価日に限らない。

### 11.1 Open の期間（集合への出入り・open_ticket_count・merge_rate）

**旧版の問題**：マージ・放棄の時刻として `updated`（その Change に最後に何かが起きた時刻）を使っていた。
マージ後にも CI の結果や Zuul の通知、人間の "Cherry Picked"（安定版への取り込み）などが付くので、
`updated` は本当に閉じた時刻より後になる。「この後に記録が付くか」という未来の情報で、集合に入るかどうかが決まっていた。

```
nova  マージ 31,463 件の 84.6%、放棄 11,719 件の 9.9% で updated が本当の決着より後
      ずれ：中央値 0 日 / 上位 10% 2.6 日 / 上位 1% 118 日
      本来は閉じているのに集合に入っている (Change, 日)：101,308 件（閉じた Change の全 (Change, 日) の 4.6%）
      そのうち正例（マージ後の人間の記録による）：1,291 件（1.3%）
```

また、放棄されたあと復活する Change がある（nova 1,880 件。最後に放棄 891 件・最後にマージ 973 件）。
旧版は作成から最後の決着までずっと Open とみなしており、放棄から復活までの閉じている間も集合に入っていた。

**新しい規則**：Change の記録を時刻順にたどり、Open の期間の列を作る。

| 出来事 | 取り方 | Open の状態 |
|---|---|---|
| 作成 | `created` | 開く |
| 放棄 | 記録の 1 行目が `Abandoned` または `Patch Set N: Abandoned` のものの `date` | 閉じる |
| 復活 | 記録の 1 行目が `Restored` または `Patch Set N: Restored` のものの `date` | 開く |
| マージ | `submitted` | 閉じる |

- T が Open の期間 `[開いた時刻, 閉じた時刻)` のどれかに入るとき、その Change は T に Open とする。
  状態が NEW のまま終わる Change は、最後の期間を閉じない。
- nova では、放棄 11,719 件の全件で放棄の記録が見つかり、マージ 31,463 件のうち 31,461 件に `submitted` がある。
  取れない場合（`submitted` がない、状態が ABANDONED なのに放棄の記録がない等）は `updated` で代用し、
  件数をログに出す。記録の書き方の揺れ（tag `autogenerated:gerrit:abandon` 等）は実装時に全件で確かめる。
- `LOOKBACK_DAYS`（`T - created <= 365 日`）は変えない。
- 同じ Open の期間を次の 3 か所で使う。
  - **集合**：T に Open な Change だけを入れる（`record_builder`）。
  - **`open_ticket_count`**：T に Open な Change の数。旧版は「作成数 − 決着数」で、復活を考えていなかった。
  - **`merge_rate` / `recent_merge_rate`**：マージの時刻を `submitted` にする（旧版は `updated` で、マージを遅れて数えていた）。

### 11.2 reviewed_lines_in_period

**旧版の問題**：「`updated` が過去 14 日以内にある Change」の「最終版の行数」を足していた。
- `updated` は収集時点から見た最後の更新なので、T の後にまた更新される Change は、14 日以内に活動していても数えられない。
  逆に数えられるのは「T の後はもう何も起きない」Change で、それは T の時点では分からない。
- 行数が最終版のもので、T の時点ではまだ存在しない版を使っていた。

```
nova の評価期間（2022-03-30〜2024-10-02、918 日）     旧版        新しい規則
  数える Change（1 日あたりの中央値）                   29 件        114 件
  行数（同）                                        2,650 行     19,278 行
  旧版の値 ÷ 新しい規則の値   中央値 0.16 / 日ごとの値の相関 0.35
```
（新しい規則の欄は「活動」に CI などの記録も含めて数えた値。採用した規則は下のとおり人間のレビューだけなので、件数はこれより少なくなる。）

**新しい規則**：
- 数える Change：`[T − 14 日, T]`（両端を含む。旧版と同じ）に**人間のレビュー**が 1 件以上ある Change。
  人間のレビューの判定は正解ラベル（§2）と同じ `review_utils.is_human_comment`（ボット・自動投稿・作成者本人を除く）。
- 行数：その Change の、T の時点で最新の版（`created <= T` のうち最新）の `lines_inserted + lines_deleted`。
- 意味：レビュアが手を付けた Change の量。CI が動いた量は含めない。

**人間のレビューの判定は変えない**：いまの判定には、レビューではない操作の記録も約 7% 入っている
（nova：本人以外によるパッチセットの追加 3.9%、編集の公開など 2.2%、放棄 0.6%、トピックの変更・取り込み・復活・取り消し 0.2%）。
何を除くかの線引きが難しいため、正解ラベルと合わせて今のままとする（旧版と同じ定義）。

### 11.3 bug_fix_confidence / refactoring_confidence

**旧版の問題**：Change の `subject` と、最新の版の説明文（`commit.message`）から判定していた。説明文は版を重ねると書き換わる。

```
nova で版が 2 つ以上ある Change 29,144 件
  全部の版に説明文が残っている        100%（収集で ALL_COMMITS を指定しているため）
  最初の版と最新の版で説明文が違う     55.8% ／ 件名が違う 23.4%
  バグ修正の判定が変わる              7.7% ／ リファクタリングの判定が変わる 6.2%
```

**新しい規則**：T の時点で最新の版（`created <= T` のうち最新）の `commit.subject` と `commit.message` から判定する。
版の選び方は、行数・ファイル数・テストの有無で使っている `_get_files_at_analysis_time` と同じにする。

### 11.4 days_to_major_release

**旧版の問題**：リリースの表（`major_releases_summary.csv`）が版番号の形（`X.0.0`、swift は `2.Y.0`）で区切りを拾っていた。
- nova・neutron・cinder・glance・keystone は、2015-10-15 より前の版（`2011.2`・`2015.1.0` などの形）が表に入っておらず、
  その期間の値が「2015-10-15 までの日数」になっていた（例：2012 年 1 月の日で約 1,370 日。本来は数か月）。
- swift は 2014-07-07 より前（`1.x`）が入っていなかった。また 2020 年以前は 1 サイクルに `2.Y.0` が 2〜3 個あり、時期によって区切りの意味が変わっていた。
- 2025-10-01 より後は次のリリースが見つからず -1 になっていた（評価する期間は 2024-10 までなので、いまは使われない）。
- 誤った大きな値が事前学習のデータに入るので、標準化の標準偏差が膨らみ、元の値が正しい評価の期間でも標準化した値が狭い範囲に縮んでいた。
  また 2011〜2015 年のこの特徴は「何年ごろか」を表す値になっていた。

**新しい規則**：
- リリースの表をサイクル単位で作り直す（作り方は [src/collectors/README.md](../../../collectors/README.md) の「メジャーリリースの表」）。
  6 件とも OpenStack の公式のサイクルのリリース日を使う。swift も公式の日付に揃える。
- その project の行の `release_date`（`status` が released・planned の両方）を日付順に並べ、**T より後の最初の日付**までの日数を返す。
  版番号は見ない。見つからなければ -1（従来どおり）。
- T がリリース日の 0 時ちょうどのときは、そのリリースは済んだものとみなす（旧版と同じ）。
- まだ出ていないサイクルは予定日を使う。予定日は T の時点ですでに公開されているので、未来の情報にはあたらない。
- `RELEASE_LEVEL`（swift だけ `"minor"`）と、版番号から区切りを判定する処理（`_is_boundary_release` / `_release_ordinal`）は不要になる。

**分析期間の区切りへの影響**：事前学習の締め（`PROJECTS` の cutoff）と評価する 5 版の期間も同じ表から引く。
`PROJECTS` の版の書き方（`25.0.0`・`2.29.0` など）と出力先の名前は変えない。版番号で表の行を引き、日付は公式の日付を使う。
- nova 型の 5 件：公式の日付と版の日付が一致しているので、期間は変わらない。
- swift：区切りが swift の版の日付から公式の日付に変わるので、各期間が 2〜3 週間後ろにずれる（例：2.29.0 の締めは 2022-02-14 → 2022-03-30）。

### 11.5 変えないもの
- 正解ラベルの定義（T から 1 日以内に人間のレビュー。観測できない期間に掛かるものは除外）と、人間のレビューの判定（§11.2）。
- 行数・ファイル数・テストの有無・版数・経過時間（すでに T の時点の版から計算している）。
- 開発者の `past_report_count` / `recent_report_count`（T までの作成数）。
- 標準化の統計（事前学習のデータだけで計算）。
- 学習の損失（BCE。下流の probe も同じ）。評価する版（各 project 5 版）。

### 11.6 やり直しの順序
リリースの表を作り直す → 事前学習エンコーダ（標準化の統計を含む）→ `lookback_window` → `concept_drift_detection` → `concept_drift_cause`（最新版）。
下流の 3 分析は集合と特徴をすべて本ディレクトリから import しているので、本ディレクトリの修正だけで揃う。
旧版の結果は `data/archive/2026-10-02_bounded-2024/` と `data/analysis/preliminary_analysis/*_old/` に退避済み。旧版の結果には §11.1〜§11.4 の問題がすべて含まれる。
