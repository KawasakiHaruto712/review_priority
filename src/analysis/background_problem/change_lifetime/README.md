# Change 生存期間（lifetime）分析モジュール

Change が「投稿されてから終了（マージ/リジェクト）するまで」どれだけ Open でいたか＝
**生存期間（lifetime）の分布**を可視化・定量化する汎用の分析ツールです。

レビュー運用では生存期間が大きく右に裾を引く（多くは短命だが一部が長期滞留する）ことが多く、
その**形・中央値・裾の重さ**を把握できます。Change のライフサイクル特性を単体で見る用途のほか、
他の分析・検証の裏づけ（例: `priority_distribution` の「検査の逆説」の確認）にも使えます。

> 設計の詳細は [`design.md`](./design.md) を参照してください。

## 🎯 何を測るか

- **生存期間**: `lifetime = decision_time − created`
  - `decision_time` … MERGED / ABANDONED 確定時刻（`updated` を採用）
  - 未決（`NEW`）は生存期間を確定できないため**除外**
- **対象 Change（リリース別）**: `cycle_start − LOOKBACK_DAYS ≤ created ≤ cycle_end`
  （各 Change を 1 回だけカウント）
- **出力する図**:
  - **頻度ヒストグラム**（横軸＝生存期間、対数軸＋対数ビン、縦軸＝件数）
  - **箱ひげ図**（リリース横断：1 リリース＝1 箱で中央値・四分位・外れ値を比較）

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `main.py` | エントリポイント。全プロジェクト×リリース×status グループを実行 |
| `lifetime_calculator.py` | 母集団抽出・生存期間の計算・ヒストグラム/要約統計の算出 |
| `utils/constants.py` | **設定の入口**（対象・lookback・ビン・単位・status グループ 等） |
| `io/result_writer.py` | ヒストグラム結果の csv / json 出力 |
| `visualization/plotter.py` | 頻度ヒストグラム / 箱ひげ図の描画 |

## 🚀 実行方法

プロジェクトのルートディレクトリ（`src/` がある場所）で実行します。

```bash
python -m src.analysis.background_problem.change_lifetime.main
```

`utils/constants.py` の `TARGET_PROJECTS` × `STATUS_GROUPS` をすべて処理します。

## ⚙️ 主な設定（`utils/constants.py`）

| 設定 | 説明 |
|---|---|
| `TARGET_PROJECTS` | 分析するプロジェクトとリリース version の一覧 |
| `LOOKBACK_DAYS` | 母集団に含める「created の遡り上限」（既定 365 日） |
| `STATUS_GROUPS` | 出力する status グループ（既定: `all` / `merged` / `abandoned`） |
| `DURATION_UNIT` | 生存期間の単位（`days` / `hours`、既定 `days`） |
| `HIST_BINS` / `HIST_LOG_BINS` | ヒストグラムのビン数 / 対数ビン（log-binning）の ON/OFF |
| `X_LOG_SCALE` | 横軸（生存期間）を対数表示にするか |
| `SUMMARY_PERCENTILES` | 要約統計に出す分位点（既定 `[10,25,50,75,90]`） |
| `MAKE_BOXPLOT` / `BOXPLOT_LOG_SCALE` | 箱ひげ図を出すか / その縦軸を対数にするか |

## 📤 出力

`data/analysis/background_problem/change_lifetime/` 配下に保存されます。

```
<project>/
├── <version>/<group>/
│   ├── histogram.csv        # bin_left, bin_right, count
│   ├── histogram.json       # meta + summary（中央値・分位点 等）+ histogram
│   └── histogram.png        # 頻度ヒストグラム
├── all_releases/<group>/    # 全リリース結合（change_number で重複排除）
│   └── histogram.{csv,json,png}
├── all_boxplot.png          # リリース横断の箱ひげ図（status=all）
├── merged_boxplot.png       # 同（status=merged）
└── abandoned_boxplot.png    # 同（status=abandoned）
```

- `<group>` は `all` / `merged` / `abandoned`。
- `histogram.json` の `summary` に **n / mean / median / 分位点 / min / max** を格納します
  （中央値は短く、平均は裾に引っ張られて長い、といった対比が確認できます）。

## 🧪 テスト

```bash
python -m pytest tests/analysis/background_problem/change_lifetime/ -q
```

## 📝 注意（解釈・記述時）

- **頻度＋対数ビン**: 対数ビンは右ほど幅が広く、頻度（件数）は「幅が広い＝多く見える」傾向があります。
  分布の“形”を厳密に比較したい場合は密度（件数÷ビン幅）が正確ですが、本モジュールは要件どおり
  **頻度**を既定にしています（裾の形は対数軸で十分読めます）。
- **未決（NEW）の除外**: 生存期間を確定できないため母集団から除外しています（件数は json の meta に記録）。
- **対数ビン/対数軸と 0 以下の値**: 対数では 0 以下を扱えないため、`updated < created` 等で
  `lifetime ≤ 0` となる Change は事前に除外しています（生存期間は本来正）。対数ビンの下限は
  `HIST_MIN_VALUE`（既定は正の最小値）で、これにより落ちる正の値は通常ありません。
- **全リリース混合の重複排除**: lookback により同一 Change が複数リリースに入りうるため、
  混合では `change_number` で重複排除して 1 回だけ数えます。
