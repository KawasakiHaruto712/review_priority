# Change 生存期間（lifetime）分析 設計書 (change_lifetime)

## 0. このドキュメントについて

本ドキュメントは、**Change の生存期間（lifetime ＝ Open でいた期間）の分布を可視化・定量化する
汎用の分析ツール** `change_lifetime` の設計書です。

各リリースの開発サイクル中に Open だった Change について、
生存期間（`lifetime = decision_time − created`）の分布をヒストグラムで描き、
中央値・分位点などの要約統計を出力する。
これにより「短命な Change が大多数だが、長命（放置）な Change の裾が長い」といった
**分布の形そのもの**を把握できる。

この分析は、Change のライフサイクル特性を単体で調べる用途のほか、
他の分析・検証の裏づけにも利用できる（用途の一例は §1.2）。

> 配置: `src/analysis/background_problem/change_lifetime/`（`priority_distribution` と並列）。

---

## 1. 背景と目的

### 1.1 目的
Change が「投稿されてから終了（マージ/リジェクト）するまで」どれだけ Open でいたか
＝**生存期間（lifetime）の分布**を把握する。
レビュー運用では生存期間は大きく右に裾を引く（多くは短命だが一部が長期滞留する）ことが多く、
その**形・中央値・裾の重さ**を定量化することを目的とする。

### 1.2 主な用途（例）
- **Change のライフサイクル特性の記述**: 典型的にどれくらいで終わるか、放置の裾はどの程度か。
- **他分析の裏づけ**: 生存期間の分布は様々な考察の根拠に使える。例えば
  `priority_distribution` の「スナップショット集合が長命 Change に偏る（検査の逆説）」という
  考察は、生存期間に**長い裾**があることと整合する、といった確認に利用できる。

### 1.3 この分析が出すもの
- 生存期間（lifetime）の **ヒストグラム**（右裾の長い分布）。
- 中央値・分位点などの **要約統計**（中央値は短い／平均は裾に引っ張られて長い、を定量化）。

---

## 2. 用語と定義

### 2.1 生存期間（lifetime）
Change が Open だった期間 = `decision_time - created`。
- `created` … 投稿時刻
- `decision_time` … マージ/リジェクト確定時刻。`priority_distribution` と同じく、
  `status` が `MERGED` / `ABANDONED` の Change で `updated` を採用（§決定事項）。
- `status` が `NEW`（未決）の Change は **lifetime を確定できないため除外**（§5 で除外件数を記録）。
  ※将来、観測打ち切り（censoring）として扱う拡張は可能（本設計では対象外）。

### 2.2 対象 Change（population）
あるリリース `R`（開発サイクル `[cycle_start, cycle_end]`、lookback `L` 日）について、
次を満たす Change を母集団とする:

```
1. status が対象（既定: MERGED または ABANDONED）。NEW 等の未決は対象外。
2. cycle_start - L <= created <= cycle_end
```

- **各 Change は 1 回だけ数える**。priority_distribution と違い、分析する値（生存期間）は
  計測時点に左右されないため、スナップショットによる重複カウントは行わない。
  よって母集団は **created の範囲だけ**で素直に定義できる（open 区間の重なり判定は不要）。
- `lookback L` は、サイクル開始よりかなり前に投稿された Change も取りこぼさないための
  上限（既定 1 年）。値の意味は「`cycle_start` の最大 L 日前までに投稿された Change まで含める」。
- lifetime は **全期間**（created→decision）で測る（サイクル外にはみ出す部分も含む）。
- **全リリース混合**では、lookback により同一 Change が複数リリースの母集団に入りうるため、
  `change_number` で **重複排除**して 1 回だけ数える（§7）。

### 2.3 status グループ
既定で次の 3 グループを別々に出力する（`constants.STATUS_GROUPS` で変更可）:
- `all` … MERGED + ABANDONED
- `merged` … MERGED のみ
- `abandoned` … ABANDONED のみ

---

## 3. 入力データ
- Change: `data/openstack_collected/<project>/changes/*.json`
- リリース日: `data/openstack_collected/major_releases_summary.csv`
- 必要な共通処理は、Change/リリース日の読み込み（`load_changes` / `load_release_dates` /
  `get_release_cycle`）と日時変換（`parse_dt` / `to_unit`）のみ。**ボット判定は不要**
  （`lifetime = decision_time − created` だけで、レビューコメントを見ないため）。
- これらの共通処理の置き場所は **§11 で決定**（推奨: `src/analysis/background_problem/common/` への
  切り出し）。本書では `load_changes(...)` 等の関数名で参照する。

---

## 4. 設定ファイル `utils/constants.py`

```python
"""change_lifetime 分析の設定値"""
from src.config.path import DEFAULT_DATA_DIR

# ── 出力 ───────────────────────────────────────────────
OUTPUT_ROOT = DEFAULT_DATA_DIR / "analysis" / "background_problem" / "change_lifetime"

# ── 分析対象プロジェクトとリリース ──────────────────────────
TARGET_PROJECTS = {
    "nova": [
        "2015.1.0",
        "12.0.0", "13.0.0", "14.0.0", "15.0.0",
        "16.0.0", "17.0.0", "18.0.0", "19.0.0", "20.0.0",
    ],
}

# ── 母集団 ─────────────────────────────────────────────
LOOKBACK_DAYS = 365           # priority_distribution と揃える
# status グループ（名前 -> 対象 status 集合）
STATUS_GROUPS = {
    "all": {"MERGED", "ABANDONED"},
    "merged": {"MERGED"},
    "abandoned": {"ABANDONED"},
}

# ── 生存期間の単位 ──────────────────────────────────────
DURATION_UNIT = "days"        # "hours" | "days"（lifetime は日単位が見やすい）

# ── ヒストグラム ────────────────────────────────────────
HIST_BINS = 50                # ビン本数
HIST_LOG_BINS = True          # True: 対数等間隔ビン（log-binning）, False: 線形等間隔
X_LOG_SCALE = True            # 横軸を対数表示にするか
# 対数ビン/対数軸で 0 以下を扱えないため、下限でクリップ（単位は DURATION_UNIT）
HIST_MIN_VALUE = None         # None なら「正の最小値」を自動採用

# ── 要約統計に出す分位点 ───────────────────────────────────
SUMMARY_PERCENTILES = [10, 25, 50, 75, 90]

# ── 箱ひげ図（リリース横断の比較図） ──────────────────────────
MAKE_BOXPLOT = True           # リリースごとの生存期間を1枚に並べた箱ひげ図を出すか
BOXPLOT_LOG_SCALE = True      # 箱ひげ図の縦軸（生存期間）を対数にするか

# ── プロット ───────────────────────────────────────────
PLOT_DPI = 150
```

---

## 5. ディレクトリ構成とモジュール責務

```
src/analysis/background_problem/change_lifetime/
├── design.md                 # 本書
├── README.md                 # 使い方
├── __init__.py
├── main.py                   # オーケストレーション（エントリポイント）
├── utils/
│   ├── __init__.py
│   └── constants.py          # §4 設定
├── metrics/
│   ├── __init__.py
│   └── lifetime_calculator.py    # 母集団抽出・lifetime 計算・ヒストグラム/要約の算出
├── io/
│   ├── __init__.py
│   └── result_writer.py      # csv / json 出力
└── visualization/
    ├── __init__.py
    └── plotter.py            # 頻度ヒストグラム / 箱ひげ図の描画
```

### 5.1 `metrics/lifetime_calculator.py`
| 関数 | 役割 |
|---|---|
| `extract_lifetimes(changes, cycle_start, cycle_end, statuses, lookback_days, unit) -> (values, change_numbers, n_excluded)` | §2.2 の母集団を抽出し、各 Change の lifetime（指定単位の float リスト）と change_number、未決/対象外の除外件数を返す |
| `make_histogram(values, bins, log_bins, min_value) -> (counts, edges)` | 頻度ヒストグラムを計算（log_bins なら `np.logspace` でビン境界を対数等間隔に） |
| `summarize(values, percentiles) -> dict` | n / mean / median / 各分位点（min,max 含む）を返す |

- `make_histogram`: `log_bins=True` のとき `edges = np.logspace(log10(lo), log10(max), bins+1)`。
  `lo` は `min_value`（None なら values の正の最小値）。0 以下の値は対数で扱えないため除外/クリップ。

### 5.2 `io/result_writer.py`
| 関数 | 役割 |
|---|---|
| `write_histogram_csv(counts, edges, path)` | 1 行 = 1 ビン。列: `bin_left, bin_right, count` |
| `write_summary_json(summary, meta, path)` | `meta`（project/version/status_group/単位/ビン設定/除外件数 等）＋ `summary`（要約統計）＋ `histogram`（bins と counts） |

### 5.3 `visualization/plotter.py`
- `plot_histogram(counts, edges, out_path, title, xlabel, ylabel, x_log, dpi)`
  - `ax.bar`（または `ax.stairs`）でビンの頻度（件数）を描画。
  - `x_log=True` なら `ax.set_xscale("log")`。対数ビンと組み合わせると棒幅が均一に見える。
  - 縦軸 = **頻度（件数）**。横軸 = 生存期間 [DURATION_UNIT]。
  - 中央値の縦線（参考）をオプションで重ねる。
- `plot_boxplot(values_by_release, out_path, title, ylabel, y_log, dpi)`
  - **リリースを横断した比較用の箱ひげ図**。`ax.boxplot([...])` で **1 リリース＝1 箱**を並べる。
  - 横軸 = リリース（version）、縦軸 = 生存期間 [DURATION_UNIT]（`y_log=True` で対数）。
  - 箱＝四分位（Q1/中央値/Q3）、ひげ、外れ値が一目で分かるため、
    「中央値は短いが裾（外れ値）が長い」構造の比較に適する。

### 5.4 `main.py`
擬似フロー:
```
for project in TARGET_PROJECTS:
    changes = load_changes(project)
    rel_df  = load_release_dates()
    for group, statuses in STATUS_GROUPS.items():
        values_by_release = {}          # version -> lifetime 配列（箱ひげ図・混合用）
        seen = set()                    # 混合の重複排除（change_number）
        mixed_values = []
        for version in TARGET_PROJECTS[project]:
            cs, ce = get_release_cycle(rel_df, project, version)
            values, cnums, n_excl = extract_lifetimes(changes, cs, ce, statuses,
                                                       LOOKBACK_DAYS, DURATION_UNIT)
            counts, edges = make_histogram(values, HIST_BINS, HIST_LOG_BINS, HIST_MIN_VALUE)
            summary = summarize(values, SUMMARY_PERCENTILES)
            write_histogram_csv / write_summary_json / plot_histogram
                -> <project>/<version>/<group>/
            values_by_release[version] = values
            for v, cn in zip(values, cnums):     # 混合は change_number で重複排除
                if cn not in seen: seen.add(cn); mixed_values.append(v)
        # 全リリース混合（重複排除済み）
        histogram / summary / plot_histogram -> <project>/all_releases/<group>/
        # リリース横断の箱ひげ図（1リリース=1箱）
        if MAKE_BOXPLOT: plot_boxplot(values_by_release) -> <project>/<group>_boxplot.png
```

---

## 6. 出力構成

```
data/analysis/background_problem/change_lifetime/
└── <project>/
    ├── <version>/                  # 例: 20.0.0
    │   ├── all/
    │   │   ├── histogram.csv       # bin_left, bin_right, count
    │   │   ├── histogram.json      # meta + summary + histogram
    │   │   └── histogram.png       # 頻度ヒストグラム
    │   ├── merged/
    │   │   └── (同上)
    │   └── abandoned/
    │       └── (同上)
    ├── all_releases/               # 全リリース結合（change_number で重複排除）
    │   └── <group>/                # group = all / merged / abandoned
    │       └── histogram.{csv,json,png}
    ├── all_boxplot.png             # リリース横断の箱ひげ図（status=all）
    ├── merged_boxplot.png          # 同（status=merged）
    └── abandoned_boxplot.png       # 同（status=abandoned）
```

### 6.1 CSV スキーマ（`histogram.csv`）
| 列 | 説明 |
|---|---|
| `bin_left` | ビン下端（DURATION_UNIT） |
| `bin_right` | ビン上端（DURATION_UNIT） |
| `count` | そのビンに入った Change 数（頻度） |

### 6.2 JSON スキーマ（`histogram.json`）
```json
{
  "meta": {
    "project": "nova",
    "version": "20.0.0",
    "status_group": "all",
    "statuses": ["MERGED", "ABANDONED"],
    "cycle_start": "2019-04-10",
    "cycle_end": "2019-10-16",
    "lookback_days": 365,
    "duration_unit": "days",
    "hist_bins": 50,
    "hist_log_bins": true,
    "x_log_scale": true,
    "n_changes": 1800,
    "n_excluded_unfinished": 120
  },
  "summary": {
    "n": 1800, "mean": 44.0, "median": 13.2,
    "min": 0.01, "max": 360.0,
    "percentiles": {"10": 1.5, "25": 4.0, "50": 13.2, "75": 45.0, "90": 120.0}
  },
  "histogram": {
    "bin_left":  [0.01, 0.02, ...],
    "bin_right": [0.02, 0.03, ...],
    "count":     [3, 7, ...]
  }
}
```

---

## 7. エッジケース・注意点
- **未決（NEW）の除外**: lifetime を確定できないため母集団から除外（件数は meta に記録）。
- **lifetime <= 0**: `updated < created` 等の異常は除外。
- **対数ビン/対数軸と 0 以下**: 値 0 以下は対数で扱えないため、`HIST_MIN_VALUE`（既定は正の最小値）で
  クリップ／除外する。下限を下回る値の扱いは meta に記録すると親切。
- **対数ビンの頻度の見え方**: 対数ビンは右ほどビン幅が広いので、頻度（件数）の棒高さは
  「ビン幅が広い＝多く見える」傾向がある。分布の“形”を厳密に比較したい場合は密度（件数÷ビン幅）が
  正確だが、本設計は要望どおり **頻度** を既定とする（裾の形は対数軸で十分読める）。
- **全リリース混合の重複**: lookback により同一 Change が複数リリースの母集団に入りうるため、
  混合は `change_number` で **重複排除**して 1 回だけ数える（§2.2）。
- **リリース日の異常エントリ**: `get_release_cycle`（§11 で決める共通モジュール）側で対処済み。

---

## 8. テスト計画（`tests/analysis/background_problem/change_lifetime/`）
- `test_lifetime_calculator.py` … 母集団抽出（created 範囲・lookback・status・NEW 除外）、lifetime 値、
  除外件数、log/linear ヒストグラム、要約統計（合成データで決定的に検証）。
- `test_result_writer.py` … csv/json スキーマ。
- `test_main.py` … tmp_path での結合テスト（`<version>/<group>/histogram.{csv,json,png}` と
  `<group>_boxplot.png` の生成）。

---

## 9. 実行方法（想定）
```bash
python -m src.analysis.background_problem.change_lifetime.main
```

---

## 10. 確定事項
1. **縦軸**: 頻度（件数）。
2. **横軸**: 生存期間（lifetime = decision_time − created）。対数軸（既定）＋対数ビン（log-binning）。
3. **decision_time**: `updated` を採用（MERGED/ABANDONED のみ）。
4. **未決（NEW）**: lifetime 確定不可のため除外。
5. **status**: `all` / `merged` / `abandoned` の 3 グループを別々に出力（`STATUS_GROUPS` で変更可）。
6. **単位**: days（既定。`DURATION_UNIT` で変更可）。
7. **対象リリース**: nova の `2015.1.0`, `12.0.0`〜`20.0.0`。
8. **母集団**: `cycle_start − lookback ≤ created ≤ cycle_end`（各 Change を 1 回だけカウント。
   open 区間の重なり判定は使わない）。
9. **箱ひげ図**: リリース横断（1 リリース＝1 箱）の比較図を status グループごとに出力。

---

## 11. 共通コードの方針

Change/リリース日の読み込み・日時変換は、`background_problem` 配下の分析で共通利用する
**共通モジュール `src/analysis/background_problem/common/`** に集約する。

- `common/data_loader.py` … `load_changes` / `load_release_dates` / `get_release_cycle`
  （＋リリース日異常エントリの除外。データの既定パスも保持）
- `common/time_utils.py` … `parse_dt` / `to_unit` / `relative_x`
- `change_lifetime` と `priority_distribution` の **両方が `common` を import** する。
  ボット判定は priority_distribution 固有のため共通化しない。
