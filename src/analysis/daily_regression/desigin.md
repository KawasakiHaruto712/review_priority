# Daily Regression Analysis（日次重回帰分析）設計書

## 1. 目的

OpenStack Nova プロジェクトにおいて、**日付ごとに重回帰分析（OLS）** を実施し、16 種類のメトリクス（説明変数）が「次のレビューまでの待ち時間（秒）」（目的変数）に与える影響の **回帰係数の時系列的遷移** を明らかにする。

---

## 2. 用語定義

| 用語 | 定義 |
|------|------|
| **分析対象日 (analysis_date)** | 重回帰分析を実行する1日（0:00:00 起点） |
| **対象Change** | 分析対象日において Open 状態の Change（後述の条件を満たすもの） |
| **目的変数 (y)** | 0時起点から「次の非botレビュー」までの秒数。当日 Open された Change は Open 時刻→最初の非botレビュー時刻。|
| **説明変数 (X)** | `src/features` で抽出する 16 種類のメトリクス |
| **打ち切り最大値** | 31,536,000 秒（= 1年 = 365日 × 86,400秒） |

---

## 3. 対象データ

### 3.1 データソース

| データ | パス | 形式 |
|--------|------|------|
| Change データ | `data/openstack_collected/nova/changes/*.json` | JSON（1ファイル=1Change） |
| メジャーリリース情報 | `data/openstack/major_releases_summary.csv` | CSV（project, version, release_date） |
| コアレビューア情報 | `data/openstack_collected/core_developers.json` | JSON |
| Bot 設定 | `src/config/gerrymanderconfig.ini` | INI（`[organization].bots`） |
| レビューキーワード | `data/processed/review_keywords.json` | JSON |
| レビューラベル | `data/processed/review_label.json` | JSON |

### 3.2 対象バージョン（10バージョン）

| # | バージョン | コードネーム | リリース日 |
|---|-----------|-------------|-----------|
| 1 | 2015.1.0 | Kilo | 2015-04-30 |
| 2 | 12.0.0 | Liberty | 2015-10-15 |
| 3 | 13.0.0 | Mitaka | 2016-04-07 |
| 4 | 14.0.0 | Newton | 2016-10-06 |
| 5 | 15.0.0 | Ocata | 2017-02-22 |
| 6 | 16.0.0 | Pike | 2017-08-30 |
| 7 | 17.0.0 | Queens | 2018-02-28 |
| 8 | 18.0.0 | Rocky | 2018-08-30 |
| 9 | 19.0.0 | Stein | 2019-04-10 |
| 10 | 20.0.0 | Train | 2019-10-16 |

### 3.3 分析期間

各バージョンについて、**当該リリース日から次のリリース日の前日まで** を日単位で分析する。

- 例: 2015.1.0（2015-04-30）→ 12.0.0（2015-10-15）の場合、2015-04-30 ～ 2015-10-14 の各日を分析
- 最終バージョン（20.0.0）は次のリリース（21.0.0）のリリース日の前日まで

### 3.4 タイムゾーン

データのタイムスタンプをそのまま使用する（タイムゾーン変換なし）。  
タイムスタンプ形式: `YYYY-MM-DD HH:MM:SS.nnnnnnnnn`

---

## 4. サンプル選定ルール

### 4.1 対象Changeの条件

分析対象日 `D`（0:00:00）に対し、以下を **すべて** 満たす Change を対象とする:

1. **Open 状態である**: `created <= D 23:59:59` かつ （`updated > D 00:00:00` または status == "NEW"）
2. **未来日に作成されていない**: `created <= D 23:59:59`
3. **当日以前に Open されている**: 上記と同義

具体的には以下の2ケースに分かれる:

| ケース | 条件 | 分析時点 (analysis_time) |
|--------|------|------------------------|
| 既にOpen | `created < D 00:00:00` | `D 00:00:00`（0時起点） |
| 当日Open | `D 00:00:00 <= created <= D 23:59:59` | `created`（Open 時刻） |

### 4.2 目的変数の算出

1. 対象 Change の `messages` フィールドから、**analysis_time 以降** の最初の非bot メッセージの `date` を取得
2. `y = (最初の非botレビュー時刻) - analysis_time` （秒単位）
3. 該当するレビューが存在しない場合:
   - `analysis_time` から 1年（31,536,000秒）以内にレビューされていない → **除外**
4. `y > 31,536,000` の場合 → `y = 31,536,000`（打ち切り）

### 4.3 除外条件

- **Bot によるレビュー**: Bot 名リスト（`gerrymanderconfig.ini`）に部分一致するレビューアのメッセージは除外
- **最初のメッセージ（Uploaded patch set）**: Change 作成者自身の最初のアップロードメッセージはレビューとしてカウントしない
- **1年以上レビューなし**: 上記 4.2.3 のとおり除外
- **サンプル数不足**: 日ごとのサンプル数が **2件未満** の場合、その日の回帰分析を **スキップ**

---

## 5. 説明変数（16メトリクス）

`src/features` で抽出する以下の 16 メトリクスを説明変数として使用する。  
メトリクス計算は `src/analysis/trend_metrics/metrics_extraction/metrics_calculator.py` の `calculate_metrics()` と同様のロジックを用いる。

| # | メトリクス名 | 説明 | 型 | データスコープ |
|---|-------------|------|-----|------------|
| 1 | `bug_fix_confidence` | バグ修正確信度（0–2） | int | period_only |
| 2 | `lines_added` | 追加行数 | int | period_only |
| 3 | `lines_deleted` | 削除行数 | int | period_only |
| 4 | `files_changed` | 変更ファイル数 | int | period_only |
| 5 | `elapsed_time` | 経過時間（分） | float | period_only |
| 6 | `revision_count` | リビジョン数 | int | period_only |
| 7 | `test_code_presence` | テストコード存在（0/1） | int | period_only |
| 8 | `past_report_count` | 過去レポート数 | int | all_data |
| 9 | `recent_report_count` | 最近のレポート数（90日以内） | int | recent_data |
| 10 | `merge_rate` | マージ率（0.0–1.0） | float | all_data |
| 11 | `recent_merge_rate` | 最近のマージ率（90日以内） | float | recent_data |
| 12 | `days_to_major_release` | メジャーリリースまでの日数 | float | period_only |
| 13 | `open_ticket_count` | オープンチケット数 | int | period_only |
| 14 | `reviewed_lines_in_period` | 期間内レビュー行数（14日間） | int | period_only |
| 15 | `refactoring_confidence` | リファクタリング確信度（0/1） | int | period_only |
| 16 | `uncompleted_requests` | 未完了リクエスト数 | int | period_only |

### 5.1 欠損値の処理

- メトリクス値が `None` のレコードは **除外** する（`trend_metrics` と同一方針）
- `elapsed_time` が `-1.0` の場合は除外

### 5.2 標準化

- **標準化は行わない**（`trend_metrics` と同一方針）

---

## 6. モデル仕様

### 6.1 回帰モデル

- **最小二乗法 (OLS: Ordinary Least Squares)** による重回帰分析
- ライブラリ: `statsmodels.api.OLS`

### 6.2 実行単位

- **日ごと × バージョンごと** に独立して OLS を実行
- 各日の回帰式: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_{16} x_{16} + \epsilon$

### 6.3 出力する統計量

| 統計量 | 説明 |
|--------|------|
| `coef` ($\beta_i$) | 各メトリクスの回帰係数 |
| `std_err` | 標準誤差 |
| `t_value` | t値 |
| `p_value` | p値 |
| `r_squared` | 決定係数 $R^2$ |
| `adj_r_squared` | 自由度調整済み $R^2$ |
| `n_samples` | サンプル数 |
| `f_statistic` | F統計量 |
| `f_pvalue` | F検定のp値 |

---

## 7. 処理パイプライン

```
┌─────────────────────────────────────────────────────┐
│                  DailyRegressionAnalyzer             │
│                                                     │
│  1. prepare_data()                                  │
│     └─ リリース情報/Change/Bot名/コアレビューア読込  │
│                                                     │
│  2. バージョンごとのループ (10バージョン)             │
│     │                                               │
│     ├─ 3. 分析期間の決定                             │
│     │     └─ 当該リリース日 ～ 次リリース前日        │
│     │                                               │
│     ├─ 4. 日付ごとのループ                           │
│     │     │                                         │
│     │     ├─ 5. 対象Change抽出                      │
│     │     │     └─ Open条件 + Bot除外               │
│     │     │                                         │
│     │     ├─ 6. 目的変数計算                         │
│     │     │     └─ 次レビューまでの秒数              │
│     │     │                                         │
│     │     ├─ 7. 説明変数計算                         │
│     │     │     └─ 16メトリクス抽出                  │
│     │     │                                         │
│     │     ├─ 8. 欠損・外れ値除外                     │
│     │     │                                         │
│     │     ├─ 9. サンプル数チェック (< 2 → skip)     │
│     │     │                                         │
│     │     └─ 10. OLS実行 → 係数・統計量を保存       │
│     │                                               │
│     ├─ 11. バージョン単位の結果保存 (CSV/JSON)      │
│     │                                               │
│     └─ 12. バージョン単位の可視化 (PNG)             │
│                                                     │
│  13. 全体サマリーの保存                              │
└─────────────────────────────────────────────────────┘
```

---

## 8. ディレクトリ構造

### 8.1 ソースコード

```
src/analysis/daily_regression/
├── desigin.md                       # 本設計書
├── README.md                        # 使い方ドキュメント
├── __init__.py                      # DailyRegressionAnalyzer をエクスポート
├── main.py                          # エントリーポイント・メインクラス
├── utils/
│   ├── __init__.py
│   ├── constants.py                 # 定数定義（バージョン一覧・メトリクス一覧等）
│   └── data_loader.py              # データ読み込み（trend_metricsのdata_loaderを再利用/参照）
├── regression/
│   ├── __init__.py
│   ├── sample_extractor.py         # 日ごとの対象Change抽出・目的変数計算
│   ├── metrics_calculator.py       # 説明変数（16メトリクス）計算
│   └── ols_executor.py             # OLS実行・統計量取得
└── visualization/
    ├── __init__.py
    └── coefficient_plotter.py      # 回帰係数の時系列プロット（メトリクスごと）
```

### 8.2 テスト

```
tests/analysis/daily_regression/
├── __init__.py
├── test_constants.py               # 定数の整合性テスト
├── test_data_loader.py             # データ読み込みテスト
├── test_sample_extractor.py        # サンプル抽出・目的変数計算テスト
├── test_metrics_calculator.py      # メトリクス計算テスト
├── test_ols_executor.py            # OLS実行テスト
├── test_coefficient_plotter.py     # 可視化テスト
└── test_main.py                    # 統合テスト
```

### 8.3 出力データ

```
data/analysis/daily_regression/
├── nova_{version}/                   # バージョンごと（例: nova_20.0.0/）
│   ├── daily_coefficients.csv       # 日ごとの回帰係数一覧
│   ├── daily_regression_stats.csv   # 日ごとの統計量（R², n_samples等）
│   ├── daily_regression_detail.json # 詳細な回帰結果（係数・p値・t値等）
│   ├── daily_samples.csv            # 日ごとのサンプルデータ（デバッグ用）
│   └── plots/                       # 可視化
│       ├── coef_bug_fix_confidence.png
│       ├── coef_lines_added.png
│       ├── coef_lines_deleted.png
│       ├── coef_files_changed.png
│       ├── coef_elapsed_time.png
│       ├── coef_revision_count.png
│       ├── coef_test_code_presence.png
│       ├── coef_past_report_count.png
│       ├── coef_recent_report_count.png
│       ├── coef_merge_rate.png
│       ├── coef_recent_merge_rate.png
│       ├── coef_days_to_major_release.png
│       ├── coef_open_ticket_count.png
│       ├── coef_reviewed_lines_in_period.png
│       ├── coef_refactoring_confidence.png
│       └── coef_uncompleted_requests.png
└── summary/
    └── analysis_summary.json        # 全バージョンのサマリー
```

---

## 9. 出力仕様

### 9.1 daily_coefficients.csv

日ごとの回帰係数を記録する。

| カラム | 型 | 説明 |
|--------|-----|------|
| `date` | str | 分析対象日（YYYY-MM-DD） |
| `n_samples` | int | サンプル数 |
| `r_squared` | float | 決定係数 |
| `adj_r_squared` | float | 自由度調整済み決定係数 |
| `coef_{metric_name}` | float | 各メトリクスの回帰係数（16カラム） |
| `pvalue_{metric_name}` | float | 各メトリクスのp値（16カラム） |

### 9.2 daily_regression_stats.csv

日ごとの回帰モデルの統計情報を記録する。

| カラム | 型 | 説明 |
|--------|-----|------|
| `date` | str | 分析対象日（YYYY-MM-DD） |
| `n_samples` | int | サンプル数 |
| `r_squared` | float | 決定係数 |
| `adj_r_squared` | float | 自由度調整済み決定係数 |
| `f_statistic` | float | F統計量 |
| `f_pvalue` | float | F検定のp値 |
| `skipped` | bool | スキップ有無 |
| `skip_reason` | str | スキップ理由 |

### 9.3 daily_regression_detail.json

```json
{
  "version": "20.0.0",
  "project": "nova",
  "analysis_period": {
    "start": "2019-10-16",
    "end": "2020-05-13"
  },
  "daily_results": {
    "2019-10-16": {
      "n_samples": 150,
      "r_squared": 0.35,
      "adj_r_squared": 0.28,
      "f_statistic": 4.57,
      "f_pvalue": 0.0001,
      "coefficients": {
        "const": {"coef": 1234.5, "std_err": 100.2, "t_value": 12.3, "p_value": 0.001},
        "bug_fix_confidence": {"coef": -500.3, "std_err": 200.1, "t_value": -2.5, "p_value": 0.013},
        ...
      }
    },
    ...
  },
  "skipped_dates": ["2019-12-25", "2019-12-31"],
  "metadata": {
    "max_censoring_seconds": 31536000,
    "min_samples": 2,
    "exclusion_window_seconds": 31536000,
    "total_dates": 210,
    "analyzed_dates": 208,
    "skipped_dates_count": 2
  }
}
```

### 9.4 可視化（PNG）

各メトリクスに対して、以下の仕様でプロットを作成する:

- **横軸**: 日付（analysis_date）
- **縦軸**: 回帰係数 ($\beta_i$)
- **プロット形式**: 折れ線グラフ（線のみ、マーカー（点）なし）
- **ファイル形式**: PNG（dpi=300）
- **図サイズ**: (16, 8)

---

## 10. クラス・関数設計

### 10.1 main.py — DailyRegressionAnalyzer

```python
class DailyRegressionAnalyzer:
    def __init__(
        self,
        project_name: str = "nova",
        versions: Optional[List[str]] = None
    ):
        """
        Args:
            project_name: プロジェクト名（デフォルト: nova）
            versions: 分析対象バージョンリスト（デフォルト: constants.pyの定義）
        """

    def run_analysis(self) -> Dict:
        """全バージョンの日次回帰分析を実行"""

    def _prepare_data(self) -> Dict:
        """データ読み込み（Change, リリース, Bot, コアレビューア）"""

    def _get_analysis_period(self, version: str) -> Tuple[date, date]:
        """バージョンの分析期間を取得"""

    def _analyze_version(self, version: str, data_context: Dict) -> Dict:
        """1バージョンの日次回帰分析を実行"""

    def _analyze_single_day(
        self, analysis_date: date, data_context: Dict
    ) -> Optional[Dict]:
        """1日分の回帰分析を実行"""

    def _save_results(self, version: str, results: Dict, output_dir: Path):
        """結果をCSV/JSONに保存"""

    def _generate_visualizations(self, version: str, results: Dict, output_dir: Path):
        """可視化を生成"""
```

### 10.2 regression/sample_extractor.py

```python
def extract_daily_samples(
    analysis_date: date,
    all_changes: List[Dict],
    bot_names: List[str],
    max_censoring_seconds: int = 31_536_000
) -> pd.DataFrame:
    """
    指定日のOpen Changeを抽出し、目的変数を算出する

    Returns:
        DataFrame: columns=['change_number', 'created', 'analysis_time',
                           'first_review_time', 'time_to_review_seconds']
    """

def _get_first_non_bot_review_time(
    change: Dict,
    after_time: datetime,
    bot_names: List[str]
) -> Optional[datetime]:
    """Change の messages から、after_time 以降の最初の非botレビュー時刻を取得"""

def _is_change_open_on_date(
    change: Dict,
    analysis_date: date
) -> bool:
    """Change が指定日に Open 状態かを判定"""
```

### 10.3 regression/metrics_calculator.py

```python
def calculate_daily_metrics(
    samples_df: pd.DataFrame,
    all_changes: List[Dict],
    all_changes_df: pd.DataFrame,
    releases_df: pd.DataFrame,
    project_name: str
) -> pd.DataFrame:
    """
    サンプルに対して16メトリクスを計算する

    Returns:
        DataFrame: 目的変数 + 16メトリクスのカラムを持つ
    """
```

### 10.4 regression/ols_executor.py

```python
def execute_ols(
    df: pd.DataFrame,
    target_col: str = 'time_to_review_seconds',
    feature_cols: Optional[List[str]] = None,
    min_samples: int = 2
) -> Optional[Dict]:
    """
    OLS重回帰分析を実行する

    Returns:
        Dict: {'coefficients': {...}, 'r_squared': float, ...} or None (スキップ時)
    """
```

### 10.5 visualization/coefficient_plotter.py

```python
def plot_coefficient_timeseries(
    daily_results: pd.DataFrame,
    metric_name: str,
    version: str,
    output_path: Path,
    significance_level: float = 0.05
):
    """メトリクスの回帰係数の時系列プロットを生成"""

def plot_all_coefficients(
    daily_results: pd.DataFrame,
    version: str,
    output_dir: Path,
    metric_columns: List[str],
    significance_level: float = 0.05
):
    """全メトリクスの回帰係数プロットを一括生成"""
```

---

## 11. 実装・運用要件

| 項目 | 内容 |
|------|------|
| 言語 | Python 3.x |
| 主要ライブラリ | `statsmodels`, `pandas`, `numpy`, `matplotlib`, `seaborn` |
| 実行方法 | `python -m src.analysis.daily_regression.main` |
| テストフレームワーク | `pytest`（`tests/analysis/` 配下に pytest クラスベース + fixture） |
| テスト方針 | `src/analysis/trend_metrics` のテストパターンを踏襲 |

---

## 12. テスト項目

### 12.1 ユニットテスト

| テスト対象 | テスト内容 |
|-----------|-----------|
| `constants.py` | バージョンリスト・メトリクスリストの整合性 |
| `data_loader.py` | 各データ読み込み関数の正常系・異常系 |
| `sample_extractor.py` | Open判定ロジック、目的変数算出、Bot除外、1年超除外、打ち切り |
| `metrics_calculator.py` | 16メトリクスの計算結果の妥当性 |
| `ols_executor.py` | OLS実行の正常系、サンプル不足時のスキップ、統計量の取得 |
| `coefficient_plotter.py` | プロット生成（ファイル出力の確認、mock使用） |
| `main.py` | パイプライン全体の統合テスト |

### 12.2 テストデータ

- モックの Change JSON データ（最小構成）を `tests/data/` または fixture 内に定義
- 期待される回帰係数の検証には、既知のデータセットで `statsmodels.OLS` の結果と比較

---

## 13. 想定される問題と対策

| 問題 | 対策 |
|------|------|
| **多重共線性** | VIF（分散膨張係数）をログ出力し、高VIFメトリクスを報告（除外は手動判断） |
| **サンプル不足** | 日ごとに最小 2 件の閾値チェック。不足日はスキップし `skipped_dates` に記録 |
| **外れ値** | 1年超の打ち切り + 目的変数の最大値制限で対応 |
| **欠損値** | メトリクス値がNoneのレコードを除外（trend_metricsと同一方針） |
| **計算コスト** | バージョンごとに結果を逐次保存。進捗ログを出力 |