# shared_ranking

イベント駆動のレビュー優先順位データセットを生成する共通モジュール。
`trend_models` と `daily_regression` の両方から参照される。

仕様の詳細は [design.md](design.md) を参照。

## 概要

各チケットの **作成イベント** と **リビジョン更新イベント** を計測時点 T として、
T 時点で「open かつ T までに非 bot レビューが付いていない」全チケットを
スナップショット母集合とし、`first_review_time - T` の昇順で順位付けする。

1 計測時点 = 1 query。母集合の各チケットが 1 行を成す。

## 公開 API

```python
from src.analysis.shared_ranking import (
    build_event_ranking_dataset,
    load_or_build_event_ranking_dataset,
    FEATURE_NAMES,
    LOGIC_VERSION,
    MAX_CENSORING_SECONDS,
    MIN_QUERY_SIZE,
)
```

### `build_event_ranking_dataset(...)`

生データから直接構築する。キャッシュは使わない。

### `load_or_build_event_ranking_dataset(...)`

優先順は **プロセス内ランタイム → ファイル → 再ビルド**。
ファイルパスは `data/analysis/shared_ranking/<project>/<release>/event_ranking_<hash>.pkl`。
`<hash>` は `(project, release, period_start, period_end, min_query_size, max_censoring_seconds, feature_names, logic_version)` から決まる。

## 出力 DataFrame の主要カラム

| カラム | 説明 |
|--------|------|
| `query_id` | 1 計測時点を識別する文字列 |
| `query_size` | 母集合サイズ |
| `change_number` | 母集合内のチケット番号 |
| `measurement_time` / `analysis_time` | 計測時点 T |
| `analysis_date` | T の `date()` |
| `event_type` | `created` または `revision_update` |
| `trigger_change_number`, `trigger_revision_number` | T を生んだイベントの主体 |
| `time_to_review_seconds` | `verification_time - T`（秒、打ち切りあり） |
| `reviewed`, `censored` | 0/1 フラグ |
| `review_priority_rank` | 1 始まり dense rank（1 が最優先） |
| `review_priority_rank_pct` | `(rank-1)/max(query_size-1, 1)` |
| 16 特徴量カラム | `FEATURE_NAMES` の各特徴量 |

## 利用側のラッパ

| 旧 API | 改修後の挙動 |
|--------|------------|
| `trend_models.ranking.daily_rank_builder.build_daily_ranking_dataset(...)` | shared_ranking を呼び、period でフィルタして `period_type` カラムを付与 |
| `daily_regression.regression.sample_extractor.extract_daily_samples(...)` | shared_ranking を呼び、`analysis_date` でフィルタ |

## キャッシュをクリアしたい場合

```bash
rm -rf data/analysis/shared_ranking/
```

`LOGIC_VERSION` をインクリメントすれば旧キャッシュは自動的に無視される。

## 注意

- shared_ranking 経由で生成されたデータには 16 特徴量がすでに含まれる。
  `daily_regression.regression.metrics_calculator.calculate_daily_metrics`
  は特徴量カラムが揃っている場合は再計算をスキップする。
- 学習結果や OLS 係数の絶対値は旧手法と異なる。比較レポートは再生成が必要。
