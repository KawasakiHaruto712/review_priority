1回目: 日次ランキング算出 → モデル結果算出 → 評価（キャッシュ保存）# Trend Models Ranking 化 設計書

## 1. 背景

現行の [src/analysis/trend_models](src/analysis/trend_models) は、各 Change を「期間内にレビューされたか（reviewed=1）」または「されなかったか（reviewed=0）」の二値分類として学習・評価している。

一方、運用上の目的は「どの Change を先にレビューすべきか」という優先順位付けであり、二値分類だけでは順位の質（上位提案の妥当性）を直接評価できない。

本設計では、[src/analysis/daily_regression](src/analysis/daily_regression) と同様の考え方で「日次の候補集合をランキング化」し、ランキング学習・評価を実施できるようにする。

## 2. 目的

1. 日次単位で候補 Change を抽出し、順位ラベルを作成する。
2. 順位ラベルを用いたモデル学習（ランキング学習）を実装する。
3. 既存の二値分類モデルと、同一データ分割条件で比較できるようにする。
4. モデル比較結果を CSV/JSON/可視化で出力する。

## 3. スコープ

### 3.1 対象

- 対象モジュール: [src/analysis/trend_models](src/analysis/trend_models)
- 対象プロジェクト: 既存設定に準拠（まずは nova）
- 対象期間: 既存の early / late / all を継続利用

### 3.2 非対象（初期リリース）

- 外部ランキングライブラリ（LightGBM ranker, XGBoost ranker）への依存追加
- オンライン推論基盤・API化

## 4. 現行課題

1. ラベルが二値（reviewed/not reviewed）のみで、順位の良さを評価できない。
2. 評価指標が Precision/Recall/F1 中心で、上位 k 件の品質を測れていない。
3. 予測出力が確率/クラスであり、日次候補群内の相対優先順位を最適化していない。

## 5. To-Be アーキテクチャ

以下の 2 系統を同時に保持する。

1. Classification Pipeline（現行互換）
2. Ranking Pipeline（新規）

比較実験では同じ期間分割・同じ特徴量で両者を走らせ、指標を並べて比較する。

## 6. データ設計（ランキング用）

### 6.1 日次クエリ単位

ランキング学習では、同一日に「同時にレビュー候補だった Change 群」を 1 クエリ（group/query）とみなす。

- query_id: project + release + period + analysis_date
- 1 query 内で順位を付与

### 6.2 候補抽出

日付 D に対して以下を候補とする。

- D 時点で open だった Change
- D より後に作成された Change は除外
- bot 由来のイベントは除外

補足: 抽出ルールは [src/analysis/daily_regression/regression/sample_extractor.py](src/analysis/daily_regression/regression/sample_extractor.py) の考え方を踏襲する。

### 6.3 目的変数

日次クエリ内で以下を算出する。

1. time_to_review_seconds
- D の analysis_time から最初の非botレビューまでの秒数

2. review_priority_rank
- time_to_review_seconds の昇順で dense rank
- 1 が最優先

3. review_priority_rank_pct（推奨）
- クエリサイズ差を吸収する正規化順位
- 例: (rank - 1) / max(n_query - 1, 1)

初期実装では review_priority_rank_pct を主目的変数とし、rank と time は補助出力として保持する。

### 6.4 学習テーブル列

最低限の列を持つ。

- query_id
- analysis_date
- release
- period_type
- change_number
- feature_16_columns
- reviewed（既存比較用）
- time_to_review_seconds
- review_priority_rank
- review_priority_rank_pct

## 7. 特徴量設計

特徴量は現行の 16 特徴量を継続利用する。

- 変更なし: [src/analysis/trend_models/features/extractor.py](src/analysis/trend_models/features/extractor.py)
- 変更点: 特徴量算出タイミングを analysis_date 時点に固定し、未来情報リークを防止

## 8. モデル設計

## 8.1 モード

実行モードを追加する。

- classification: 現行二値分類のみ
- ranking: 新規ランキングのみ
- both: 両方実行して比較

## 8.2 ランキング学習方式（初期）

Pointwise 回帰で順位スコアを学習する。

- 学習ターゲット: review_priority_rank_pct（小さいほど優先）
- 予測時は score = -pred_rank_pct として降順ソート

採用モデル（初期）

- random_forest_regressor
- gradient_boosting_regressor
- linear_regression（比較基準）

既存 classification モデルはそのまま維持。

## 8.3 モデル共通インタフェース

新規クラス案:

- RankingPredictor
- fit(X, y, group=None)
- predict_score(X)

注: 初期の pointwise 回帰では group 未使用だが、将来の pairwise/listwise 対応のため引数は残す。

## 9. 評価設計

### 9.1 ランキング指標（主）

日次 query ごとに算出し、最終的に平均する。

- NDCG@5, NDCG@10, NDCG@20
- MRR
- Spearman 相関（pred order vs true order）
- Pairwise Accuracy

### 9.2 既存互換指標（副）

比較のため二値指標も残す。

- Precision / Recall / F1

算出方法:
- 各日で予測上位 k% を positive とみなす閾値化
- 既存 reviewed ラベルと比較

### 9.3 分割戦略

現行と同様に Leave-One-Release-Out を維持する。

- eval_release を 1 つ除外
- 残り release で train
- train_period × eval_period（early, late, all の組み合わせ）で比較

## 10. 可視化・出力設計

## 10.1 出力ディレクトリ

既存を踏襲。

- data/analysis/trend_models/{project}/{model}/

## 10.2 追加ファイル

- rank_detail_*.csv
  - query 単位・日次単位の詳細指標
- rank_summary_*.csv
  - period 組み合わせ別の平均指標
- ranking_results_*.json
  - ランキング評価の集約結果
- figures/rank_ndcg_*.png
  - NDCG 比較棒グラフ
- figures/rank_heatmap_*.png
  - train/eval 組み合わせヒートマップ

## 10.3 既存ファイルとの整合

- cv_detail/cv_summary/results は classification 用として残す
- both モード時は classification と ranking を同じ timestamp で出力

## 11. 変更対象ファイル（計画）

### 11.1 既存修正

- [src/analysis/trend_models/main.py](src/analysis/trend_models/main.py)
  - task_mode, ranking_label, k_values オプション追加
  - ranking パイプライン呼び出し追加

- [src/analysis/trend_models/utils/constants.py](src/analysis/trend_models/utils/constants.py)
  - TASK_MODE, RANKING_MODEL_TYPES, RANKING_K_VALUES 追加

- [src/analysis/trend_models/evaluation/evaluator.py](src/analysis/trend_models/evaluation/evaluator.py)
  - RankingEvaluationResult, RankingCVResult 追加
  - ranking 用評価関数追加

- [src/analysis/trend_models/evaluation/visualizer.py](src/analysis/trend_models/evaluation/visualizer.py)
  - ranking 指標の可視化関数追加

### 11.2 新規追加

- src/analysis/trend_models/ranking/__init__.py
- src/analysis/trend_models/ranking/daily_rank_builder.py
  - 日次 query データ作成
- src/analysis/trend_models/ranking/ranking_predictor.py
  - pointwise ranking モデル実装
- src/analysis/trend_models/ranking/ranking_dataset.py
  - 学習用データ整形（X, y, group）

## 12. 実装ステップ

1. Step 1: データ生成
- daily_rank_builder を実装
- 日次 query + rank 系ラベルを生成
- CSV ダンプで検証

2. Step 2: 学習器実装
- RankingPredictor（pointwise）実装
- 3 モデル（RF/GBR/Linear）対応

3. Step 3: 評価実装
- NDCG/MRR/Spearman/Pairwise Accuracy を実装
- rank_summary 出力

4. Step 4: main 統合
- --task-mode {classification,ranking,both}
- --ranking-label {rank,rank_pct,time}
- --k-values 5,10,20

5. Step 5: 可視化と比較
- classification と ranking の比較レポート生成

## 13. リスクと対策

1. 日次 query サイズの偏り
- 対策: rank_pct を主ターゲットにし、query サイズ依存を低減

2. 情報リーク
- 対策: 特徴量は analysis_date 時点までの情報のみ利用

3. 計算コスト増
- 対策: query 単位の中間キャッシュ、期間絞り込みオプションを追加

4. 指標解釈の複雑化
- 対策: 主指標を NDCG@10 に固定し、他は補助指標として扱う

## 14. 受け入れ基準

1. ranking モードで end-to-end 実行できる。
2. rank_detail/rank_summary/ranking_results が出力される。
3. NDCG@10 を含むランキング指標が算出される。
4. both モードで classification と ranking の比較表が生成される。
5. 既存 classification モードの挙動を壊さない。

## 15. 将来拡張

- Pairwise 学習（差分特徴量）
- Listwise 学習（学習 to rank 専用ライブラリ）
- オンライン推論向けの daily top-k 提案 API
