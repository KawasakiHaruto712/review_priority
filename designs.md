# Release Impact Analysis System Design

## 1. 概要

本システムは、OpenStackプロジェクトにおいて、リリース前後でレビュー対象の変更（Change）の特性がどのように変化するかを分析するためのものです。具体的には、同一バージョンのライフサイクル内で、**リリース直後の初期期間**（early period）と**次のリリース直前の後期期間**（late period）を比較し、16種類のfeatureメトリクスの分布を統計的に評価します。

### 分析の目的
- リリース直後（バージョンが新しい時期）と次リリース直前（バージョンが成熟した時期）でメトリクスがどう変化するか
- レビュー済み（reviewed）と未レビュー（not reviewed）の変更で傾向が異なるか

## 2. 分析期間の定義

### 2.1 基本概念

各リリースペアに対して、以下の4つの期間グループを定義します：

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│リリース〜1ヶ月後  │1ヶ月前〜リリース  │1ヶ月前〜リリース  │リリース〜1ヶ月後  │
│(レビューされた)   │(レビューされなかった)│(レビューされた)   │(レビューされなかった)│
│                 │                 │                 │                 │
│  early_reviewed │early_not_reviewed│  late_reviewed │late_not_reviewed│
│                 │                 │                 │                 │
│  ← 期間A →     │  ← 期間A →     │  ← 期間B →     │  ← 期間B →     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### 2.2 期間の具体的な計算

#### Early Period（初期期間）
- **基準日**: Current Release の日付
- **対象期間**: 基準日 + 0日 ～ 基準日 + 30日
- **対象Change**: この期間に「オープン状態があった」すべてのChange
  - `created_at`が期間開始前でも、期間中にオープンであればカウント
  - 期間終了前や後にクローズされても問題なし

#### Late Period（後期期間）
- **基準日**: Next Release の日付
- **対象期間**: 基準日 - 30日 ～ 基準日 + 0日
- **対象Change**: この期間に「オープン状態があった」すべてのChange

### 2.3 Review Status による分類

各期間内のChangeを以下の2グループに分類：

1. **reviewed**: `review_count > 0` のChange
2. **not_reviewed**: `review_count == 0` のChange

### 2.4 4つの分析グループ

最終的に以下の4グループを比較：

1. `early_reviewed`: 初期期間のレビュー済みChange
2. `early_not_reviewed`: 初期期間の未レビューChange
3. `late_reviewed`: 後期期間のレビュー済みChange
4. `late_not_reviewed`: 後期期間の未レビューChange

## 3. 対象メトリクス

### 3.1 16種類のFeatureメトリクス

`ReviewPriorityDataProcessor`から抽出される以下のメトリクス：

#### Bug Metrics（バグメトリクス）
1. `bug_fix_confidence`: バグ修正の確信度（0-2のスコア）

#### Change Metrics（変更メトリクス）
2. `lines_added`: 追加行数
3. `lines_deleted`: 削除行数
4. `files_changed`: 変更ファイル数
5. `elapsed_time`: PRが作成されてから分析時点までの経過時間（分数）
6. `revision_count`: 分析時点までのリビジョン数
7. `test_code_presence`: テストコードの存在確認（0/1のフラグ）

#### Developer Metrics（開発者メトリクス）
8. `past_report_count`: 開発者が過去に報告したチケット数
9. `recent_report_count`: 開発者が最近（過去3ヶ月）に報告したチケット数
10. `merge_rate`: 開発者の全体的なマージ率（0.0-1.0）
11. `recent_merge_rate`: 開発者の最近のマージ率（0.0-1.0）

#### Project Metrics（プロジェクトメトリクス）
12. `days_to_major_release`: 次のメジャーリリースまでの残り日数
13. `open_ticket_count`: 分析時点でオープンなチケット数
14. `reviewed_lines_in_period`: 過去2週間にレビューされた行数の合計

#### Refactoring Metrics（リファクタリングメトリクス）
15. `refactoring_confidence`: リファクタリングの確信度（0/1のフラグ）

#### Review Metrics（レビューメトリクス）
16. `uncompleted_requests`: PRの未完了修正要求数

## 4. システム構成

### 4.1 ディレクトリ構造

```
src/release_impact/
├── __init__.py
├── README.md
├── designs.md                    # 本設計書
├── metrics_comparator.py         # メイン分析ロジック（エントリーポイント）
├── metrics_analysis/                     # 分析モジュール
    ├── __init__.py
    ├── statistical_analyzer.py  # 統計分析
    └── visualizer.py            # 可視化

src/features/
├── bug_metrics.py               # バグメトリクス
├── change_metrics.py            # Change概要メトリクス
├── developer_metrics.py         # 開発者メトリクス
├── project_metrics.py           # プロジェクトメトリクス
├── refactoring_metrics.py       # リファクタリングメトリクス
├── review_metrics.py            # レビューメトリクス

data/openstack/
├── {project}
│   ├── changes # 入力データ（Changeのデータがjsonで保存されている）

data/release_impact/              # 出力先
├── {project}_{release_pair}/
│   ├── metrics_data.csv         # 全メトリクスデータ
│   ├── summary_statistics.json  # 統計サマリー
│   ├── test_results.json        # 統計検定結果
│   ├── boxplots_4x4.pdf         # ボックスプロット（4×4グリッド）
│   └── heatmap.pdf              # ヒートマップ
```

### 4.2 定数定義（constants.py）

#### RELEASE_IMPACT_ANALYSIS

```python
RELEASE_IMPACT_ANALYSIS = {
    'nova': {
        'target_release': [
            '15.0.0',  # 15.0.0のリリース
            # ... 他のペア
        ]
    },
    'neutron': {
        # 同様の構造
    },
    # ... 他のプロジェクト
}
```

**注**: `target_release_pairs`は連続する2つのリリースバージョンのペアを表します。
- 第1要素（current_release）: Early期間の基準となるリリース（このリリース直後の30日間を分析）
- 第2要素（next_release）: Late期間の基準となるリリース（このリリース直前の30日間を分析）

つまり、**同一バージョンのライフサイクル**（current_releaseからnext_releaseまでの期間）において、初期期間と終期期間を比較します。

#### RELEASE_ANALYSIS_PERIODS

```python
RELEASE_ANALYSIS_PERIODS = {
    'early_reviewed': {
        'base_date': 'current_release',
        'offset_start': 0,
        'offset_end': 30,
        'review_status': 'reviewed'
    },
    'early_not_reviewed': {
        'base_date': 'current_release',
        'offset_start': 0,
        'offset_end': 30,
        'review_status': 'not_reviewed'
    },
    'late_reviewed': {
        'base_date': 'next_release',
        'offset_start': -30,
        'offset_end': 0,
        'review_status': 'reviewed'
    },
    'late_not_reviewed': {
        'base_date': 'next_release',
        'offset_start': -30,
        'offset_end': 0,
        'review_status': 'not_reviewed'
    }
}
```

## 5. クラス設計

### 5.1 ReleaseMetricsComparator

#### 責務
- リリースペアごとのメトリクス抽出
- 4期間グループへのデータ分類
- データの保存と管理

### 5.2 StatisticalAnalyzer

#### 責務
- 記述統計量の計算
- Mann-Whitney U検定の実行
- 統計結果の保存

### 5.3 MetricsVisualizer

#### 責務
- ボックスプロットの生成（4×4グリッド）
- グラフの保存

## 6. 出力形式

### 6.1 metrics_data.csv

全メトリクスの生データ：

```csv
change_number,component,period_group,created,bug_fix_confidence,lines_added,lines_deleted,files_changed,elapsed_time,revision_count,test_code_presence,past_report_count,recent_report_count,merge_rate,recent_merge_rate,days_to_major_release,open_ticket_count,reviewed_lines_in_period,refactoring_confidence,uncompleted_requests
1234,nova,early_reviewed,2024-01-15,1,150,80,5,1440.5,3,1,25,5,0.85,0.90,45,120,5000,0,2
5678,neutron,early_not_reviewed,2024-01-16,0,50,30,2,720.0,1,0,10,2,0.60,0.65,44,125,4800,1,0
...
```

### 6.2 summary_statistics.json

記述統計量：

```json
{
  "early_reviewed": {
    "lines_added": {
      "count": 500,
      "mean": 125.5,
      "std": 45.3,
      "min": 10,
      "25%": 80,
      "50%": 120,
      "75%": 160,
      "max": 300
    },
    "bug_fix_confidence": {
      "count": 500,
      "mean": 0.65,
      "std": 0.72,
      "min": 0,
      "25%": 0,
      "50%": 1,
      "75%": 1,
      "max": 2
    }
  }
}
```

### 6.3 test_results.json

統計検定結果：

```json
{
  "early_reviewed_vs_late_reviewed": {
    "lines_added": {
      "statistic": 12345.0,
      "p_value": 0.023,
      "significant": true,
      "effect_size": 0.15
    },
    "elapsed_time": {
      "statistic": 9876.0,
      "p_value": 0.001,
      "significant": true,
      "effect_size": 0.28
    }
  },
  "early_reviewed_vs_early_not_reviewed": {
    "uncompleted_requests": {
      "statistic": 5432.0,
      "p_value": 0.045,
      "significant": true,
      "effect_size": 0.12
    }
  }
}
```

## 7. 使用技術

### 7.1 Python ライブラリ

- **pandas**: データ操作
- **numpy**: 数値計算
- **scipy.stats**: 統計検定（Mann-Whitney U test）
- **matplotlib**: 基本的な描画
- **seaborn**: 統計的可視化

### 7.2 統計手法

- **Mann-Whitney U test**: ノンパラメトリック検定（分布を仮定しない）
- **記述統計量**: mean, median, std, quartiles等
- **効果量**: Cohen's d or rank-biserial correlation

## 8. メトリクスの詳細説明

### 8.1 Bug Metrics
- **bug_fix_confidence**: バグ修正パターン（バグID、特定キーワード）に基づく確信度スコア

### 8.2 Change Metrics
- **lines_added / lines_deleted**: コードベースへの影響度を示す基本的な指標
- **files_changed**: 変更の広がりを示す
- **elapsed_time**: レビューの緊急性や優先度に影響
- **revision_count**: レビューのイテレーション回数
- **test_code_presence**: 品質保証の有無

### 8.3 Developer Metrics
- **past_report_count / recent_report_count**: 開発者の経験値と活動度
- **merge_rate / recent_merge_rate**: 開発者の信頼性と実績

### 8.4 Project Metrics
- **days_to_major_release**: リリース計画との関連性
- **open_ticket_count**: プロジェクトの作業負荷
- **reviewed_lines_in_period**: レビュアーのキャパシティ

### 8.5 Refactoring Metrics
- **refactoring_confidence**: コード品質改善の意図の検出

### 8.6 Review Metrics
- **uncompleted_requests**: レビューの完了度と対応状況

## 9. データ収集と前処理

### 9.1 データソース
- **OpenStack Gerrit**: Change（PR）の詳細情報
- **リリース情報**: `releases_summary.csv`
- **レビューコメント**: `review_keywords.json`, `review_label.json`

### 9.2 前処理ステップ
1. 日時データの正規化（タイムゾーン統一、フォーマット変換）
2. 欠損値の処理（デフォルト値の設定）
3. 外れ値の検出と処理
4. メトリクスの正規化とスケーリング

### 9.3 データクオリティチェック
- Change数の妥当性確認
- 各期間の十分なサンプル数の確保
- メトリクス値の範囲チェック

## 10. 考慮事項

### 10.1 データ品質

- 期間内にChangeが少ない場合の対処
- 外れ値の扱い
- 欠損値の処理

### 10.2 統計的妥当性

- 多重検定補正（Bonferroni, FDR等）の必要性
- サンプルサイズの確認
- 効果量の解釈

### 10.3 拡張性

- 新しいメトリクスの追加
- 他のプロジェクトへの適用
- 異なる期間設定での分析

### 10.4 可視化の調整

- **対数軸の使用**: データの範囲が広いメトリクス（例: `lines_added`, `lines_deleted`, `elapsed_time`, `open_ticket_count`）については、対数軸を使用することで分布を見やすくします
  - 自動判定機能: データの最大値/最小値の比率が100以上の場合に自動的に対数軸を適用
  - 手動指定: `log_scale_metrics`パラメータで明示的に指定可能
  - 注意事項: 0や負の値を含むメトリクスには対数軸を適用しない

- **スケールの正規化**: メトリクス間でスケールが大きく異なるため、統計分析前に適切な正規化が必要
  - 行数系メトリクス（`lines_added`, `lines_deleted`）: 1-10,000の範囲
  - 比率系メトリクス（`merge_rate`, `recent_merge_rate`）: 0.0-1.0の範囲
  - フラグ系メトリクス（`test_code_presence`, `refactoring_confidence`）: 0/1または0-2

- **外れ値の処理**: 特に`lines_added`や`revision_count`などで極端な外れ値が発生する可能性があるため、四分位範囲（IQR）を用いた外れ値検出を検討

## 11. 今後の拡張可能性

### 11.1 短期的な拡張

- 複数のリリースペアをまとめた集計分析
- プロジェクト間の比較分析
- 時系列的な傾向分析

### 11.2 長期的な拡張

- 機械学習モデルとの統合
- リアルタイム分析システム
- Webダッシュボード化

---

## 付録A: 具体例

### A.1 メトリクス比較の解釈例

もし`lines_added`が以下のような結果なら：

- early_reviewed: median=120
- late_reviewed: median=95
- p_value=0.001 (有意)

→ **解釈**: リリース直後（early）は追加行数が多く、次リリース直前（late）では少ない傾向。バージョンが成熟するにつれて小規模な修正が増える可能性。

### A.2 対数軸の適用例

`lines_added`メトリクスの分布が以下のような場合：

```
Min: 1行
Median: 120行
Max: 10,000行
範囲比: 10,000 / 1 = 10,000
```

この場合、範囲が非常に広いため、対数軸（log scale）を使用することで：
- 小規模な変更（1-100行）の分布が見やすくなる
- 大規模な外れ値の影響を軽減
- グループ間の違いをより明確に可視化

### A.3 特徴量の相関例

以下のような相関が観察される可能性があります：

- `lines_added` と `lines_deleted`: 正の相関（大きな変更は追加も削除も多い）
- `elapsed_time` と `revision_count`: 正の相関（時間が経つほどリビジョンが増える）
- `recent_merge_rate` と `uncompleted_requests`: 負の相関（マージ率が高い開発者は未完了要求が少ない）
- `days_to_major_release` と `open_ticket_count`: 負の相関（リリース直前は未完了チケットが減る）

---

**文書作成日**: 2025-01-10  
**最終更新日**: 2025-01-10  
**バージョン**: 2.0  
**作成者**: GitHub Copilot  
**更新内容**: 実装されているメトリクスに合わせて設計書を修正
