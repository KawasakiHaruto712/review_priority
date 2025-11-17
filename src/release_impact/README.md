# Release Impact Analysis

リリース前後でレビュー対象の変更（Change）の特性がどのように変化するかを分析するモジュールです。

## 📋 概要

本モジュールは、OpenStackプロジェクトにおいて、同一バージョンのライフサイクル内で：
- **リリース直後の初期期間**（early period）
- **次のリリース直前の後期期間**（late period）

を比較し、16種類のfeatureメトリクスの分布を統計的に評価します。

## 🔍 分析対象

### 4つの期間グループ

1. **early_reviewed**: 初期期間（リリース直後30日間）のレビュー済みChange
2. **early_not_reviewed**: 初期期間の未レビューChange
3. **late_reviewed**: 後期期間（次リリース直前30日間）のレビュー済みChange
4. **late_not_reviewed**: 後期期間の未レビューChange

### 16種類のメトリクス

1. `bug_fix_confidence`: バグ修正の確信度
2. `lines_added`: 追加行数
3. `lines_deleted`: 削除行数
4. `files_changed`: 変更ファイル数
5. `elapsed_time`: 経過時間
6. `revision_count`: リビジョン数
7. `test_code_presence`: テストコード存在
8. `past_report_count`: 過去レポート数
9. `recent_report_count`: 最近レポート数
10. `merge_rate`: マージ率
11. `recent_merge_rate`: 最近マージ率
12. `days_to_major_release`: リリースまで日数
13. `open_ticket_count`: オープンチケット数
14. `reviewed_lines_in_period`: 期間内レビュー行数
15. `refactoring_confidence`: リファクタリング確信度
16. `uncompleted_requests`: 未完了リクエスト数

## 🚀 使用方法

### 基本的な使用例

```python
from src.release_impact import ReleaseMetricsComparator

# 分析の実行
comparator = ReleaseMetricsComparator('nova')
comparator.run_analysis()
```

### コマンドライン実行

```bash
# 単一プロジェクトの分析
python -m src.release_impact.metrics_comparator

# または直接実行
cd src/release_impact
python metrics_comparator.py
```

## 📊 出力ファイル

各リリース期間について、以下のファイルが生成されます：

```
data/release_impact/{project}_{release_version}_period/
├── metrics_data.csv         # 全メトリクスデータ
├── summary_statistics.json  # 記述統計量
├── test_results.json        # Mann-Whitney U検定結果
├── boxplots_4x4.pdf        # ボックスプロット(4×4グリッド)
├── heatmap.pdf             # p値ヒートマップ
└── summary_plot.pdf        # 平均値比較プロット
```

### 出力例

**metrics_data.csv**
```csv
change_number,component,period_group,created,bug_fix_confidence,lines_added,...
1234,nova,early_reviewed,2024-01-15,1,150,...
```

**summary_statistics.json**
```json
{
  "early_reviewed": {
    "lines_added": {
      "count": 500,
      "mean": 125.5,
      "std": 45.3,
      "50%": 120
    }
  }
}
```

**test_results.json**
```json
{
  "early_reviewed_vs_late_reviewed": {
    "lines_added": {
      "statistic": 12345.0,
      "p_value": 0.023,
      "significant": true,
      "effect_size": 0.15
    }
  }
}
```

## 📈 統計手法

- **Mann-Whitney U検定**: ノンパラメトリック検定（分布を仮定しない）
- **記述統計量**: mean, median, std, quartiles
- **効果量**: Rank-biserial correlation

## 🎨 可視化

### ボックスプロット
- 4×4グリッドで16メトリクスを表示
- 4つの期間グループを比較
- 自動的に対数軸を適用（範囲が広いメトリクス）

### ヒートマップ
- p値を色分けして表示
- 有意な差がある箇所を一目で把握

### サマリープロット
- 各グループの平均値を棒グラフで比較
- エラーバーで標準偏差を表示

## ⚙️ 設定のカスタマイズ

### 対象リリースの変更

`src/config/release_constants.py` で設定：

```python
RELEASE_IMPACT_ANALYSIS = {
    'nova': {
        'target_release': [
            '15.0.0',
            '16.0.0',
            # ... 追加
        ]
    }
}
```

### 分析期間の変更

```python
RELEASE_ANALYSIS_PERIODS = {
    'early_reviewed': {
        'base_date': 'current_release',
        'offset_start': 0,
        'offset_end': 30,  # 日数を変更
        'review_status': 'reviewed'
    }
}
```

## 🔧 モジュール構成

```
src/release_impact/
├── __init__.py
├── README.md
├── metrics_comparator.py        # メイン分析ロジック
└── metrics_analysis/
    ├── __init__.py
    ├── statistical_analyzer.py  # 統計分析
    └── visualizer.py           # 可視化
```

## 📝 ログ出力

分析の進行状況とエラーは標準出力に記録されます：

```
2024-01-15 10:00:00 - INFO - ReleaseMetricsComparator initialized for project: nova
2024-01-15 10:00:01 - INFO - Loaded 15 releases for nova
2024-01-15 10:00:02 - INFO - === リリース期間分析開始: 15.0.0 (終期基準: 16.0.0) ===
...
```

## ⚠️ 注意事項

1. **データの前提条件**
   - `data/openstack/{project}/changes/` にChangeデータが必要
   - `data/openstack/releases_summary.csv` にリリース情報が必要

2. **メモリ使用量**
   - 大量のChangeを処理する場合、メモリ使用量が多くなる可能性があります

3. **処理時間**
   - プロジェクトとリリース数によって、数分から数十分かかる場合があります

## 🐛 トラブルシューティング

### データが見つからない

```
FileNotFoundError: リリースファイルが見つかりません
```

→ `data/openstack/releases_summary.csv` が存在するか確認

### メトリクスが抽出できない

```
WARNING - すべての期間でメトリクスの抽出に失敗しました
```

→ 指定した期間内にChangeが存在するか確認

### グラフが生成されない

→ matplotlib, seabornがインストールされているか確認

```bash
pip install matplotlib seaborn
```

## 📚 関連ドキュメント

- [設計書](designs.md): 詳細な設計仕様
- [Features README](../features/README.md): メトリクスの詳細
- [Learning README](../learning/README.md): データ処理の詳細