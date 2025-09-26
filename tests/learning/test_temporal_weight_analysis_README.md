# temporal_weight_analysis.py テストドキュメント

## 概要
このファイルは `src/learning/temporal_weight_analysis.py` の包括的なテストスイートです。時系列重み分析機能の正確性と信頼性を確保するために作成されました。

## テスト構成

### 1. TestTemporalWeightAnalyzer クラス
**主要機能のユニットテスト**

#### テスト項目：
- `test_init()`: クラスの初期化テスト
- `test_generate_time_windows()`: スライディングウィンドウ生成の正確性
- `test_load_bot_names_default()`: デフォルトボット名読み込み
- `test_load_bot_names_from_config()`: 設定ファイルからのボット名読み込み
- `test_analyze_window_success()`: ウィンドウ分析の成功ケース
- `test_analyze_window_no_changes()`: Changeデータなし時の処理
- `test_analyze_window_no_events()`: 学習イベントなし時の処理
- `test_save_results()`: CSV/JSON結果保存機能
- `test_create_weight_visualization_*()`: PDF可視化機能

### 2. TestTemporalWeightAnalysisIntegration クラス
**統合テスト**

#### テスト項目：
- `test_run_temporal_analysis_no_data()`: データなし時の全体処理
- `test_run_temporal_analysis_success()`: 成功時の全体処理フロー

### 3. TestEdgeCases クラス
**エッジケースのテスト**

#### テスト項目：
- `test_empty_window_size()`: ウィンドウサイズ0の処理
- `test_large_sliding_step()`: 大きなスライディングステップ
- `test_invalid_date_range()`: 無効な日付範囲

## テスト対象機能

### ✅ 実装済み機能
1. **スライディングウィンドウ生成**
   - 2週間ウィンドウ、1日ステップの正確な生成
   - 日付範囲の境界条件処理

2. **設定管理**
   - 設定ファイルからのボット名読み込み
   - デフォルト値のフォールバック処理

3. **ウィンドウ分析**
   - 成功/失敗ステータスの適切な設定
   - エラーメッセージの正確な記録
   - IRLモデル学習の実行

4. **結果保存**
   - CSV形式：ウィンドウ毎の結果とステータス
   - JSON形式：詳細な分析結果
   - PDF形式：時系列重み変動グラフ

5. **エラーハンドリング**
   - データ不足時の適切な失敗処理
   - 例外発生時のエラーメッセージ記録

## テスト実行方法

### 個別実行
```bash
cd /Users/haruto-k/review_priority
/Users/haruto-k/review_priority/.venv/bin/python -m pytest tests/learning/test_temporal_weight_analysis.py -v
```

### 特定テストクラス実行
```bash
/Users/haruto-k/review_priority/.venv/bin/python -m pytest tests/learning/test_temporal_weight_analysis.py::TestTemporalWeightAnalyzer -v
```

### 特定テストメソッド実行
```bash
/Users/haruto-k/review_priority/.venv/bin/python -m pytest tests/learning/test_temporal_weight_analysis.py::TestTemporalWeightAnalyzer::test_generate_time_windows -v
```

## モック使用箇所

### データプロセッサ
- `ReviewPriorityDataProcessor`: OpenStackデータ読み込み
- `extract_features()`: 特徴量抽出

### 学習関連
- `extract_learning_events()`: 学習イベント抽出
- `get_open_changes_at_time()`: 時刻指定Change取得
- `calculate_review_priorities()`: 優先度計算

### 可視化
- `matplotlib/seaborn`: PDF生成機能の条件分岐テスト

## テスト結果の確認

### 成功時の出力例
```
15 passed in 2.85s
```

### カバレッジ確認（オプション）
```bash
/Users/haruto-k/review_priority/.venv/bin/python -m pytest tests/learning/test_temporal_weight_analysis.py --cov=src.learning.temporal_weight_analysis --cov-report=html
```

## 今後の拡張予定

### 追加テスト項目
1. **性能テスト**
   - 大量データでの処理時間測定
   - メモリ使用量の監視

2. **並列処理テスト**
   - マルチプロセッシング機能のテスト

3. **データ整合性テスト**
   - CSV/JSON出力の一貫性確認
   - 重み値の数値精度テスト

### 改善点
1. パラメータテスト（pytest.mark.parametrize）の追加
2. フィクスチャの活用でテストデータの共通化
3. より詳細なエラーケースのテスト追加

## 依存関係

### 必須パッケージ
- `pytest`: テストフレームワーク
- `pytest-mock`: モック機能
- `pandas`: データフレーム操作
- `numpy`: 数値計算
- `matplotlib`, `seaborn`: 可視化（オプション）

### テスト専用パッケージ
- `unittest.mock`: モック作成
- `tempfile`: 一時ファイル/ディレクトリ
- `pathlib`: パス操作
