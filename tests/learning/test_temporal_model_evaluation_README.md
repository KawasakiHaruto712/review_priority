# temporal_model_evaluation.py のテスト

このディレクトリには、`src/learning/temporal_model_evaluation.py`の機能をテストするユニットテストが含まれています。

## テスト対象

### `TemporalModelEvaluator`クラス
- **初期化**: ウィンドウサイズ、スライディングステップ、乱数シードの設定
- **時間ウィンドウ生成**: 分析期間のウィンドウ分割
- **ボット名読み込み**: 設定ファイルからのボット名取得
- **正負例抽出**: ウィンドウ期間中のレビュー有無に基づくラベル付け
- **ウィンドウ評価**: Balanced Random Forestによる学習・評価
- **結果保存**: CSV/JSON/PDF形式での結果出力

### `run_temporal_model_evaluation`関数
- **成功ケース**: 正常な評価実行とファイル保存
- **エラーハンドリング**: 例外発生時の適切な処理

## テストクラス構成

### 1. `TestTemporalModelEvaluator`
主要なクラス機能のテスト
- `test_init`: 初期化の検証
- `test_generate_time_windows`: ウィンドウ生成ロジックの検証
- `test_load_bot_names`: ボット名読み込みの検証
- `test_extract_window_labels_positive_case`: 正例抽出の検証
- `test_extract_window_labels_negative_case`: 負例抽出の検証
- `test_extract_window_labels_bot_review`: ボットレビューの除外検証
- `test_evaluate_window_success`: ウィンドウ評価の成功ケース
- `test_evaluate_window_no_open_changes`: オープンなChangeなしの処理
- `test_save_results_csv`: CSV保存機能の検証
- `test_save_results_json`: JSON保存機能の検証
- `test_save_results_summary_statistics`: サマリー統計の計算検証
- `test_save_results_pdf`: PDF生成機能の検証

### 2. `TestRunTemporalModelEvaluation`
メイン実行関数のテスト
- `test_run_temporal_model_evaluation_success`: 正常実行の検証
- `test_run_temporal_model_evaluation_error`: エラーハンドリングの検証

### 3. `TestEdgeCases`
エッジケースのテスト
- `test_empty_evaluation_results`: 空の評価結果の処理
- `test_all_failed_evaluations`: 全ウィンドウ失敗時の処理
- `test_single_window`: 単一ウィンドウのみの場合

## テストデータ

### モックデータ生成関数
```python
create_mock_changes_data(n_changes=100, start_date="2024-01-01", end_date="2024-01-31")
```
- テスト用のChangeデータを生成
- ランダムなレビューメッセージを含む

```python
create_mock_releases_data()
```
- テスト用のリリースデータを生成

```python
create_mock_features_data(n_samples=50)
```
- テスト用の特徴量データを生成
- 16種類のメトリクスを含む

## 実行方法

### 全テストの実行
```bash
cd /Users/haruto-k/review_priority
python -m pytest tests/learning/test_temporal_model_evaluation.py -v
```

### 特定のテストクラスのみ実行
```bash
python -m pytest tests/learning/test_temporal_model_evaluation.py::TestTemporalModelEvaluator -v
```

### 特定のテストメソッドのみ実行
```bash
python -m pytest tests/learning/test_temporal_model_evaluation.py::TestTemporalModelEvaluator::test_init -v
```

### カバレッジレポート付きで実行
```bash
python -m pytest tests/learning/test_temporal_model_evaluation.py --cov=src.learning.temporal_model_evaluation --cov-report=html
```

## テストのポイント

### 1. 正負例の定義テスト
- **正例（ラベル=1）**: ウィンドウ期間中に人間によるレビューがあったPR
- **負例（ラベル=0）**: ウィンドウ期間中にレビューがなかったPR、またはボットのみのレビュー

### 2. データ分割の検証
- PR単位で8:2に分割されることを確認
- `random_state=42`で再現性が保証されることを確認

### 3. 評価指標の検証
- Precision（適合率）
- Recall（再現率）
- F1 Score（F値）

### 4. ファイル出力の検証
- CSV: ウィンドウごとの詳細評価指標
- JSON: サマリー統計を含む全体結果
- PDF: 6つのグラフを含む可視化レポート

## モックの使用

複雑な依存関係を持つ機能は`unittest.mock`を使用してモック化：
- `get_open_changes_at_time`: オープンなChange取得
- `extract_features`: 特徴量抽出
- `create_evaluation_visualization`: PDF生成

## 期待される結果

全てのテストが成功すれば、以下が保証されます：
- ✅ ウィンドウ生成が正しく動作する
- ✅ 正負例の抽出が正確である
- ✅ Balanced Random Forestでの評価が実行できる
- ✅ 結果が正しく保存される（CSV/JSON/PDF）
- ✅ エラーハンドリングが適切である

## 依存関係

テスト実行には以下のパッケージが必要です：
- pytest >= 7.0.0
- numpy
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib (PDF生成用)
- seaborn (PDF生成用)

## トラブルシューティング

### テストが失敗する場合
1. 依存パッケージがインストールされているか確認
   ```bash
   pip install -r requirements.txt
   ```

2. Pythonパスが正しく設定されているか確認
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/Users/haruto-k/review_priority"
   ```

3. テストの詳細ログを確認
   ```bash
   python -m pytest tests/learning/test_temporal_model_evaluation.py -v -s
   ```

## 今後の拡張

- [ ] 実データを使用した統合テスト
- [ ] パフォーマンステスト（大量データでの実行時間）
- [ ] PDF生成内容の詳細検証
- [ ] 異なるウィンドウサイズでの動作検証
- [ ] 特徴量の重要度分析テスト
