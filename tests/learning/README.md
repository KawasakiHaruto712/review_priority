# IRLモデル テストガイド

このディレクトリには、`src/learning/irl_models.py`のテストファイルが含まれています。

## ファイル構成

```
tests/learning/
├── __init__.py                # パッケージ初期化
├── test_irl_models.py        # メインのテストファイル（pytest対応）
├── test_helpers.py           # テスト用ヘルパー関数とモックデータ生成
├── test_performance.py      # パフォーマンステスト
├── run_tests.py             # 単体テスト実行スクリプト（pytest不要）
└── README.md                # このファイル
```

## テスト実行方法

### 方法1: pytestを使用した実行

```bash
# プロジェクトルートディレクトリで実行
cd /Users/haruto-k/review_priority

# 全ての学習関連テストを実行
pytest tests/learning/ -v

# 特定のテストファイルを実行
pytest tests/learning/test_irl_models.py -v

# パフォーマンステスト（時間がかかるものを除く）
pytest tests/learning/test_performance.py -v -m "not slow"

# すべてのパフォーマンステスト（時間がかかる）
pytest tests/learning/test_performance.py -v
```

### 方法2: 単体テストスクリプトを使用した実行

```bash
# pytestがインストールされていない環境での実行
cd /Users/haruto-k/review_priority
python tests/learning/run_tests.py
```

### 方法3: 個別モジュールのテスト

```bash
# テストヘルパーの動作確認
python tests/learning/test_helpers.py

# パフォーマンステストの実行例
python tests/learning/test_performance.py
```

## テスト内容

### 1. test_irl_models.py - 主要機能テスト

- **MaxEntIRLModelクラス**
  - モデルの初期化
  - 分配関数の計算
  - 期待特徴量の計算
  - 学習プロセス
  - 予測機能
  - モデルの保存・読み込み

- **ユーティリティ関数**
  - `is_bot_author()`: ボット判定
  - `extract_learning_events()`: 学習イベント抽出
  - `get_open_changes_at_time()`: 指定時刻でのオープンChange取得
  - `calculate_review_priorities()`: 優先順位計算

- **ReviewPriorityDataProcessor**
  - データ読み込み
  - 特徴量抽出

- **時系列IRL分析**
  - `run_temporal_irl_analysis()`: 時系列学習機能

- **統合テスト**
  - 完全なワークフローのテスト

### 2. test_helpers.py - テストユーティリティ

- モックデータ生成関数
- テスト用のChangeデータ作成
- 特徴量データフレーム生成
- 学習イベントシナリオ作成
- 検証用アサート関数

### 3. test_performance.py - パフォーマンステスト

- **実行時間テスト**
  - 小規模データ（100サンプル）
  - 中規模データ（1000サンプル）
  - 大規模データ（5000サンプル）

- **メモリ使用量テスト**
  - 学習時のメモリ使用量
  - メモリリークの検出

- **スケーラビリティテスト**
  - 特徴量次元数に対するスケーラビリティ
  - データサイズに対するスケーラビリティ

### 4. run_tests.py - 軽量テストランナー

- pytest非依存の基本テスト実行
- 重要な機能の動作確認
- 開発環境での簡易テスト

## 期待される結果

### 正常なテスト実行時の出力例

```
=== IRLモデル テスト実行 ===

✓ テストヘルパーモジュールのインポート成功
✓ IRLモデルモジュールのインポート成功

実行中: ボット判定機能
✓ ボット判定機能 - PASSED

実行中: MaxEntIRLModel基本機能
✓ MaxEntIRLModel基本機能 - PASSED

実行中: 優先順位計算
✓ 優先順位計算 - PASSED

実行中: モックデータ生成
✓ モックデータ生成 - PASSED

実行中: 学習イベントシナリオ
✓ 学習イベントシナリオ - PASSED

実行中: データプロセッサー初期化
✓ データプロセッサー初期化 - PASSED

=== テスト結果サマリー ===
実行: 6
成功: 6
失敗: 0

🎉 すべてのテストが成功しました！
```

## トラブルシューティング

### 依存関係のエラー

```bash
# 必要なパッケージのインストール
pip install pytest numpy pandas scipy scikit-learn

# パフォーマンステスト用（オプション）
pip install psutil
```

### インポートエラー

```bash
# Pythonパスの確認
export PYTHONPATH=/Users/haruto-k/review_priority:$PYTHONPATH

# または、プロジェクトルートから実行
cd /Users/haruto-k/review_priority
python -m tests.learning.run_tests
```

### メモリ不足エラー

大規模なテストでメモリ不足が発生する場合：

```bash
# 大規模テストをスキップ
pytest tests/learning/test_performance.py -v -m "not slow"

# または、テストサイズを縮小
# test_performance.py内のサンプル数を調整
```

## テストカバレッジ

このテストスイートは以下をカバーしています：

- ✅ MaxEntIRLModelクラスの全メソッド
- ✅ 優先順位計算アルゴリズム
- ✅ 時系列学習ワークフロー
- ✅ データ処理パイプライン
- ✅ エラーハンドリング
- ✅ パフォーマンス特性
- ✅ メモリ使用量

## 継続的な改善

### 新しいテストの追加

1. `test_irl_models.py`に新しいテストクラス/メソッドを追加
2. `test_helpers.py`に必要なヘルパー関数を追加
3. `run_tests.py`に基本的な動作確認テストを追加

### パフォーマンスベンチマーク

定期的にパフォーマンステストを実行して、性能劣化がないことを確認：

```bash
# ベンチマーク実行
pytest tests/learning/test_performance.py::TestPerformance::test_model_training_performance_medium -v --tb=short

# 結果をログに保存
pytest tests/learning/test_performance.py -v > performance_log_$(date +%Y%m%d).txt
```

## 注意事項

- パフォーマンステストは実行環境によって結果が大きく変わる可能性があります
- 大規模データテストは十分なメモリ（4GB以上推奨）が必要です
- CI/CD環境では`-m "not slow"`オプションを使用して高速テストのみ実行することを推奨します
