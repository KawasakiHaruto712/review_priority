# Review Priority プロジェクト テストディレクトリ

このディレクトリには、Review Priority プロジェクトのユニットテストと統合テストが含まれています。

## ディレクトリ構造

- `collectors/` - データ収集モジュールのテスト
- `features/` - 特徴量抽出モジュールのテスト
- `learning/` - 機械学習・IRL（逆強化学習）モジュールのテスト
- `preprocessing/` - データ前処理モジュールのテスト
- `data/` - テスト用サンプルデータファイル
- `test_integration.py` - 実データを使用した統合テスト

## テストの実行方法

### 前提条件

テスト用依存関係をインストール：
```bash
pip install -e .[test]
```

### 全テスト実行
```bash
pytest tests/ -v
```

### モジュール別テスト実行
```bash
pytest tests/collectors/ -v      # データ収集テスト
pytest tests/features/ -v       # 特徴量抽出テスト
pytest tests/learning/ -v       # 機械学習テスト
pytest tests/preprocessing/ -v  # データ前処理テスト
```

### 統合テストのみ実行
```bash
pytest tests/test_integration.py -v
```

### Makefileを使用した実行
```bash
make test           # 全テスト
make test-unit      # ユニットテストのみ
make test-integration # 統合テストのみ
```

### run_tests.pyスクリプトを使用した実行
```bash
python run_tests.py                # 依存関係インストール＋全テスト実行
python run_tests.py install        # テスト依存関係のインストール
python run_tests.py unit          # ユニットテスト実行
python run_tests.py integration   # 統合テスト実行
```

## テストカテゴリ

### ユニットテスト
- 各モジュールの個別機能をテスト
- モックデータと依存関係の模擬を使用
- 高速実行
- エッジケースとエラーハンドリングをテスト

### 統合テスト（`test_integration.py`）
- `data/openstack/`の実データファイルを使用
- 実際の変更データでの動作確認
- 実行時間は長いが、より現実的なテスト
- データディレクトリが存在しない場合はスキップ可能

## テストデータ

- `tests/data/sample_change.json` - ユニットテスト用サンプル変更データ
- `data/openstack/{project}/changes/` の実データは統合テストで使用

## 各モジュールの詳細

詳細なテスト内容については、各ディレクトリのREADME.mdを参照してください：

- [collectors/README.md](collectors/README.md) - データ収集テスト
- [features/README.md](features/README.md) - 特徴量抽出テスト  
- [learning/README.md](learning/README.md) - 機械学習テスト
- [preprocessing/README.md](preprocessing/README.md) - データ前処理テスト
