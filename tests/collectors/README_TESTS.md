# テストコードについて

## 概要

design.mdに基づいて作成されたモジュールの包括的なテストコードを作成しました。
ただし、実際の実装とテストコードの間にいくつかの乖離があることが判明しました。

## 作成したテストファイル

### 基底クラスのテスト
- `tests/collectors/base/test_retry_handler.py` - RetryConfigとリトライハンドラーのテスト
- `tests/collectors/base/test_base_api_client.py` - BaseAPIClientの抽象基底クラスのテスト

### エンドポイントのテスト
- `tests/collectors/endpoints/test_changes_endpoint.py` - ChangesEndpointのテスト
- `tests/collectors/endpoints/test_change_detail_endpoint.py` - ChangeDetailEndpointのテスト
- `tests/collectors/endpoints/test_comments_endpoint.py` - CommentsEndpointのテスト
- `tests/collectors/endpoints/test_reviewers_endpoint.py` - ReviewersEndpointのテスト
- `tests/collectors/endpoints/test_file_content_endpoint.py` - FileContentEndpointのテスト
- `tests/collectors/endpoints/test_file_diff_endpoint.py` - FileDiffEndpointのテスト
- `tests/collectors/endpoints/test_commit_endpoint.py` - CommitEndpointのテスト
- `tests/collectors/endpoints/test_commit_parents_endpoint.py` - CommitParentsEndpointのテスト

### 設定管理のテスト
- `tests/collectors/config/test_collector_config.py` - CollectorConfigのテスト

### ストレージのテスト
- `tests/collectors/storage/test_base_storage.py` - BaseStorageの抽象基底クラスのテスト
- `tests/collectors/storage/test_change_storage.py` - ChangeStorageのテスト
- `tests/collectors/storage/test_commit_storage.py` - CommitStorageのテスト

### 統合テスト
- `tests/collectors/test_change_collector.py` - ChangeCollectorの統合テスト

## テストが想定していた実装 vs 実際の実装の違い

### 1. RetryConfig
**想定**: `initial_wait`, `max_wait` パラメータ
**実際**: `base_delay`, `max_delay` パラメータ

### 2. BaseAPIClient
**想定**: `base_url`, `username`, `password` を受け取る初期化
**実際**: 異なる抽象メソッド構造 (`fetch`, `get_endpoint_path` が必要)

### 3. CollectorConfig
**想定**: `config_data` 属性、`default_limit` などの設定キー
**実際**: `config` 属性、`output_dir` などの異なる設定キー構造

### 4. BaseStorage
**想定**: `base_dir` パラメータを受け取る
**実際**: `save_component_data` 抽象メソッドが必要

### 5. エンドポイントクラスのメソッド名
**想定**: `get_changes()`, `get_change_detail()` などのメソッド
**実際**: `fetch()` メソッドを使用する実装

## テストコードの調整が必要な項目

1. **RetryConfig のパラメータ名を修正**
   - `initial_wait` → `base_delay`
   - `max_wait` → `max_delay`

2. **BaseAPIClient の具象クラス実装を修正**
   - `fetch()` と `get_endpoint_path()` メソッドを実装

3. **CollectorConfig のアサーションを修正**
   - 実際の設定キー名に合わせる
   - `config_data` → `config`
   - `base_dir` → `output_dir`

4. **BaseStorage の抽象メソッドを実装**
   - テスト用具象クラスに `save_component_data()` を追加

5. **エンドポイントクラスのモック対象を修正**
   - 実際のメソッド名に合わせてモックを作成

6. **ChangeCollector の初期化パラメータを確認**
   - 実際のコンストラクタシグネチャに合わせる

## テストの実行方法

実装に合わせてテストを修正後:

```bash
# 全テストの実行
python -m pytest tests/collectors/ -v

# 特定のモジュールのテスト実行
python -m pytest tests/collectors/base/ -v
python -m pytest tests/collectors/endpoints/ -v
python -m pytest tests/collectors/config/ -v
python -m pytest tests/collectors/storage/ -v

# カバレッジ付きでテスト実行
python -m pytest tests/collectors/ --cov=src/collectors --cov-report=html
```

## 次のステップ

1. 実際の実装コードを詳細に確認
2. テストコードを実装に合わせて修正
3. モックの設定を実際のクラス構造に合わせて調整
4. テストを実行して通過を確認
5. カバレッジレポートを確認して不足部分を補完

## テストカバレッジ

各テストファイルには以下のテストケースが含まれています:

### 基底クラス (50+ テストケース)
- 正常系のテスト
- 異常系のテスト
- エッジケースのテスト
- リトライロジックのテスト

### エンドポイント (80+ テストケース)
- 各エンドポイントの基本機能
- パラメータ指定のバリエーション
- エラーハンドリング
- レスポンスパースのテスト

### 設定・ストレージ (40+ テストケース)
- 設定の読み込みと検証
- データの保存と読み込み
- ファイル操作のテスト

### 統合テスト (20+ テストケース)
- エンドツーエンドのワークフロー
- 複数コンポーネントの連携
- 環境変数の統合

**合計: 約170個のテストケース**

## 参考資料

- `design.md` - モジュール設計ドキュメント
- 実装されたソースコード (`src/collectors/`)
- pytest公式ドキュメント: https://docs.pytest.org/
- unittest.mockドキュメント: https://docs.python.org/3/library/unittest.mock.html
