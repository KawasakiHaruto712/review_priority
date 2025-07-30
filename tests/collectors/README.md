# Collectors テストスイート

このディレクトリには、データ収集モジュール（`src.collectors`）のテストが含まれています。

## テスト対象モジュール

- `src.collectors.openstack` - OpenStack Gerritからのデータ収集
- `src.collectors.release_collector` - リリース情報の収集

## テストファイル構成

### `test_openstack.py`
OpenStack Gerrit収集機能の包括的なテスト：

#### テストクラス
- **TestOpenStackGerritCollector** - メインクラスのテスト
  - 初期化とプロジェクト設定
  - Change情報の取得
  - Commit情報の取得
  - メタデータ生成
  - 進捗表示
  - エラーハンドリング

- **TestOpenStackGerritCollectorIntegration** - 統合テスト
  - 実際のAPIとの連携テスト（モック使用）
  - データ収集プロセス全体のテスト

- **TestHelperFunctions** - ヘルパー関数のテスト
  - 日付範囲生成
  - プロジェクト設定読み込み
  - その他ユーティリティ関数

### `test_release_collector.py`
リリース収集機能の包括的なテスト：

#### テストクラス
- **TestReleaseCollector** - メインクラスのテスト
  - GitHubリポジトリのクローン
  - リリース情報の解析
  - プロジェクト・コンポーネントのマッピング
  - メタデータ生成

- **TestReleaseCollectorIntegration** - 統合テスト
  - Git操作の統合テスト
  - YAML解析の統合テスト

- **TestHelperFunctions** - ヘルパー関数のテスト
  - ファイル操作
  - データ変換ユーティリティ

## テスト内容

### 機能テスト
- ✅ データ収集APIの正常動作
- ✅ 各種パラメータでの動作確認
- ✅ データ変換とフォーマット処理
- ✅ ファイル保存とディレクトリ操作

### エラーハンドリングテスト
- ✅ ネットワークエラー時の対応
- ✅ 無効なAPIレスポンスの処理
- ✅ ファイルI/Oエラーの対応
- ✅ 設定ファイルエラーの処理

### エッジケーステスト
- ✅ 空のデータセット
- ✅ 無効な日付範囲
- ✅ 存在しないプロジェクト
- ✅ 権限エラー

### パフォーマンステスト
- ✅ 大量データの処理
- ✅ メモリ使用量の確認
- ✅ タイムアウト処理

## テスト実行方法

### 全テスト実行
```bash
pytest tests/collectors/ -v
```

### 個別ファイル実行
```bash
pytest tests/collectors/test_openstack.py -v
pytest tests/collectors/test_release_collector.py -v
```

### 特定のテストクラス実行
```bash
pytest tests/collectors/test_openstack.py::TestOpenStackGerritCollector -v
pytest tests/collectors/test_release_collector.py::TestReleaseCollector -v
```

### 統合テストのみ実行
```bash
pytest tests/collectors/ -k "Integration" -v
```

### カバレッジ付き実行
```bash
pytest tests/collectors/ --cov=src.collectors --cov-report=html
```

## モック使用について

テストでは以下のライブラリをモックして使用：
- `requests` - HTTP API呼び出し
- `git` - Git操作
- `pathlib.Path` - ファイルシステム操作
- `configparser` - 設定ファイル読み込み

これにより、外部依存関係なしでテストを実行できます。

## テストデータ

テストで使用するサンプルデータ：
- Change情報のJSONサンプル
- Commit情報のJSONサンプル
- プロジェクト設定のサンプル
- リリース情報のYAMLサンプル

## 注意事項

- テストは外部APIに依存せず、完全にモック化されています
- 実際のGerrit APIを使用したい場合は、環境変数で設定を変更してください
- 統合テストは実際のファイルシステムを使用する場合があります

## 継続的改善

新しい機能を追加する際は、対応するテストケースも追加してください：
1. 正常系のテスト
2. エラーハンドリングのテスト
3. エッジケースのテスト
4. パフォーマンステスト（必要に応じて）
