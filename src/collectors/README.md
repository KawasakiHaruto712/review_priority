# Collectors モジュール

データ収集機能を提供するモジュールです。OpenStackのGerritシステムからレビューデータを取得し、分析に必要な形式で保存します。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `openstack.py` | OpenStack Gerritからのデータ収集クラス |
| `release_collector.py` | リリース情報の収集機能 |

## 🔧 主要機能

### OpenStackDataCollector (`openstack.py`)
- **Gerrit API連携**: OpenStackのGerritシステムからレビューデータを取得
- **Change収集**: コードレビューの詳細情報を収集
- **Commit収集**: 関連するコミット情報を取得
- **リトライ機能**: ネットワークエラー時の自動再試行
- **レート制限対応**: API制限を考慮した適切な間隔での取得

### ReleaseCollector (`release_collector.py`)
- **リリース情報取得**: OpenStackの各プロジェクトのリリース情報を収集
- **バージョン管理**: メジャーリリースの日程情報を管理

## 📊 収集データ形式

### Changeデータ
```json
{
  "change_number": 12345,
  "id": "I1234567890abcdef",
  "subject": "Fix memory leak in nova compute",
  "status": "MERGED",
  "owner": {"name": "developer", "email": "dev@example.com"},
  "created": "2024-01-01T10:00:00Z",
  "updated": "2024-01-01T15:00:00Z",
  "messages": [...],
  "revisions": {...}
}
```

### Commitデータ
```json
{
  "commit": "abc123def456",
  "author": "Developer Name",
  "date": "2024-01-01T10:00:00Z",
  "message": "Fix bug in authentication",
  "files": [...]
}
```

## 🚀 使用方法

### 基本的な使用例

```python
from src.collectors.openstack import OpenStackDataCollector
from src.collectors.release_collector import ReleaseCollector

# データ収集の初期化
collector = OpenStackDataCollector()

# 特定期間のChangeデータを収集
changes = collector.collect_changes(
    start_date="2024-01-01",
    end_date="2024-01-31",
    project="nova"
)

# リリース情報の収集
release_collector = ReleaseCollector()
releases = release_collector.collect_release_data()
```

### 環境設定

```bash
# 必要な環境変数（.envファイル）
GERRIT_USERNAME=your_username
GERRIT_PASSWORD=your_password
```

## ⚡ パフォーマンス

- **並列処理**: 複数プロジェクトの同時収集
- **増分更新**: 前回収集以降の差分のみ取得
- **データ圧縮**: 大量データの効率的な保存

## 🔍 ログ出力

収集過程は詳細にログ出力されます：

```
2024-01-01 10:00:00 - INFO - Collecting changes for project: nova
2024-01-01 10:05:00 - INFO - Collected 150 changes
2024-01-01 10:05:00 - WARNING - Rate limit approaching, waiting...
```

## ⚠️ 注意事項

1. **API制限**: Gerrit APIのレート制限に注意
2. **大量データ**: 長期間のデータ収集は時間がかかります
3. **ネットワーク**: 安定したインターネット接続が必要
4. **認証**: 適切なGerritアカウントの設定が必要

## 📈 収集統計

収集完了後、以下の統計情報が出力されます：
- 収集対象期間
- 取得したChange数
- エラー件数
- 実行時間
