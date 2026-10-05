# Collectors モジュール

データ収集機能を提供するモジュールです。OpenStackのGerritシステムからレビューデータを取得し、分析に必要な形式で保存します。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `openstack.py` | OpenStack Gerritからのデータ収集クラス |
| `release_collector.py` | リリース情報の収集機能（`releases_summary.csv` を出力） |
| `major_release_storage.py` | 公式のサイクル一覧から、サイクル単位の `major_releases_summary.csv` を生成（2026-10 改訂。下の「メジャーリリースの表」） |

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
- **出力**: 全リリースを `releases_summary.csv`（列：`component, version, release_date, yaml_url`）に保存

### メジャーリリース抽出 (`major_release_storage.py`)
- **目的**: `releases_summary.csv` から**メジャーリリースだけを抽出**して `major_releases_summary.csv`（列：`project, version, release_date, yaml_url`）を生成する（分析側 `load_release_dates` が使う）。
- **背景**: この CSV は生成コードが残っていなかったため、`releases_summary.csv` から再生成できるようにする。
- **注記**: swift は版付け方式が異なる（`X.0.0` でなく `2.Y.0` で刻む）ため、swift だけ別ルールで抽出する。
- **入出力**: 入力 `data/openstack_collected/releases_summary.csv` → 出力 `data/openstack_collected/major_releases_summary.csv`。
- **注記（2026-10）**: 上の版番号で拾う方式は廃止する。理由と新しい作り方は次の「メジャーリリースの表」。

## 📅 メジャーリリースの表（`major_releases_summary.csv`、2026-10 改訂）

特徴量 `days_to_major_release` と、分析期間の区切り（事前学習の締め・評価する版の期間）に使う表。
特徴量側の扱いは [pretrained_encoders/design.md](../analysis/preliminary_analysis/pretrained_encoders/design.md) §11.4。

### 旧方式の問題
版番号の形（nova 型は `X.0.0`、swift は `2.Y.0`）で、全リリースの一覧から区切りを拾っていた。
- 版の付け方が途中で変わっている（nova：`2011.2` → `2015.1.0` → `12.0.0`）。`X.0.0` だけを拾うので、2015-10-15 より前が抜けていた。
- swift は 2014-07-07 より前（`1.x`）が抜けていた。2020 年以前は 1 サイクルに `2.Y.0` が 2〜3 個あった。
- 表の最後が 2025-10-01 で、それより後のサイクルがなかった（releases リポジトリが 2025-12-18 の状態）。
- 元の一覧 `releases_summary.csv` の日付は「その版が releases リポジトリに書き込まれた日」を git の履歴から取っている。
  austin・bexar はまとめて登録した日（2018-02-21）になっていた（公式は 2010-10-21・2011-02-03）。

### 新しい作り方：サイクルを単位にする
OpenStack の公式のサイクル一覧（`releases_repo/data/series_status.yaml`）から、サイクル名とリリース日を読む。
版番号を解釈しないので、メジャー・マイナー・パッチを見分ける必要がない。

| 列 | 中身 |
|---|---|
| `project` | nova など |
| `series` | サイクル名（`yoga` など） |
| `version` | その project がそのサイクルで最初に出した正式版（rc・b・`-eom`・`-eol` などは除く）。例：nova yoga → `25.0.0`、nova kilo → `2015.1.0`、swift yoga → `2.29.0`。まだ出ていないサイクルは空 |
| `release_date` | `series_status.yaml` の `initial-release`（公式のサイクルのリリース日）。6 件共通。**swift もこの日付**（swift の版の日付ではない） |
| `status` | `released`（そのサイクルで正式版が出ている）／`planned`（まだ出ていない。日付は予定日） |
| `yaml_url` | そのサイクルの deliverables ファイルの URL（従来どおり） |

- 対象のサイクル：その project の deliverables ファイルが最初に現れたサイクルから、一覧にある最新のサイクル（予定のものを含む）まで。
  例：cinder は folsom から、keystone は essex から。
- **予定であることは `status` 列で持つ**。サイクル名には印を付けない（分析の設定は名前で行を引くため、名前が変わると見つからなくなる）。
  図や表に出すときは `gazpacho (planned)` のように表示する。
- 分析の設定（`PROJECTS` の cutoff・versions）と出力先の名前は、従来どおり版番号（`25.0.0`・`2.29.0` など）を使う。
  版番号で行を引き、日付は公式の日付を使う。
- 確認済み（2026-10-02、nova・swift）：nova の各サイクルの最初の正式版の日付は、cactus 以降すべて公式の日付と一致する。
  swift の正式版は公式の日付より 2〜3 週間早い。

### OpenStack 以外の行
`release_tags_collector.py` が同じ CSV に qtbase・qtcreator・libreoffice の行を追記している。
いまの `MajorReleaseStorage.save()` は CSV 全体を上書きするので、**作り直すときは OpenStack 以外の行を残す**。
OpenStack 以外のプロジェクトの区切り（何を「メジャーリリース」とみなすか）は、プロジェクトの選定のときに決める。
区切りが公開されているプロジェクトだけを使う。

### 手順（コードで作り、手で編集しない）
```bash
# 1. releases リポジトリを最新にする（git pull 1 回）
python -m src.collectors.release_collector
# 2. 表を作り直す
python -m src.collectors.major_release_storage
```
- `release_collector.py` の `__main__` は保存先を `data/openstack` にしているが、releases リポジトリと表は
  `data/openstack_collected` にある。実装の際に `data/openstack_collected` に直す。
- 作り直したときは、project ごとの行数・最初と最後のサイクル・planned の行をログに出す。

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
