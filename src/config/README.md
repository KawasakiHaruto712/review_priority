# Config モジュール

アプリケーションの設定管理機能を提供するモジュールです。パス設定、設定ファイルの読み込み、各種定数の管理を行います。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `path.py` | ファイルパスとディレクトリの設定管理 |
| `release_constants.py` | リリース分析用の定数定義 |
| `gerrymanderconfig.ini` | Gerrit接続用の設定ファイル（ボット名一覧 `[organization] bots` を含む） |
| `third_party_ci_accounts.csv` | サードパーティCIアカウント一覧（`name,email`）。ボット判定の補完用 |
| `extra_bots.txt` | 追加のボット/自動アカウント名（`zuul` / `jenkins` 等。1行1名） |

## 🔧 主要機能

### Path管理 (`path.py`)
- **ディレクトリパス定義**: データ保存先、設定ファイル場所などの統一管理
- **相対パス解決**: プロジェクトルートからの相対パス計算
- **クロスプラットフォーム対応**: Windows/Linux/macOS対応のパス処理

### 設定ファイル管理 (`gerrymanderconfig.ini`)
- **Gerrit接続設定**: APIエンドポイント、認証情報
- **ボット名設定**: 自動化ツールの識別用設定（`[organization]` セクションの `bots`）
- **タイムアウト設定**: API接続のタイムアウト値

### ボット/自動アカウント一覧（3つのソース）
レビューコメントから「人間のレビュー」を抽出する際に除外する、ボット・CI などの
自動アカウントを定義します。利用側（例: `priority_distribution` の `data_loader.load_bot_names`）は
次の **3 つの和集合** を用い、`name` / `username` / `email` を**大文字小文字を区別せず**照合します。

| ソース | 内容 | 由来 |
|--------|------|------|
| `gerrymanderconfig.ini` の `[organization] bots` | CI/bot の username（カンマ区切り） | OpenStack Wiki「GerrymanderConfig」由来 |
| `third_party_ci_accounts.csv` | サードパーティ CI の `name,email`（ヘッダ付き） | サードパーティ CI アカウント一覧から抽出 |
| `extra_bots.txt` | 上記未登録の補完（`zuul` / `jenkins` 等） | 本リポジトリで追加 |

> 注: これらは当時の表記揺れや新規 CI を完全には網羅しません。取りこぼしは
> `third_party_ci_accounts.csv` / `extra_bots.txt` への追記で補えます。

## 📋 設定項目

### パス設定 (`path.py`)
```python
# 主要なディレクトリパス
DEFAULT_DATA_DIR: Path        # データ保存ディレクトリ
DEFAULT_CONFIG: Path          # 設定ファイルディレクトリ
DEFAULT_RESULTS_DIR: Path     # 結果出力ディレクトリ
```

### Gerrit設定 (`gerrymanderconfig.ini`)
```ini
[gerrit]
host = review.opendev.org
port = 443
scheme = https
bot_name = jenkins, zuul, ci-bot

[api]
timeout = 30
max_retries = 3
```

## 🚀 使用方法

### パス設定の利用

```python
from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG

# データファイルの保存
data_file = DEFAULT_DATA_DIR / "openstack" / "nova.json"

# 設定ファイルの読み込み
config_file = DEFAULT_CONFIG / "gerrymanderconfig.ini"
```

### 設定ファイルの読み込み

```python
import configparser
from src.config.path import DEFAULT_CONFIG

config = configparser.ConfigParser()
config.read(DEFAULT_CONFIG / "gerrymanderconfig.ini")

# 設定値の取得
host = config.get('gerrit', 'host')
bot_names = config.get('gerrit', 'bot_name').split(', ')
```

## 📁 ディレクトリ構造

```
data/                          # DEFAULT_DATA_DIR
├── openstack/                 # OpenStackデータ
│   ├── component_summary.csv
│   ├── releases_summary.csv
│   └── [project]/
│       ├── changes/
│       └── commits/
├── processed/                 # 処理済みデータ
└── results/                   # 分析結果

src/config/                        # DEFAULT_CONFIG
├── path.py                        # パス定義
├── release_constants.py           # リリース分析用の定数
├── gerrymanderconfig.ini          # Gerrit設定 + ボット名（[organization] bots）
├── third_party_ci_accounts.csv    # サードパーティCIアカウント一覧（name,email）
└── extra_bots.txt                 # 追加のボット/自動アカウント名
```

## ⚙️ 環境変数

以下の環境変数で設定をオーバーライド可能：

```bash
# データディレクトリの変更
export REVIEW_PRIORITY_DATA_DIR="/custom/data/path"

# 設定ディレクトリの変更
export REVIEW_PRIORITY_CONFIG_DIR="/custom/config/path"
```

## 🔧 カスタマイズ

### 新しいパスの追加

```python
# path.py に追加
CUSTOM_OUTPUT_DIR = DEFAULT_DATA_DIR / "custom_output"
```

### 設定セクションの追加

```ini
# gerrymanderconfig.ini に追加
[custom]
feature_flag = true
batch_size = 100
```

## 📝 設定ファイルの例

### 開発環境用設定
```ini
[gerrit]
host = review.opendev.org
port = 443
scheme = https
bot_name = jenkins, zuul

[api]
timeout = 30
max_retries = 3
batch_size = 50

[logging]
level = DEBUG
file = openstack_collector.log
```

### 本番環境用設定
```ini
[gerrit]
host = review.opendev.org
port = 443
scheme = https
bot_name = jenkins, zuul, ci-bot, gate-bot

[api]
timeout = 60
max_retries = 5
batch_size = 100

[logging]
level = INFO
file = production.log
```

## ⚠️ 注意事項

1. **設定ファイルの権限**: 認証情報を含む場合は適切な権限設定
2. **パスの存在確認**: 指定されたパスが存在しない場合の自動作成
3. **設定の検証**: 不正な設定値に対するエラーハンドリング
4. **環境依存**: OS固有のパス区切り文字への対応
