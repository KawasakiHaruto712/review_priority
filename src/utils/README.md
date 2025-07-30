# Utils モジュール

プロジェクト全体で使用される共通ユーティリティ機能を提供するモジュールです。定数定義、言語識別、共通処理関数などが含まれます。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `constants.py` | プロジェクト全体で使用する定数定義 |
| `lang_identifiyer.py` | ファイル拡張子から言語を識別する機能 |

## 🔧 主要機能

### 定数管理 (`constants.py`)
- **OpenStackコンポーネント**: 分析対象プロジェクトの定義
- **日付範囲**: データ収集・分析期間の設定
- **設定値**: システム全体で使用する共通値

### 言語識別 (`lang_identifiyer.py`)
- **ファイル拡張子判定**: 拡張子から言語を自動識別
- **マルチ言語対応**: 主要プログラミング言語に対応
- **エラーハンドリング**: 未対応拡張子の適切な処理

## 📊 定数定義

### OpenStackコンポーネント
```python
OPENSTACK_CORE_COMPONENTS = [
    "nova",        # コンピュート（仮想マシン管理）
    "neutron",     # ネットワーキング（仮想ネットワーク）
    "swift",       # オブジェクトストレージ（ファイル保存）
    "cinder",      # ブロックストレージ（ディスク管理）
    "keystone",    # 認証（ユーザー・権限管理）
    "glance",      # イメージサービス（OS画像管理）
]
```

### 分析期間設定
```python
START_DATE = "2015-01-01"    # 分析開始日
END_DATE = "2024-12-31"      # 分析終了日
LABELLED_CHANGE_COUNT = 383  # 手動ラベル付けデータ数
```

## 🔧 言語識別機能

### サポート言語
```python
lang_map = {
    "ts": "JavaScript",      # TypeScript
    "js": "JavaScript",      # JavaScript
    "py": "Python",          # Python
    "java": "Java",          # Java
    "c": "CPP",              # C/C++
    "php": "PHP",            # PHP
    "dart": "Dart",          # Dart
    "r": "R",                # R
}
```

### 使用例
```python
from src.utils.lang_identifiyer import identify_lang_from_file

# ファイルから言語を識別
language = identify_lang_from_file("main.py")  # "Python"
language = identify_lang_from_file("app.js")   # "JavaScript"
language = identify_lang_from_file("test.java") # "Java"
```

## 🚀 使用方法

### 定数の利用

```python
from src.utils.constants import OPENSTACK_CORE_COMPONENTS, START_DATE, END_DATE

# 全プロジェクトの処理
for project in OPENSTACK_CORE_COMPONENTS:
    print(f"Processing {project}...")
    process_project_data(project, START_DATE, END_DATE)

# 期間フィルタリング
filtered_data = data[
    (data['date'] >= START_DATE) & 
    (data['date'] <= END_DATE)
]
```

### 言語識別の活用

```python
from src.utils.lang_identifiyer import identify_lang_from_file

def analyze_code_files(file_paths):
    language_stats = {}
    
    for file_path in file_paths:
        try:
            language = identify_lang_from_file(file_path)
            language_stats[language] = language_stats.get(language, 0) + 1
        except ValueError as e:
            print(f"Warning: {e}")
    
    return language_stats

# 使用例
files = ["src/main.py", "tests/test.py", "config.js", "README.md"]
stats = analyze_code_files(files)
# Output: {"Python": 2, "JavaScript": 1}
```

## 📈 実用的な応用

### 1. プロジェクト別統計

```python
def get_project_statistics():
    stats = {}
    for project in OPENSTACK_CORE_COMPONENTS:
        stats[project] = {
            'total_changes': count_changes(project),
            'date_range': f"{START_DATE} to {END_DATE}",
            'status': 'active'
        }
    return stats
```

### 2. 言語別分析

```python
def analyze_change_by_language(changes_df):
    language_metrics = {}
    
    for _, change in changes_df.iterrows():
        files = change.get('files', [])
        for file_info in files:
            try:
                lang = identify_lang_from_file(file_info['path'])
                if lang not in language_metrics:
                    language_metrics[lang] = {
                        'changes': 0,
                        'lines_added': 0,
                        'lines_deleted': 0
                    }
                
                language_metrics[lang]['changes'] += 1
                language_metrics[lang]['lines_added'] += file_info.get('lines_added', 0)
                language_metrics[lang]['lines_deleted'] += file_info.get('lines_deleted', 0)
                
            except ValueError:
                continue  # 未対応拡張子はスキップ
    
    return language_metrics
```

### 3. 設定の動的管理

```python
def get_analysis_config():
    return {
        'projects': OPENSTACK_CORE_COMPONENTS,
        'date_range': {
            'start': START_DATE,
            'end': END_DATE
        },
        'sample_size': LABELLED_CHANGE_COUNT,
        'supported_languages': list(lang_map.values())
    }
```

## 🔧 拡張方法

### 新しい定数の追加

```python
# constants.py に追加
OPENSTACK_OPTIONAL_COMPONENTS = [
    "horizon",     # ダッシュボード
    "ceilometer",  # 監視・測定
    "heat",        # オーケストレーション
]

# 新しい設定値
DEFAULT_BATCH_SIZE = 100
MAX_RETRY_COUNT = 3
TIMEOUT_SECONDS = 30
```

### 言語サポートの拡張

```python
# lang_identifiyer.py の拡張
extended_lang_map = {
    **lang_map,  # 既存のマッピング
    "go": "Go",
    "rs": "Rust", 
    "kt": "Kotlin",
    "swift": "Swift",
    "rb": "Ruby",
    "cpp": "CPP",
    "h": "CPP",
    "cs": "CSharp",
    "scala": "Scala"
}
```

### 環境別設定

```python
import os

# 環境変数による動的設定
def get_environment_config():
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return {
            'start_date': "2020-01-01",
            'batch_size': 1000,
            'timeout': 60
        }
    else:  # development
        return {
            'start_date': "2024-01-01", 
            'batch_size': 100,
            'timeout': 30
        }
```

## 📊 統計情報の例

### プロジェクト別データ量
```
OpenStackコンポーネント別統計:
├── nova: 45,234 changes (25.1%)
├── neutron: 38,567 changes (21.4%)  
├── swift: 23,891 changes (13.3%)
├── cinder: 28,445 changes (15.8%)
├── keystone: 22,156 changes (12.3%)
└── glance: 21,789 changes (12.1%)
```

### 言語別分布
```
プログラミング言語別分布:
├── Python: 78,234 files (72.5%)
├── JavaScript: 15,678 files (14.5%)
├── Java: 8,923 files (8.3%)
├── CPP: 3,456 files (3.2%)
└── Other: 1,621 files (1.5%)
```

## ⚡ パフォーマンス

### 計算量
- **定数アクセス**: O(1) - 瞬時
- **言語識別**: O(1) - ファイル名のみで判定
- **プロジェクト一覧**: O(n) - リスト長に比例

### メモリ使用量
- **定数**: 数KB（文字列のみ）
- **言語マップ**: 1KB未満
- **実行時オーバーヘッド**: ほぼゼロ

## ⚠️ 注意事項

1. **拡張子の限界**: ファイル内容は確認しない
2. **設定の一貫性**: 定数変更時の影響範囲に注意
3. **言語の曖昧性**: 同一拡張子で複数言語の場合
4. **エラーハンドリング**: 未対応拡張子への適切な対応

## 🔍 デバッグ・テスト

### 定数の検証
```python
def validate_constants():
    assert len(OPENSTACK_CORE_COMPONENTS) == 6
    assert START_DATE < END_DATE
    assert LABELLED_CHANGE_COUNT > 0
    print("✓ All constants are valid")
```

### 言語識別のテスト
```python
def test_language_identification():
    test_cases = [
        ("test.py", "Python"),
        ("app.js", "JavaScript"), 
        ("Main.java", "Java"),
        ("script.unknown", ValueError)
    ]
    
    for file_path, expected in test_cases:
        try:
            result = identify_lang_from_file(file_path)
            assert result == expected, f"Expected {expected}, got {result}"
        except ValueError:
            assert expected == ValueError
    
    print("✓ Language identification tests passed")
```
