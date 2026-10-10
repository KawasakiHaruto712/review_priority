# Preprocessing モジュール

レビューコメントからキーワード抽出と前処理を行うモジュールです。checklistデータからレビューパターンを分析し、修正要求・修正確認キーワードを自動抽出します。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `review_comment_processor.py` | レビューコメントのキーワード抽出・分析機能 |

## 🔧 主要機能

### レビューコメント処理 (`review_comment_processor.py`)
- **キーワード抽出**: checklistデータから修正要求・修正確認のキーワードを抽出
- **N-gram生成**: 1〜10語のフレーズパターンを自動生成
- **ボット除外**: `src/utils/bot_detection.py` の判定でボットのコメントを除外（checklist.csv には投稿者の名前しかないため、当てられるのは名前の決まり（③）だけ）
- **テキスト前処理**: URL除去、記号除去、プロジェクト名除去
- **統計分析**: キーワードの出現頻度と精度による品質評価

## 📊 処理データ形式

### 入力データ (checklist.csv)
```csv
コメント,修正要求,修正確認
"Please fix the memory leak in line 42",1,0
"Looks good to me",0,1
"The implementation looks correct",0,1
"This needs to be changed",1,0
```

### 出力データ (review_keywords.json)
```json
{
  "修正要求": [
    "fix",
    "change",
    "needs",
    "should",
    "please fix"
  ],
  "修正確認": [
    "good",
    "looks good",
    "correct",
    "approve"
  ]
}
```

## 🚀 使用方法

### 基本的なキーワード抽出

```python
from src.preprocessing.review_comment_processor import extract_and_save_review_keywords
from src.config.path import DEFAULT_DATA_DIR
from src.utils.bot_detection import BotDetector, host_of

# 入力ファイルのパス設定
checklist_path = DEFAULT_DATA_DIR / "processed" / "checklist.csv"
output_path = DEFAULT_DATA_DIR / "processed" / "review_keywords.json"
bot_detector = BotDetector(host_of("nova"))  # checklist.csv は nova の Change
label_path = DEFAULT_DATA_DIR / "processed" / "review_label.json"

# キーワード抽出の実行
extract_and_save_review_keywords(
    checklist_path=checklist_path,
    output_keywords_path=output_path,
    bot_detector=bot_detector,
    review_label_path=label_path,
    min_comment_count=10,        # 最小出現回数
    min_precision_ratio=0.90,    # 最小精度（90%以上）
    ngram_min=1,                 # 1語のキーワードから
    ngram_max=10                 # 10語のフレーズまで
)
```

### パラメータの調整

```python
# より厳格な条件でキーワード抽出
extract_and_save_review_keywords(
    checklist_path,
    output_path,
    config_path,
    label_path,
    min_comment_count=20,     # 20回以上出現
    min_precision_ratio=0.95, # 95%以上の精度
    ngram_min=1,
    ngram_max=5               # 5語までのフレーズ
)

# より緩い条件でより多くのキーワードを抽出
extract_and_save_review_keywords(
    checklist_path,
    output_path,
    config_path,
    label_path,
    min_comment_count=5,      # 5回以上出現
    min_precision_ratio=0.80, # 80%以上の精度
    ngram_min=1,
    ngram_max=15              # 15語までのフレーズ
)
```

## 📝 テキスト前処理パイプライン

### 1. 基本的な清浄化
```python
# URL除去
processed_comment = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+|[^\s<>"]+\.[a-zA-Z]{2,}', " ", comment)

# パッチセット情報除去
processed_comment = re.sub(r"patch set \d+:", " ", processed_comment, flags=re.IGNORECASE)

# インラインコメント情報除去
processed_comment = re.sub(r"\(\d+\s*(?:inline\s+)?comments?\)", " ", processed_comment)

# 記号除去（文字、数字、+、-のみ残す）
processed_comment = re.sub(r"[^a-zA-Z'0-9\+-]+", " ", processed_comment)

# プロジェクト名除去
processed_comment = re.sub(r"\b(nova|neutron|cinder|horizon|keystone|swift|glance|openstack|ci)\b", " ", processed_comment)
```

### 2. N-gram生成
```python
def _generate_ngrams(words: list[str], n_min: int, n_max: int) -> list[str]:
    """単語のリストからN-gramを生成"""
    ngrams = []
    num_words = len(words)
    for n in range(n_min, n_max + 1):
        for i in range(num_words - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)
    return ngrams
```

### 3. 包含関係の除去
```python
def summarize_keywords_by_inclusion(keywords: list[str]) -> list[str]:
    """包含関係にある冗長なキーワードを削除"""
    # 例: ['good', 'looks good', 'looks good to me'] -> ['good']
    sorted_keywords = sorted(keywords, key=len, reverse=False)
    kept_keywords_set = set()
    
    for current_keyword in sorted_keywords:
        is_redundant = False
        for kept_keyword in kept_keywords_set:
            if kept_keyword in current_keyword and current_keyword != kept_keyword:
                is_redundant = True
                break
        
        if not is_redundant:
            kept_keywords_set.add(current_keyword)
            
    return sorted(list(kept_keywords_set))
```

## 🔧 設定ファイル

### bot_accounts.csv
ボットの一覧は `src/config/bot_accounts.csv`（説明は `src/config/README.md`）。

### review_label.json
```json
{
  "category1": {
    "positive": ["approve", "lgtm", "good"],
    "negative": ["reject", "fix", "change"]
  }
}
```

## 📊 分析結果の例

### 抽出されたキーワード統計
```
修正要求キーワード: 156個
├── 1語: 45個 (fix, change, needs, should, ...)
├── 2語: 38個 (please fix, needs to, should be, ...)
├── 3語: 28個 (needs to be, should be changed, ...)
└── 4語以上: 45個 (this needs to be fixed, ...)

修正確認キーワード: 89個
├── 1語: 23個 (good, approve, ok, ...)
├── 2語: 31個 (looks good, sounds good, ...)
├── 3語: 22個 (looks good to, ...)
└── 4語以上: 13個 (looks good to me, ...)
```

### 品質メトリクス
```
キーワード品質:
├── 平均精度: 92.4%
├── 最小出現回数: 10回
├── 最高精度キーワード: "fix" (98.2%)
└── 最低精度キーワード: "please" (90.1%)
```

## ⚡ パフォーマンス最適化

### メモリ効率化
```python
# セットを使用してコメント内重複を避ける
for phrase in set(phrases): 
    vocab_counts[phrase]['total'] += 1
```

### 処理時間短縮
```python
# 長いラベルから順にソートして効率的なパターンマッチング
all_labels.sort(key=len, reverse=True)
```

## 🎯 実用的な応用

### 1. リアルタイムコメント分類
```python
def classify_comment(comment_text, keywords):
    """コメントを修正要求/確認に分類"""
    for request_keyword in keywords['修正要求']:
        if request_keyword in comment_text.lower():
            return "修正要求"
    
    for confirm_keyword in keywords['修正確認']:
        if confirm_keyword in comment_text.lower():
            return "修正確認"
    
    return "その他"
```

### 2. 自動優先順位付け
```python
def calculate_urgency_score(comment_text, keywords):
    """コメントの緊急度スコアを計算"""
    urgency_keywords = ['urgent', 'critical', 'fix', 'error', 'bug']
    score = 0
    for keyword in urgency_keywords:
        if keyword in comment_text.lower():
            score += 1
    return min(score / len(urgency_keywords), 1.0)
```

## ⚠️ 注意事項

1. **NLTK依存の除去**: 現在はNLTKに依存せず、シンプルなテキスト処理を使用
2. **言語固有**: 英語のレビューコメントに特化した処理
3. **データ品質**: checklistデータの品質が抽出キーワードの品質に直結
4. **精度設定**: `min_precision_ratio`が低すぎるとノイズが混入

## 🔧 カスタマイズ

### 新しいプロジェクト名の除去
```python
# プロジェクト名リストの拡張
project_names = r"\b(nova|neutron|cinder|horizon|keystone|swift|glance|openstack|ci|your_project)\b"
processed_comment = re.sub(project_names, " ", processed_comment)
```

### カスタム前処理パターン
```python
# 特定パターンの除去
custom_patterns = [
    r"patch set \d+:",           # パッチセット情報
    r"\(\d+\s*comments?\)",      # コメント数情報
    r"build \w+ \w+",            # ビルド情報
]

for pattern in custom_patterns:
    processed_comment = re.sub(pattern, " ", processed_comment, flags=re.IGNORECASE)
```
