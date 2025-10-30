# Classification モジュール

コミットメッセージの分類と分析を行うモジュールです。

## 概要

このモジュールは、OpenStackプロジェクトのChangeデータを分析し、コミットメッセージをゼロショット分類することで、レビュープロセスの特性を明らかにします。

## ファイル構成

| ファイル名 | 説明 |
|-----------|------|
| `commit_classifier.py` | コミットメッセージの分類とレビュー時間分析 |

## commit_classifier.py

### 機能

1. **ゼロショット分類**
   - Hugging Faceの事前学習済みモデル `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` を使用
   - コミットメッセージを10種類のラベルに自動分類
   
2. **レビュー時間分析**
   - Change作成から最初のレビューまでの日数を計算
   - ボット（Jenkins, Zuul等）によるメッセージを除外
   
3. **統計分析**
   - ラベルごとのレビュー時間分布を集計
   - 時間区間別の件数と割合を算出

### ラベル定義

| ラベル | 説明 |
|-------|------|
| `feat` | コードベースに新機能を導入する変更 |
| `fix` | バグや不具合を修正する変更 |
| `refactor` | バグ修正や機能追加を伴わない、コードの構造を改善する変更 |
| `docs` | ドキュメントのみを修正する変更 |
| `style` | コードの意味に影響を与えない、フォーマットや可読性に関する変更 |
| `test` | テストの追加や既存テストの修正 |
| `perf` | パフォーマンスを向上させるコード変更 |
| `ci` | CI（継続的インテグレーション）の設定ファイルやスクリプトに関する変更 |
| `build` | ビルドシステムや外部依存関係に影響を与える変更 |
| `chore` | 上記のいずれにも当てはまらない、その他の雑多なタスク |

### 使用方法

#### 基本的な使用

```python
from src.classification.commit_classifier import classify_and_analyze

# デフォルト設定で実行（全プロジェクト、constants.pyの期間）
results = classify_and_analyze()

# 特定のプロジェクトのみ分析
results = classify_and_analyze(projects=['nova', 'neutron'])

# カスタム期間で実行
results = classify_and_analyze(
    start_date='2024-01-01',
    end_date='2024-03-31'
)
```

#### コマンドライン実行

```bash
# Docker環境で実行
python src/classification/commit_classifier.py
```

#### 個別の分類器使用

```python
from src.classification.commit_classifier import CommitMessageClassifier

# 分類器を初期化
classifier = CommitMessageClassifier()

# 単一のメッセージを分類
result = classifier.classify("Add new feature for volume encryption")
print(f"Label: {result['label']}, Score: {result['score']:.2f}")

# 複数のメッセージを一括分類
messages = [
    "Fix bug in network allocation",
    "Update documentation for API endpoints",
    "Refactor database connection pooling"
]
results = classifier.classify_batch(messages, batch_size=8)
```

### 出力ファイル

実行すると、以下のファイルが `data/commit_classification/` ディレクトリに生成されます:

#### 1. 詳細データ (`classification_detail_YYYYMMDD_YYYYMMDD.csv`)

全Changeの詳細情報を含むCSVファイル。

| カラム名 | 説明 |
|---------|------|
| `change_number` | Change番号 |
| `project` | プロジェクト名 |
| `subject` | コミットサブジェクト |
| `commit_message` | コミットメッセージ全文 |
| `created` | Change作成日時 |
| `updated` | 最終更新日時 |
| `merged` | マージ日時 |
| `label` | 分類されたラベル |
| `label_score` | 分類の信頼度スコア (0.0-1.0) |
| `time_to_first_review` | 最初のレビューまでの日数 |
| `time_bin` | 時間区間 (~1日, 1~2日, etc.) |

#### 2. サマリー (`classification_summary_YYYYMMDD_YYYYMMDD.csv`)

ラベルと時間区間ごとの集計結果。

| カラム名 | 説明 |
|---------|------|
| `label` | ラベル名 |
| `time_bin` | 時間区間 |
| `count` | 該当件数 |
| `ratio` | 割合 (0.0-1.0) |

#### 3. ピボットテーブル (`classification_pivot_YYYYMMDD_YYYYMMDD.csv`)

ラベル別・時間区間別の割合を一覧表示。

```
label      total_count  ~1日  1~2日  2~3日  3~4日  4~5日  5日~
feat       1234         0.35  0.25   0.15   0.10   0.08   0.07
fix        2345         0.42  0.28   0.14   0.08   0.05   0.03
refactor   567          0.30  0.22   0.18   0.12   0.10   0.08
...
```

### 依存パッケージ

```
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
```

requirements.txtに以下を追加してインストール:

```bash
pip install transformers torch
```

### パフォーマンス

- **処理速度**: バッチサイズ8で、約100件/分（CPUの場合）
- **メモリ使用量**: モデルロード時に約1.5GB、推論時に追加で500MB程度
- **GPU使用**: `CommitMessageClassifier`の初期化時に`device=0`を指定することでGPUを使用可能

```python
# GPU使用例
classifier = CommitMessageClassifier()
# classifier.classifier.device = 0  # 手動でGPU設定も可能
```

### 注意事項

1. **初回実行時**: モデルのダウンロードに時間がかかります（約2GB）
2. **ボットフィルタリング**: Jenkins, Zuul, Elasticrecheckのメッセージは自動的に除外されます
3. **日付フィルタリング**: `constants.py`の`START_DATE`と`END_DATE`を使用します
4. **エラーハンドリング**: 分類に失敗した場合は`chore`ラベルが割り当てられます

### カスタマイズ

#### ラベルの追加・変更

`CommitMessageClassifier.LABELS`を編集してラベルを変更できます:

```python
LABELS = {
    'feat': '新機能の追加',
    'fix': 'バグ修正',
    'custom_label': 'カスタムラベルの説明',
    # ...
}
```

#### 時間区間の変更

`classify_and_analyze`関数内の`bins`と`labels`を編集:

```python
bins = [0, 0.5, 1, 2, 5, float('inf')]
labels = ['~12h', '12h~1d', '1~2d', '2~5d', '5d~']
```

#### ボット名の追加

`calculate_time_to_first_review`の`bot_names`引数を指定:

```python
bot_names = ['jenkins', 'zuul', 'elasticrecheck', 'custom_bot']
```

## 使用例

### 例1: 特定プロジェクトの分析

```python
from src.classification.commit_classifier import classify_and_analyze

# NovaとNeutronのみを分析
results = classify_and_analyze(
    projects=['nova', 'neutron'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# ピボットテーブルを表示
print(results['pivot'])
```

### 例2: カスタム出力ディレクトリ

```python
from pathlib import Path
from src.classification.commit_classifier import classify_and_analyze

# カスタムディレクトリに保存
output_dir = Path('/path/to/custom/output')
results = classify_and_analyze(output_dir=output_dir)
```

### 例3: 分類結果の詳細分析

```python
from src.classification.commit_classifier import classify_and_analyze

results = classify_and_analyze()

# 詳細データを取得
detail_df = results['detail']

# 信頼度スコアの分布を確認
print(detail_df['label_score'].describe())

# 各ラベルの平均レビュー時間を計算
avg_review_time = detail_df.groupby('label')['time_to_first_review'].mean()
print(avg_review_time.sort_values())
```

## トラブルシューティング

### モデルのダウンロードに失敗する

```bash
# プロキシ設定が必要な場合
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### メモリ不足エラー

バッチサイズを小さくしてください:

```python
results = classifier.classify_batch(messages, batch_size=4)
```

### GPU使用時のエラー

CUDA対応のPyTorchをインストール:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```