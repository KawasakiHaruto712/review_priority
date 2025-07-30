# Review Priority - ソースコードディレクトリ

このディレクトリには、Review Priority プロジェクトのメインソースコードが含まれています。OpenStackプロジェクトのコードレビューデータを収集・分析し、機械学習（IRL: 逆強化学習）を用いてレビュー優先順位を学習・予測するシステムです。

## 🏗️ アーキテクチャ概要

```
src/
├── collectors/     # データ収集
├── preprocessing/ # データ前処理
├── features/      # 特徴量抽出
├── learning/      # 機械学習・IRL
├── config/        # 設定管理
└── utils/         # 共通ユーティリティ
```

## 📁 モジュール構成

### 📊 [collectors/](collectors/README.md) - データ収集
OpenStackのGerritシステムとGitHubから必要なデータを収集
- **OpenStack Gerrit**: コードレビューデータの取得
- **リリース情報**: プロジェクトリリース履歴の収集
- **メタデータ**: プロジェクト統計情報の生成

### 🔧 [preprocessing/](preprocessing/README.md) - データ前処理
収集したデータの前処理とキーワード抽出
- **レビューコメント処理**: コメントのテキスト前処理
- **キーワード抽出**: 修正要求・確認の自動分類
- **N-gram生成**: フレーズレベルでの分析

### 📈 [features/](features/README.md) - 特徴量抽出
機械学習モデル用の特徴量計算
- **Bug Metrics**: バグ修正関連の特徴量
- **Change Metrics**: コード変更規模・複雑度
- **Developer Metrics**: 開発者の経験・専門性
- **Project Metrics**: プロジェクト全体の特性
- **Refactoring Metrics**: リファクタリング検出
- **Review Metrics**: レビュープロセスの特徴

### 🤖 [learning/](learning/README.md) - 機械学習・IRL
逆強化学習によるレビュー優先順位の学習
- **IRL Models**: Maximum Entropy IRLの実装
- **時系列分析**: 時間経過に伴う優先順位変化
- **モデル評価**: 学習結果の検証と可視化

### ⚙️ [config/](config/README.md) - 設定管理
プロジェクト設定とパス管理
- **パス設定**: データディレクトリの管理
- **設定ファイル**: Gerrit接続設定等

### 🛠️ [utils/](utils/README.md) - 共通ユーティリティ
プロジェクト全体で使用される共通機能
- **定数定義**: プロジェクト共通の定数
- **言語識別**: プログラミング言語の判定

## 🔄 データフロー

```
1. データ収集 (collectors/)
   ↓
2. データ前処理 (preprocessing/)
   ↓
3. 特徴量抽出 (features/)
   ↓
4. 機械学習 (learning/)
   ↓
5. 優先順位予測
```

### 詳細なワークフロー

1. **収集フェーズ**
   - `collectors.openstack`: Gerritからchange/commit情報を取得
   - `collectors.release_collector`: リリース情報を収集
   - 結果: `data/openstack/` にJSONファイルとして保存

2. **前処理フェーズ**
   - `preprocessing.review_comment_processor`: コメントからキーワード抽出
   - 結果: `data/processed/review_keywords.json` 等を生成

3. **特徴量抽出フェーズ**
   - `features.*_metrics`: 各種特徴量を計算
   - 結果: 機械学習用のDataFrame生成

4. **学習・予測フェーズ**
   - `learning.irl_models`: IRLモデルでの学習・予測
   - 結果: `data/results/` に学習結果とモデルを保存

## 🎯 主要機能

### データ収集機能
- **自動データ取得**: Gerrit APIを使用した効率的な収集
- **増分更新**: 新しいデータのみの取得
- **エラー処理**: ネットワークエラー・API制限への対応
- **進捗表示**: 大量データ収集時の進捗可視化

### 特徴量エンジニアリング
- **多次元特徴量**: 16種類の特徴量を自動計算
- **時系列特徴量**: 時間的変化を考慮した特徴量
- **テキスト特徴量**: コメント・コミットメッセージの分析
- **メタ特徴量**: プロジェクト全体の特性を反映

### 機械学習・IRL
- **Maximum Entropy IRL**: 報酬関数の学習
- **時系列学習**: 時間窓での段階的学習
- **モデル評価**: 学習曲線・特徴量重要度の可視化
- **予測機能**: 新しいレビューの優先順位予測

## 💡 使用例

### 基本的な使用方法

```python
# 1. データ収集
from src.collectors.openstack import OpenStackGerritCollector
collector = OpenStackGerritCollector()
collector.collect_project_data('nova', start_date='2024-01-01')

# 2. 特徴量抽出
from src.features.change_metrics import calculate_change_features
features = calculate_change_features(change_data)

# 3. IRL学習
from src.learning.irl_models import run_temporal_irl_analysis
results = run_temporal_irl_analysis(
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### プロジェクト別分析

```python
# OpenStackプロジェクト別の分析
projects = ['nova', 'neutron', 'cinder', 'keystone']
for project in projects:
    results = run_temporal_irl_analysis(
        start_date='2024-01-01',
        end_date='2024-12-31',
        projects=[project]
    )
    print(f"{project}: 学習完了")
```

## 🔧 設定・環境

### 必要な設定ファイル
- `src/config/gerrymanderconfig.ini`: Gerrit接続設定
- `data/processed/review_label.json`: レビューラベル定義

### 環境変数
```bash
# Gerrit認証（オプション）
export GERRIT_USERNAME="your_username"
export GERRIT_PASSWORD="your_password"
```

### データディレクトリ
```
data/
├── openstack/          # 収集された生データ
│   ├── {project}/
│   │   ├── changes/    # Change情報
│   │   └── commits/    # Commit情報
├── processed/          # 前処理済みデータ
│   ├── checklist.csv
│   └── review_keywords.json
└── results/            # 学習結果
    ├── irl_analysis_*.json
    └── irl_model_*.pkl
```

## 📊 出力・結果

### 学習結果
- **特徴量重要度**: どの特徴量が優先順位に重要か
- **学習統計**: 収束状況、最終目的関数値
- **時系列変化**: 時間経過による優先順位基準の変化

### 予測結果
- **優先順位スコア**: 各レビューの数値的優先度
- **ランキング**: 優先順位でソートされたレビューリスト
- **信頼度**: 予測の確信度

## 🧪 テスト

各モジュールには包括的なテストスイートがあります：
- **ユニットテスト**: 個別機能のテスト
- **統合テスト**: モジュール間連携のテスト
- **パフォーマンステスト**: 大量データでの性能テスト

詳細は [tests/README.md](../tests/README.md) を参照してください。

## 📝 開発ガイドライン

### コーディング規約
- **PEP 8**: Python標準のコーディングスタイル
- **型ヒント**: 可能な限り型注釈を使用
- **ドキュメント**: 全ての公開関数にdocstring

### 新機能追加
1. **設計**: 機能設計と影響範囲の検討
2. **実装**: テスト駆動開発（TDD）
3. **テスト**: ユニット・統合テストの作成
4. **ドキュメント**: README・コメントの更新

### パフォーマンス考慮
- **メモリ効率**: 大量データ処理時の配慮
- **並列処理**: 可能な箇所での並列化
- **キャッシュ**: 計算結果の適切なキャッシュ

## 🔗 関連リソース

- **プロジェクトルート**: [../README.md](../README.md)
- **テストスイート**: [../tests/README.md](../tests/README.md)
- **設定例**: [../config/](../config/)
- **サンプルデータ**: [../data/](../data/)

## 📧 サポート・貢献

バグ報告、機能リクエスト、プルリクエストを歓迎します。  
詳細はプロジェクトルートの README.md を参照してください。
