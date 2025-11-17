# Review Priority 📊

OpenStackプロジェクトのコードレビューデータを分析し、**逆強化学習（IRL: Inverse Reinforcement Learning）** を用いてレビュー優先順位を自動学習・予測するシステムです。

## 🎯 概要

このシステムは、OpenStackのGerritレビューシステムから大量のレビューデータを収集し、機械学習によってレビューの重要度や優先順位を学習します。開発チームがより効率的にレビューリソースを配分できるよう支援します。

### 主な機能
- 📥 **自動データ収集**: OpenStack Gerrit APIからのレビューデータ取得
- 🔄 **データ前処理**: レビューコメントの分析とキーワード抽出
- 📊 **特徴量抽出**: 16種類の多次元特徴量を自動計算
- 🤖 **IRL学習**: Maximum Entropy IRLによる優先順位学習
- 📈 **時系列分析**: 時間経過に伴う優先順位基準の変化を分析
- 🎯 **優先順位予測**: 新しいレビューに対する優先度予測

## 🚀 クイックスタート

### 前提条件
- Python 3.12+
- Git
- 十分なディスク容量（OpenStackデータは数GB）

### インストール
```bash
# リポジトリのクローン
git clone <repository-url>
cd review_priority

# 依存関係のインストール
pip install -e .

# テスト用依存関係（オプション）
pip install -e .[test]
```

### 基本的な使用方法
```bash
# システム状態の確認
python main.py status

# 全工程の実行（データ収集→前処理→学習・分析）
python main.py full

# 特定プロジェクトのみ分析
python main.py analyze --project nova

# 個別工程の実行
python main.py collect      # データ収集のみ
python main.py preprocess   # 前処理のみ
python main.py train        # 学習のみ
```

## 📁 プロジェクト構成

```
review_priority/
├── src/                    # メインソースコード
│   ├── collectors/         # データ収集モジュール
│   ├── preprocessing/      # データ前処理
│   ├── features/          # 特徴量抽出
│   ├── learning/          # 機械学習・IRL
│   ├── release_impact/    # リリース影響分析
│   ├── config/            # 設定管理
│   └── utils/             # 共通ユーティリティ
├── tests/                 # テストスイート
├── data/                  # データディレクトリ
│   ├── openstack/         # 収集された生データ
│   ├── processed/         # 前処理済みデータ
│   ├── results/           # 分析結果
│   └── release_impact/    # リリース影響分析結果
├── main.py               # メインエントリーポイント
└── README.md             # このファイル
```

## 🔄 ワークフロー

### 1. データ収集（Collectors）
OpenStack Gerritシステムから以下のデータを収集：
- **Change情報**: レビュー対象の変更内容
- **Commit情報**: コミット履歴とメタデータ
- **レビューコメント**: 人間のレビュアーによるフィードバック
- **プロジェクトメタデータ**: 統計情報とリリース履歴

### 2. データ前処理（Preprocessing）
収集したデータの前処理と整理：
- **コメント分析**: 修正要求・確認コメントの自動分類
- **キーワード抽出**: N-gramベースのフレーズ抽出
- **ノイズ除去**: ボットコメントやシステムメッセージの除外

### 3. 特徴量抽出（Features）
機械学習用の16種類の特徴量を計算：
- **Bug Metrics (1個)**: バグ修正の確信度
- **Change Metrics (5個)**: 追加行数、削除行数、変更ファイル数、経過時間、リビジョン数
- **Test Metrics (1個)**: テストコード存在確認
- **Developer Metrics (4個)**: 過去報告数、最近報告数、マージ率、最近マージ率
- **Project Metrics (3個)**: リリースまでの日数、オープンチケット数、期間内レビュー行数
- **Refactoring Metrics (1個)**: リファクタリング確信度
- **Review Metrics (1個)**: 未完了修正要求数

### 4. 機械学習（Learning）
逆強化学習による優先順位学習：
- **Maximum Entropy IRL**: 報酬関数の学習
- **時系列分析**: 時間窓での段階的学習
- **モデル評価**: 収束性と特徴量重要度の分析

### 5. リリース影響分析（Release Impact）
リリース前後でのメトリクス変化を統計的に分析：
- **期間比較**: リリース直後（early）vs リリース直前（late）
- **レビュー状態比較**: レビュー済み vs 未レビュー
- **統計検定**: Mann-Whitney U検定による有意差検定
- **可視化**: ボックスプロット、ヒートマップの自動生成

```bash
# リリース影響分析の実行
python -m src.release_impact.metrics_comparator

# または特定プロジェクトのテスト
python src/release_impact/test_release_impact.py
```

## 📊 出力・結果

### 学習結果ファイル
- `data/results/irl_analysis_YYYYMMDD_YYYYMMDD.json`: 学習統計と特徴量重要度
- `data/results/irl_model_YYYYMMDD_YYYYMMDD.pkl`: 学習済みモデル

### リリース影響分析結果
各リリースペアについて以下のファイルが生成されます：
- `data/release_impact/{project}_{release_pair}/metrics_data.csv`: 全メトリクスデータ
- `data/release_impact/{project}_{release_pair}/summary_statistics.json`: 記述統計量
- `data/release_impact/{project}_{release_pair}/test_results.json`: Mann-Whitney U検定結果
- `data/release_impact/{project}_{release_pair}/boxplots_4x4.pdf`: ボックスプロット（4×4グリッド）
- `data/release_impact/{project}_{release_pair}/heatmap.pdf`: p値ヒートマップ
- `data/release_impact/{project}_{release_pair}/summary_plot.pdf`: 平均値比較プロット

### 結果の解釈
```json
{
  "project_results": {
    "nova": {
      "feature_weights": {
        "bug_fix_confidence": 0.245,
        "lines_added": 0.123,
        "developer_experience": 0.089,
        // ... 他の特徴量
      },
      "training_stats": {
        "converged": true,
        "iterations": 45,
        "final_objective": -1234.56
      }
    }
  }
}
```

## 🧪 テスト

包括的なテストスイートが用意されています：

```bash
# 全テストの実行
pytest tests/ -v

# モジュール別テスト
pytest tests/collectors/ -v     # データ収集
pytest tests/features/ -v      # 特徴量抽出
pytest tests/learning/ -v      # 機械学習
pytest tests/preprocessing/ -v # データ前処理

# カバレッジ付きテスト
pytest tests/ --cov=src --cov-report=html
```

## ⚙️ 設定

### 必要な設定ファイル
```bash
# Gerrit接続設定（オプション）
src/config/gerrymanderconfig.ini

# レビューラベル定義
data/processed/review_label.json
```

### 環境変数（オプション）
```bash
export GERRIT_USERNAME="your_username"
export GERRIT_PASSWORD="your_password"
```

## 🛠️ 開発

### 新機能の追加
1. **特徴量追加**: `src/features/` に新しいメトリクス関数を追加
2. **テスト作成**: 対応するテストを `tests/` に追加
3. **ドキュメント更新**: README.mdの更新

### コーディング規約
- **PEP 8**: Python標準スタイル
- **型ヒント**: 可能な限り型注釈を使用
- **ドキュメント**: 全関数にdocstring

## 📚 詳細ドキュメント

- [src/README.md](src/README.md) - ソースコード詳細
- [tests/README.md](tests/README.md) - テスト詳細
- [src/collectors/README.md](src/collectors/README.md) - データ収集
- [src/features/README.md](src/features/README.md) - 特徴量抽出
- [src/learning/README.md](src/learning/README.md) - 機械学習・IRL
- [src/preprocessing/README.md](src/preprocessing/README.md) - データ前処理
- [src/release_impact/README.md](src/release_impact/README.md) - リリース影響分析
- [src/release_impact/designs.md](src/release_impact/designs.md) - リリース影響分析設計書

## 🙏 謝辞

- OpenStackコミュニティ：貴重なレビューデータの提供
- 研究コミュニティ：逆強化学習アルゴリズムの開発