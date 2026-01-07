"""
Trend Models Analysis - 定数定義

本モジュールでは、トレンドモデル分析で使用する定数を定義します。
"""

# 分析対象プロジェクトとリリース
TREND_MODEL_CONFIG = {
    'project': {
        'nova': [
            '2015.1.0',
            '12.0.0',
            '13.0.0',
            '14.0.0',
            '15.0.0',
            '16.0.0',
            '17.0.0',
            '18.0.0',
            '19.0.0',
            '20.0.0',
        ],
        # 他のプロジェクトを追加可能
        # 'neutron': ['...'],
        # 'cinder': ['...'],
    }
}

# 分析期間の定義
ANALYSIS_PERIODS = {
    'early': {
        'base_date': 'current_release',
        'offset_start': 0,
        'offset_end': 30,
        'description': 'リリース直後30日間'
    },
    'late': {
        'base_date': 'next_release',
        'offset_start': -30,
        'offset_end': 0,
        'description': '次リリース直前30日間'
    },
    'all': {
        'base_date': 'current_release',
        'offset_start': 0,
        'offset_end': 'next_release',
        'description': '当該リリースの全期間'
    }
}

# モデルパラメータ
MODEL_PARAMS = {
    'random_forest': {
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    },
    'gradient_boosting': {
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    'svm': {
        'random_state': 42,
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True  # predict_probaを有効化
    },
    'tabnet': {
        # TabNet固有パラメータ
        'n_d': 8,           # 決定ステップの次元
        'n_a': 8,           # アテンションの次元
        'n_steps': 3,       # 決定ステップ数
        'gamma': 1.3,       # 特徴量再利用係数
        'n_independent': 2, # 独立層の数
        'n_shared': 2,      # 共有層の数
        'lambda_sparse': 1e-3,  # スパース正則化
        'seed': 42,
        # 学習パラメータ
        'max_epochs': 100,
        'patience': 15,
        'batch_size': 256,
        'virtual_batch_size': 128,
    },
    'ft_transformer': {
        # FT-Transformer（MLP代替実装）パラメータ
        'hidden_dims': [128, 64, 32],  # 隠れ層の次元
        'dropout_rate': 0.3,           # ドロップアウト率
        'lr': 1e-3,                    # 学習率
        'weight_decay': 1e-4,          # 重み減衰
        'batch_size': 256,             # バッチサイズ
        'epochs': 100,                 # エポック数
        'patience': 15,                # Early stopping
    }
}

# 特徴量名リスト（16種類）
FEATURE_NAMES = [
    # Bug Metrics
    'bug_fix_confidence',
    
    # Change Metrics
    'lines_added',
    'lines_deleted',
    'files_changed',
    'elapsed_time',
    'revision_count',
    'test_code_presence',
    
    # Developer Metrics
    'past_report_count',
    'recent_report_count',
    'merge_rate',
    'recent_merge_rate',
    
    # Project Metrics
    'days_to_major_release',
    'open_ticket_count',
    'reviewed_lines_in_period',
    
    # Refactoring Metrics
    'refactoring_confidence',
    
    # Review Metrics
    'uncompleted_requests',
]

# 期間の長さ（日数）
PERIOD_DURATION_DAYS = 30

# 使用するモデルタイプのリスト（複数指定可能）
# コメントアウトで無効化可能
MODEL_TYPES = [
    'random_forest',
    'gradient_boosting',
    'logistic_regression',
    'svm',
    # 深層学習モデル（オプショナル: pytorch-tabnet, torchが必要）
    'tabnet',
    'ft_transformer',
]

# デフォルトモデルタイプ（MODEL_TYPESが空の場合に使用）
DEFAULT_MODEL_TYPE = 'random_forest'

# 出力ディレクトリ名
OUTPUT_DIR_NAME = 'trend_models'
