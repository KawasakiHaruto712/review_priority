# OpenStackコアコンポーネント一覧
OPENSTACK_CORE_COMPONENTS = [
    "nova",        # コンピュート
    "neutron",     # ネットワーキング
    "swift",       # オブジェクトストレージ 
    "cinder",      # ブロックストレージ
    "keystone",    # 認証
    "glance",      # イメージサービス
]

# 日付範囲
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# ラベル付けしたChange数
LABELLED_CHANGE_COUNT = 383

# スライディングウィンドウ日数
SLIDING_WINDOW_DAYS = 14 # ウィンドウサイズ（2週間）
SLIDING_WINDOW_STEP_DAYS = 1 # ウィンドウをずらす間隔（1日）