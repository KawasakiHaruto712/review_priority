# OpenStackコアコンポーネント一覧
OPENSTACK_CORE_COMPONENTS = [
    "nova",        # コンピュート
    "neutron",     # ネットワーキング
    "swift",       # オブジェクトストレージ
    "cinder",      # ブロックストレージ
    "keystone",    # 認証
    "glance",      # イメージサービス
]

# 収集対象プロジェクトの登録表
#   キー   : 保存ディレクトリ名（data/openstack_collected/<キー>/）かつ解析側の project 名。
#            Gerrit 上の名前はスラッシュを含むので、キーは平たい名前にする。
#   host   : Gerrit の API ベース URL。OpenStack は認証つき（/a）、公開ホストは認証なし。
#   path   : Gerrit 上のプロジェクト名（クエリの project: に渡す値）。
#   auth   : True なら Basic 認証する。認証つきのパスは `/a/` を前置した URL になる。
#   env    : 資格情報を読む .env のキー接頭辞（既定 "GERRIT" → GERRIT_USERNAME / GERRIT_PASSWORD）。
#            HTTP パスワードは Gerrit インスタンスごとに発行されるため、OpenStack の資格情報は
#            Qt では使えない。Qt は QT_GERRIT_USERNAME / QT_GERRIT_PASSWORD を使う。
#   tag_pattern / tag_series : リリース版の抽出規則（release_collector が使う）。
#            tag_pattern に一致したタグから tag_series で「版シリーズ」を作り、
#            同じシリーズの最初のタグ日をそのリリース日とする。
#   batch_size : 1 リクエストで要求する件数（省略時は endpoint_config.yaml の値）。
#            Gerrit は要求した n のぶんだけ候補を評価するため、大きすぎる n を投げると
#            サーバ側の実行時間制限に触れて 500 が返る。実行時は失敗するたびに n を半分に
#            落として続行するので、ここは「安定して通る値」を入れておけばよい。
GERRIT_PROJECTS = {
    # ── OpenStack（収集済み。リリース情報は openstack/releases リポジトリから取得）──
    "nova":     {"host": "https://review.opendev.org/a", "path": "openstack/nova",     "auth": True},
    "neutron":  {"host": "https://review.opendev.org/a", "path": "openstack/neutron",  "auth": True},
    "swift":    {"host": "https://review.opendev.org/a", "path": "openstack/swift",    "auth": True},
    "cinder":   {"host": "https://review.opendev.org/a", "path": "openstack/cinder",   "auth": True},
    "keystone": {"host": "https://review.opendev.org/a", "path": "openstack/keystone", "auth": True},
    "glance":   {"host": "https://review.opendev.org/a", "path": "openstack/glance",   "auth": True},

    # ── Qt ──
    # 匿名だと 1 ページ 10 件しか返らない（queryLimit が匿名グループに低く設定されている）。
    # 認証すると上限が上がる。ただし n=100 は qtcreator で実行時間制限に掛かるため 50 にする
    # （qtbase の n=100 も所要 3.1〜9.2 秒と不安定だった）。匿名の 10 件に対し 5 倍。
    "qtbase": {
        "host": "https://codereview.qt-project.org/a", "path": "qt/qtbase",
        "auth": True, "env": "QT_GERRIT", "batch_size": 50,
        "tag_pattern": r"^v(\d+)\.(\d+)\.0$", "tag_series": r"\1.\2.0",
    },
    "qtcreator": {
        "host": "https://codereview.qt-project.org/a", "path": "qt-creator/qt-creator",
        "auth": True, "env": "QT_GERRIT", "batch_size": 50,
        "tag_pattern": r"^v(\d+)\.0\.0$", "tag_series": r"\1.0.0",
    },

    # ── LibreOffice（タグは RC まで刻むので、シリーズ（X.Y）の最初のタグ日を採用）──
    "libreoffice": {
        "host": "https://gerrit.libreoffice.org", "path": "core", "auth": False,
        "tag_pattern": r"^libreoffice-(\d+)\.(\d+)\.0\.\d+$", "tag_series": r"\1.\2",
    },
}

# 新規に収集する対象（OpenStack は収集済みなので既定には入れない）
NEW_COLLECT_TARGETS = ["qtbase", "qtcreator", "libreoffice"]

# 日付範囲
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# ラベル付けしたChange数
LABELLED_CHANGE_COUNT = 383

# スライディングウィンドウ日数
SLIDING_WINDOW_DAYS = 14 # ウィンドウサイズ（2週間）
SLIDING_WINDOW_STEP_DAYS = 1 # ウィンドウをずらす間隔（1日）