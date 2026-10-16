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
#   chunk_days : 1 回の問い合わせで扱う日数（省略時は checkpoint_years ぶんを一度に問い合わせる）。
#            件数の多いリポジトリで 1 年ぶんを一度に問い合わせると、サーバの応答が 120 秒を
#            超えてタイムアウトする（chromium/src で発生）。第 1 段階のランキングの件数から、
#            1 区間がおよそ 4,000 件になるよう決める。完了マーカーの単位は変わらない。
#   delay  : 1 リクエストの後に空ける秒数（省略時は change_collector.DEFAULT_REQUEST_DELAY）。
#            アクセス頻度に制限のあるインスタンス用。Wikimedia は短時間に集中すると
#            HTTP 403（Too many requests）を返し、しばらく解けない。3 秒では足りず
#            8 秒で安定した。制限に掛かると再試行の待ち時間に入って時間を無駄にするので、
#            最初から間隔を空けておくほうが速い。
#   former_main_branches : 昔の本流（API の HEAD では今の本流しか取れないため、出どころつきで書く）。
#            main_branch_collector が「昔の本流（kind=former）」として本流の表に入れる。
#            until は改名の日（分からなければ空）。pretrained_encoders/design.md §12.2・src/collectors/README.md「本流の表」

# Qt（qtbase・qtdeclarative）の昔の本流。Qt 5 のときに master から dev に移った（実際の日は未確認）
QT_FORMER_MASTER = {
    "branch": "master", "until": "",
    "source": "https://lists.qt-project.org/pipermail/development/2012-September/006691.html",
    "note": "2012-09-27 に告知（10 月中旬に dev を開く予定）。実際の日は未確認で、2012 年末〜2013-09 の間とみられる",
}

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
        # 昔の本流（API では取れないので出どころつきで書く。main_branch_collector が「昔の本流」として表に入れる）
        "former_main_branches": [QT_FORMER_MASTER],
    },
    "qtcreator": {
        "host": "https://codereview.qt-project.org/a", "path": "qt-creator/qt-creator",
        "auth": True, "env": "QT_GERRIT", "batch_size": 50,
        # 機能リリース x.y.0（2021 年より前の 4.x を含む。2026-10-15 にメジャー版 v<X>.0.0 だけから改めた）。
        # qds/・qp-・csa- で始まる別製品のタグは頭が v でないので拾わない
        "tag_pattern": r"^v(\d+)\.(\d+)\.0$", "tag_series": r"\1.\2.0",
    },

    # ── LibreOffice（タグは RC まで刻むので、シリーズ（X.Y）の最初のタグ日を採用）──
    "libreoffice": {
        "host": "https://gerrit.libreoffice.org", "path": "core", "auth": False,
        "tag_pattern": r"^libreoffice-(\d+)\.(\d+)\.0\.\d+$", "tag_series": r"\1.\2",
    },

    # ── 選定の第 2 段階で収集する候補（project_selection/design.md §4）────────────
    # 第 1 段階（data/project_selection/ranking_period.csv）の上位 12 件のうち、
    # まだローカルに無いもの。採用が決まるまでは「候補」であり、分析対象ではない。
    # tag_pattern / tag_series は採用が決まってから調べて足す（収集には不要）。
    # 件数は 2022-01-01〜2024-12-31 に作成された Change（第 1 段階の実測）。
    "chromium_src": {                       # 658,962 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromium/src", "auth": False, "chunk_days": 7,
    },
    "chromiumos_overlay": {                 # 263,911 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/overlays/chromiumos-overlay", "auth": False, "chunk_days": 14,
    },
    "chromiumos_kernel": {                  # 86,417 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/third_party/kernel", "auth": False, "chunk_days": 30,
    },
    "android_kernel_common": {              # 68,881 件
        "host": "https://android-review.googlesource.com",
        "path": "kernel/common", "auth": False, "chunk_days": 60,
    },
    "chromiumos_infra_superproject": {      # 63,464 件
        "host": "https://chromium-review.googlesource.com",
        "path": "infra/infra_superproject", "auth": False, "chunk_days": 60,
    },
    "chromiumos_zephyr": {                  # 49,426 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/third_party/zephyr", "auth": False, "chunk_days": 60,
    },
    "android_frameworks_support": {         # 49,332 件
        "host": "https://android-review.googlesource.com",
        "path": "platform/frameworks/support", "auth": False, "chunk_days": 60,
    },
    "v8": {                                 # 32,977 件
        "host": "https://chromium-review.googlesource.com",
        "path": "v8/v8", "auth": False, "chunk_days": 90,
    },
    "chromiumos_tast_tests": {              # 28,368 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/platform/tast-tests", "auth": False, "chunk_days": 90,
    },
    "chromiumos_infra": {                   # 28,051 件
        "host": "https://chromium-review.googlesource.com",
        "path": "infra/infra", "auth": False, "chunk_days": 90,
    },
    "chromiumos_platform2": {               # 28,259 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/platform2", "auth": False, "chunk_days": 90,
    },

    # ── 2026-10-13 に候補を上位 15 件に広げて足したもの（project_selection/design.md §4
    #    「上位 15 件に広げる」）。件数は 2023-01-01〜2025-12-31 に作成された Change ──
    # zephyrproject は 2025-11〜2026-01 に 1 日約 2,000 件と集中しているので、
    # 1 区間がおよそ 4,000 件になるよう 2 日で区切る
    "chromiumos_zephyrproject": {           # 82,560 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/third_party/zephyrproject", "auth": False, "chunk_days": 2,
    },
    "chromiumos_infra_recipes": {           # 24,737 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/infra/recipes", "auth": False, "chunk_days": 90,
    },
    # Wikimedia は短時間に集中すると 403 を返すので、第 1 段階と同じく 8 秒空ける。
    # User-Agent の連絡先は .env の CONTACT から読む（project_selection/design.md §6.2）
    "wikimedia_puppet": {                   # 26,326 件
        "host": "https://gerrit.wikimedia.org/r",
        "path": "operations/puppet", "auth": False, "chunk_days": 90, "delay": 8.0,
    },

    # ── 2026-10-14 に候補を上位 30 件に広げて足したもの（project_selection/design.md §4
    #    「上位 30 件に広げる」）。件数は 2023-01-01〜2025-12-31 に作成された Change ──
    # どれも年 4,000〜8,000 件なので、1 区間がおよそ 4,000 件になるよう 90 日で区切る。
    # web-workload だけは 2024 年に 13,235 件と多いので 30 日
    "android_cts": {                        # 22,520 件
        "host": "https://android-review.googlesource.com",
        "path": "platform/cts", "auth": False, "chunk_days": 90,
    },
    "chromiumos_coreboot": {                # 22,397 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/third_party/coreboot", "auth": False, "chunk_days": 90,
    },
    "chromium_devtools_frontend": {         # 18,805 件
        "host": "https://chromium-review.googlesource.com",
        "path": "devtools/devtools-frontend", "auth": False, "chunk_days": 90,
    },
    # Wikimedia は puppet と同じく 8 秒空ける（User-Agent の連絡先も同じ）。
    # 正式版のタグは 1.N.0。版の名前もタグのまま（project_selection/design.md §5.2「追記：区切りの日をタグから取る」）
    "wikimedia_mediawiki_core": {           # 18,624 件
        "host": "https://gerrit.wikimedia.org/r",
        "path": "mediawiki/core", "auth": False, "chunk_days": 90, "delay": 8.0,
        "tag_pattern": r"^(1\.\d+\.0)$", "tag_series": r"\1",
    },
    "chromiumos_ec": {                      # 17,223 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/platform/ec", "auth": False, "chunk_days": 90,
    },
    "chromium_web_workload": {              # 17,170 件
        "host": "https://chromium-review.googlesource.com",
        "path": "web-workload", "auth": False, "chunk_days": 30,
    },
    # Qt は qtbase と同じく認証つき・1 回 50 件。区切りの日は qtbase と同じ決め方で、自分のタグから取る
    # （project_selection/design.md §5.2「追記：区切りの日をタグから取る」）
    "qtdeclarative": {                      # 16,685 件
        "host": "https://codereview.qt-project.org/a", "path": "qt/qtdeclarative",
        "auth": True, "env": "QT_GERRIT", "batch_size": 50, "chunk_days": 90,
        "tag_pattern": r"^v(\d+)\.(\d+)\.0$", "tag_series": r"\1.\2.0",
        "former_main_branches": [QT_FORMER_MASTER],   # qtbase と同じ（Qt 5 のときに master → dev）
    },
    "android_manifest": {                   # 16,462 件
        "host": "https://android-review.googlesource.com",
        "path": "platform/manifest", "auth": False, "chunk_days": 90,
    },
    "chromiumos_board_overlays": {          # 14,247 件
        "host": "https://chromium-review.googlesource.com",
        "path": "chromiumos/overlays/board-overlays", "auth": False, "chunk_days": 90,
    },
    # 正式版のタグは go1・go1.N（go1.20 まで）・go1.N.0（go1.21 から）。版の名前は go1・go1.N に揃える。
    # rc・beta と go1.N.M（M≥1）は拾わない（project_selection/design.md §5.2「追記：区切りの日をタグから取る」）
    "go": {                                 # 14,153 件
        "host": "https://go-review.googlesource.com",
        "path": "go", "auth": False, "chunk_days": 90,
        "tag_pattern": r"^(go1(?:\.\d+)?)(?:\.0)?$", "tag_series": r"\1",
    },
    "chromium_emscripten_releases": {       # 13,490 件
        "host": "https://chromium-review.googlesource.com",
        "path": "emscripten-releases", "auth": False, "chunk_days": 90,
    },
    "android_bluetooth": {                  # 13,412 件
        "host": "https://android-review.googlesource.com",
        "path": "platform/packages/modules/Bluetooth", "auth": False, "chunk_days": 90,
    },
    "typo3_cms": {                          # 12,733 件
        "host": "https://review.typo3.org",
        "path": "Packages/TYPO3.CMS", "auth": False, "chunk_days": 90,
        # 機能リリース x.y.0（sprint release を含む）。時期によってタグの形が TYPO3_X-Y-0・X.Y.0・vX.Y.0 と違うので、
        # 「または」でつなぐ。前半に当たれば数は \1・\2、後半なら \3・\4 に入り、当たらなかった側は空の文字になる
        # ので、どの形でも "X.Y.0" になる（project_selection/design.md §5.2「区切りの日をタグから取る」）
        "tag_pattern": r"^(?:TYPO3_(\d+)-(\d+)-0|v?(\d+)\.(\d+)\.0)$", "tag_series": r"\1\3.\2\4.0",
        # 昔の本流（2021-11-29 に master → main。pretrained_encoders/design.md §12.2）
        "former_main_branches": [{
            "branch": "master", "until": "2021-11-29",
            "source": "https://news.typo3.com/archive/typo3-core-development-to-change-branch-name",
            "note": "2021-11-28 に告知。改名の時点で Open だった約 220 件は手元のデータでは main と記録されている（推測）",
        }],
    },
}

# 新規に収集する対象（OpenStack は収集済みなので既定には入れない）
NEW_COLLECT_TARGETS = ["qtbase", "qtcreator", "libreoffice"]

# 区切りの日を正式版のタグから取る対象（release_tags_collector の既定。
# 今回の論文で分析する 6 件。project_selection/design.md §4「論文の対象」・
# §5.2「追記：区切りの日をタグから取る」。2026-10-15 に 3 件から 6 件へ広げた）
RELEASE_TAG_TARGETS = ["qtbase", "qtcreator", "wikimedia_mediawiki_core", "qtdeclarative", "go", "typo3_cms"]

# 選定の候補（2023〜2025 年の上位 15 件のうち、全期間が手元にない 12 件。
# project_selection/design.md §4「上位 15 件に広げる」）。
# core（libreoffice）・qt/qtbase・qt-creator は全期間を収集済みなので含めない。
# 2026-10-13 に上位 12 件から変更。外れた tast-tests・platform2 はデータと登録を予備として残す
# 2026-10-14 に上位 30 件へ広げた（design.md §4「上位 30 件に広げる」）。予備だった platform2・tast-tests も戻し、27 件
SELECTION_CANDIDATES = [
    "chromium_src", "chromiumos_overlay", "chromiumos_infra_superproject",
    "android_kernel_common", "chromiumos_zephyrproject", "android_frameworks_support",
    "chromiumos_kernel", "chromiumos_zephyr", "v8",
    "chromiumos_infra", "wikimedia_puppet", "chromiumos_infra_recipes",
    # 16〜30 位
    "android_cts", "chromiumos_coreboot", "chromiumos_platform2", "chromiumos_tast_tests",
    "chromium_devtools_frontend", "wikimedia_mediawiki_core", "chromiumos_ec",
    "chromium_web_workload", "qtdeclarative", "android_manifest",
    "chromiumos_board_overlays", "go", "chromium_emscripten_releases",
    "android_bluetooth", "typo3_cms",
]

# 日付範囲
START_DATE = "2000-01-01"
END_DATE = "2027-04-01"

# ラベル付けしたChange数
LABELLED_CHANGE_COUNT = 383

# スライディングウィンドウ日数
SLIDING_WINDOW_DAYS = 14 # ウィンドウサイズ（2週間）
SLIDING_WINDOW_STEP_DAYS = 1 # ウィンドウをずらす間隔（1日）