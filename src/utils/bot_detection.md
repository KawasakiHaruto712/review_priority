# bot_detection 設計書（ボットの判定）

## 0. 位置づけ・要旨

レビューのコメントが人間によるものか、チケット（Change）を作ったのがボットかを判定する仕組みを、**1 か所にまとめる**。

これまでは、ボットの判定が 5 か所に別々にあり、使う一覧も 3 つのファイルに分かれていた（§6.1）。中身は OpenStack 専用で、新しく選定したリポジトリ（ChromiumOS・Android・LibreOffice・Qt）には使えなかった。また、一覧の照合に表示名を使っていたため、同じ名前の人間をボットと誤判定していた（§7.2）。

本設計では、判定の根拠を**公式の定義と、出どころの分かる外部の資料**に限り、こちらで決める部分をできるだけ小さくする。

- **判定は 4 つの決まりの和**（§1）。Gerrit 公式の印・Google Cloud 公式のアカウントの形・OpenDev 公式の命名規則・出どころ付きの一覧。
- **一覧は 1 つのファイル**（`src/config/bot_accounts.csv`）。各行に出どころの URL を必ず書く（§2）。
- **判定のコードは 1 か所**（`src/utils/bot_detection.py`）。今の 5 か所はすべてこれを使うように直す（§6）。
- **5 か所すべてを、実装のときに一斉に新しい判定へ切り替える**（§6.2）。今の分析の結果は今後使わず、特徴量を増やして回し直す予定のため。古い一覧ファイル 3 つも、実装のときに削除する（§6.3）。

---

## 1. 判定の決まり（アカウント単位）

次の ①〜④ のどれかに当てはまるアカウントを、ボットとみなす。

| 順 | 決まり | 根拠 |
|---|---|---|
| ① | Gerrit の `SERVICE_USER` の印がある | Gerrit 公式（§1.1） |
| ② | メールアドレスが `gserviceaccount.com` で終わる | Google Cloud 公式（§1.2） |
| ③ | `name` か `display_name` の最後の語が Bot・CI | OpenDev 公式の命名規則（§1.3） |
| ④ | 一覧ファイルに載っている | 行ごとに書いた出どころの資料（§2） |

### 1.1 ① SERVICE_USER の印

Gerrit には「Service Users」という既定のグループがあり、各ホストの管理者が、機械のアカウントをそこに登録する。Gerrit の公式ドキュメントは、このグループを次のように定義している。

> "The Service Users group is used to identify service users (aka bots). All service users should be added to this group so that they can be identified as such."
> （access-control.html#service_users）

登録されたアカウントには、REST API のアカウント情報に `"tags": ["SERVICE_USER"]` が付く（rest-api-accounts.html#account-info。Gerrit 3.3 以降）。この印は、問い合わせのオプション `DETAILED_ACCOUNTS` を付けたときだけ返る。収集では付けている（`ChangesEndpoint.FULL_OPTIONS`）。**今後の収集でも、このオプションを外さないこと。**

**印は、問い合わせた時点のグループの中身から作られる**（Gerrit のソースコード `InternalAccountDirectory` / `ServiceUserClassifierImpl` で確認）。記録として保存されているわけではないので、分析期間中に動いていて、その後にグループから外れたボットには印が付かない（§7.1）。

**グループの旧名（2026-10-14 追記）**：このグループは、Gerrit 3.3 より前は「**Non-Interactive Users**」という名前だった。3.3 で「Service Users」に名前が変わったもので、グループそのもの（グループの ID）は同じである（出どころ：Gerrit 3.3 のリリースノート。原文は未確認）。古い Gerrit の説明書・ホストの設定・メーリングリストなどで「Non-Interactive Users」と書かれていれば、このグループのことである。印（`SERVICE_USER`）は問い合わせた時点のグループの中身から作られるので、旧名の時代に登録されたアカウントにも、今も登録されていれば印が付く。

### 1.2 ② gserviceaccount.com

Google Cloud のサービスアカウントは、人ではなくアプリケーションなどが使うアカウントで、メールアドレスは `…@….iam.gserviceaccount.com` などの形になる（Google Cloud 公式ドキュメント service-account-overview / service-account-types）。ChromiumOS と Android の機械のアカウントの多くがこの形である。

大文字・小文字を区別せず、メールアドレスの末尾で判定する。

### 1.3 ③ 名前の最後の語が Bot・CI

OpenDev の公式ドキュメント（サードパーティ CI の手引き）は、CI のアカウントの名前を次のように定めている。

> "The name should have three pieces Organization Product/technology CI designator. … the CI designator is used to denote this is a CI system so that automatic Gerrit comment parsers can filter these comments out. This value should be CI for most CI systems but can be Bot if you are not performing continuous integration."

この決まりの趣旨（名前の最後の語で、CI・ボットであることを示す）に従い、次のどちらかに当てはまれば ③ とする。

| 区切り方 | 判定 | 例 |
|---|---|---|
| 空白・`_`・`-` で区切った最後の語が `ci` か `bot` | 大文字・小文字を区別しない | Qt CI Bot、Cloudbase Nova Hyper-V CI、huawei-cinder-ci |
| 区切りがなく、大文字の `Bot` か `CI` が、小文字か数字の直後で終わる | 大文字・小文字を区別する | QtCIAnalysisBot、QtWelcomeBot |

```python
SEP_PATTERN = re.compile(r"(?:^|[\s_\-])(ci|bot)$", re.IGNORECASE)
CAMEL_PATTERN = re.compile(r"[a-z0-9](Bot|CI)$")
```

- 名前の前後の空白は取り除いてから判定する。
- 区切りなしの決まりで大文字・小文字を区別するのは、「Abbot」「Talbot」のような、小文字の `bot` で終わる人名を拾わないため。
- **当てるのは `name` と `display_name` だけ**。ユーザー名とメールアドレスには当てない（公式の決まりは名前についてのものであるため）。
- `display_name` は、`name` とは別に画面に出す名前として設定できる欄。OpenStack では、`name` が人名で、`display_name` だけを CI にして運用しているアカウントがある（例：name「Senthil Vasudevan」、display_name「HPE AlletraMP FC CI」）。

**③は、名前を「本人の名乗り」として使う。**名前の形だけを見て、ほかのアカウントの名前とは比べない。④の照合で名前を使わないこと（§2.2）とは、目的が異なる。

---

## 2. 一覧ファイル（④）

### 2.1 形

`src/config/bot_accounts.csv`。ボットの一覧はこのファイル 1 つだけとする。

```text
host,account_id,username,email,name,source,url,note
review.opendev.org,,jenkins,,Jenkins,GerrymanderConfig,https://web.archive.org/web/20240624040816/https://wiki.openstack.org/wiki/GerrymanderConfig,
```

| 列 | 内容 | 必須 |
|---|---|---|
| `host` | Gerrit のホスト名（`review.opendev.org` など。スキームと `/a` は付けない） | 必須 |
| `account_id` | そのホストでのアカウント番号 | `account_id`・`username`・`email` のうち 1 つ以上 |
| `username` | ユーザー名 | 同上 |
| `email` | メールアドレス | 同上 |
| `name` | 人が読むための名前。**照合には使わない** | 任意 |
| `source` | 出どころの資料の名前 | 必須 |
| `url` | 出どころの資料の URL | 必須 |
| `note` | 補足（出どころとアカウントのつながりの根拠など） | 任意 |

### 2.2 照合のしかた

- **ホストを必ず区別する。**アカウント番号はホストごとに別々に振られており、同じ名前の機械のアカウントも、ホストによって別のアカウントである（例：gwsq は ChromiumOS と Android で別）。
- そのうえで、`account_id`（文字列として完全一致）・`username`（大文字・小文字を区別しない）・`email`（同）の、**埋まっている列のどれか 1 つでも一致**すれば当てはまる。
- **名前（`name`・`display_name`）では照合しない。**名前は人と重なりうるため、アカウントを見分ける手がかりにはしない。以前の一覧では、CI の担当者の名前の行（例：`Mikhail Khodos`）が、同じ名前の人間のアカウントと一致し、人間をボットと誤判定していた（§7.2）。

### 2.3 載せる条件

**①〜③ で拾えず、かつ出どころの資料が見つかったものだけ**を載せる。出どころの分からない行は置かない。①〜③ で拾えるアカウントは、一覧に書かない。

### 2.4 載せるアカウント（2026-10-11 時点）

| ホスト | アカウント | 照合に使う値 | 出どころ | 補足 |
|---|---|---|---|---|
| review.opendev.org | Jenkins | username `jenkins` | GerrymanderConfig（OpenStack Wiki。Web アーカイブの保存版） | 昔の CI。今の Service Users グループに入っていない |
| review.opendev.org | Elastic Recheck | username `elasticrecheck` | 同上 | |
| review.opendev.org | LaunchpadSync | username `launchpadsync` | 同上 | |
| review.opendev.org | Trivial Rebase | username `trivial-rebase` | 同上 | |
| chromium-review.googlesource.com | v8 autoroll | account_id 1113489、email `v8-autoroll@chromium.org` | Chromium のビルド基盤のコード（V8 の自動更新の処理で、このアカウントを作成者・レビュアに指定） | |
| chromium-review.googlesource.com | adservices-enrollment | account_id 3533182、email `mdb.adservices-enrollment@google.com` | chromium/src の OWNERS（「For bot updates to …」とこのアカウントを記載） | |
| chromium-review.googlesource.com | CopyBot Service Account | account_id 1526902、email `copybot.service@gmail.com` | CopyBot の README（外部のリポジトリから Gerrit へコミットを自動でコピーする道具） | アカウントとのつながりは、投稿の定型文（`Tested-by: CopyBot Service Account <copybot.service@gmail.com>`）による |
| chromium-review.googlesource.com | Copybara Service | account_id 1312627、email `copybara-worker-blackhole@google.com` | Copybara の README（Google の、リポジトリ間でコードを移す道具） | アカウントとのつながりは、投稿の定型文（`This CL was generated by a Copybara workflow.`）による |
| gerrit.libreoffice.org | Weblate | account_id 1002285、username `weblate` | Weblate の公式ドキュメント（コミットする者の名前の既定値が「Weblate」）、TDF の翻訳の Wiki | アカウントとのつながりは、投稿の定型文（`Translated using Weblate (…)`）による |
| codereview.qt-project.org | Qt Continuous Integration System（2026-10-16 追加） | **account_id 1000060、username `qtintegration`**（2026-10-16 の実装で、手元のデータで確かめた。qtbase の 2011〜2016 年のメッセージ 65,993 件の作者。email は空）。~~実装のときに手元のデータで確かめて書く~~（名前では照合しないため、account_id・username・email のどれか）。Qt の Gerrit には問い合わせず、手元の収集データのメッセージの作者から探す（2026-10-16 本人判断。結果は `data/feature_survey/qt_ci_system_account_result.txt`） | Qt の CI の報告のメーリングリスト（lists.qt-project.org の ci-reports。送り主が「Qt Continuous Integration System」 ci-noreply@qt-project.org で、本文は CI が Gerrit に `staging-approve` を送る記録） | 2014〜2016 年の CI の結果（`QtBase_…_Integration #N: SUCCESS`・`Compilation failed :(` など）を書いていたアカウント。名前の最後の語が System なので ③ に当たらない。メールの送り主と Gerrit のアカウントが同じものだという直接の記述は見ていない（`data/feature_survey/source_check9_untagged_messages_20261016.md` §5） |

各行の正確な URL は、一覧ファイルと README（§5）に書く。

### 2.5 一覧を変えるときの決まり

- 行を足すときは、§2.3 の条件を満たすことを確かめ、`source`・`url` を必ず書く。
- **一覧の正本は `bot_accounts.csv`。README の表はその写し**（§5）なので、一覧を変えたら README の表も同時に直す。

---

## 3. メッセージ単位の決まり

次のメッセージは、人間のレビューとして数えない（今の判定と同じ）。

| 決まり | 根拠 |
|---|---|
| `tag` が `autogenerated:` で始まる | Gerrit 公式（rest-api-changes.html。「CI などの自動システムが、人間のレビューと区別するために使ってよい」） |
| 投稿者がボット（§1） | 本設計 |
| 投稿者がチケットの作成者本人 | 本人のコメントはレビューではないため（今の判定と同じ） |

Gerrit 自身も、人が操作したときのメッセージ（新しい版の投稿・放棄など）に `autogenerated:gerrit:` を付ける。これらはレビューではないので、数えないままでよい。一方、`autogenerated:` の印は、アカウントの判定には使わない（人が操作したメッセージにも付くため）。

**2026-10-16 改めた：印のない自動の記録も、人間のレビューとして数えない（§3.1）。**上の「印で除く」だけでは、印の付かない記録が人間のレビューに混ざっていた。
Gerrit が印を付けるのは 2.14（2017 年ごろ）以降で、それより前の記録には印がない。Qt のプラグインも v3.5 以降は印を付けない。
自動の記録は**操作した人の名前で残る**（例：作者以外が版を足すと「Uploaded patch set 3.」がその人の名前で残る）ため、ボットの判定でも除けない。
手元の 6 件で、印がなくボットでもないメッセージのうち、Gerrit 本体の文は 10〜26%、Qt のプラグインの文は Qt で 9〜12% あった
（`data/feature_survey/untagged_residual_result.txt`）。

### 3.1 印のない自動の記録（2026-10-16 追加）

メッセージの 1 行目（前に付く `Patch Set N: ` は外して照らす）が次に当たるものを、印がなくても自動の記録とし、人間のレビューとして数えない。

**2026-10-16 改めた：前置き `Patch Set N: ` を外して照らすのは Gerrit 本体の文だけ。**Qt のプラグイン・Qt の古い改造版の文は、1 行目をそのまま照らす
（ソースで確かめた文には前置きが付かないため）。前置きの付いた形（例：2013〜2014 年の `Patch Set N: Deferred`）は出どころを確かめられていないので、
決まり（出どころ不明の文は除かない）どおり人のメッセージとして残す。Open の期間でもこの形では閉じない（`pretrained_encoders/design.md` §11.1）ので、扱いがそろう（本人判断）。
**公式の資料（ソース・説明書）で文を確かめたものだけ**を使う。出どころを確かめられなかった文は除かない（本人判断）。

| 種類 | 文（先頭一致） | 当てるホスト | 根拠 |
|---|---|---|---|
| Gerrit 本体 | `Uploaded patch set `、`Commit message was updated`、`Patch Set M was rebased`、`Published edit on patch set `、`Cherry Picked`、`Change has been successfully merged`、`Change has been successfully cherry-picked`（後ろが ` as ` か行末）、`Change has been successfully rebased and submitted as `、`Change has been successfully pushed`、`Abandoned`、`Restored`、`Created a revert of this change as `、`Set Work In Progress`、`Set Ready For Review`、`Reverted`、`Topic set to `、`Topic changed from `、`Topic … removed`、`Removed the following votes:`、`Removed <投票> by ` | すべて | Gerrit のソース（2.8・2.11・2.13・2.14・master）。`source_check7_gerrit_messages_20261016.md` §4、`source_check9_untagged_messages_20261016.md` §4 |
| Qt のプラグイン | `Staged for CI`、`Added to build `、`Merge conflict in staging branch`、`Unstaged`、`Change merged into branch '`、`Precheck requested with params `、`Deferred`、`Reopened`、`Assigned to staging queue, waiting for promotion to CI` | Qt（codereview.qt-project.org） | qtqa/gerrit-plugin-qt-workflow のソース。`source_check8_qt_plugin_20261016.md`、`source_check9…` §2 |
| Qt の古い改造版 | `Change has been successfully cherry-picked to the staging branch as `、`Change has been successfully merged into the staging branch.`、`Your change could not be merged due to a path conflict.` | Qt | qtqa/gerrit（V2.2.1-NQT-010）のソース。`source_check9…` §1・§3 |
| 書いた人がいない | `author` がないメッセージはすべて | すべて | Gerrit の REST の説明書「Unset if written by the Gerrit system.」。`source_check9…` §3 |

- **除かないもの**（出どころを確かめられなかった文。`source_check9…` §1・§4）：Qt の「Change has been successfully cherry-picked to the staging branch.」（as のない形、2014〜2019 年、約 13 万件）、
  「Patch Set N: Rebased」、小文字の「Patch set N: …」、「Removed reviewer」、「Change message removed」、「Assignee was added.」など。
- **人が入力した文は残す**：`recheck` などのコメントは、人が自分で入力した文なので人間のメッセージのまま。

### 3.2 理由の文と、記録の中の投票（2026-10-16 追加）

- 放棄・復活・後回し・再開（`Abandoned`・`Restored`・`Deferred`・`Reopened`）の記録の後（2 行目以降）に、操作した人が入力した理由の文が続くものは、
  **人のメッセージとして扱う**（本人判断）。理由の文がなければ自動の記録。
- 自動の記録の 1 行目にある投票（例：go の `Uploaded patch set 2: Code-Review+1.`）は、メッセージは人間のレビューとして数えないが、**投票は人の投票として数える**（本人判断）。
  投票を使う特徴量（`pretrained_encoders/design.md` §13.2）のための決まり。

### 3.3 使う場所

同じ決まりを、正解ラベル（`is_human_review_message`）と、新しい特徴量の「人のコメント」（`pretrained_encoders/design.md` §13.2）の両方で使う。
選定の E2（人のコメントの数）も同じ関数を使うので値が変わるが、**E2 の数え直しは論文の後に回す**（2026-10-16 本人判断。ほかのリポジトリのボット・自動の記録の扱いもそのときに決める）。

---

## 4. ボットが作ったチケットの扱い

- **予測の対象（その日の集合）に入れる。**「ボットが多くの作業を担っていること」と、「ボットのチケットをレビューするかどうか」は別の話であり、後者は予測で扱う対象に含まれるため。
- 選定の E3 では、チケットの作成者（`owner`）を §1 で判定し、ボットが作ったチケットの割合を出す。

---

## 5. README に載せるもの

`src/config/README.md` のボットの一覧の節を、次の形に書き換える。

- 判定の決まり（§1・§3）の要約。
- **参照した URL の一覧**を、出どころの種類ごとの折りたたみ（`<details>`）で載せる。
  - Gerrit 公式（①・メッセージの `autogenerated:`）
  - Google Cloud 公式（②）
  - OpenDev 公式（③）
  - 一覧ファイルの出どころ（④。OpenStack・ChromiumOS・LibreOffice）
- 一覧の出どころの折りたたみの、**さらに内側の折りたたみ**に、`bot_accounts.csv` の中身を表で載せる（§2.5 のとおり、CSV が正本）。

---

## 6. コードの置き場所と移行

### 6.1 判定のコード

`src/utils/bot_detection.py` の 1 か所にまとめる。

```python
BOT_ACCOUNTS_CSV = DEFAULT_CONFIG / "bot_accounts.csv"

def host_of(component: str) -> str:
    """GERRIT_PROJECTS[component]["host"] からホスト名を返す（スキームと /a を除く）。"""

class BotDetector:
    """1 つのホストのボット判定（§1）。一覧はホストで絞って読み込む。"""
    def __init__(self, host: str, accounts_csv: Path = BOT_ACCOUNTS_CSV): ...
    def is_bot(self, account: dict) -> bool: ...
    def reason(self, account: dict) -> str | None:
        """当てはまった決まり（"service_user" / "gserviceaccount" / "name" / "list"）。E3 の集計や確認に使う。"""

def is_human_review_message(message: dict, change: dict, detector: BotDetector) -> bool:
    """§3 の決まりで、人間のレビューのメッセージか。"""

# 2026-10-16 追加（§3.1・§3.2）。is_human_review_message と、新しい特徴量の「人のコメント」の両方がこれを使う
def is_auto_record(message: dict, host: str) -> bool:
    """印のない自動の記録も含めて、自動の記録か（§3.1。理由の文が続く放棄・復活などは False。§3.2）。"""

def votes_of(message: dict) -> list[tuple[str, int]]:
    """1 行目の投票（数字の形と、古い形式の既定の文の読み替え。pretrained_encoders/design.md §13.2）。自動の記録の中の投票も返す。"""

def is_owner(account: dict, change: dict) -> bool:
    """チケットの作成者本人か（_account_id → email → username/name の順。今の review_utils と同じ）。"""
```

ホストは、収集のプロジェクト名（`nova`・`chromium_src` など）から `host_of` で決める。判定はホストごとに `BotDetector` を作って使う。

今、ボットの判定を使っている次の 5 か所を、すべてこれを使うように直す。

| 箇所 | 今の判定 | 使っているもの |
|---|---|---|
| `src/analysis/preliminary_analysis/pretrained_encoders/utils/review_utils.py` | 3 つの一覧の和 | 事前学習・窓長の分析・距離×時期行列・特徴量の寄与度（正解ラベルと特徴量） |
| `src/analysis/background_problem/priority_distribution/utils/data_loader.py` | 3 つの一覧の和 | 優先度の分布の分析 |
| `src/collectors/project_selection/metrics.py` | 3 つの一覧の和 ＋ メールの形 ＋ インスタンスごとの追加 | 選定の E1〜E3 |
| `src/features/review_metrics.py` | GerrymanderConfig だけ | 今の分析からは呼ばれていない |
| `src/preprocessing/review_comment_processor.py` | GerrymanderConfig だけ | 今の分析からは呼ばれていない |

**呼んでいる側も直す。**今は「ボットの名前の集合（`bot_names`）」を受け取って渡しているので、「プロジェクトのホストに合わせた `BotDetector`」を受け取って渡す形に変える。分析の中身（特徴量や正解ラベルの作り方）は変えない。

| 種類 | ファイル |
|---|---|
| 判定を呼んでいる | `pretrained_encoders/build_encoders.py`、`pretrained_encoders/features/feature_builder.py`、`pretrained_encoders/dataset/record_builder.py`、`lookback_window/main.py`、`lookback_window/sweep/window_sweep.py`、`concept_drift_detection/main.py`、`concept_drift_cause/main.py`、`background_problem/priority_distribution/main.py`、`collectors/project_selection/main.py` |
| 一覧ファイルのパスを持っている | `pretrained_encoders/utils/constants.py`、`background_problem/priority_distribution/utils/constants.py`（`GERRYMANDER_CONFIG` などを消す） |

上の表のほかにも、実装のときに全コードを検索し、古い判定の関数・一覧ファイル名を参照している箇所が残っていないことを確かめる（§6.3 の手順 3）。

### 6.2 切り替える時期

**§6.1 の 5 か所すべてを、実装のときに一斉に新しい判定へ切り替える。**今の判定のコード（`load_bot_names` などの各実装）は、切り替えと同時に削除する。判定が 2 種類並ぶ期間は作らない。

- 今の分析（事前学習・窓長の分析・距離×時期行列・特徴量の寄与度）と background_problem は、実装のあとに動かすと、新しい判定で正解ラベルと特徴量を作る。§8.1 のとおり正解ラベルが変わるが、**今の分析の結果は今後使わず、特徴量を増やして回し直す予定のため問題ない**（2026-10-11 本人判断）。
- 実装の時点で、今の判定で動いている処理はない。寄与度の計算（古い 4 版）は、2026-10-11 に止めた（nova 26.0.0 まで保存済み）。この計算はプロジェクトが変わるたびに一覧を読み直すため、動かしたまま切り替えると、途中から判定が変わってしまう。**今後、長い計算を動かしている間は、判定のコードと一覧ファイルを変えないこと。**
- 今の判定で作った保存物（事前学習のエンコーダ、窓長の分析・距離×時期行列・寄与度の結果）は削除しない。ただし、新しい判定とは正解ラベルが違うので、回し直した結果と混ぜて使わない。

### 6.3 古い一覧ファイルの削除

実装のときに、新しい判定に切り替えて結果を確かめてから、次の 3 つを削除する。

| ファイル | 削除してよい理由 |
|---|---|
| `src/config/third_party_ci_accounts.csv` | ボットの判定にしか使われていない。中身のほとんどは ③ で拾え、残りの人名の行は誤判定の原因になっていた |
| `src/config/extra_bots.txt` | zuul は ①、jenkins は ④（GerrymanderConfig 由来）で拾える |
| `src/config/gerrymanderconfig.ini` | Gerrit の接続設定やチームの一覧も入っているが、コードから読まれているのは `[organization] bots` の行だけ（全コードで確認）。必要な 4 つは、出どころ付きで一覧ファイルに移す |

手順は次のとおり。

1. 5 か所を新しい判定に切り替える（§6.2）。
2. §9 のテストと、§8 の確認の結果の再現（OpenStack で 244 個など）を確かめる。
3. 3 つのファイルを読んでいる箇所が、コードに残っていないことを確かめる（全コードを検索する）。
4. 3 つのファイルを削除する。

---

## 7. 限界

論文の妥当性の脅威に書く。

### 7.1 ① の印は、収集した時点の登録状態

印は、問い合わせた時点の Service Users グループの中身から作られる（§1.1）。分析期間（2022〜2024 年）に動いていて、その後にグループから外れたボットや、削除されたアカウントには印が付かない。その取りこぼしは ②〜④ で補う（例：OpenStack の Jenkins は ④ で拾う）。

また、グループへの登録は各ホストの管理者の手作業であり、公式のドキュメントも「登録すべき」という書き方にとどまる。印がないことは、人間であることを意味しない（Qt では登録の漏れが多い）。

### 7.2 取りこぼし

①〜④ のどれにも当てはまらず、出どころの資料も見つからなかったボットは、取りこぼしている。

| ホスト | アカウント | 影響（分析対象期間 2022〜2024 年） |
|---|---|---|
| Android | gwsq（レビュアの割り当ての通知） | kernel/common：71,067 件中 33 件、frameworks/support：50,353 件中 156 件のチケットにメッセージ（0.3% 以下） |
| ChromiumOS | floss-automerger | 作成 178 件・メッセージ 168 件（0.1% 未満） |
| Qt・LibreOffice | ciautostagebot 以外の、小文字だけで区切りのない名前のもの（mrprobot など） | 十数件 |
| OpenStack | rocktown（account_id 10934。Intel の CI と見られる） | 2014 年のメッセージ 2 件（分析対象期間の外） |

rocktown は GerrymanderConfig の bots の行に載っているが、データ上のアカウントに `username` がなく、名前（`rocktown`）でしか当たらない。④ は名前で照合しないので、一覧の GerrymanderConfig の行には加えていない（§8.1）。

ciautostagebot は `display_name`（CI Auto Stage Bot）で ③ に当てはまるので、取りこぼしではない。

### 7.3 自動の記録の取りこぼし（2026-10-16 追加）

- §3.1 は、公式の資料で文を確かめたものだけを除く。出どころを確かめられなかった文（Qt の as のないステージングの文 約 13 万件など）は、人間のメッセージとして残る。
  いずれも評価期間（2021〜2025 年）より前の記録で、過去を使う特徴量（中核の開発者の索引など）にだけ効く。
- TYPO3 の「Mr. Jenkins」（2011〜2019 年、CI の結果の投票と見られる）は、ボットである根拠の資料が見つからなかったので、ボットに加えていない（本人判断）。
- 各プロジェクト独自の古い形式の投票の文（Qt の `Sanity review passed`、TYPO3 の `Verified by team member` など）は点数に読み替えない。
- 今回の決まりは 6 件（論文の対象）を見て作った。ほかのリポジトリのボット・自動の記録は、論文の後に改めて調べる。

---

## 8. 確認の結果（2026-10-11、手元のデータの全件）

### 8.1 OpenStack 6 件：今の判定との比較

| | ボットのアカウント |
|---|---|
| 今の判定（3 つの一覧の和） | 155 |
| 新しい判定（§1 のとおり。④ は §2.4 の一覧） | 244 |

- **今ボットとしている 155 個のうち 151 個は、新しい判定でもボット。**zuul・jenkins も、`extra_bots.txt` なしで拾える。外れる 4 個は次のとおり。
  - 今の一覧の照合で、名前だけが一致して、人間をボットとしていた 3 個（Mikhail Khodos・lakshman・Eddie Lin。メッセージ計 227 件）。新しい判定では人間になる（意図どおり）。
  - rocktown（GerrymanderConfig に載っているが、名前でしか当たらない。§7.2）。
- **新たにボットになるのは 93 個**（サードパーティの CI、OpenStack Proposal Bot・Release Bot など）。全件の名前を確かめ、すべて本物の CI・ボットだった。今の分析では、これらのメッセージ（約 37 万件）を人間のレビューとして数え、Proposal Bot・Release Bot が作った約 2,900 件のチケットを人間が作ったものとしていた。**回し直しで正解ラベル（翌日までに人間のレビューが付いたか）が変わる。**
- 決まりごとの内訳は、① 138 個・③ 102 個・④ 4 個（④ の 4 行はすべて実データのアカウントに当たった）。
- 実装前の確認では 248 個としていたが、これは ④ を古い一覧（名前でも照合）で数えた値で、上の 4 個を含んでいた。実装後に新しいコードで数え直した値が 244 個である（2026-10-11）。

### 8.2 候補のリポジトリ

| ホスト | ボットのアカウント | ボットのメッセージの割合 | ボットが作ったチケットの割合 |
|---|---|---|---|
| ChromiumOS（8 件） | 91 | 23% | 55% |
| Android（2 件） | 23 | 52% | 31% |
| LibreOffice | 8 | 56% | ほぼ 0% |
| Qt（qtbase・qtcreator） | 17 | 35% | 12% |

この表は ③ を `name` だけに、区切りのある形だけで当てたときの値。決まりの全体を当てると、次が加わる。

- ③ の区切りなしの形：ChromiumOS で CopyBot（`cros.copybot.ota@gmail.com`）・AI TestBot（`ai.code.reviewer.for.ibm@gmail.com`）・ChromeBot（`chrome-bot@google.com`）、Qt で QtCIAnalysisBot・QtSecurityBot・QtAPIReviewBot・Qt LanceBot・QtWelcomeBot・QtStaticAnalysisBot・QtGitHubBot。いずれもメールアドレスや活動からボットと分かるもので、人間の誤判定はなかった。
- ③ を `display_name` にも当てる：ChromiumOS で rt-node-js-v8（表示名「🤖 rt-node-js-v8 Bot」）、Qt で ciautostagebot。
- ④ の一覧（§2.4）：CopyBot Service Account・Copybara Service・v8 autoroll・adservices-enrollment・Weblate。

実装後に新しいコードで数え直した値（2026-10-11。メッセージの割合は `autogenerated:` の印のないメッセージで数えた値）。ボットのアカウントの数は、上の表に加わる分を足した数と一致した。

| ホスト | ボットのアカウント | ボットのメッセージの割合 | ボットが作ったチケットの割合 |
|---|---|---|---|
| ChromiumOS（8 件） | 99（91 ＋ 区切りなし 3 ＋ 表示名 1 ＋ 一覧 4） | 23.5% | 56.2% |
| Android（2 件） | 23 | 51.8% | 31.2% |
| LibreOffice | 9（8 ＋ 一覧 1） | 56.4% | 0.1% |
| Qt（qtbase・qtcreator） | 25（17 ＋ 区切りなし 7 ＋ 表示名 1） | 36.1% | 12.0% |

### 8.3 `display_name` にも ③ を当てたときの影響

`name` だけに当てた場合と結果が変わるアカウントは、全ホストで 14 個。すべて表示名で CI・Bot を名乗るアカウントで、**人間の誤判定はなかった**。うち 3 個（Android Autosubmit Bot・Skia Gold Bot・ZadaraStorage VPSA CI）は、①・② ですでに拾えていた。

### 8.4 ③ で人間らしい名前を拾った例

「Rz _ci」（ChromiumOS、活動 14 回）の 1 件だけ。区切りのある最後の語が `ci` のため ③ に当たる。人間かボットかは確かめられていない。影響が小さいので、決まりの例外は設けない。

---

## 9. テスト

新しいテストは `tests/utils/test_bot_detection.py` に作る。既存のテストは扱わない（古い判定を参照しているもの、対象のコードがすでにないものを含め、手を付けない。2026-10-11 本人判断）。古い判定の関数を削除するので、`tests/analysis/background_problem/priority_distribution/test_data_loader.py` などは動かなくなるが、それでよい。

実装のときに、次を確かめる。

| 対象 | 確かめること |
|---|---|
| ③ の決まり | 当てはまる：`Qt CI Bot`・`Cloudbase Nova Hyper-V CI`・`huawei-cinder-ci`・`QtCIAnalysisBot`・`🤖 rt-node-js-v8 Bot`。当てはまらない：`Abbot`・`Talbot`・`Mikhail Khodos`・`ciautostagebot`（区切りなし・小文字） |
| ④ の照合 | ホストが違えば当たらない。名前だけが一致しても当たらない。`account_id`・`username`・`email` のどれか 1 つで当たる |
| 一覧ファイル | 全行に `host`・`source`・`url` があり、`account_id`・`username`・`email` のどれかが埋まっている |
| 全体 | §8 の確認の結果（OpenStack で 244 個、今の 155 個のうち 151 個を含む、など）を、新しいコードで再現できる |
| 自動の記録（2026-10-16 追加） | 印がなくても §3.1 の文は自動の記録、`Patch Set N: ` が前に付いても同じ。Qt の文は Qt のホストだけ。書いた人がいないものは自動の記録。理由の文が続く `Abandoned` は人のメッセージ。as のないステージングの文・`recheck` は人のメッセージ。自動の記録の中の投票は `votes_of` で取れる。古い形式の既定の文は点数に、独自の文は読まない |

---

## 10. 決定の記録

| 日付 | 決定 |
|---|---|
| 2026-10-11 | 判定を ①〜④ の和にする。③ はゆるい版（区切りありは大小区別なし、区切りなしは大文字の Bot・CI のみ）で、`name`・`display_name` に当てる |
| 2026-10-11 | 一覧はホスト必須・名前では照合しない。①〜③ で拾えず出どころのあるものだけ載せる。道具の公式資料＋投稿の定型文でつながるもの（CopyBot・Copybara・Weblate）も載せる |
| 2026-10-11 | ボットが作ったチケットは予測の対象に入れる |
| 2026-10-11 | ① が収集時点の状態であることと、取りこぼし（§7.2）は限界として書く |
| 2026-10-11 | 5 か所すべて（今の分析・background_problem・選定・呼ばれていない古い実装）を、実装のときに一斉に切り替える。今の分析の結果は今後使わないため、動かなくなってもよい（当初は「回し直しのときに切り替える」としていたが、寄与度の計算を止めたので改めた） |
| 2026-10-11 | 古い一覧ファイル 3 つは、実装のときに、切り替えて結果を確かめてから削除する |
| 2026-10-11 | 古い判定の関数（`load_bot_names`・`is_bot`・`_load_gerrymander_bots` など、§6.1 の 5 か所にある各実装）も、実装のときに削除する（本人指示） |
| 2026-10-11 | 参照した URL と一覧の中身は `src/config/README.md` に折りたたみで載せる。CSV が正本 |
| 2026-10-11 | 実装。新しいコードで数え直し、OpenStack は 244 個（実装前の 248 は古い一覧で照合した値）。§8.1・§9 の数を直した。rocktown は名前でしか当たらないため、一覧に加えず取りこぼしとして扱う（§7.2。本人判断）。古い一覧ファイル 3 つを削除 |
| 2026-10-16 | 印のない自動の記録も除く（§3.1）。公式の資料で確かめた文だけ（Gerrit 本体・Qt のプラグイン・Qt の古い改造版）と、書いた人がいないメッセージ。確かめられなかった文は除かない |
| 2026-10-16 | 放棄・復活・後回し・再開の後に理由の文が続くものは人のメッセージ。自動の記録の中の投票は人の投票として数える（§3.2） |
| 2026-10-16 | 正解ラベルと新しい特徴量で同じ決まりを使う。選定の E2 の数え直しは論文の後（§3.3） |
| 2026-10-16 | 前置き `Patch Set N: ` を外して照らすのは Gerrit 本体の文だけにする。Qt の文は 1 行目をそのまま照らす（§3.1） |
| 2026-10-16 | ④ に Qt Continuous Integration System を加える（照合の値は実装のときに確かめる）。Mr. Jenkins は根拠がないので加えない（§2.4・§7.3） |
