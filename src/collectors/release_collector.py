import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

# --- モジュールのインポート設定 ---
# このスクリプトが単体で実行された場合でも、
# プロジェクト内の他モジュールを正しくインポートするための設定
try:
    # 通常のパッケージとしてのインポートを試みる
    from ..config import path as app_path
    from ..utils import constants
except ImportError:
    # スクリプトを直接実行した場合（例: `python src/collectors/release_collector.py`）
    # プロジェクトのルートディレクトリをシステムパスに追加する
    # (このファイルの2つ上の親ディレクトリがプロジェクトルート)
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from src.config import path as app_path
    from src.utils import constants

# --- ロガー設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- 定数 ---
# OpenStackのリリース情報を管理するリポジトリ 
RELEASES_REPO_URL = "https://opendev.org/openstack/releases"

class ReleaseCollector:
    """
    OpenStackのリリース情報を `openstack/releases` Gitリポジトリから収集し、
    CSVファイルとして保存するクラス。

    Attributes:
        data_dir (Path): 収集したデータを保存するディレクトリ。
        local_repo_path (Path): `openstack/releases` リポジトリのローカルクローンパス。
        output_path (Path): 出力するCSVファイルのパス。
        target_components (set[str]): 収集対象のコアコンポーネント名セット。
    """

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir (Path): データ保存用のベースディレクトリ。
        """
        self.data_dir = data_dir
        self.local_repo_path = self.data_dir / "releases_repo"
        self.output_path = self.data_dir / "releases_summary.csv"
        self.target_components = set(constants.OPENSTACK_CORE_COMPONENTS)
        logging.info(f"対象コンポーネント: {self.target_components}")
        logging.info(f"リポジトリパス: {self.local_repo_path}")
        logging.info(f"出力ファイルパス: {self.output_path}")

    def _run_git_command(self, command: list[str], cwd: Path) -> str:
        """指定されたディレクトリでgitコマンドを実行し、標準出力を返す。"""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            return result.stdout.strip()
        except FileNotFoundError:
            logging.error("Gitがインストールされていないか、PATHが通っていません。")
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"Gitコマンドの実行に失敗しました: git {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr.strip()}")
            raise

    def _prepare_repository(self):
        """リポジトリが存在しない場合はクローンし、存在する場合は更新する。"""
        if self.local_repo_path.exists():
            logging.info(f"既存のリポジトリを更新中: {self.local_repo_path}")
            self._run_git_command(["pull", "origin", "master"], cwd=self.local_repo_path)
        else:
            logging.info(f"{RELEASES_REPO_URL} からリポジトリをクローン中...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._run_git_command(
                ["clone", RELEASES_REPO_URL, str(self.local_repo_path)],
                cwd=self.data_dir
            )
        logging.info("リポジトリの準備が完了しました。")

    def _get_release_date(self, file_path: Path, version: str) -> str | None:
        """
        Gitログから、特定のバージョンがファイルに追加されたコミットの日時を取得する。
        """
        try:
            # `-S`オプションで、指定文字列の変更があったコミットを検索
            # `--pretty=format:%cI`でコミッターの日時をISO 8601形式で取得
            iso_date_str = self._run_git_command(
                [
                    "log",
                    "--all",
                    "-S", f"version: {version}",
                    "-n", "1",
                    "--pretty=format:%cI",
                    "--", str(file_path.relative_to(self.local_repo_path))
                ],
                cwd=self.local_repo_path
            )
            if iso_date_str:
                return datetime.fromisoformat(iso_date_str).strftime('%Y-%m-%d')
            logging.warning(f"リリース日が見つかりませんでした: version={version}, file={file_path.name}")
            return None
        except subprocess.CalledProcessError:
            logging.warning(f"リリース日の取得に失敗しました: version={version}, file={file_path.name}")
            return None

    def collect_and_save_releases(self):
        """
        リリース情報を収集し、整形してCSVファイルに保存するメインメソッド。
        """
        self._prepare_repository()

        deliverables_path = self.local_repo_path / "deliverables"
        all_releases = []

        logging.info(f"{deliverables_path} 内のDeliverableファイルを走査します...")
        for series_dir in deliverables_path.iterdir():
            if not series_dir.is_dir():
                continue

            for yaml_file in series_dir.glob("*.yaml"):
                component_name = yaml_file.stem
                if component_name not in self.target_components:
                    continue

                logging.info(f"処理中: {yaml_file.relative_to(self.local_repo_path)}")
                with open(yaml_file, "r", encoding="utf-8") as f:
                    try:
                        data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        logging.error(f"YAMLファイルのパースに失敗しました: {yaml_file}: {e}")
                        continue

                if "releases" not in data or not isinstance(data["releases"], list):
                    continue
                
                # YAMLファイルからリリース情報を抽出 
                for release in data["releases"]:
                    version = release.get("version")
                    if not version:
                        continue
                    
                    # このリリースが対象コンポーネントのものか最終確認
                    # `releases.projects.repo`に`openstack/{component_name}`が含まれるかチェック
                    projects = release.get("projects", [])
                    repo_name_prefix = "openstack/"
                    is_target_repo = any(
                        p.get("repo") == f"{repo_name_prefix}{component_name}" for p in projects
                    )
                    
                    if not is_target_repo:
                        continue

                    release_date = self._get_release_date(yaml_file, version)
                    all_releases.append({
                        "component": component_name,
                        "version": version,
                        "release_date": release_date,
                        "release_notes_url": data.get("release-notes")
                    })

        if not all_releases:
            logging.warning("リリース情報が1件も収集されませんでした。")
            return

        df = pd.DataFrame(all_releases)
        df.sort_values(by=["component", "release_date"], ascending=[True, False], inplace=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        logging.info(f"計 {len(df)} 件のリリース情報を収集し、CSVファイルに保存しました。")
        logging.info(f"出力先: {self.output_path}")


if __name__ == "__main__":
    """
    このスクリプトを直接実行した際のエントリポイント。
    """
    logging.info("===== OpenStackリリース情報収集スクリプトを開始します =====")
    
    # `path.py`で定義されたデータディレクトリを基準とする
    # 例: data/openstack/
    output_base_dir = app_path.DEFAULT_DATA_DIR / "openstack"

    try:
        collector = ReleaseCollector(data_dir=output_base_dir)
        collector.collect_and_save_releases()
    except Exception as e:
        logging.critical(f"スクリプトの実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)

    logging.info("===== スクリプトの実行が正常に終了しました =====")