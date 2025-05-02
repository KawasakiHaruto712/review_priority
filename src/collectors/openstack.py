"""
OpenStack Data Collector

このモジュールはOpenStackのコアコンポーネントからコードレビュー関連データを収集します。
期間: constants.pyで定義されたSTART_DATEからEND_DATEまで
"""

import argparse
import os
import sys
import json
import csv
import time
import requests
import logging
import configparser
from pathlib import Path
from git import Repo, Git
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from config.path import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_REPOS_DIR
from utils.constants import OPENSTACK_CORE_COMPONENTS, START_DATE, END_DATE

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openstack_data_collector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class OpenStackDataCollector:
    """OpenStackコアコンポーネントからデータを収集するクラス"""
    
    # Gerritのベースエンドポイント
    GERRIT_API_BASE = "https://review.opendev.org/changes/"
    
    # GitHubのベースエンドポイント
    GITHUB_API_BASE = "https://api.github.com/repos/openstack/"
    
    def __init__(self, config_path=None):
        """
        初期化メソッド
        
        Args:
            config_path: 設定ファイルのパスを指定します
        """
        
        # 設定ファイルの読み込み
        self.config = self._load_config(DEFAULT_CONFIG)
        
        # 出力ディレクトリの設定
        self.data_dir = Path(DEFAULT_DATA_DIR)
        self.repos_dir = Path(DEFAULT_REPOS_DIR)
        
        # 出力ディレクトリの作成
        self._ensure_directories()
        
        # GitHub API トークン
        load_dotenv()
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        self.github_token = GITHUB_TOKEN
        if self.github_token == "your_github_token_here" or not self.github_token:
            self.github_token = self.config.get('API', 'github_token', fallback=None)

        # OpenStackコアコンポーネントのリストを初期化
        self.openstack_core_components = OPENSTACK_CORE_COMPONENTS
        
    def _load_config(self, config_path=None):
        """
        設定ファイルを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            configparser.ConfigParser: 設定オブジェクト
        """
        config = configparser.ConfigParser()
        
        # 設定ファイルがあれば読み込み
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                logger.info(f"設定ファイルを読み込み中: {config_path}")
                config.read(config_path)
            else:
                logger.warning(f"設定ファイルが見つかりません: {config_path}")
                # デフォルト設定のファイルを作成
                with open(config_path, 'w') as f:
                    config.write(f)
                logger.info(f"デフォルト設定を作成しました: {config_path}")
        
        return config
        
    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        dirs = [
            self.data_dir,
            self.repos_dir,
            self.data_dir / "commits",
            self.data_dir / "reviews",
            self.data_dir / "pull_requests",
            self.data_dir / "issues",
            self.data_dir / "stats",
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"ディレクトリを確認/作成: {directory}")
    
    def run(self):
        """データ収集の実行"""
        logger.info("OpenStackデータ収集を開始します")
        
        for component in self.openstack_core_components:
            logger.info(f"コンポーネント '{component}' の処理を開始します")
            
            # Gitリポジトリのクローンとコミットデータの収集
            self.collect_git_data(component)
            
            # Gerritレビューデータの収集
            self.collect_gerrit_data(component)
            
            # GitHubのIssueとPull Requestの収集
            self.collect_github_data(component)
            
            # 統計情報の集計
            self.generate_stats(component)
            
        logger.info("全てのデータ収集が完了しました")

    def collect_git_data(self, component):
        """
        Gitリポジトリからコミットデータを収集
        
        Args:
            component: コンポーネント名
        """
        repo_path = self.repos_dir / component
        repo_url = f"https://opendev.org/openstack/{component}"
        
        logger.info(f"{component}: Gitリポジトリデータの収集を開始")
        
        # リポジトリのクローンまたは更新
        if repo_path.exists():
            logger.info(f"{component}: リポジトリを更新します")
            repo = Repo(repo_path)
            repo.git.fetch(all=True)
            repo.git.reset('--hard', 'origin/master')
        else:
            logger.info(f"{component}: リポジトリをクローンします: {repo_url}")
            repo = Repo.clone_from(repo_url, repo_path)
        
        # 期間内のコミットを取得
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"{component}: {start_date_str}から{end_date_str}までのコミットを取得します")
        
        # コミットログの取得（日付フィルタ付き）
        git_cmd = [
            'git', 'log',
            f'--after={start_date_str}',
            f'--before={end_date_str}',
            '--pretty=format:%H|%an|%ae|%ad|%s',
            '--date=iso'
        ]
        git = Git(repo_path)
        commits_raw = git.execute(git_cmd)
        
        # 結果を解析してCSVに保存
        commits_file = self.data_dir / "commits" / f"{component}_commits.csv"
        
        with open(commits_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['commit_hash', 'author_name', 'author_email', 'commit_date', 'subject'])
            
            if commits_raw:
                for commit_line in commits_raw.split('\n'):
                    if commit_line:
                        writer.writerow(commit_line.split('|'))
        
        logger.info(f"{component}: コミットデータを保存しました: {commits_file}")
        
        # ファイル変更の統計を収集
        self.collect_file_changes(component, repo_path)

    def collect_file_changes(self, component, repo_path):
        """
        期間内のファイル変更統計を収集
        
        Args:
            component: コンポーネント名
            repo_path: リポジトリのパス
        """
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"{component}: ファイル変更統計を収集します")
        
        # ファイル変更の統計を取得
        git = Git(repo_path)
        changes_cmd = [
            'git', 'log',
            f'--after={start_date_str}',
            f'--before={end_date_str}',
            '--numstat',
            '--pretty=%H'
        ]
        changes_raw = git.execute(changes_cmd)
        
        changes_file = self.data_dir / "commits" / f"{component}_file_changes.csv"
        
        current_commit = None
        
        with open(changes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['commit_hash', 'added_lines', 'deleted_lines', 'file_path'])
            
            if changes_raw:
                for line in changes_raw.split('\n'):
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # コミットハッシュの行
                    if not line[0].isdigit():
                        current_commit = line
                    else:
                        # ファイル変更の行
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            added = parts[0] if parts[0] != '-' else '0'
                            deleted = parts[1] if parts[1] != '-' else '0'
                            file_path = parts[2]
                            writer.writerow([current_commit, added, deleted, file_path])

        logger.info(f"{component}: ファイル変更データを保存しました: {changes_file}")

    def collect_gerrit_data(self, component):
        """
        Gerritからコードレビューデータを収集
        
        Args:
            component: コンポーネント名
        """
        logger.info(f"{component}: Gerritレビューデータの収集を開始")
        
        # 開始と終了のタイムスタンプ（秒）
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Gerrit APIのクエリパラメータ
        query_params = [
            f"project:openstack/{component}",
            f"after:{start_timestamp}",
            f"before:{end_timestamp}",
        ]
        
        query_str = "+".join(query_params)
        
        # ページネーション用の変数
        start = 0
        limit = 100  # 一度に取得する最大数
        more_changes = True
        all_reviews = []
        
        reviews_file = self.data_dir / "reviews" / f"{component}_reviews.json"
        
        # Gerrit APIでデータ取得
        while more_changes:
            url = f"{self.GERRIT_API_BASE}?q={query_str}&o=DETAILED_ACCOUNTS&o=DETAILED_LABELS&o=MESSAGES&n={limit}&S={start}"
            logger.debug(f"Gerrit APIリクエスト: {url}")
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Gerritの応答は ")]}'" で始まるため、これを除去する
                data = response.text[4:] if response.text.startswith(")]}'") else response.text
                changes = json.loads(data)
                
                if not changes:
                    more_changes = False
                else:
                    all_reviews.extend(changes)
                    
                    # 次のページがあるかチェック
                    if len(changes) < limit:
                        more_changes = False
                    else:
                        start += limit
                
                # APIレート制限を避けるための待機
                time.sleep(float(self.config.get('API', 'request_delay', fallback='1.0')))
                
            except Exception as e:
                logger.error(f"Gerrit APIリクエスト中にエラーが発生しました: {e}")
                more_changes = False
        
        # 結果をJSONファイルに保存
        with open(reviews_file, 'w', encoding='utf-8') as f:
            json.dump(all_reviews, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{component}: {len(all_reviews)}件のレビューデータを保存しました: {reviews_file}")
        
        # レビュー統計の抽出と保存
        self.extract_review_stats(component, all_reviews)

    def extract_review_stats(self, component, reviews):
        """
        レビューデータから統計情報を抽出
        
        Args:
            component: コンポーネント名
            reviews: レビューデータのリスト
        """
        reviews_stats_file = self.data_dir / "stats" / f"{component}_review_stats.csv"
        
        with open(reviews_stats_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'change_id', 'subject', 'owner', 'created', 'updated', 'status',
                'reviewers_count', 'comments_count', 'revisions_count', 'time_to_merge_days'
            ])
            
            for review in reviews:
                if 'id' not in review:
                    continue
                    
                # 基本情報
                change_id = review.get('change_id', '')
                subject = review.get('subject', '')
                owner = review.get('owner', {}).get('name', '')
                created = review.get('created', '')
                updated = review.get('updated', '')
                status = review.get('status', '')
                
                # レビュアー数
                reviewers = set()
                for label in review.get('labels', {}).values():
                    for vote in label.get('all', []):
                        if vote.get('value') and vote.get('name'):
                            reviewers.add(vote.get('name'))
                
                # コメント数
                comments_count = len(review.get('messages', []))
                
                # リビジョン数
                revisions_count = len(review.get('revisions', {}))
                
                # マージまでの時間（日数）
                time_to_merge_days = ''
                if status == 'MERGED' and created:
                    created_date = datetime.datetime.fromisoformat(created.replace('Z', '+00:00'))
                    updated_date = datetime.datetime.fromisoformat(updated.replace('Z', '+00:00'))
                    time_to_merge = updated_date - created_date
                    time_to_merge_days = time_to_merge.total_seconds() / (24 * 3600)
                
                writer.writerow([
                    change_id, subject, owner, created, updated, status,
                    len(reviewers), comments_count, revisions_count, time_to_merge_days
                ])
        
        logger.info(f"{component}: レビュー統計データを保存しました: {reviews_stats_file}")

    def collect_github_data(self, component):
        """
        GitHubからIssueやPull Requestのデータを収集
        
        Args:
            component: コンポーネント名
        """
        logger.info(f"{component}: GitHub IssueとPRデータの収集を開始")
        
        # 期間の設定
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Issue取得
        self._collect_github_issues(component, start_date_str, end_date_str)
        
        # Pull Request取得
        self._collect_github_prs(component, start_date_str, end_date_str)

    def _collect_github_issues(self, component, start_date_str, end_date_str):
        """
        GitHubからIssueを収集
        
        Args:
            component: コンポーネント名
            start_date_str: 開始日（YYYY-MM-DD）
            end_date_str: 終了日（YYYY-MM-DD）
        """
        issues_file = self.data_dir / "issues" / f"{component}_issues.json"
        
        # GitHub APIのヘッダ設定
        headers = {}
        if self.github_token:
            headers['Authorization'] = f"token {self.github_token}"
        
        # ページネーション用の変数
        page = 1
        per_page = 100
        all_issues = []
        has_more = True

        while has_more:
            # クエリパラメータを構築
            query = f"repo:openstack/{component} is:issue created:{start_date_str}..{end_date_str}"
            url = f"https://api.github.com/search/issues?q={query}&sort=created&order=asc&page={page}&per_page={per_page}"
            
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                items = data.get('items', [])
                all_issues.extend(items)
                
                # 次のページがあるかチェック
                total_count = data.get('total_count', 0)
                if page * per_page >= total_count or not items:
                    has_more = False
                else:
                    page += 1
                
                # APIレート制限を避けるための待機
                time.sleep(float(self.config.get('API', 'request_delay', fallback='1.0')))
                
                # レート制限の確認
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining <= 1:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        sleep_time = max(reset_time - time.time() + 1, 0)
                        logger.warning(f"GitHub APIレート制限に達しました。{sleep_time}秒待機します...")
                        time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"GitHub Issues APIリクエスト中にエラーが発生しました: {e}")
                has_more = False
        
        # 結果をJSONファイルに保存
        with open(issues_file, 'w', encoding='utf-8') as f:
            json.dump(all_issues, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{component}: {len(all_issues)}件のIssueデータを保存しました: {issues_file}")

    def _collect_github_prs(self, component, start_date_str, end_date_str):
        """
        GitHubからPull Requestを収集
        
        Args:
            component: コンポーネント名
            start_date_str: 開始日（YYYY-MM-DD）
            end_date_str: 終了日（YYYY-MM-DD）
        """
        prs_file = self.data_dir / "pull_requests" / f"{component}_prs.json"
        
        # GitHub APIのヘッダ設定
        headers = {}
        if self.github_token:
            headers['Authorization'] = f"token {self.github_token}"
        
        # ページネーション用の変数
        page = 1
        per_page = 100
        all_prs = []
        has_more = True
        
        while has_more:
            # クエリパラメータを構築
            query = f"repo:openstack/{component} is:pr created:{start_date_str}..{end_date_str}"
            url = f"https://api.github.com/search/issues?q={query}&sort=created&order=asc&page={page}&per_page={per_page}"
            
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                items = data.get('items', [])
                all_prs.extend(items)
                
                # 次のページがあるかチェック
                total_count = data.get('total_count', 0)
                if page * per_page >= total_count or not items:
                    has_more = False
                else:
                    page += 1
                
                # APIレート制限を避けるための待機
                time.sleep(float(self.config.get('API', 'request_delay', fallback='1.0')))
                
                # レート制限の確認
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining <= 1:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        sleep_time = max(reset_time - time.time() + 1, 0)
                        logger.warning(f"GitHub APIレート制限に達しました。{sleep_time}秒待機します...")
                        time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"GitHub PRs APIリクエスト中にエラーが発生しました: {e}")
                has_more = False
        
        # 結果をJSONファイルに保存
        with open(prs_file, 'w', encoding='utf-8') as f:
            json.dump(all_prs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{component}: {len(all_prs)}件のPull Requestデータを保存しました: {prs_file}")

    def generate_stats(self, component):
        """
        コンポーネントの統計情報を生成
        
        Args:
            component: コンポーネント名
        """
        logger.info(f"{component}: 統計情報の生成を開始")
        
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(END_DATE, '%Y-%m-%d')

        stats = {
            'component': component,
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
        }
        
        # コミット統計
        commits_file = self.data_dir / "commits" / f"{component}_commits.csv"
        if commits_file.exists():
            commit_count = sum(1 for _ in open(commits_file)) - 1  # ヘッダー行を除く
            stats['commit_count'] = commit_count
        
        # ファイル変更統計
        changes_file = self.data_dir / "commits" / f"{component}_file_changes.csv"
        if changes_file.exists():
            with open(changes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total_added = 0
                total_deleted = 0
                file_types = {}
                
                for row in reader:
                    total_added += int(row['added_lines']) if row['added_lines'].isdigit() else 0
                    total_deleted += int(row['deleted_lines']) if row['deleted_lines'].isdigit() else 0
                    
                    # ファイル拡張子の統計
                    file_path = row['file_path']
                    if '.' in file_path:
                        ext = file_path.split('.')[-1].lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
            
            stats['lines_added'] = total_added
            stats['lines_deleted'] = total_deleted
            stats['file_types'] = file_types
        
        # レビュー統計
        reviews_file = self.data_dir / "reviews" / f"{component}_reviews.json"
        if reviews_file.exists():
            try:
                with open(reviews_file, 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
                    
                stats['review_count'] = len(reviews)
                
                # ステータス別の集計
                status_counts = {}
                for review in reviews:
                    status = review.get('status', 'UNKNOWN')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                stats['review_status_counts'] = status_counts
                
                # マージされたレビューのみの統計
                merged_reviews = [r for r in reviews if r.get('status') == 'MERGED']
                stats['merged_review_count'] = len(merged_reviews)
                
            except Exception as e:
                logger.error(f"レビューデータの読み込み中にエラーが発生しました: {e}")
        
        # レビュー詳細統計
        review_stats_file = self.data_dir / "stats" / f"{component}_review_stats.csv"
        if review_stats_file.exists():
            with open(review_stats_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                reviewers_counts = []
                comments_counts = []
                revisions_counts = []
                time_to_merge_days = []
                
                for row in reader:
                    reviewers_counts.append(int(row['reviewers_count']) if row['reviewers_count'] else 0)
                    comments_counts.append(int(row['comments_count']) if row['comments_count'] else 0)
                    revisions_counts.append(int(row['revisions_count']) if row['revisions_count'] else 0)
                    
                    if row['time_to_merge_days']:
                        try:
                            time_to_merge_days.append(float(row['time_to_merge_days']))
                        except ValueError:
                            pass
            
            if reviewers_counts:
                stats['avg_reviewers_per_change'] = sum(reviewers_counts) / len(reviewers_counts)
            
            if comments_counts:
                stats['avg_comments_per_change'] = sum(comments_counts) / len(comments_counts)
            
            if revisions_counts:
                stats['avg_revisions_per_change'] = sum(revisions_counts) / len(revisions_counts)
            
            if time_to_merge_days:
                stats['avg_time_to_merge_days'] = sum(time_to_merge_days) / len(time_to_merge_days)
        
        # GitHub Issue統計
        issues_file = self.data_dir / "issues" / f"{component}_issues.json"
        if issues_file.exists():
            try:
                with open(issues_file, 'r', encoding='utf-8') as f:
                    issues = json.load(f)
                    
                stats['issue_count'] = len(issues)
                
                # オープン/クローズの集計
                open_issues = sum(1 for issue in issues if issue.get('state') == 'open')
                closed_issues = sum(1 for issue in issues if issue.get('state') == 'closed')
                
                stats['open_issue_count'] = open_issues
                stats['closed_issue_count'] = closed_issues
                
            except Exception as e:
                logger.error(f"Issueデータの読み込み中にエラーが発生しました: {e}")
        
        # GitHub PR統計
        prs_file = self.data_dir / "pull_requests" / f"{component}_prs.json"
        if prs_file.exists():
            try:
                with open(prs_file, 'r', encoding='utf-8') as f:
                    prs = json.load(f)
                    
                stats['pr_count'] = len(prs)
                
                # オープン/クローズ/マージの集計
                open_prs = sum(1 for pr in prs if pr.get('state') == 'open')
                closed_prs = sum(1 for pr in prs if pr.get('state') == 'closed')
                
                stats['open_pr_count'] = open_prs
                stats['closed_pr_count'] = closed_prs
                
            except Exception as e:
                logger.error(f"PRデータの読み込み中にエラーが発生しました: {e}")
        
        # 統計結果をJSONファイルに保存
        stats_file = self.data_dir / "stats" / f"{component}_summary.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{component}: 統計情報を保存しました: {stats_file}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='OpenStack コアコンポーネントデータ収集')
    parser.add_argument('--config', dest='config_path', default='src/config/path.py',
                      help='設定ファイルのパス (デフォルト: src/config/path.py)')
    
    args = parser.parse_args()
    
    # データ収集の実行
    collector = OpenStackDataCollector(config_path=args.config_path)
    collector.run()