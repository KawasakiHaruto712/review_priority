import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
import warnings

# warningsを無視
warnings.filterwarnings('ignore', category=FutureWarning)

class TrajectoryBuilder:
    """
    OpenStackのレビューデータから専門家の軌跡を構築するクラス
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.pr_details = {}

    def _load_pr_details(self, component: str):
        """各PRの詳細情報（特にメッセージ）をJSONから読み込む"""
        changes_dir = self.data_dir / component / "changes"
        for json_file in changes_dir.glob("change_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Gerritのタイムスタンプ形式をdatetimeオブジェクトに変換
                    for msg in data.get('messages', []):
                        ts_str = msg.get('date', '').split('.')[0]
                        if ts_str:
                            msg['datetime'] = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                    self.pr_details[data['change_number']] = data
            except (json.JSONDecodeError, KeyError) as e:
                # 警告メッセージを日本語に変更
                print(f"警告: {json_file} の読み込みまたは解析に失敗しました: {e}")

    def build(self, component: str):
        """
        指定されたコンポーネントの軌跡データを構築する
        """
        component_dir = self.data_dir / component / "changes"
        csv_path = component_dir / "changes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} が見つかりません。先にopenstack.pyを実行してください。")

        df_prs = pd.read_csv(csv_path)
        df_prs['created'] = pd.to_datetime(df_prs['created'].str.split('.').str[0])
        df_prs['merged'] = pd.to_datetime(df_prs['merged'].str.split('.').str[0])

        self._load_pr_details(component)

        review_events = []
        for pr_num, details in self.pr_details.items():
            for msg in details.get('messages', []):
                author_name = msg.get('author', {}).get('name', '')
                if 'Zuul' not in author_name and 'Jenkins' not in author_name and 'Gerrit' not in author_name:
                    if 'datetime' in msg:
                        review_events.append({
                            'time': msg['datetime'],
                            'reviewed_pr': pr_num,
                            'author': author_name
                        })
        
        review_events.sort(key=lambda x: x['time'])

        trajectories = []
        for event in review_events:
            event_time = event['time']
            reviewed_pr_num = event['reviewed_pr']

            open_prs_df = df_prs[
                (df_prs['created'] <= event_time) & (df_prs['merged'] > event_time)
            ]

            if len(open_prs_df) < 2:
                continue
            
            if reviewed_pr_num not in open_prs_df['change_number'].values:
                continue

            state_pr_numbers = open_prs_df['change_number'].tolist()
            action_pr_number = reviewed_pr_num
            
            trajectories.append({
                "state": state_pr_numbers,
                "action": action_pr_number,
                "timestamp": event_time
            })
            
        return trajectories, df_prs

class FeatureExtractor:
    """
    PRデータから特徴量ベクトルを生成するクラス
    """
    def __init__(self, pr_df, pr_details):
        self.pr_df = pr_df.set_index('change_number')
        self.pr_details = pr_details
        # 特徴量の名前を定義
        self.feature_names = [
            "time_since_creation",  # 作成からの経過時間(日)
            "time_since_last_update", # 最終更新からの経過時間(日)
            "lines_inserted",       # 追加行数
            "lines_deleted",        # 削除行数
            "files_changed",        # 変更ファイル数
            "comments_count",       # 現在までのコメント数
            "is_merge_conflict",    # マージコンフリクトの可能性
            "bias"                  # バイアス項
        ]

    def _get_pr_data(self, pr_number):
        """PR番号から基本データと詳細データを取得"""
        basic_data = self.pr_df.loc[pr_number]
        detail_data = self.pr_details.get(pr_number, {})
        return basic_data, detail_data

    def extract(self, pr_number, current_time):
        """単一のPRから特徴量ベクトルを抽出"""
        pr_data, pr_details = self._get_pr_data(pr_number)

        time_since_creation = (current_time - pr_data['created']).total_seconds() / (3600 * 24)

        last_update_time = pr_data['created']
        if pr_details and 'messages' in pr_details:
            message_times = [msg['datetime'] for msg in pr_details['messages'] if 'datetime' in msg and msg['datetime'] <= current_time]
            if message_times:
                last_update_time = max(message_times)
        time_since_last_update = (current_time - last_update_time).total_seconds() / (3600 * 24)

        lines_inserted = pr_data['lines_inserted']
        lines_deleted = pr_data['lines_deleted']
        files_changed = pr_data['files_changed']
        
        comments_count = 0
        if pr_details and 'messages' in pr_details:
            comments_count = len([msg for msg in pr_details['messages'] if msg['datetime'] <= current_time])

        is_merge_conflict = 1 if 'merge conflict' in pr_data.get('subject', '').lower() else 0
        bias = 1.0
        
        features = np.array([
            np.log1p(time_since_creation),
            np.log1p(time_since_last_update),
            np.log1p(lines_inserted),
            np.log1p(lines_deleted),
            np.log1p(files_changed),
            np.log1p(comments_count),
            is_merge_conflict,
            bias
        ])
        
        return features

class MaxEntIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) を実装するクラス
    """
    def __init__(self, feature_dim, learning_rate=0.01, epochs=100):
        self.theta = np.random.rand(feature_dim)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def fit(self, trajectories, feature_extractor: FeatureExtractor):
        """軌跡データを用いて報酬関数のパラメータθを学習する"""
        expert_feature_expectation = np.zeros_like(self.theta)
        for traj in trajectories:
            action_features = feature_extractor.extract(traj['action'], traj['timestamp'])
            expert_feature_expectation += action_features
        expert_feature_expectation /= len(trajectories)
        
        # 学習開始のメッセージ
        print("--- 学習開始 ---")
        for epoch in range(self.epochs):
            grad = np.zeros_like(self.theta)
            
            for traj in trajectories:
                state_features = np.array([feature_extractor.extract(s, traj['timestamp']) for s in traj['state']])
                action_features = feature_extractor.extract(traj['action'], traj['timestamp'])
                rewards = state_features @ self.theta
                probabilities = self._softmax(rewards)
                policy_feature_expectation = np.sum(state_features * probabilities[:, np.newaxis], axis=0)
                gradient_for_traj = action_features - policy_feature_expectation
                grad += gradient_for_traj
            
            self.theta += self.learning_rate * (grad / len(trajectories))
            
            if (epoch + 1) % 10 == 0:
                loss = np.linalg.norm(grad / len(trajectories))
                # 学習進捗のメッセージ
                print(f"エポック {epoch+1}/{self.epochs}, 勾配ノルム: {loss:.4f}")

        return self.theta