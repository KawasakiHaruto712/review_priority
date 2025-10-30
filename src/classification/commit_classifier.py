"""
コミットメッセージ分類器
Zero-shot分類を用いてChangeのcommit_messageからラベルを付与
"""
import os
import sys
import json
import logging
import configparser
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from transformers import pipeline

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from src.config.path import DEFAULT_DATA_DIR
from src.utils.constants import START_DATE, END_DATE, OPENSTACK_CORE_COMPONENTS

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_bot_names_from_config(config_path: Path = None) -> List[str]:
    """
    gerrymanderconfig.iniからボット名のリストを読み込む
    
    Args:
        config_path: 設定ファイルのパス（Noneの場合はデフォルトパス）
        
    Returns:
        List[str]: ボット名のリスト
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "gerrymanderconfig.ini"
    
    if not config_path.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return ['jenkins', 'zuul', 'elasticrecheck']  # デフォルト値
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        
        if 'organization' in config and 'bots' in config['organization']:
            bots_str = config['organization']['bots']
            # カンマ区切りで分割し、空白を除去
            bots = [bot.strip() for bot in bots_str.split(',') if bot.strip()]
            logger.info(f"設定ファイルから{len(bots)}個のボット名を読み込みました")
            return bots
        else:
            logger.warning("設定ファイルに[organization]セクションまたはbotsキーが見つかりません")
            return ['jenkins', 'zuul', 'elasticrecheck']
    
    except Exception as e:
        logger.error(f"設定ファイルの読み込みエラー: {e}")
        return ['jenkins', 'zuul', 'elasticrecheck']


class CommitMessageClassifier:
    """
    コミットメッセージをゼロショット分類するクラス
    """
    
    # ラベル定義
    LABELS = {
        'feat': 'コードベースに新機能（内部向け・ユーザー向け問わず）を導入する変更',
        'fix': 'コードベース内のバグや不具合を修正する変更',
        'refactor': 'バグ修正や機能追加を伴わない、コードの構造を改善する変更。保守性の向上を目的とします',
        'docs': 'ドキュメントのみを修正する変更',
        'style': 'コードの意味に影響を与えない、フォーマットや可読性に関する変更（インデント、変数名など）',
        'test': 'テストの追加や既存テストの修正',
        'perf': 'パフォーマンスを向上させるコード変更',
        'ci': 'CI（継続的インテグレーション）の設定ファイルやスクリプトに関する変更',
        'build': 'ビルドシステムや外部依存関係に影響を与える変更',
        'chore': '上記のいずれにも当てはまらない、その他の雑多なタスク（ツール設定など）'
    }
    
    def __init__(self, model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"):
        """
        Args:
            model_name: 使用するゼロショット分類モデル
        """
        self.model_name = model_name
        logger.info(f"ゼロショット分類モデルをロード中: {model_name}")
        
        try:
            # デバイスの自動選択（GPUが利用可能ならGPU、なければCPU）
            import torch
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"
            logger.info(f"使用デバイス: {device_name}")
            
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device
            )
            logger.info("モデルのロードが完了しました")
        except Exception as e:
            logger.error(f"モデルのロードに失敗: {e}")
            raise
    
    def classify(self, commit_message: str) -> Dict[str, any]:
        """
        コミットメッセージを分類
        
        Args:
            commit_message: 分類対象のコミットメッセージ
            
        Returns:
            Dict: 分類結果（ラベル、スコア、全スコアを含む）
        """
        
        # ラベル候補
        candidate_labels = list(self.LABELS.keys())
        
        try:
            # ゼロショット分類を実行
            result = self.classifier(
                commit_message,
                candidate_labels,
                multi_label=False
            )
            
            # 結果を整形
            classification = {
                'label': result['labels'][0],
                'score': result['scores'][0],
                'all_scores': dict(zip(result['labels'], result['scores']))
            }
            
            return classification
            
        except Exception as e:
            logger.error(f"分類エラー: {e}")
            return {
                'label': 'chore',
                'score': 0.0,
                'all_scores': {},
                'error': str(e)
            }
    
    def classify_batch(self, commit_messages: List[str], batch_size: int = 8) -> List[Dict[str, any]]:
        """
        複数のコミットメッセージを一括分類
        
        Args:
            commit_messages: 分類対象のコミットメッセージリスト
            batch_size: バッチサイズ
            
        Returns:
            List[Dict]: 分類結果のリスト
        """
        results = []
        total = len(commit_messages)
        
        for i in range(0, total, batch_size):
            batch = commit_messages[i:i+batch_size]
            logger.info(f"分類進捗: {i}/{total}")
            
            for message in batch:
                result = self.classify(message)
                results.append(result)
        
        logger.info(f"分類完了: {total}件")
        return results


def load_openstack_changes(projects: List[str] = None,
                           start_date: str = START_DATE,
                           end_date: str = END_DATE) -> pd.DataFrame:
    """
    OpenStackのChangeデータを読み込む
    
    Args:
        projects: プロジェクトリスト（Noneの場合は全プロジェクト）
        start_date: 開始日
        end_date: 終了日
        
    Returns:
        pd.DataFrame: Changeデータ
    """
    target_projects = projects if projects else OPENSTACK_CORE_COMPONENTS
    all_changes = []
    
    for project in target_projects:
        project_dir = DEFAULT_DATA_DIR / "openstack" / project / "changes"
        
        if not project_dir.exists():
            logger.warning(f"プロジェクトディレクトリが見つかりません: {project_dir}")
            continue
        
        logger.info(f"プロジェクト '{project}' のデータを読み込み中...")
        
        # JSONファイルを読み込み
        for json_file in project_dir.glob("change_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    change_data = json.load(f)
                
                # 基本情報を抽出
                change = {
                    'change_number': change_data.get('change_number'),
                    'project': change_data.get('project'),
                    'subject': change_data.get('subject'),
                    'commit_message': change_data.get('commit_message'),
                    'created': change_data.get('created'),
                    'updated': change_data.get('updated'),
                    'merged': change_data.get('merged'),
                    'messages': change_data.get('messages', [])
                }
                
                all_changes.append(change)
                
            except Exception as e:
                logger.error(f"ファイル読み込みエラー ({json_file}): {e}")
                continue
        
        logger.info(f"プロジェクト '{project}': {len([c for c in all_changes if project in c.get('project', '')])}件")
    
    # DataFrameに変換
    df = pd.DataFrame(all_changes)
    
    if df.empty:
        logger.warning("データが見つかりませんでした")
        return df
    
    # 日付型に変換
    df['created'] = pd.to_datetime(df['created'])
    df['updated'] = pd.to_datetime(df['updated'])
    df['merged'] = pd.to_datetime(df['merged'])
    
    # 期間でフィルタリング
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    mask = (df['created'] >= start_dt) & (df['created'] <= end_dt)
    df = df[mask]
    
    logger.info(f"期間フィルタ後: {len(df)}件 ({start_date} ～ {end_date})")
    
    return df


def calculate_time_to_first_review(messages: List[Dict], created_time: datetime,
                                   bot_names: List[str] = None) -> Optional[float]:
    """
    最初のレビューまでの時間を計算（日数）
    
    Args:
        messages: メッセージリスト
        created_time: Change作成時刻
        bot_names: ボット名のリスト
        
    Returns:
        float: 最初のレビューまでの日数（レビューがない場合はNone）
    """
    if bot_names is None:
        bot_names = load_bot_names_from_config()
    
    if not messages:
        return None
    
    first_review_time = None
    
    for message in messages:
        try:
            msg_date = pd.to_datetime(message.get('date'))
            author_name = message.get('author', {}).get('name', '')
            
            # ボットによるメッセージは除外
            is_bot = any(bot in author_name.lower() for bot in bot_names)
            
            if not is_bot:
                if first_review_time is None or msg_date < first_review_time:
                    first_review_time = msg_date
        
        except (ValueError, AttributeError, TypeError):
            continue
    
    if first_review_time is None:
        return None
    
    # 日数を計算
    time_diff = (first_review_time - created_time).total_seconds() / 86400.0  # 秒を日に変換
    
    return max(0, time_diff)  # 負の値は0にする


def classify_and_analyze(projects: List[str] = None,
                        start_date: str = START_DATE,
                        end_date: str = END_DATE,
                        output_dir: Path = None,
                        bot_names: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Changeを分類し、レビュー時間を分析
    
    Args:
        projects: プロジェクトリスト
        start_date: 開始日
        end_date: 終了日
        output_dir: 出力ディレクトリ
        bot_names: ボット名のリスト（Noneの場合は設定ファイルから読み込み）
        
    Returns:
        Dict[str, pd.DataFrame]: 分析結果
    """
    # ボット名を読み込み
    if bot_names is None:
        bot_names = load_bot_names_from_config()
    
    logger.info(f"使用するボットフィルタ: {len(bot_names)}個")
    
    # データ読み込み
    logger.info("=== Changeデータの読み込み ===")
    df = load_openstack_changes(projects, start_date, end_date)
    
    if df.empty:
        logger.error("データが見つかりません")
        return {}
    
    # 分類器を初期化
    logger.info("=== 分類器の初期化 ===")
    classifier = CommitMessageClassifier()
    
    # コミットメッセージを分類
    logger.info("=== コミットメッセージの分類 ===")
    commit_messages = df['commit_message'].fillna('').tolist()
    classifications = classifier.classify_batch(commit_messages, batch_size=8)
    
    # 分類結果をDataFrameに追加
    df['label'] = [c['label'] for c in classifications]
    df['label_score'] = [c['score'] for c in classifications]
    
    # 最初のレビューまでの時間を計算
    logger.info("=== 最初のレビューまでの時間を計算 ===")
    df['time_to_first_review'] = df.apply(
        lambda row: calculate_time_to_first_review(row['messages'], row['created'], bot_names),
        axis=1
    )
    
    # レビューがあったもののみに絞る
    df_with_review = df[df['time_to_first_review'].notna()].copy()
    
    logger.info(f"レビューがあったChange: {len(df_with_review)}件 / 全体: {len(df)}件")
    
    # 時間区間を定義
    bins = [0, 1, 2, 3, 4, 5, float('inf')]
    labels = ['~1日', '1~2日', '2~3日', '3~4日', '4~5日', '5日~']
    
    df_with_review['time_bin'] = pd.cut(
        df_with_review['time_to_first_review'],
        bins=bins,
        labels=labels,
        right=False
    )
    
    # ラベルごとに集計
    logger.info("=== ラベルごとの集計 ===")
    summary_results = {}
    
    for label in CommitMessageClassifier.LABELS.keys():
        label_df = df_with_review[df_with_review['label'] == label]
        
        if len(label_df) == 0:
            logger.warning(f"ラベル '{label}': データなし")
            continue
        
        # 時間区間ごとの件数と割合を計算
        time_distribution = label_df['time_bin'].value_counts().sort_index()
        time_ratio = (time_distribution / len(label_df)).round(2)
        
        summary = pd.DataFrame({
            'count': time_distribution,
            'ratio': time_ratio
        })
        
        summary_results[label] = summary
        
        logger.info(f"ラベル '{label}' ({len(label_df)}件):")
        logger.info(f"\n{summary}")
    
    # 結果を保存
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR / "commit_classification"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 詳細データを保存
    detail_path = output_dir / f"classification_detail_{start_date}_{end_date}.csv"
    df_with_review.to_csv(detail_path, index=False, encoding='utf-8')
    logger.info(f"詳細データ保存: {detail_path}")
    
    # サマリーを保存
    summary_path = output_dir / f"classification_summary_{start_date}_{end_date}.csv"
    
    summary_rows = []
    for label, summary_df in summary_results.items():
        for time_bin, row in summary_df.iterrows():
            summary_rows.append({
                'label': label,
                'time_bin': time_bin,
                'count': int(row['count']),
                'ratio': float(row['ratio'])
            })
    
    summary_all = pd.DataFrame(summary_rows)
    summary_all.to_csv(summary_path, index=False, encoding='utf-8')
    logger.info(f"サマリー保存: {summary_path}")
    
    # ピボットテーブル形式でも保存（画像のフォーマット）
    pivot_path = output_dir / f"classification_pivot_{start_date}_{end_date}.csv"
    pivot_table = summary_all.pivot(index='label', columns='time_bin', values='ratio').fillna(0)
    
    # 件数も追加
    label_counts = df_with_review['label'].value_counts()
    pivot_table.insert(0, 'total_count', pivot_table.index.map(lambda x: label_counts.get(x, 0)))
    
    pivot_table.to_csv(pivot_path, encoding='utf-8')
    logger.info(f"ピボットテーブル保存: {pivot_path}")
    
    return {
        'detail': df_with_review,
        'summary': summary_all,
        'pivot': pivot_table
    }


if __name__ == "__main__":
    """
    メイン実行部分
    """
    logger.info("=" * 60)
    logger.info("コミットメッセージ分類とレビュー時間分析")
    logger.info("=" * 60)
    
    try:
        results = classify_and_analyze()
        
        logger.info("\n" + "=" * 60)
        logger.info("分析完了")
        logger.info("=" * 60)
        
        if 'pivot' in results:
            logger.info("\n=== ピボットテーブル（割合） ===")
            logger.info(f"\n{results['pivot']}")
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        raise
