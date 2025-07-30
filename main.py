#!/usr/bin/env python3
"""
Review Priority - メインエントリーポイント

OpenStackプロジェクトのコードレビューデータを分析し、
逆強化学習（IRL）を用いて優先順位付けを行うシステム

使用方法:
    python main.py collect                    # データ収集
    python main.py preprocess                 # データ前処理
    python main.py train                      # IRLモデル学習
    python main.py analyze [--project nova]   # 分析実行
    python main.py full                       # 全工程実行
"""

import argparse
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.collectors.openstack import OpenStackGerritCollector
from src.preprocessing.review_comment_processor import extract_and_save_review_keywords
from src.learning.irl_models import run_temporal_irl_analysis
from src.config.path import DEFAULT_DATA_DIR, DEFAULT_CONFIG

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data():
    """OpenStackからデータを収集"""
    logger.info("データ収集を開始します...")
    try:
        collector = OpenStackGerritCollector()
        collector.collect_data()
        logger.info("データ収集が完了しました")
        return True
    except Exception as e:
        logger.error(f"データ収集エラー: {e}")
        return False


def preprocess_data():
    """収集したデータの前処理"""
    logger.info("データ前処理を開始します...")
    try:
        checklist_path = DEFAULT_DATA_DIR / "processed" / "checklist.csv"
        output_path = DEFAULT_DATA_DIR / "processed" / "review_keywords.json"
        config_path = DEFAULT_CONFIG / "gerrymanderconfig.ini"
        label_path = DEFAULT_DATA_DIR / "processed" / "review_label.json"
        
        if not checklist_path.exists():
            logger.error(f"checklistファイルが見つかりません: {checklist_path}")
            return False
            
        extract_and_save_review_keywords(
            checklist_path=checklist_path,
            output_keywords_path=output_path,
            gerrymander_config_path=config_path,
            review_label_path=label_path
        )
        logger.info("データ前処理が完了しました")
        return True
    except Exception as e:
        logger.error(f"データ前処理エラー: {e}")
        return False


def train_model(projects=None):
    """IRLモデルの学習"""
    logger.info("IRLモデル学習を開始します...")
    try:
        results = run_temporal_irl_analysis(projects=projects)
        
        if "error" not in results:
            logger.info("IRLモデル学習が完了しました")
            logger.info(f"分析成功プロジェクト: {results['data_summary']['projects_analyzed']}件")
            logger.info(f"総学習データ: {results['data_summary']['total_training_samples']}件")
            return True
        else:
            logger.error(f"学習エラー: {results['error']}")
            return False
    except Exception as e:
        logger.error(f"IRLモデル学習エラー: {e}")
        return False


def analyze_project(project=None):
    """特定プロジェクトまたは全プロジェクトの分析"""
    projects = [project] if project else None
    project_name = project if project else "全プロジェクト"
    
    logger.info(f"{project_name}の分析を開始します...")
    return train_model(projects)


def run_full_pipeline(project=None):
    """全工程を順次実行"""
    logger.info("Review Priority 分析パイプライン全工程を開始します...")
    
    # 1. データ収集
    if not collect_data():
        logger.error("データ収集に失敗しました")
        return False
    
    # 2. データ前処理
    if not preprocess_data():
        logger.error("データ前処理に失敗しました")
        return False
    
    # 3. モデル学習・分析
    if not analyze_project(project):
        logger.error("分析に失敗しました")
        return False
    
    logger.info("全工程が正常に完了しました！")
    return True


def show_status():
    """システムの状態を表示"""
    logger.info("Review Priority システム状態チェック...")
    
    # データディレクトリの確認
    data_dir = DEFAULT_DATA_DIR
    logger.info(f"データディレクトリ: {data_dir}")
    logger.info(f"  存在: {'✓' if data_dir.exists() else '✗'}")
    
    # OpenStackデータの確認
    openstack_dir = data_dir / "openstack"
    if openstack_dir.exists():
        projects = [d.name for d in openstack_dir.iterdir() if d.is_dir()]
        logger.info(f"  収集済みプロジェクト: {len(projects)}個 {projects}")
    else:
        logger.info("  収集済みプロジェクト: なし")
    
    # 前処理データの確認
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        files = [f.name for f in processed_dir.iterdir() if f.is_file()]
        logger.info(f"  前処理済みファイル: {files}")
    else:
        logger.info("  前処理済みファイル: なし")
    
    # 結果データの確認
    results_dir = data_dir / "results"
    if results_dir.exists():
        files = [f.name for f in results_dir.iterdir() if f.is_file()]
        logger.info(f"  分析結果: {files}")
    else:
        logger.info("  分析結果: なし")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Review Priority - OpenStackレビュー優先順位分析システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py collect                    # データ収集のみ
  python main.py preprocess                 # 前処理のみ
  python main.py train                      # 学習のみ
  python main.py analyze                    # 全プロジェクト分析
  python main.py analyze --project nova     # Novaプロジェクトのみ分析
  python main.py full                       # 全工程実行
  python main.py status                     # システム状態確認
        """
    )
    
    parser.add_argument(
        'command',
        choices=['collect', 'preprocess', 'train', 'analyze', 'full', 'status'],
        help='実行するコマンド'
    )
    parser.add_argument(
        '--project',
        help='分析対象プロジェクト (nova, neutron, cinder, etc.)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Review Priority システム開始: {args.command}")
    
    try:
        if args.command == 'collect':
            success = collect_data()
        elif args.command == 'preprocess':
            success = preprocess_data()
        elif args.command == 'train':
            success = train_model()
        elif args.command == 'analyze':
            success = analyze_project(args.project)
        elif args.command == 'full':
            success = run_full_pipeline(args.project)
        elif args.command == 'status':
            show_status()
            success = True
        else:
            logger.error(f"不明なコマンド: {args.command}")
            success = False
        
        if success:
            logger.info("処理が正常に完了しました")
            sys.exit(0)
        else:
            logger.error("処理が失敗しました")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
