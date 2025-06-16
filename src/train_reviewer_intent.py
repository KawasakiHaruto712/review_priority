import pandas as pd
from pathlib import Path
from src.config.path import DEFAULT_DATA_DIR
from src.learning.irl_models import TrajectoryBuilder, FeatureExtractor, MaxEntIRL

def main():
    """学習プロセスのメイン関数"""
    # --- 1. データの準備 ---
    DATA_DIRECTORY = DEFAULT_DATA_DIR / "openstack"
    TARGET_COMPONENT = "nova"

    # ステップ1のメッセージ
    print("ステップ1: 専門家の軌跡データを構築します...")
    trajectory_builder = TrajectoryBuilder(DATA_DIRECTORY)
    expert_trajectories, all_prs_df = trajectory_builder.build(TARGET_COMPONENT)
    
    if not expert_trajectories:
        # エラーメッセージ
        print("有効な軌跡データが見つかりませんでした。処理を終了します。")
        return

    # 軌跡データの構築完了メッセージ
    print(f"専門家の軌跡データを {len(expert_trajectories)} 件構築しました。")

    # --- 2. 特徴量抽出器の準備 ---
    # ステップ2のメッセージ
    print("\nステップ2: 特徴量抽出器を初期化します...")
    feature_extractor = FeatureExtractor(all_prs_df, trajectory_builder.pr_details)
    feature_dimension = len(feature_extractor.feature_names)
    # 特徴量情報のメッセージ
    print(f"特徴量の次元数: {feature_dimension}")
    print(f"特徴量: {feature_extractor.feature_names}")

    # --- 3. MaxEnt IRLによる学習 ---
    # ステップ3のメッセージ
    print("\nステップ3: Maximum Entropy IRLによる学習を開始します...")
    irl_model = MaxEntIRL(feature_dim=feature_dimension, learning_rate=0.05, epochs=50)
    learned_theta = irl_model.fit(expert_trajectories, feature_extractor)

    # --- 4. 結果の表示と解釈 ---
    # 学習完了のメッセージ
    print("\n--- 学習が完了しました ---")
    print("学習された報酬の重み (theta):")
    
    results_df = pd.DataFrame({
        'Feature': feature_extractor.feature_names,
        'Weight': learned_theta
    }).sort_values(by='Weight', ascending=False)
    
    print(results_df)

    # 結果の解釈のメッセージ
    print("\n--- 結果の解釈 ---")
    print("'Weight'（重み）は、各特徴量がPRの優先度にどの程度貢献しているかを示します。")
    print("- 正の重み: この特徴量の値が大きいほど、PRの優先度が高くなる傾向があります。")
    print("- 負の重み: この特徴量の値が大きいほど、PRの優先度が低くなる傾向があります。")

if __name__ == '__main__':
    main()