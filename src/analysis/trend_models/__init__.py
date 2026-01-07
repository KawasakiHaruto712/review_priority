"""
Trend Models Analysis System

OpenStackプロジェクトにおいて、レビュー優先度予測モデルを構築・評価するためのモジュール。
複数のリリース期間のデータを用いたLeave-One-Out交差検証により、
リリース期間の特性（前期/後期/全期間）がモデル性能に与える影響を分析します。
"""

from src.analysis.trend_models.main import TrendModelsAnalyzer

__all__ = ['TrendModelsAnalyzer']
