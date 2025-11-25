"""
Visualization module for Trend Metrics Analysis
"""
from .trend_plotter import plot_boxplots_8groups, plot_trend_lines, plot_metric_changes
from .heatmap_generator import generate_heatmap

__all__ = [
    'plot_boxplots_8groups',
    'plot_trend_lines',
    'plot_metric_changes',
    'generate_heatmap'
]
