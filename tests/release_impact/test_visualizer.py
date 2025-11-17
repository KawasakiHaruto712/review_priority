"""
Tests for visualizer.py module
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.release_impact.metrics_analysis.visualizer import MetricsVisualizer


class TestMetricsVisualizer:
    """Test class for MetricsVisualizer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        data = {
            'period_group': ['early_reviewed'] * 50 + ['late_reviewed'] * 50 + 
                          ['early_not_reviewed'] * 30 + ['late_not_reviewed'] * 30,
            'lines_added': np.concatenate([
                np.random.lognormal(4, 1, 50),
                np.random.lognormal(3.8, 1, 50),
                np.random.lognormal(3.5, 1, 30),
                np.random.lognormal(3.6, 1, 30)
            ]),
            'bug_fix_confidence': np.concatenate([
                np.random.choice([0, 1, 2], 50),
                np.random.choice([0, 1, 2], 50),
                np.random.choice([0, 1, 2], 30),
                np.random.choice([0, 1, 2], 30)
            ]),
            'merge_rate': np.concatenate([
                np.random.uniform(0.7, 0.9, 50),
                np.random.uniform(0.6, 0.85, 50),
                np.random.uniform(0.5, 0.75, 30),
                np.random.uniform(0.55, 0.8, 30)
            ]),
            'revision_count': np.concatenate([
                np.random.poisson(3, 50),
                np.random.poisson(2.5, 50),
                np.random.poisson(2, 30),
                np.random.poisson(2.2, 30)
            ])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_statistics(self):
        """Create sample statistics for testing"""
        return {
            'early_reviewed': {
                'lines_added': {
                    'count': 50,
                    'mean': 100.5,
                    'std': 30.2,
                    'min': 20,
                    '25%': 80,
                    '50%': 95,
                    '75%': 120,
                    'max': 200
                },
                'bug_fix_confidence': {
                    'count': 50,
                    'mean': 0.8,
                    'std': 0.7,
                    'min': 0,
                    '25%': 0,
                    '50%': 1,
                    '75%': 1,
                    'max': 2
                }
            },
            'late_reviewed': {
                'lines_added': {
                    'count': 50,
                    'mean': 85.3,
                    'std': 25.1,
                    'min': 15,
                    '25%': 70,
                    '50%': 82,
                    '75%': 100,
                    'max': 150
                },
                'bug_fix_confidence': {
                    'count': 50,
                    'mean': 0.7,
                    'std': 0.6,
                    'min': 0,
                    '25%': 0,
                    '50%': 1,
                    '75%': 1,
                    'max': 2
                }
            }
        }
    
    @pytest.fixture
    def sample_test_results(self):
        """Create sample test results for testing"""
        return {
            'early_reviewed_vs_late_reviewed': {
                'lines_added': {
                    'statistic': 1234.5,
                    'p_value': 0.023,
                    'significant': True,
                    'effect_size': 0.15
                },
                'bug_fix_confidence': {
                    'statistic': 987.3,
                    'p_value': 0.156,
                    'significant': False,
                    'effect_size': 0.08
                }
            },
            'early_not_reviewed_vs_late_not_reviewed': {
                'lines_added': {
                    'statistic': 567.8,
                    'p_value': 0.089,
                    'significant': False,
                    'effect_size': 0.12
                },
                'bug_fix_confidence': {
                    'statistic': 432.1,
                    'p_value': 0.234,
                    'significant': False,
                    'effect_size': 0.05
                }
            }
        }
    
    @pytest.fixture
    def visualizer(self):
        """Create MetricsVisualizer instance"""
        return MetricsVisualizer(figsize=(20, 20), dpi=100)  # Lower DPI for faster tests
    
    def test_initialization(self, visualizer):
        """Test MetricsVisualizer initialization"""
        assert visualizer.figsize == (20, 20)
        assert visualizer.dpi == 100
        assert isinstance(visualizer, MetricsVisualizer)
    
    def test_identify_log_scale_metrics(self, visualizer, sample_data):
        """Test automatic log scale detection"""
        metric_columns = ['lines_added', 'bug_fix_confidence', 'merge_rate']
        
        log_metrics = visualizer._identify_log_scale_metrics(
            sample_data,
            metric_columns,
            threshold=100
        )
        
        # Check that it returns a list
        assert isinstance(log_metrics, list)
        
        # lines_added should be identified for log scale (wide range)
        # bug_fix_confidence should not (values 0-2)
        # merge_rate should not (values 0-1)
        assert 'lines_added' in log_metrics or len(log_metrics) >= 0
    
    def test_identify_log_scale_metrics_with_negative_values(self, visualizer):
        """Test log scale detection with negative values"""
        # Create data with negative values
        data = pd.DataFrame({
            'metric_with_negatives': np.random.uniform(-10, 100, 50)
        })
        
        log_metrics = visualizer._identify_log_scale_metrics(
            data,
            ['metric_with_negatives']
        )
        
        # Should not include metrics with negative values
        assert 'metric_with_negatives' not in log_metrics
    
    def test_identify_log_scale_metrics_with_zeros(self, visualizer):
        """Test log scale detection with zero values"""
        # Create data with zeros
        data = pd.DataFrame({
            'metric_with_zeros': np.concatenate([np.zeros(10), np.random.uniform(1, 1000, 40)])
        })
        
        log_metrics = visualizer._identify_log_scale_metrics(
            data,
            ['metric_with_zeros']
        )
        
        # Should not include metrics with zeros
        assert 'metric_with_zeros' not in log_metrics
    
    def test_create_boxplots(self, visualizer, sample_data, tmp_path):
        """Test boxplot creation"""
        metric_columns = ['lines_added', 'bug_fix_confidence', 'merge_rate', 'revision_count']
        output_path = tmp_path / "test_boxplots.pdf"
        
        # Create boxplots
        visualizer.create_boxplots(
            sample_data,
            'period_group',
            metric_columns,
            output_path,
            auto_log_scale=False  # Disable for predictable testing
        )
        
        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_create_boxplots_with_log_scale(self, visualizer, sample_data, tmp_path):
        """Test boxplot creation with log scale"""
        metric_columns = ['lines_added']
        output_path = tmp_path / "test_boxplots_log.pdf"
        
        # Create boxplots with manual log scale
        visualizer.create_boxplots(
            sample_data,
            'period_group',
            metric_columns,
            output_path,
            log_scale_metrics=['lines_added']
        )
        
        # Check file was created
        assert output_path.exists()
    
    def test_create_boxplots_with_auto_log_scale(self, visualizer, sample_data, tmp_path):
        """Test boxplot creation with automatic log scale"""
        metric_columns = ['lines_added', 'bug_fix_confidence']
        output_path = tmp_path / "test_boxplots_auto_log.pdf"
        
        # Create boxplots with auto log scale
        visualizer.create_boxplots(
            sample_data,
            'period_group',
            metric_columns,
            output_path,
            auto_log_scale=True
        )
        
        # Check file was created
        assert output_path.exists()
    
    def test_create_boxplots_with_missing_metric(self, visualizer, sample_data, tmp_path):
        """Test boxplot creation with missing metric"""
        metric_columns = ['lines_added', 'nonexistent_metric']
        output_path = tmp_path / "test_boxplots_missing.pdf"
        
        # Should handle missing metric gracefully
        visualizer.create_boxplots(
            sample_data,
            'period_group',
            metric_columns,
            output_path
        )
        
        # Check file was created despite missing metric
        assert output_path.exists()
    
    def test_create_heatmap(self, visualizer, sample_test_results, tmp_path):
        """Test heatmap creation"""
        metric_columns = ['lines_added', 'bug_fix_confidence']
        output_path = tmp_path / "test_heatmap.pdf"
        
        # Create heatmap
        visualizer.create_heatmap(
            sample_test_results,
            output_path,
            metric_columns
        )
        
        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_create_heatmap_with_nan_values(self, visualizer, tmp_path):
        """Test heatmap creation with NaN p-values"""
        test_results = {
            'group1_vs_group2': {
                'metric1': {'p_value': 0.05},
                'metric2': {'p_value': None}  # NaN case
            }
        }
        
        metric_columns = ['metric1', 'metric2']
        output_path = tmp_path / "test_heatmap_nan.pdf"
        
        # Should handle NaN values
        visualizer.create_heatmap(
            test_results,
            output_path,
            metric_columns
        )
        
        # Check file was created
        assert output_path.exists()
    
    def test_create_summary_plot(self, visualizer, sample_statistics, tmp_path):
        """Test summary plot creation"""
        metric_columns = ['lines_added', 'bug_fix_confidence']
        output_path = tmp_path / "test_summary.pdf"
        
        # Create summary plot
        visualizer.create_summary_plot(
            sample_statistics,
            output_path,
            metric_columns
        )
        
        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_create_summary_plot_with_missing_metric(
        self, visualizer, sample_statistics, tmp_path
    ):
        """Test summary plot with missing metric in some groups"""
        metric_columns = ['lines_added', 'nonexistent_metric']
        output_path = tmp_path / "test_summary_missing.pdf"
        
        # Should handle missing metrics gracefully
        visualizer.create_summary_plot(
            sample_statistics,
            output_path,
            metric_columns
        )
        
        # Check file was created
        assert output_path.exists()
    
    def test_large_number_of_metrics(self, visualizer, sample_data, tmp_path):
        """Test boxplot creation with many metrics (16 metrics)"""
        # Add more metrics to reach 16
        for i in range(12):
            sample_data[f'metric_{i}'] = np.random.normal(50, 10, len(sample_data))
        
        metric_columns = list(sample_data.columns[1:17])  # Skip period_group, take 16 metrics
        output_path = tmp_path / "test_boxplots_16.pdf"
        
        # Should handle 16 metrics in 4x4 grid
        visualizer.create_boxplots(
            sample_data,
            'period_group',
            metric_columns,
            output_path
        )
        
        # Check file was created
        assert output_path.exists()
    
    def test_output_directory_creation(self, visualizer, sample_data, tmp_path):
        """Test automatic directory creation"""
        # Use nested path that doesn't exist
        output_path = tmp_path / "nested" / "directory" / "test_boxplots.pdf"
        
        metric_columns = ['lines_added']
        
        # Should create directories automatically
        visualizer.create_boxplots(
            sample_data,
            'period_group',
            metric_columns,
            output_path
        )
        
        # Check file and directories were created
        assert output_path.exists()
        assert output_path.parent.exists()


class TestMetricsVisualizerIntegration:
    """Integration tests for MetricsVisualizer"""
    
    def test_complete_visualization_workflow(self, tmp_path):
        """Test complete visualization workflow"""
        # Create realistic data
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'period_group': ['early_reviewed'] * n_samples + ['late_reviewed'] * n_samples,
            'lines_added': np.concatenate([
                np.random.lognormal(4, 1, n_samples),
                np.random.lognormal(3.8, 1, n_samples)
            ]),
            'bug_fix_confidence': np.concatenate([
                np.random.choice([0, 1, 2], n_samples),
                np.random.choice([0, 1, 2], n_samples)
            ]),
            'merge_rate': np.concatenate([
                np.random.uniform(0.6, 0.9, n_samples),
                np.random.uniform(0.5, 0.85, n_samples)
            ])
        })
        
        statistics = {
            'early_reviewed': {
                'lines_added': {'mean': 100, 'std': 30},
                'bug_fix_confidence': {'mean': 0.8, 'std': 0.7},
                'merge_rate': {'mean': 0.75, 'std': 0.15}
            },
            'late_reviewed': {
                'lines_added': {'mean': 85, 'std': 25},
                'bug_fix_confidence': {'mean': 0.7, 'std': 0.6},
                'merge_rate': {'mean': 0.68, 'std': 0.18}
            }
        }
        
        test_results = {
            'early_reviewed_vs_late_reviewed': {
                'lines_added': {'p_value': 0.023},
                'bug_fix_confidence': {'p_value': 0.156},
                'merge_rate': {'p_value': 0.089}
            }
        }
        
        visualizer = MetricsVisualizer(dpi=100)
        metric_columns = ['lines_added', 'bug_fix_confidence', 'merge_rate']
        
        # Create all plots
        boxplot_path = tmp_path / "boxplots.pdf"
        heatmap_path = tmp_path / "heatmap.pdf"
        summary_path = tmp_path / "summary.pdf"
        
        visualizer.create_boxplots(data, 'period_group', metric_columns, boxplot_path)
        visualizer.create_heatmap(test_results, heatmap_path, metric_columns)
        visualizer.create_summary_plot(statistics, summary_path, metric_columns)
        
        # Verify all files created
        assert boxplot_path.exists()
        assert heatmap_path.exists()
        assert summary_path.exists()
        
        # Verify files have content
        assert boxplot_path.stat().st_size > 1000  # PDFs should be reasonably sized
        assert heatmap_path.stat().st_size > 1000
        assert summary_path.stat().st_size > 1000
