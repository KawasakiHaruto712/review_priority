"""
Tests for statistical_analyzer.py module
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile

from src.release_impact.metrics_analysis.statistical_analyzer import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test class for StatisticalAnalyzer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        data = {
            'period_group': ['early_reviewed'] * 50 + ['late_reviewed'] * 50 + 
                          ['early_not_reviewed'] * 30 + ['late_not_reviewed'] * 30,
            'lines_added': np.concatenate([
                np.random.normal(100, 30, 50),
                np.random.normal(80, 25, 50),
                np.random.normal(60, 20, 30),
                np.random.normal(70, 22, 30)
            ]),
            'bug_fix_confidence': np.concatenate([
                np.random.choice([0, 1, 2], 50, p=[0.4, 0.4, 0.2]),
                np.random.choice([0, 1, 2], 50, p=[0.5, 0.3, 0.2]),
                np.random.choice([0, 1, 2], 30, p=[0.6, 0.3, 0.1]),
                np.random.choice([0, 1, 2], 30, p=[0.5, 0.4, 0.1])
            ]),
            'merge_rate': np.concatenate([
                np.random.uniform(0.7, 0.9, 50),
                np.random.uniform(0.6, 0.85, 50),
                np.random.uniform(0.5, 0.75, 30),
                np.random.uniform(0.55, 0.8, 30)
            ])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def analyzer(self):
        """Create StatisticalAnalyzer instance"""
        return StatisticalAnalyzer(significance_level=0.05)
    
    def test_initialization(self, analyzer):
        """Test StatisticalAnalyzer initialization"""
        assert analyzer.significance_level == 0.05
        assert isinstance(analyzer, StatisticalAnalyzer)
    
    def test_calculate_descriptive_statistics(self, analyzer, sample_data):
        """Test descriptive statistics calculation"""
        metric_columns = ['lines_added', 'bug_fix_confidence', 'merge_rate']
        
        statistics = analyzer.calculate_descriptive_statistics(
            sample_data,
            'period_group',
            metric_columns
        )
        
        # Check structure
        assert isinstance(statistics, dict)
        assert 'early_reviewed' in statistics
        assert 'late_reviewed' in statistics
        
        # Check metrics
        for group in statistics:
            assert 'lines_added' in statistics[group]
            assert 'bug_fix_confidence' in statistics[group]
            assert 'merge_rate' in statistics[group]
        
        # Check statistics fields
        early_stats = statistics['early_reviewed']['lines_added']
        assert 'count' in early_stats
        assert 'mean' in early_stats
        assert 'std' in early_stats
        assert 'min' in early_stats
        assert '25%' in early_stats
        assert '50%' in early_stats
        assert '75%' in early_stats
        assert 'max' in early_stats
        
        # Check values are reasonable
        assert early_stats['count'] == 50
        assert early_stats['min'] <= early_stats['25%']
        assert early_stats['25%'] <= early_stats['50%']
        assert early_stats['50%'] <= early_stats['75%']
        assert early_stats['75%'] <= early_stats['max']
    
    def test_calculate_descriptive_statistics_with_missing_metric(
        self, analyzer, sample_data
    ):
        """Test handling of missing metrics"""
        metric_columns = ['lines_added', 'nonexistent_metric']
        
        statistics = analyzer.calculate_descriptive_statistics(
            sample_data,
            'period_group',
            metric_columns
        )
        
        # Should still work for valid metric
        assert 'lines_added' in statistics['early_reviewed']
        # Should not have stats for missing metric
        assert 'nonexistent_metric' not in statistics['early_reviewed']
    
    def test_perform_mann_whitney_test(self, analyzer, sample_data):
        """Test Mann-Whitney U test"""
        metric_columns = ['lines_added', 'merge_rate']
        
        test_results = analyzer.perform_mann_whitney_test(
            sample_data,
            'period_group',
            'early_reviewed',
            'late_reviewed',
            metric_columns
        )
        
        # Check structure
        assert isinstance(test_results, dict)
        assert 'lines_added' in test_results
        assert 'merge_rate' in test_results
        
        # Check result fields
        result = test_results['lines_added']
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'effect_size' in result
        assert 'n1' in result
        assert 'n2' in result
        
        # Check value types
        assert isinstance(result['statistic'], float)
        assert isinstance(result['p_value'], float)
        assert isinstance(result['significant'], bool)
        assert isinstance(result['effect_size'], float)
        
        # Check value ranges
        assert 0 <= result['p_value'] <= 1
        assert -1 <= result['effect_size'] <= 1
        assert result['n1'] == 50
        assert result['n2'] == 50
    
    def test_perform_mann_whitney_test_insufficient_samples(
        self, analyzer, sample_data
    ):
        """Test handling of insufficient sample size"""
        # Create small sample
        small_data = sample_data.head(4)
        
        test_results = analyzer.perform_mann_whitney_test(
            small_data,
            'period_group',
            'early_reviewed',
            'late_reviewed',
            ['lines_added']
        )
        
        # Should return error result
        assert 'lines_added' in test_results
        assert test_results['lines_added']['statistic'] is None
        assert test_results['lines_added']['p_value'] is None
        assert test_results['lines_added']['significant'] is False
        assert 'error' in test_results['lines_added']
    
    def test_calculate_rank_biserial_correlation(self, analyzer):
        """Test rank-biserial correlation calculation"""
        # Test with known values
        u_statistic = 500
        n1 = 50
        n2 = 50
        
        r = analyzer._calculate_rank_biserial_correlation(u_statistic, n1, n2)
        
        # Check result is in valid range
        assert -1 <= r <= 1
        
        # Test edge cases
        r_min = analyzer._calculate_rank_biserial_correlation(0, n1, n2)
        assert r_min == 1.0  # Maximum effect
        
        r_max = analyzer._calculate_rank_biserial_correlation(n1 * n2, n1, n2)
        assert r_max == -1.0  # Minimum effect
    
    def test_perform_multiple_comparisons(self, analyzer, sample_data):
        """Test multiple comparison execution"""
        metric_columns = ['lines_added', 'merge_rate']
        comparison_pairs = [
            ('early_reviewed', 'late_reviewed'),
            ('early_not_reviewed', 'late_not_reviewed')
        ]
        
        all_results = analyzer.perform_multiple_comparisons(
            sample_data,
            'period_group',
            metric_columns,
            comparison_pairs
        )
        
        # Check structure
        assert isinstance(all_results, dict)
        assert 'early_reviewed_vs_late_reviewed' in all_results
        assert 'early_not_reviewed_vs_late_not_reviewed' in all_results
        
        # Check each comparison has results for all metrics
        for comparison in all_results:
            assert 'lines_added' in all_results[comparison]
            assert 'merge_rate' in all_results[comparison]
    
    def test_save_statistics(self, analyzer, sample_data, tmp_path):
        """Test saving statistics to JSON"""
        metric_columns = ['lines_added', 'bug_fix_confidence']
        
        statistics = analyzer.calculate_descriptive_statistics(
            sample_data,
            'period_group',
            metric_columns
        )
        
        # Save to temporary file
        output_path = tmp_path / "test_statistics.json"
        analyzer.save_statistics(statistics, output_path)
        
        # Check file exists
        assert output_path.exists()
        
        # Load and verify content
        with open(output_path, 'r') as f:
            loaded_stats = json.load(f)
        
        assert loaded_stats == statistics
        assert 'early_reviewed' in loaded_stats
    
    def test_save_test_results(self, analyzer, sample_data, tmp_path):
        """Test saving test results to JSON"""
        metric_columns = ['lines_added']
        
        test_results = analyzer.perform_mann_whitney_test(
            sample_data,
            'period_group',
            'early_reviewed',
            'late_reviewed',
            metric_columns
        )
        
        # Wrap in comparison key
        all_results = {'early_reviewed_vs_late_reviewed': test_results}
        
        # Save to temporary file
        output_path = tmp_path / "test_results.json"
        analyzer.save_test_results(all_results, output_path)
        
        # Check file exists
        assert output_path.exists()
        
        # Load and verify content
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert 'early_reviewed_vs_late_reviewed' in loaded_results
        assert 'lines_added' in loaded_results['early_reviewed_vs_late_reviewed']
    
    def test_significance_threshold(self, analyzer, sample_data):
        """Test significance level threshold application"""
        # Test with different significance levels
        analyzer_strict = StatisticalAnalyzer(significance_level=0.01)
        analyzer_lenient = StatisticalAnalyzer(significance_level=0.10)
        
        # Perform test
        results_strict = analyzer_strict.perform_mann_whitney_test(
            sample_data,
            'period_group',
            'early_reviewed',
            'late_reviewed',
            ['lines_added']
        )
        
        results_lenient = analyzer_lenient.perform_mann_whitney_test(
            sample_data,
            'period_group',
            'early_reviewed',
            'late_reviewed',
            ['lines_added']
        )
        
        # Check that significance determination uses correct threshold
        p_value = results_strict['lines_added']['p_value']
        
        if 0.01 < p_value < 0.10:
            # Should be significant for lenient but not strict
            assert not results_strict['lines_added']['significant']
            assert results_lenient['lines_added']['significant']
    
    def test_empty_data_handling(self, analyzer):
        """Test handling of empty data"""
        empty_data = pd.DataFrame({
            'period_group': [],
            'lines_added': []
        })
        
        statistics = analyzer.calculate_descriptive_statistics(
            empty_data,
            'period_group',
            ['lines_added']
        )
        
        # Should return empty dict or handle gracefully
        assert isinstance(statistics, dict)


class TestStatisticalAnalyzerIntegration:
    """Integration tests for StatisticalAnalyzer"""
    
    def test_full_workflow(self, tmp_path):
        """Test complete analysis workflow"""
        # Create realistic data
        np.random.seed(42)
        data = pd.DataFrame({
            'period_group': ['early_reviewed'] * 100 + ['late_reviewed'] * 100,
            'lines_added': np.concatenate([
                np.random.lognormal(4, 1, 100),
                np.random.lognormal(3.8, 1, 100)
            ]),
            'bug_fix_confidence': np.concatenate([
                np.random.choice([0, 1, 2], 100),
                np.random.choice([0, 1, 2], 100)
            ])
        })
        
        analyzer = StatisticalAnalyzer()
        metric_columns = ['lines_added', 'bug_fix_confidence']
        
        # Calculate statistics
        statistics = analyzer.calculate_descriptive_statistics(
            data, 'period_group', metric_columns
        )
        
        # Perform tests
        test_results = analyzer.perform_mann_whitney_test(
            data, 'period_group', 'early_reviewed', 'late_reviewed', metric_columns
        )
        
        # Save results
        stats_path = tmp_path / "statistics.json"
        test_path = tmp_path / "test_results.json"
        
        analyzer.save_statistics(statistics, stats_path)
        analyzer.save_test_results(
            {'early_reviewed_vs_late_reviewed': test_results},
            test_path
        )
        
        # Verify files created
        assert stats_path.exists()
        assert test_path.exists()
        
        # Verify content is valid JSON
        with open(stats_path, 'r') as f:
            loaded_stats = json.load(f)
        with open(test_path, 'r') as f:
            loaded_tests = json.load(f)
        
        assert len(loaded_stats) > 0
        assert len(loaded_tests) > 0
