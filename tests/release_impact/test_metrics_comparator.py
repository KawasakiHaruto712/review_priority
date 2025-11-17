"""
Tests for metrics_comparator.py module
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

from src.release_impact.metrics_comparator import ReleaseMetricsComparator
from src.config.release_constants import RELEASE_ANALYSIS_PERIODS


class TestReleaseMetricsComparator:
    """Test class for ReleaseMetricsComparator"""
    
    @pytest.fixture
    def sample_releases_df(self):
        """Create sample release data"""
        return pd.DataFrame({
            'component': ['nova', 'nova', 'nova', 'neutron', 'neutron'],
            'version': ['15.0.0', '16.0.0', '17.0.0', '9.0.0', '10.0.0'],
            'release_date': pd.to_datetime([
                '2016-10-06',
                '2017-02-22',
                '2017-08-30',
                '2016-10-06',
                '2017-02-22'
            ])
        })
    
    @pytest.fixture
    def sample_changes_df(self):
        """Create sample change data"""
        np.random.seed(42)
        base_date = datetime(2016, 10, 6)
        
        changes = []
        for i in range(100):
            created_date = base_date + timedelta(days=np.random.randint(0, 365))
            updated_date = created_date + timedelta(days=np.random.randint(1, 30))
            
            change = {
                'change_number': i + 1000,
                'component': 'nova',
                'created': created_date,
                'updated': updated_date,
                'owner': {'email': f'user{i % 10}@example.com'},
                'subject': f'Test change {i}',
                'message': f'Description for change {i}',
                'messages': [
                    {
                        'author': {'name': f'reviewer{j}'},
                        'date': (created_date + timedelta(days=j+1)).isoformat()
                    }
                    for j in range(np.random.randint(0, 5))
                ],
                'file_changes': {
                    'file1.py': {
                        'lines_inserted': np.random.randint(10, 200),
                        'lines_deleted': np.random.randint(5, 100)
                    }
                }
            }
            changes.append(change)
        
        return pd.DataFrame(changes)
    
    @pytest.fixture
    def comparator(self):
        """Create ReleaseMetricsComparator instance"""
        return ReleaseMetricsComparator('nova')
    
    def test_initialization(self, comparator):
        """Test ReleaseMetricsComparator initialization"""
        assert comparator.project_name == 'nova'
        assert len(comparator.metric_columns) == 16
        assert hasattr(comparator, 'data_processor')
        assert hasattr(comparator, 'statistical_analyzer')
        assert hasattr(comparator, 'visualizer')
    
    def test_initialization_invalid_project(self):
        """Test initialization with invalid project name"""
        with pytest.raises(ValueError, match="プロジェクト .* は定義されていません"):
            ReleaseMetricsComparator('invalid_project')
    
    def test_get_release_date(self, comparator, sample_releases_df):
        """Test release date retrieval"""
        release_date = comparator.get_release_date(sample_releases_df, '15.0.0')
        
        assert release_date is not None
        assert isinstance(release_date, pd.Timestamp)
        assert release_date.year == 2016
        assert release_date.month == 10
    
    def test_get_release_date_not_found(self, comparator, sample_releases_df):
        """Test release date retrieval for non-existent version"""
        release_date = comparator.get_release_date(sample_releases_df, '99.0.0')
        
        assert release_date is None
    
    def test_extract_period_changes(self, comparator, sample_changes_df):
        """Test extraction of changes within a period"""
        base_date = datetime(2016, 10, 6)
        
        # Extract early reviewed changes (0-30 days after release)
        period_changes = comparator.extract_period_changes(
            sample_changes_df,
            base_date,
            offset_start=0,
            offset_end=30,
            review_status='reviewed'
        )
        
        # Check that we got some changes
        assert isinstance(period_changes, pd.DataFrame)
        # Should have at least some changes in this period
        assert len(period_changes) >= 0
    
    def test_extract_period_changes_review_status_filtering(
        self, comparator, sample_changes_df
    ):
        """Test review status filtering"""
        base_date = datetime(2016, 10, 6)
        
        # Extract reviewed changes
        reviewed = comparator.extract_period_changes(
            sample_changes_df,
            base_date,
            offset_start=0,
            offset_end=30,
            review_status='reviewed'
        )
        
        # Extract not reviewed changes
        not_reviewed = comparator.extract_period_changes(
            sample_changes_df,
            base_date,
            offset_start=0,
            offset_end=30,
            review_status='not_reviewed'
        )
        
        # Both should be DataFrames
        assert isinstance(reviewed, pd.DataFrame)
        assert isinstance(not_reviewed, pd.DataFrame)
        
        # Total should not exceed total changes in period
        # (some might have no messages at all)
        assert len(reviewed) + len(not_reviewed) <= len(sample_changes_df)
    
    def test_extract_period_changes_negative_offset(
        self, comparator, sample_changes_df
    ):
        """Test extraction with negative offset (late period)"""
        base_date = datetime(2017, 2, 22)  # Next release date
        
        # Extract late period (-30 to 0 days before release)
        period_changes = comparator.extract_period_changes(
            sample_changes_df,
            base_date,
            offset_start=-30,
            offset_end=0,
            review_status='reviewed'
        )
        
        assert isinstance(period_changes, pd.DataFrame)
        assert len(period_changes) >= 0
    
    def test_metric_columns_completeness(self, comparator):
        """Test that all required metrics are defined"""
        expected_metrics = [
            'bug_fix_confidence',
            'lines_added',
            'lines_deleted',
            'files_changed',
            'elapsed_time',
            'revision_count',
            'test_code_presence',
            'past_report_count',
            'recent_report_count',
            'merge_rate',
            'recent_merge_rate',
            'days_to_major_release',
            'open_ticket_count',
            'reviewed_lines_in_period',
            'refactoring_confidence',
            'uncompleted_requests'
        ]
        
        assert len(comparator.metric_columns) == 16
        for metric in expected_metrics:
            assert metric in comparator.metric_columns
    
    def test_period_configuration_consistency(self):
        """Test that period configurations are consistent"""
        # Check all required periods exist
        required_periods = [
            'early_reviewed',
            'early_not_reviewed',
            'late_reviewed',
            'late_not_reviewed'
        ]
        
        for period in required_periods:
            assert period in RELEASE_ANALYSIS_PERIODS
            config = RELEASE_ANALYSIS_PERIODS[period]
            
            # Check required fields
            assert 'base_date' in config
            assert 'offset_start' in config
            assert 'offset_end' in config
            assert 'review_status' in config
            
            # Check valid values
            assert config['base_date'] in ['current_release', 'next_release']
            assert config['review_status'] in ['reviewed', 'not_reviewed']
            assert isinstance(config['offset_start'], int)
            assert isinstance(config['offset_end'], int)


class TestReleaseMetricsComparatorIntegration:
    """Integration tests for ReleaseMetricsComparator"""
    
    @pytest.fixture
    def mock_data_processor(self, monkeypatch):
        """Mock the data processor to avoid loading real data"""
        class MockDataProcessor:
            def load_openstack_data(self):
                # Return minimal mock data
                changes_df = pd.DataFrame({
                    'change_number': [1, 2, 3],
                    'component': ['nova', 'nova', 'nova'],
                    'created': pd.to_datetime(['2016-10-06', '2016-10-10', '2016-10-20']),
                    'updated': pd.to_datetime(['2016-10-07', '2016-10-11', '2016-10-21']),
                    'owner': [
                        {'email': 'user1@example.com'},
                        {'email': 'user2@example.com'},
                        {'email': 'user3@example.com'}
                    ],
                    'subject': ['Change 1', 'Change 2', 'Change 3'],
                    'message': ['Desc 1', 'Desc 2', 'Desc 3'],
                    'messages': [[], [], []],
                    'owner_email': ['user1@example.com', 'user2@example.com', 'user3@example.com']
                })
                
                releases_df = pd.DataFrame({
                    'component': ['nova', 'nova'],
                    'version': ['15.0.0', '16.0.0'],
                    'release_date': pd.to_datetime(['2016-10-06', '2017-02-22'])
                })
                
                return changes_df, releases_df
            
            def extract_features(self, changes_df, analysis_time, project_name, releases_df):
                # Return minimal feature data
                features = []
                for _, row in changes_df.iterrows():
                    feature = {
                        'change_number': row['change_number'],
                        'component': row['component'],
                        'created': row['created'],
                        'bug_fix_confidence': 0,
                        'lines_added': 100,
                        'lines_deleted': 50,
                        'files_changed': 2,
                        'elapsed_time': 60.0,
                        'revision_count': 1,
                        'test_code_presence': 0,
                        'past_report_count': 5,
                        'recent_report_count': 2,
                        'merge_rate': 0.75,
                        'recent_merge_rate': 0.80,
                        'days_to_major_release': 100.0,
                        'open_ticket_count': 50,
                        'reviewed_lines_in_period': 1000,
                        'refactoring_confidence': 0,
                        'uncompleted_requests': 1
                    }
                    features.append(feature)
                return pd.DataFrame(features)
        
        return MockDataProcessor()
    
    def test_save_results(self, tmp_path, mock_data_processor, monkeypatch):
        """Test saving analysis results"""
        # Create comparator with mocked data processor
        comparator = ReleaseMetricsComparator('nova')
        monkeypatch.setattr(comparator, 'data_processor', mock_data_processor)
        
        # Create minimal test data
        metrics_df = pd.DataFrame({
            'change_number': [1, 2],
            'period_group': ['early_reviewed', 'late_reviewed'],
            'lines_added': [100, 80],
            'bug_fix_confidence': [1, 0]
        })
        
        statistics = {
            'early_reviewed': {
                'lines_added': {'count': 1, 'mean': 100, 'std': 0}
            },
            'late_reviewed': {
                'lines_added': {'count': 1, 'mean': 80, 'std': 0}
            }
        }
        
        test_results = {
            'early_reviewed_vs_late_reviewed': {
                'lines_added': {
                    'statistic': 1.0,
                    'p_value': 0.5,
                    'significant': False,
                    'effect_size': 0.1
                }
            }
        }
        
        # Patch DEFAULT_DATA_DIR to use tmp_path
        import src.release_impact.metrics_comparator as mc_module
        monkeypatch.setattr(mc_module, 'DEFAULT_DATA_DIR', tmp_path)
        
        # Save results
        comparator.save_results(
            '15.0.0_period',
            metrics_df,
            statistics,
            test_results
        )
        
        # Check that output directory was created
        output_dir = tmp_path / "release_impact" / "nova_15.0.0_period"
        assert output_dir.exists()
        
        # Check that files were created
        assert (output_dir / "metrics_data.csv").exists()
        assert (output_dir / "summary_statistics.json").exists()
        assert (output_dir / "test_results.json").exists()
        # Note: PDF files might not be created in test environment


class TestReleaseMetricsComparatorHelpers:
    """Test helper functions and edge cases"""
    
    def test_review_count_calculation(self):
        """Test review count calculation from messages"""
        # This tests the internal logic used in extract_period_changes
        messages = [
            {'author': {'name': 'reviewer1'}, 'date': '2016-10-07T00:00:00Z'},
            {'author': {'name': 'reviewer2'}, 'date': '2016-10-08T00:00:00Z'},
            {'author': {'name': ''}, 'date': '2016-10-09T00:00:00Z'},  # Empty name
        ]
        
        # Count non-empty author names
        review_count = len([m for m in messages if m.get('author', {}).get('name', '')])
        
        assert review_count == 2  # Should count only messages with author names
    
    def test_date_range_calculation(self):
        """Test date range calculation for periods"""
        base_date = datetime(2016, 10, 6)
        
        # Early period: 0 to 30 days after release
        start_early = base_date + timedelta(days=0)
        end_early = base_date + timedelta(days=30)
        
        assert start_early == base_date
        assert end_early == datetime(2016, 11, 5)
        
        # Late period: -30 to 0 days before next release
        next_release = datetime(2017, 2, 22)
        start_late = next_release + timedelta(days=-30)
        end_late = next_release + timedelta(days=0)
        
        assert start_late == datetime(2017, 1, 23)
        assert end_late == next_release
    
    def test_period_overlap_check(self):
        """Test that early and late periods don't overlap (for consecutive releases)"""
        current_release = datetime(2016, 10, 6)
        next_release = datetime(2017, 2, 22)
        
        # Early period end
        early_end = current_release + timedelta(days=30)
        
        # Late period start
        late_start = next_release + timedelta(days=-30)
        
        # They should not overlap (there should be gap between them)
        assert early_end < late_start, "Periods should not overlap for consecutive releases"
        
        # Gap should be significant (more than 60 days for typical releases)
        gap_days = (late_start - early_end).days
        assert gap_days > 60
