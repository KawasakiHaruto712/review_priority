"""
Tests for bug_metrics.py module
"""
import pytest
from src.features.bug_metrics import calculate_bug_fix_confidence


class TestBugMetrics:
    """Test class for bug metrics functions"""
    
    def test_calculate_bug_fix_confidence_with_bug_number_patterns(self):
        """Test bug fix confidence calculation with bug number patterns"""
        # Test various bug number patterns
        test_cases = [
            ("Fix bug #123", None, 1),  # bug # pattern
            ("Resolve pr 456", None, 1),  # pr pattern
            ("Fix for show_bug.cgi?id=789", None, 1),  # show_bug.cgi pattern
            ("Address issue [101]", None, 1),  # bracket pattern
            ("BUG 202 fixed", None, 1),  # case insensitive
        ]
        
        for title, description, expected in test_cases:
            result = calculate_bug_fix_confidence(title, description)
            assert result >= 1, f"Expected at least 1 for '{title}', got {result}"

    def test_calculate_bug_fix_confidence_with_keywords(self):
        """Test bug fix confidence calculation with fix keywords"""
        test_cases = [
            ("Fix authentication issue", None, 1),
            ("Fixed the memory bug", None, 1), 
            ("Fixes various bugs", None, 1),
            ("Address defect in parser", None, 1),
            ("Apply patch for security", None, 1),
            ("Fix defects in code", None, 1),
        ]
        
        for title, description, expected in test_cases:
            result = calculate_bug_fix_confidence(title, description)
            assert result >= 1, f"Expected at least 1 for '{title}', got {result}"

    def test_calculate_bug_fix_confidence_with_numeric_patterns(self):
        """Test bug fix confidence calculation with numeric-only patterns"""
        test_cases = [
            ("123456", None, 1),  # numbers only
            ("12.34.56", None, 1),  # numbers with dots
            ("789-012", None, 1),  # numbers with dashes
            ("[123] 456", None, 1),  # numbers with brackets and spaces
        ]
        
        for title, description, expected in test_cases:
            result = calculate_bug_fix_confidence(title, description)
            assert result >= 1, f"Expected at least 1 for '{title}', got {result}"

    def test_calculate_bug_fix_confidence_max_score(self):
        """Test that bug fix confidence is capped at 2"""
        # Case with both bug number pattern and keyword
        title = "Fix bug #123 - critical defect"
        result = calculate_bug_fix_confidence(title, None)
        assert result == 2, f"Expected max score of 2, got {result}"

    def test_calculate_bug_fix_confidence_zero_score(self):
        """Test bug fix confidence calculation with no matching patterns"""
        test_cases = [
            ("Add new feature", None),
            ("Implement user authentication", None),
            ("Update documentation", None),
            ("Refactor code structure", None),
            ("Add tests for module", None),
        ]
        
        for title, description in test_cases:
            result = calculate_bug_fix_confidence(title, description)
            assert result == 0, f"Expected 0 for '{title}', got {result}"

    def test_calculate_bug_fix_confidence_with_description(self):
        """Test bug fix confidence calculation using description text"""
        title = "Update module"
        description = "This change fixes bug #456 in the authentication system"
        result = calculate_bug_fix_confidence(title, description)
        assert result >= 1, f"Expected at least 1 when description contains bug pattern"

    def test_calculate_bug_fix_confidence_with_none_description(self):
        """Test bug fix confidence calculation with None description"""
        title = "Fix authentication bug"
        result = calculate_bug_fix_confidence(title, None)
        assert result >= 1, f"Expected at least 1 for title with fix keyword"

    def test_calculate_bug_fix_confidence_case_insensitive(self):
        """Test that pattern matching is case insensitive"""
        test_cases = [
            ("FIX BUG #123", None),
            ("FIXES DEFECT", None),
            ("Fixed Bugs", None),
            ("fix issue", None),
        ]
        
        for title, description in test_cases:
            result = calculate_bug_fix_confidence(title, description)
            assert result >= 1, f"Expected at least 1 for case insensitive '{title}'"

    def test_calculate_bug_fix_confidence_edge_cases(self):
        """Test edge cases for bug fix confidence calculation"""
        # Empty strings
        assert calculate_bug_fix_confidence("", None) == 0
        assert calculate_bug_fix_confidence("", "") == 0
        
        # Only spaces
        assert calculate_bug_fix_confidence("   ", None) == 0
        
        # Special characters without meaningful patterns
        assert calculate_bug_fix_confidence("!@#$%^&*()", None) == 0

    def test_calculate_bug_fix_confidence_combined_patterns(self):
        """Test combinations of different patterns"""
        # Bug number + keyword
        title1 = "Fix bug #123 and patch defects"
        result1 = calculate_bug_fix_confidence(title1, None)
        assert result1 == 2
        
        # Numeric pattern only (without keywords) 
        title2 = "123456"
        result2 = calculate_bug_fix_confidence(title2, None)
        assert result2 == 1
        
        # Keyword only (without bug number pattern)
        title3 = "This fixes the issue"
        result3 = calculate_bug_fix_confidence(title3, None)
        assert result3 == 1
        
        # Bug number in title, keyword in description
        title4 = "Change for bug #789"
        description4 = "This fixes the authentication problem"
        result4 = calculate_bug_fix_confidence(title4, description4)
        assert result4 == 2
        
        # Numeric only + keyword in separate locations
        title5 = "123456"
        description5 = "fixes issue"
        result5 = calculate_bug_fix_confidence(title5, description5)
        assert result5 == 2


if __name__ == "__main__":
    pytest.main([__file__])
