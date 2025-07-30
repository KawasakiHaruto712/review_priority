"""
Tests for refactoring_metrics.py module
"""
import pytest
from src.features.refactoring_metrics import calculate_refactoring_confidence


class TestRefactoringMetrics:
    """Test class for refactoring metrics functions"""
    
    def test_calculate_refactoring_confidence_basic_patterns(self):
        """Test refactoring confidence with basic SAR patterns"""
        test_cases = [
            ("Refactor authentication module", None, 1),
            ("Move utility functions", None, 1),
            ("Split large class into smaller ones", None, 1),
            ("Fix code structure", None, 1),
            ("Introduce new interface", None, 1),
            ("Decompose complex method", None, 1),
            ("Reorganize project structure", None, 1),
            ("Extract common functionality", None, 1),
            ("Merge duplicate classes", None, 1),
            ("Rename variables for clarity", None, 1),
        ]
        
        for title, description, expected in test_cases:
            result = calculate_refactoring_confidence(title, description)
            assert result == expected, f"Expected {expected} for '{title}', got {result}"

    def test_calculate_refactoring_confidence_advanced_patterns(self):
        """Test refactoring confidence with advanced SAR patterns"""
        test_cases = [
            ("Code cleanup and optimization", None, 1),
            ("Remove redundant code", None, 1),
            ("Improve naming consistency", None, 1),
            ("Fix technical debt issues", None, 1),
            ("Clean up unnecessary code", None, 1),
            ("Code reformatting & reordering", None, 1),
            ("Simplify code redundancies", None, 1),
            ("Enhanced code beauty", None, 1),
            ("Fix code smell", None, 1),
            ("Improve code quality", None, 1),
        ]
        
        for title, description, expected in test_cases:
            result = calculate_refactoring_confidence(title, description)
            assert result == expected, f"Expected {expected} for '{title}', got {result}"

    def test_calculate_refactoring_confidence_case_insensitive(self):
        """Test that pattern matching is case insensitive"""
        test_cases = [
            ("REFACTOR CODE", None, 1),
            ("Move Files", None, 1),
            ("split module", None, 1),
            ("FIX Structure", None, 1),
            ("IMPROVE naming", None, 1),
        ]
        
        for title, description, expected in test_cases:
            result = calculate_refactoring_confidence(title, description)
            assert result == expected, f"Expected {expected} for case insensitive '{title}'"

    def test_calculate_refactoring_confidence_with_description(self):
        """Test refactoring confidence using description text"""
        title = "Update module"
        description = "This change refactors the authentication system to improve maintainability"
        result = calculate_refactoring_confidence(title, description)
        assert result == 1, "Expected 1 when description contains refactoring pattern"
        
        # Test with pattern in title but not description
        title2 = "Refactor code"
        description2 = "Add new feature for users"
        result2 = calculate_refactoring_confidence(title2, description2)
        assert result2 == 1, "Expected 1 when title contains refactoring pattern"

    def test_calculate_refactoring_confidence_no_match(self):
        """Test refactoring confidence with no matching patterns"""
        test_cases = [
            ("Implement user authentication", None),
            ("Update documentation", None),
            ("Version bump to 2.0", None),
            ("Initial commit", None),
            ("Bump version", None),
            ("Update README", None),
        ]
        
        for title, description in test_cases:
            result = calculate_refactoring_confidence(title, description)
            assert result == 0, f"Expected 0 for non-refactoring '{title}', got {result}"

    def test_calculate_refactoring_confidence_with_none_description(self):
        """Test refactoring confidence with None description"""
        title = "Refactor authentication system"
        result = calculate_refactoring_confidence(title, None)
        assert result == 1, "Expected 1 for refactoring title with None description"

    def test_calculate_refactoring_confidence_word_boundaries(self):
        """Test that patterns respect word boundaries"""
        # These should NOT match because the patterns are not at word boundaries
        non_matching_cases = [
            "preferences", # contains "refactor" but not as whole word
            "movement",    # contains "move" but not as whole word  
            "prefix",      # contains "fix" but not as whole word
        ]
        
        for title in non_matching_cases:
            result = calculate_refactoring_confidence(title, None)
            assert result == 0, f"Expected 0 for non-word-boundary '{title}', got {result}"
        
        # These SHOULD match because they are proper word boundaries
        matching_cases = [
            "refactor preferences",
            "move old files", 
            "fix the structure",
        ]
        
        for title in matching_cases:
            result = calculate_refactoring_confidence(title, None)
            assert result == 1, f"Expected 1 for word-boundary '{title}', got {result}"

    def test_calculate_refactoring_confidence_complex_patterns(self):
        """Test complex multi-word patterns"""
        test_cases = [
            ("Remove poor coding practice", None, 1),
            ("Pull some code up", None, 1),
            ("Use better name", None, 1),
            ("Replace it with new implementation", None, 1),
            ("Make maintenance easier", None, 1),
            ("Minor simplification of logic", None, 1),
            ("Reorganize project structures", None, 1),
            ("Fix quality issue in module", None, 1),
            ("Getting code out of legacy system", None, 1),
            ("Deleting a lot of old stuff", None, 1),
        ]
        
        for title, description, expected in test_cases:
            result = calculate_refactoring_confidence(title, description)
            assert result == expected, f"Expected {expected} for complex pattern '{title}'"

    def test_calculate_refactoring_confidence_edge_cases(self):
        """Test edge cases for refactoring confidence calculation"""
        # Empty strings
        assert calculate_refactoring_confidence("", None) == 0
        assert calculate_refactoring_confidence("", "") == 0
        
        # Only spaces
        assert calculate_refactoring_confidence("   ", None) == 0
        
        # Special characters without meaningful patterns
        assert calculate_refactoring_confidence("!@#$%^&*()", None) == 0
        
        # Very long text with pattern
        long_text = "This is a very long description that contains the word refactor somewhere in the middle of a lot of other text"
        assert calculate_refactoring_confidence(long_text, None) == 1

    def test_calculate_refactoring_confidence_partial_word_patterns(self):
        """Test patterns that include word variations"""
        # Test patterns with optional endings like "refactor(ing|ed)?"
        test_cases = [
            ("refactoring", 1),
            ("refactored", 1),
            ("refactor", 1),
            ("moving", 1),
            ("moved", 1),
            ("move", 1),
            ("splitting", 1),
            ("split", 1),
            ("fixing", 1),
            ("fixed", 1),
            ("fixes", 1),
            ("fix", 1),
        ]
        
        for title, expected in test_cases:
            result = calculate_refactoring_confidence(title, None)
            assert result == expected, f"Expected {expected} for word variation '{title}'"

    def test_calculate_refactoring_confidence_whitespace_handling(self):
        """Test that multi-word patterns handle whitespace correctly"""
        test_cases = [
            ("remove   redundant   code", 1),  # Multiple spaces
            ("code\tcleanup", 1),               # Tab character
            ("improve\nnaming\nconsistency", 1), # Newlines
            ("fix  quality   issue", 1),        # Mixed spacing
        ]
        
        for title, expected in test_cases:
            result = calculate_refactoring_confidence(title, None)
            assert result == expected, f"Expected {expected} for whitespace test '{title}'"


if __name__ == "__main__":
    pytest.main([__file__])
