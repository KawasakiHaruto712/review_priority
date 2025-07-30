# Test directory for review-priority project

This directory contains unit tests and integration tests for the project.

## Structure

- `features/` - Tests for feature extraction modules
- `data/` - Sample test data files
- `test_integration.py` - Integration tests using real data

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e .[test]
```

### All Tests
```bash
pytest tests/ -v
```

### Unit Tests Only
```bash
pytest tests/features/ -v
```

### Integration Tests Only
```bash
pytest tests/test_integration.py -v
```

### Using Makefile
```bash
make test           # All tests
make test-unit      # Unit tests only
make test-integration # Integration tests only
```

### Using the run_tests.py Script
```bash
python run_tests.py                # Install deps and run all tests
python run_tests.py install        # Install test dependencies
python run_tests.py unit          # Run unit tests
python run_tests.py integration   # Run integration tests
```

## Test Categories

### Unit Tests (`tests/features/`)
- Test individual functions in isolation
- Use mock data and mocked dependencies
- Fast execution
- Test edge cases and error handling

### Integration Tests (`test_integration.py`)
- Test with real data files from `data/openstack/`
- Verify that functions work with actual change data
- Slower execution but more realistic testing
- Can be skipped if data directories don't exist

## Test Data

- `tests/data/sample_change.json` - Sample change data for unit tests
- Real data in `data/openstack/{project}/changes/` is used for integration tests
