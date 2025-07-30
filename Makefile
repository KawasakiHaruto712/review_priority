# Makefile for review-priority project

.PHONY: help install test test-unit test-integration clean

help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies and test dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  clean          - Clean up generated files"

install:
	pip install -e .[test]

test: install
	pytest tests/ -v

test-unit: install
	pytest tests/features/ -v

test-integration: install
	pytest tests/test_integration.py -v

clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
