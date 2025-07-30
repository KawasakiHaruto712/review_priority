#!/usr/bin/env python3
"""
Test runner script for the review-priority project
"""
import subprocess
import sys
from pathlib import Path


def install_test_dependencies():
    """Install test dependencies"""
    print("Installing test dependencies...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", ".[test]"
    ], cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("Failed to install test dependencies")
        sys.exit(1)


def run_tests():
    """Run all tests"""
    print("Running tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v"
    ], cwd=Path(__file__).parent)
    
    return result.returncode


def run_unit_tests():
    """Run only unit tests (exclude integration tests)"""
    print("Running unit tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/features/", "-v"
    ], cwd=Path(__file__).parent)
    
    return result.returncode


def run_integration_tests():
    """Run only integration tests"""
    print("Running integration tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/test_integration.py", "-v"
    ], cwd=Path(__file__).parent)
    
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "install":
            install_test_dependencies()
        elif sys.argv[1] == "unit":
            sys.exit(run_unit_tests())
        elif sys.argv[1] == "integration":
            sys.exit(run_integration_tests())
        else:
            print("Usage: python run_tests.py [install|unit|integration]")
            sys.exit(1)
    else:
        # Install dependencies and run all tests
        install_test_dependencies()
        sys.exit(run_tests())
