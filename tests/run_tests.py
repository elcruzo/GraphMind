#!/usr/bin/env python3
"""
Test runner for GraphMind project

Runs all unit tests and generates coverage report.
"""

import sys
import pytest
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_all_tests():
    """Run all tests with coverage"""
    print("="*60)
    print("Running GraphMind Test Suite")
    print("="*60)
    
    # Test arguments
    args = [
        # Test discovery
        'tests/',
        
        # Verbosity
        '-v',
        
        # Coverage
        '--cov=src',
        '--cov-report=html',
        '--cov-report=term-missing',
        
        # Output
        '--tb=short',
        
        # Parallel execution (if pytest-xdist installed)
        # '-n', 'auto',
        
        # Stop on first failure (optional)
        # '-x',
        
        # Show local variables in tracebacks
        '-l',
        
        # Capture output
        '-s',
    ]
    
    # Run tests
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\n📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed!")
        print("="*60)
    
    return exit_code

def run_specific_module(module_name):
    """Run tests for a specific module"""
    print(f"Running tests for module: {module_name}")
    
    test_file = f"tests/test_{module_name}.py"
    
    if not Path(test_file).exists():
        print(f"Error: Test file {test_file} not found!")
        return 1
    
    args = [
        test_file,
        '-v',
        '--tb=short',
        '-l',
    ]
    
    return pytest.main(args)

def run_performance_tests():
    """Run performance/benchmark tests"""
    print("Running performance tests...")
    
    args = [
        'tests/',
        '-v',
        '-k', 'performance or benchmark or scalability',
        '--tb=short',
    ]
    
    return pytest.main(args)

def run_integration_tests():
    """Run integration tests only"""
    print("Running integration tests...")
    
    args = [
        'tests/',
        '-v',
        '-k', 'integration or Integration',
        '--tb=short',
    ]
    
    return pytest.main(args)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GraphMind Test Runner')
    parser.add_argument('--module', type=str, help='Run tests for specific module')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (no coverage)')
    
    args = parser.parse_args()
    
    if args.module:
        exit_code = run_specific_module(args.module)
    elif args.performance:
        exit_code = run_performance_tests()
    elif args.integration:
        exit_code = run_integration_tests()
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)