#!/usr/bin/env python
"""
Test Runner for IdiotAutoLabeler
================================

Run all tests or specific test categories to verify functionality.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Run quick tests only (skip slow ones)
    python run_tests.py --verbose    # Run with verbose output
    python run_tests.py --category metrics    # Run only metrics tests
    python run_tests.py --coverage   # Run with coverage report

Categories:
    metrics       - Training metrics and loss functions
    preprocessing - Image/mask preprocessing
    inference     - Model loading and inference
    kalman        - Kalman filter tracking
    features      - Trigger and aimbot logic
    config        - Configuration persistence
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for IdiotAutoLabeler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick tests only (skip slow model tests)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--category", "-c",
        choices=["metrics", "preprocessing", "inference", "kalman", "features", "config"],
        help="Run only a specific test category"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )

    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="Stop on first failure"
    )

    parser.add_argument(
        "--last-failed", "-lf",
        action="store_true",
        help="Re-run only tests that failed last time"
    )

    parser.add_argument(
        "--markers", "-m",
        help="Run tests matching given marker expression (e.g., 'not slow')"
    )

    args = parser.parse_args()

    # Use the same Python interpreter that's running this script
    python_exe = sys.executable

    # Build pytest command
    cmd = [python_exe, "-m", "pytest", "tests/"]

    # Verbose
    if args.verbose:
        cmd.append("-v")

    # Fail fast
    if args.failfast:
        cmd.append("-x")

    # Last failed
    if args.last_failed:
        cmd.append("--lf")

    # Category filter
    if args.category:
        cmd[-1] = f"tests/test_{args.category}.py"

    # Quick mode - skip slow tests
    if args.quick:
        cmd.extend(["-m", "not slow"])

    # Markers
    if args.markers:
        cmd.extend(["-m", args.markers])

    # Coverage
    if args.coverage:
        cmd = [python_exe, "-m", "pytest", "--cov=src", "--cov-report=term-missing"] + cmd[3:]

    # Add color output
    cmd.append("--color=yes")

    # Print command
    print(f"\n{'='*60}")
    print("Running: " + " ".join(cmd))
    print(f"{'='*60}\n")

    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    # Print summary
    print(f"\n{'='*60}")
    if result.returncode == 0:
        print("All tests passed!")
    else:
        print(f"Tests failed with exit code {result.returncode}")
    print(f"{'='*60}\n")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
