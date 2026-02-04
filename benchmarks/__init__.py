"""
CapibaraGPT Benchmark Suite
===========================

Automated benchmarking system for performance tracking and CI/CD integration.

Features:
- Configurable benchmark suites
- Historical result tracking
- Performance regression detection
- JSON/HTML report generation
- CI/CD pipeline integration
"""

from .runner import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkSuite,
    run_benchmarks
)
from .comparator import (
    BenchmarkComparator,
    compare_results,
    detect_regressions
)
from .reporter import (
    BenchmarkReporter,
    generate_report
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkSuite",
    "run_benchmarks",
    "BenchmarkComparator",
    "compare_results",
    "detect_regressions",
    "BenchmarkReporter",
    "generate_report",
]
