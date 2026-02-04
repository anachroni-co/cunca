"""
Tests for the Automated Benchmark System
========================================

Unit tests for benchmark runner, comparator, and reporter.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkSuite,
    TimingStats,
    benchmark,
    run_benchmarks,
    benchmark_context
)
from benchmarks.comparator import (
    BenchmarkComparator,
    ComparisonResult,
    RegressionReport,
    compare_results,
    detect_regressions,
    update_baseline
)
from benchmarks.reporter import (
    BenchmarkReporter,
    generate_report
)


class TestTimingStats:
    """Tests for TimingStats dataclass."""

    def test_timing_stats_creation(self):
        """Test creating timing statistics."""
        stats = TimingStats(
            min_ms=1.0,
            max_ms=10.0,
            mean_ms=5.0,
            median_ms=4.5,
            std_ms=2.0,
            iterations=100,
            warmup_iterations=3
        )

        assert stats.min_ms == 1.0
        assert stats.max_ms == 10.0
        assert stats.mean_ms == 5.0
        assert stats.iterations == 100


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test creating a benchmark result."""
        timing = TimingStats(1, 10, 5, 4.5, 2, 100, 3)
        result = BenchmarkResult(
            name="test_benchmark",
            group="core",
            timing=timing,
            memory_before_mb=100.0,
            memory_after_mb=150.0,
            memory_peak_mb=200.0,
            timestamp="2024-01-01T00:00:00",
            hardware_info={"cpu": "test"},
            tags=["core", "fast"]
        )

        assert result.name == "test_benchmark"
        assert result.group == "core"
        assert result.throughput_ops_per_sec == pytest.approx(200.0)

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        timing = TimingStats(1, 10, 5, 4.5, 2, 100, 3)
        result = BenchmarkResult(
            name="test",
            group="default",
            timing=timing,
            memory_before_mb=0,
            memory_after_mb=0,
            memory_peak_mb=0,
            timestamp="",
            hardware_info={}
        )

        data = result.to_dict()
        assert "name" in data
        assert "timing" in data
        assert "throughput_ops_per_sec" in data


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_runner_creation(self):
        """Test creating a benchmark runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                warmup_iterations=2,
                min_iterations=3,
                results_dir=tmpdir
            )

            assert runner.warmup_iterations == 2
            assert runner.min_iterations == 3

    def test_add_benchmark(self):
        """Test adding a benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(results_dir=tmpdir)

            def simple_benchmark():
                return sum(range(1000))

            runner.add_benchmark("simple", simple_benchmark, group="test")

            assert "simple" in runner._benchmarks

    def test_run_benchmark(self):
        """Test running a simple benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                warmup_iterations=1,
                min_iterations=2,
                max_iterations=5,
                max_time_seconds=1.0,
                results_dir=tmpdir
            )

            def fast_benchmark():
                return sum(range(100))

            runner.add_benchmark("fast", fast_benchmark)
            results = runner.run()

            assert len(results) == 1
            assert results[0].name == "fast"
            assert results[0].passed
            assert results[0].timing.iterations >= 2

    def test_run_multiple_benchmarks(self):
        """Test running multiple benchmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                warmup_iterations=1,
                min_iterations=2,
                max_time_seconds=1.0,
                results_dir=tmpdir
            )

            runner.add_benchmark("bench1", lambda: sum(range(100)), group="a")
            runner.add_benchmark("bench2", lambda: sum(range(200)), group="b")
            runner.add_benchmark("bench3", lambda: sum(range(300)), group="a")

            results = runner.run()
            assert len(results) == 3

    def test_filter_by_group(self):
        """Test filtering benchmarks by group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                warmup_iterations=1,
                min_iterations=2,
                max_time_seconds=1.0,
                results_dir=tmpdir
            )

            runner.add_benchmark("bench1", lambda: 1, group="include")
            runner.add_benchmark("bench2", lambda: 2, group="exclude")

            results = runner.run(filter_groups=["include"])
            assert len(results) == 1
            assert results[0].name == "bench1"

    def test_save_results(self):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                warmup_iterations=1,
                min_iterations=2,
                max_time_seconds=1.0,
                results_dir=tmpdir
            )

            runner.add_benchmark("test", lambda: 42)
            runner.run()

            results_path = runner.save_results(append_timestamp=False)
            assert results_path.exists()

            with open(results_path) as f:
                data = json.load(f)

            assert "results" in data
            assert "hardware_info" in data
            assert len(data["results"]) == 1

    def test_benchmark_failure(self):
        """Test handling benchmark failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                warmup_iterations=1,
                min_iterations=2,
                results_dir=tmpdir
            )

            def failing_benchmark():
                raise ValueError("Test error")

            runner.add_benchmark("failing", failing_benchmark)
            results = runner.run()

            assert len(results) == 1
            assert not results[0].passed
            assert "Test error" in results[0].error_message


class TestBenchmarkDecorator:
    """Tests for the @benchmark decorator."""

    def test_decorator_marks_function(self):
        """Test that decorator adds metadata."""
        @benchmark(group="test", tags=["fast"])
        def my_benchmark():
            return 42

        assert my_benchmark._benchmark_group == "test"
        assert "fast" in my_benchmark._benchmark_tags


class TestBenchmarkContext:
    """Tests for benchmark_context context manager."""

    def test_context_manager_timing(self):
        """Test context manager captures timing."""
        with benchmark_context("test") as timer:
            time.sleep(0.1)

        assert timer.elapsed_ms >= 100
        assert timer.elapsed_ms < 200


class TestBenchmarkComparator:
    """Tests for BenchmarkComparator class."""

    def test_comparator_creation(self):
        """Test creating a comparator."""
        comparator = BenchmarkComparator(
            regression_threshold_percent=15.0,
            improvement_threshold_percent=15.0
        )

        assert comparator.regression_threshold == 15.0

    def test_compare_results(self):
        """Test comparing two result sets."""
        baseline = {
            "results": [
                {"name": "bench1", "timing": {"mean_ms": 10.0, "iterations": 10}},
                {"name": "bench2", "timing": {"mean_ms": 20.0, "iterations": 10}},
            ]
        }

        current = {
            "results": [
                {"name": "bench1", "timing": {"mean_ms": 12.0, "iterations": 10}},  # +20%
                {"name": "bench2", "timing": {"mean_ms": 15.0, "iterations": 10}},  # -25%
            ]
        }

        comparator = BenchmarkComparator(
            regression_threshold_percent=10.0,
            improvement_threshold_percent=10.0
        )
        report = comparator.compare(baseline, current)

        assert report.total_benchmarks == 2
        assert len(report.regressions) == 1
        assert len(report.improvements) == 1
        assert report.regressions[0].benchmark_name == "bench1"
        assert report.improvements[0].benchmark_name == "bench2"

    def test_has_regressions_property(self):
        """Test has_regressions property."""
        report = RegressionReport(
            timestamp="",
            baseline_file="",
            current_file="",
            total_benchmarks=1,
            regressions=[
                ComparisonResult("test", 10, 15, 5, 50, True, False, 10, 10, "high")
            ],
            improvements=[],
            unchanged=[],
            regression_threshold_percent=10.0
        )

        assert report.has_regressions
        assert report.regression_count == 1


class TestDetectRegressions:
    """Tests for detect_regressions function."""

    def test_detect_no_regressions(self):
        """Test when no regressions exist."""
        baseline = {
            "results": [
                {"name": "bench1", "timing": {"mean_ms": 10.0, "iterations": 10}},
            ]
        }
        current = {
            "results": [
                {"name": "bench1", "timing": {"mean_ms": 10.5, "iterations": 10}},  # +5%
            ]
        }

        has_regression, report = detect_regressions(
            baseline, current, threshold_percent=10.0, fail_on_regression=False
        )

        assert not has_regression

    def test_detect_with_regressions(self):
        """Test when regressions exist."""
        baseline = {
            "results": [
                {"name": "bench1", "timing": {"mean_ms": 10.0, "iterations": 10}},
            ]
        }
        current = {
            "results": [
                {"name": "bench1", "timing": {"mean_ms": 15.0, "iterations": 10}},  # +50%
            ]
        }

        has_regression, report = detect_regressions(
            baseline, current, threshold_percent=10.0, fail_on_regression=False
        )

        assert has_regression


class TestUpdateBaseline:
    """Tests for update_baseline function."""

    def test_update_baseline(self):
        """Test updating baseline file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial results
            results = {
                "timestamp": "2024-01-01",
                "results": [{"name": "test", "timing": {"mean_ms": 10.0}}]
            }

            results_path = Path(tmpdir) / "results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f)

            # Update baseline
            baseline_path = update_baseline(results_path, tmpdir, "baseline.json")

            assert baseline_path.exists()
            with open(baseline_path) as f:
                data = json.load(f)
            assert data["results"][0]["name"] == "test"


class TestBenchmarkReporter:
    """Tests for BenchmarkReporter class."""

    def test_reporter_creation(self):
        """Test creating a reporter."""
        reporter = BenchmarkReporter(title="Test Report")
        assert reporter.title == "Test Report"

    def test_generate_html_report(self):
        """Test generating HTML report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "timestamp": "2024-01-01T00:00:00",
                "hardware_info": {"cpu": "test"},
                "config": {"warmup_iterations": 3},
                "results": [
                    {
                        "name": "bench1",
                        "group": "core",
                        "timing": {
                            "mean_ms": 10.0,
                            "median_ms": 9.5,
                            "std_ms": 1.0,
                            "min_ms": 8.0,
                            "max_ms": 12.0,
                            "iterations": 10
                        },
                        "passed": True
                    }
                ],
                "summary": {"total_benchmarks": 1, "passed": 1, "failed": 0}
            }

            reporter = BenchmarkReporter()
            output_path = Path(tmpdir) / "report.html"
            reporter.generate_html(results, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "<html" in content
            assert "bench1" in content

    def test_generate_markdown_report(self):
        """Test generating Markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "timestamp": "2024-01-01",
                "hardware_info": {},
                "config": {},
                "results": [
                    {
                        "name": "bench1",
                        "timing": {"mean_ms": 10.0, "std_ms": 1.0, "iterations": 10},
                        "passed": True
                    }
                ],
                "summary": {}
            }

            reporter = BenchmarkReporter()
            output_path = Path(tmpdir) / "report.md"
            reporter.generate_markdown(results, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "# " in content
            assert "bench1" in content

    def test_generate_json_report(self):
        """Test generating JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "timestamp": "2024-01-01",
                "results": [{"name": "test"}]
            }

            reporter = BenchmarkReporter()
            output_path = Path(tmpdir) / "report.json"
            reporter.generate_json(results, output_path)

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
            assert "results" in data


class TestGenerateReport:
    """Tests for generate_report convenience function."""

    def test_generate_report_html(self):
        """Test generate_report with HTML format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "timestamp": "2024-01-01",
                "hardware_info": {},
                "config": {},
                "results": [],
                "summary": {}
            }

            output_path = generate_report(
                results,
                Path(tmpdir) / "report.html",
                format="html"
            )

            assert output_path.exists()

    def test_generate_report_invalid_format(self):
        """Test generate_report with invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {"results": []}

            with pytest.raises(ValueError) as exc_info:
                generate_report(
                    results,
                    Path(tmpdir) / "report.xyz",
                    format="xyz"
                )

            assert "Unknown format" in str(exc_info.value)


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_suite_creation(self):
        """Test creating a benchmark suite."""
        def bench1():
            return 1

        def bench2():
            return 2

        suite = BenchmarkSuite(
            name="test_suite",
            description="Test suite",
            benchmarks=[bench1, bench2],
            tags=["core"]
        )

        assert suite.name == "test_suite"
        assert len(suite.benchmarks) == 2
        assert "core" in suite.tags

    def test_suite_with_setup_teardown(self):
        """Test suite with setup and teardown."""
        setup_called = []
        teardown_called = []

        def setup():
            setup_called.append(True)

        def teardown():
            teardown_called.append(True)

        suite = BenchmarkSuite(
            name="test",
            benchmarks=[lambda: 1],
            setup=setup,
            teardown=teardown
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                warmup_iterations=1,
                min_iterations=1,
                results_dir=tmpdir
            )
            runner.add_suite(suite)
            runner.run()

        assert len(setup_called) == 1
        assert len(teardown_called) == 1
