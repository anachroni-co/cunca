"""
Tests for the Memory Profiler Module
====================================

Unit tests for memory profiling utilities.
"""

import gc
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from utils.memory_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    AllocationRecord,
    FunctionProfile,
    LeakReport,
    get_profiler,
    profile_memory,
    memory_profile_block,
    check_for_leaks,
    print_memory_summary,
    TrainingMemoryTracker
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_bytes=1024 * 1024 * 100,  # 100 MB
            vms_bytes=1024 * 1024 * 200,  # 200 MB
            heap_bytes=1024 * 1024 * 50,   # 50 MB
            gpu_bytes=0,
            num_objects=10000,
            label="test"
        )

        assert snapshot.rss_mb == pytest.approx(100.0, rel=0.01)
        assert snapshot.gpu_mb == 0.0
        assert snapshot.label == "test"
        assert snapshot.num_objects == 10000

    def test_snapshot_mb_conversion(self):
        """Test byte to MB conversion properties."""
        snapshot = MemorySnapshot(
            timestamp=0,
            rss_bytes=1024 * 1024,  # 1 MB
            vms_bytes=0,
            heap_bytes=0,
            gpu_bytes=1024 * 1024 * 2,  # 2 MB
            num_objects=0
        )

        assert snapshot.rss_mb == pytest.approx(1.0, rel=0.01)
        assert snapshot.gpu_mb == pytest.approx(2.0, rel=0.01)


class TestAllocationRecord:
    """Tests for AllocationRecord dataclass."""

    def test_allocation_creation(self):
        """Test creating an allocation record."""
        record = AllocationRecord(
            size_bytes=1024 * 1024 * 10,  # 10 MB
            timestamp=time.time(),
            traceback="test.py:10 in func",
            dtype="float32",
            shape=(1000, 1000),
            device="cpu"
        )

        assert record.size_mb == pytest.approx(10.0, rel=0.01)
        assert record.dtype == "float32"
        assert record.shape == (1000, 1000)


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_singleton_pattern(self):
        """Test that MemoryProfiler is a singleton."""
        profiler1 = MemoryProfiler()
        profiler2 = MemoryProfiler()
        assert profiler1 is profiler2

    def test_profiler_start_stop(self):
        """Test starting and stopping the profiler."""
        profiler = get_profiler()

        # Reset state
        profiler._running = False
        profiler._snapshots.clear()

        profiler.start()
        assert profiler._running is True

        # Give it time to take a snapshot
        time.sleep(0.5)

        profiler.stop()
        assert profiler._running is False

    def test_take_snapshot(self):
        """Test taking a manual snapshot."""
        profiler = get_profiler()
        profiler._snapshots.clear()

        snapshot = profiler.take_snapshot("manual_test")

        assert snapshot.label == "manual_test"
        assert snapshot.timestamp > 0
        assert len(profiler._snapshots) >= 1

    def test_get_stats(self):
        """Test getting profiler statistics."""
        profiler = get_profiler()
        profiler._snapshots.clear()

        # Take a snapshot to have data
        profiler.take_snapshot()

        stats = profiler.get_stats()

        assert "running" in stats
        assert "snapshot_count" in stats
        assert stats["snapshot_count"] >= 1

    def test_max_snapshots_limit(self):
        """Test that snapshots are trimmed to max_snapshots."""
        profiler = get_profiler()
        original_max = profiler.max_snapshots
        profiler.max_snapshots = 5
        profiler._snapshots.clear()

        # Take more snapshots than max
        for i in range(10):
            profiler.take_snapshot(f"test_{i}")

        assert len(profiler._snapshots) <= 5

        # Restore
        profiler.max_snapshots = original_max

    def test_get_top_allocations(self):
        """Test getting top allocations."""
        profiler = get_profiler()
        profiler._allocations.clear()

        # Add some fake allocations
        for i in range(5):
            profiler._allocations[i] = AllocationRecord(
                size_bytes=(i + 1) * 1024 * 1024,
                timestamp=time.time(),
                traceback=f"test_{i}",
                dtype="float32"
            )

        top = profiler.get_top_allocations(3)

        assert len(top) == 3
        assert top[0]["size_mb"] > top[1]["size_mb"]
        assert top[1]["size_mb"] > top[2]["size_mb"]


class TestProfileMemoryDecorator:
    """Tests for the @profile_memory decorator."""

    def test_decorator_basic(self):
        """Test basic decorator functionality."""
        profiler = get_profiler()
        profiler._function_profiles.clear()

        @profile_memory
        def test_function():
            data = [0] * 1000
            return sum(data)

        result = test_function()

        assert result == 0
        assert len(profiler._function_profiles) >= 1

    def test_decorator_call_count(self):
        """Test that decorator tracks call count."""
        profiler = get_profiler()
        profiler._function_profiles.clear()

        @profile_memory
        def counted_function():
            return 42

        # Call multiple times
        for _ in range(5):
            counted_function()

        # Find the profile for this function
        func_profiles = [
            p for p in profiler._function_profiles.values()
            if p.function_name == "counted_function"
        ]

        assert len(func_profiles) == 1
        assert func_profiles[0].call_count == 5

    def test_decorator_preserves_return_value(self):
        """Test that decorator preserves function return value."""
        @profile_memory
        def returning_function(x, y):
            return x + y

        result = returning_function(3, 4)
        assert result == 7


class TestMemoryProfileBlock:
    """Tests for memory_profile_block context manager."""

    def test_context_manager_basic(self):
        """Test basic context manager functionality."""
        profiler = get_profiler()
        profiler._snapshots.clear()

        with memory_profile_block("test_block"):
            data = [0] * 10000

        # Should have taken at least 2 snapshots (start and end)
        labels = [s.label for s in profiler._snapshots]
        assert any("start:test_block" in l for l in labels)
        assert any("end:test_block" in l for l in labels)

    def test_context_manager_with_exception(self):
        """Test context manager handles exceptions."""
        profiler = get_profiler()
        snapshot_count_before = len(profiler._snapshots)

        with pytest.raises(ValueError):
            with memory_profile_block("error_block"):
                raise ValueError("Test error")

        # Should still have taken snapshots
        assert len(profiler._snapshots) > snapshot_count_before


class TestTrainingMemoryTracker:
    """Tests for TrainingMemoryTracker class."""

    def test_tracker_creation(self):
        """Test creating a training tracker."""
        tracker = TrainingMemoryTracker(log_interval=10)
        assert tracker.log_interval == 10

    def test_epoch_tracking(self):
        """Test tracking an epoch."""
        tracker = TrainingMemoryTracker()

        tracker.start_epoch(0)
        time.sleep(0.1)
        epoch_data = tracker.end_epoch()

        assert epoch_data["epoch"] == 0
        assert "duration_seconds" in epoch_data
        assert "memory_growth_mb" in epoch_data
        assert epoch_data["duration_seconds"] >= 0.1

    def test_step_tracking(self):
        """Test tracking training steps."""
        tracker = TrainingMemoryTracker(log_interval=1)

        tracker.start_epoch(0)

        for i in range(5):
            tracker.start_step()
            time.sleep(0.01)
            tracker.end_step(loss=0.5 - i * 0.1)

        tracker.end_epoch()

        # Should have step data
        assert len(tracker._step_data) >= 1

    def test_report_generation(self):
        """Test generating a training report."""
        tracker = TrainingMemoryTracker()

        # Run a mini training loop
        for epoch in range(2):
            tracker.start_epoch(epoch)
            for _ in range(3):
                tracker.start_step()
                tracker.end_step(loss=0.5)
            tracker.end_epoch()

        report = tracker.report()

        assert report["total_epochs"] == 2
        assert "memory_trend" in report
        assert "leak_detected" in report


class TestReportGeneration:
    """Tests for report generation."""

    def test_json_report(self):
        """Test generating a JSON report."""
        profiler = get_profiler()
        profiler._snapshots.clear()
        profiler._start_time = time.time()

        # Add some data
        for i in range(5):
            profiler.take_snapshot(f"test_{i}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            profiler.generate_report(output_path, format="json")

            assert output_path.exists()

            import json
            with open(output_path) as f:
                data = json.load(f)

            assert "stats" in data
            assert "memory_timeline" in data

    def test_html_report(self):
        """Test generating an HTML report."""
        profiler = get_profiler()
        profiler._snapshots.clear()
        profiler._start_time = time.time()

        # Add some data
        for i in range(5):
            profiler.take_snapshot(f"test_{i}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            profiler.generate_report(output_path, format="html")

            assert output_path.exists()

            with open(output_path) as f:
                content = f.read()

            assert "<html>" in content
            assert "Memory Profile Report" in content
            assert "Chart" in content


class TestLeakDetection:
    """Tests for memory leak detection."""

    def test_leak_analysis_insufficient_data(self):
        """Test leak analysis with insufficient data."""
        profiler = get_profiler()
        profiler._snapshots.clear()
        profiler._leak_reports.clear()

        # Only a few snapshots
        for i in range(3):
            profiler.take_snapshot()

        profiler._analyze_leaks()

        # Should not report leaks with insufficient data
        # (may or may not have reports depending on growth)

    def test_check_for_leaks_function(self):
        """Test the check_for_leaks helper function."""
        profiler = get_profiler()
        profiler._snapshots.clear()
        profiler._leak_reports.clear()

        leaks = check_for_leaks(threshold_mb=1000.0)

        # Should return a list (possibly empty)
        assert isinstance(leaks, list)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_profiler(self):
        """Test get_profiler returns singleton."""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        assert profiler1 is profiler2

    def test_print_memory_summary(self, capsys):
        """Test print_memory_summary outputs data."""
        print_memory_summary()

        captured = capsys.readouterr()
        assert "MEMORY PROFILE SUMMARY" in captured.out
        assert "RSS Memory" in captured.out


class TestFunctionProfile:
    """Tests for FunctionProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a function profile."""
        profile = FunctionProfile(
            function_name="test_func",
            module="test_module",
            call_count=10,
            total_allocated_bytes=1024 * 1024,
            peak_memory_bytes=2 * 1024 * 1024,
            total_time_seconds=1.5
        )

        assert profile.function_name == "test_func"
        assert profile.call_count == 10
        assert profile.total_time_seconds == 1.5


class TestLeakReport:
    """Tests for LeakReport dataclass."""

    def test_leak_report_creation(self):
        """Test creating a leak report."""
        report = LeakReport(
            location="test_location",
            growth_bytes=100 * 1024 * 1024,
            growth_rate_bytes_per_sec=1024 * 1024,
            sample_count=100,
            confidence=0.85,
            first_seen=time.time() - 100,
            last_seen=time.time()
        )

        assert report.location == "test_location"
        assert report.confidence == 0.85
        assert report.sample_count == 100
