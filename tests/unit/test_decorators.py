"""
Unit Tests for Optimization Decorators

Tests the core/decorators.py module functionality.
"""

import time
import numpy as np
import pytest
from unittest.mock import patch


class TestCachedComputation:
    """Test cached_computation decorator."""

    def test_basic_caching(self):
        """Test that results are cached."""
        from core.decorators import cached_computation

        call_count = 0

        @cached_computation(maxsize=10)
        def expensive_fn(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_fn(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = expensive_fn(5)
        assert result2 == 10
        assert call_count == 1  # No additional call

        # Different arg - should compute
        result3 = expensive_fn(10)
        assert result3 == 20
        assert call_count == 2

    def test_ttl_expiry(self):
        """Test TTL-based cache expiry."""
        from core.decorators import cached_computation

        call_count = 0

        @cached_computation(maxsize=10, ttl_seconds=0.1)
        def fn_with_ttl(x):
            nonlocal call_count
            call_count += 1
            return x

        # First call
        fn_with_ttl(1)
        assert call_count == 1

        # Immediate second call - cached
        fn_with_ttl(1)
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should recompute after TTL
        fn_with_ttl(1)
        assert call_count == 2

    def test_cache_info(self):
        """Test cache_info method."""
        from core.decorators import cached_computation

        @cached_computation(maxsize=5, ttl_seconds=60)
        def fn(x):
            return x

        fn(1)
        fn(2)

        info = fn.cache_info()
        assert info["size"] == 2
        assert info["maxsize"] == 5
        assert info["ttl"] == 60

    def test_clear_cache(self):
        """Test clear_cache method."""
        from core.decorators import cached_computation

        call_count = 0

        @cached_computation(maxsize=10)
        def fn(x):
            nonlocal call_count
            call_count += 1
            return x

        fn(1)
        assert call_count == 1

        fn(1)
        assert call_count == 1

        fn.clear_cache()

        fn(1)
        assert call_count == 2

    def test_maxsize_eviction(self):
        """Test that oldest entries are evicted when maxsize is reached."""
        from core.decorators import cached_computation

        @cached_computation(maxsize=3)
        def fn(x):
            return x

        fn(1)
        fn(2)
        fn(3)

        info = fn.cache_info()
        assert info["size"] == 3

        # Add one more - should evict oldest
        fn(4)
        info = fn.cache_info()
        assert info["size"] == 3


class TestJitIfAvailable:
    """Test JIT decorators."""

    def test_passthrough_without_jax(self):
        """Test that decorator passes through when JAX not available."""
        from core.decorators import jit_if_available

        @jit_if_available()
        def my_fn(x):
            return x * 2

        result = my_fn(5)
        assert result == 10


class TestCachedMask:
    """Test cached mask functionality."""

    def test_get_causal_mask(self):
        """Test causal mask generation and caching."""
        from core.decorators import get_causal_mask

        mask1 = get_causal_mask(4)

        # Check shape
        assert mask1.shape == (4, 4)

        # Check values (lower triangular)
        expected = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ], dtype=np.float32)
        np.testing.assert_array_equal(mask1, expected)

        # Second call should return same object (cached)
        mask2 = get_causal_mask(4)
        assert mask1 is mask2  # Same object from cache

    def test_different_sizes_cached_separately(self):
        """Test that different sequence lengths are cached separately."""
        from core.decorators import get_causal_mask

        mask4 = get_causal_mask(4)
        mask8 = get_causal_mask(8)

        assert mask4.shape == (4, 4)
        assert mask8.shape == (8, 8)


class TestProfileExecution:
    """Test execution profiling."""

    def test_profiling_disabled_by_default(self):
        """Test that profiling doesn't collect stats when disabled."""
        from core.decorators import (
            profile_execution, get_profile_stats, clear_profile_stats,
            enable_profiling
        )

        enable_profiling(False)
        clear_profile_stats()

        @profile_execution("test_fn")
        def fn():
            return 42

        fn()

        stats = get_profile_stats()
        assert "test_fn" not in stats

    def test_profiling_when_enabled(self):
        """Test that profiling collects stats when enabled."""
        from core.decorators import (
            profile_execution, get_profile_stats, clear_profile_stats,
            enable_profiling
        )

        enable_profiling(True)
        clear_profile_stats()

        @profile_execution("profiled_fn")
        def fn():
            time.sleep(0.01)
            return 42

        fn()
        fn()
        fn()

        stats = get_profile_stats()
        assert "profiled_fn" in stats
        assert stats["profiled_fn"]["calls"] == 3
        assert stats["profiled_fn"]["total_time"] > 0.03

        # Cleanup
        enable_profiling(False)

    def test_profile_stats_accuracy(self):
        """Test that profile stats are accurate."""
        from core.decorators import (
            profile_execution, get_profile_stats, clear_profile_stats,
            enable_profiling
        )

        enable_profiling(True)
        clear_profile_stats()

        @profile_execution("timed_fn")
        def fn():
            time.sleep(0.05)

        fn()

        stats = get_profile_stats()
        assert stats["timed_fn"]["min_time"] >= 0.04
        assert stats["timed_fn"]["max_time"] >= 0.04

        # Cleanup
        enable_profiling(False)


class TestValidateShapes:
    """Test shape validation decorator."""

    def test_valid_shapes_pass(self):
        """Test that valid shapes pass validation."""
        from core.decorators import validate_shapes

        @validate_shapes(x=(-1, 4), y=(-1, 4))
        def fn(x, y):
            return x + y

        x = np.zeros((2, 4))
        y = np.ones((2, 4))

        result = fn(x, y)
        np.testing.assert_array_equal(result, np.ones((2, 4)))

    def test_invalid_rank_raises(self):
        """Test that wrong rank raises ValueError."""
        from core.decorators import validate_shapes

        @validate_shapes(x=(-1, 4))
        def fn(x):
            return x

        x = np.zeros((2, 4, 8))  # 3D instead of 2D

        with pytest.raises(ValueError, match="rank"):
            fn(x)

    def test_invalid_dimension_raises(self):
        """Test that wrong dimension raises ValueError."""
        from core.decorators import validate_shapes

        @validate_shapes(x=(-1, 4))
        def fn(x):
            return x

        x = np.zeros((2, 8))  # 8 instead of 4

        with pytest.raises(ValueError, match="dim 1"):
            fn(x)


class TestLazyInit:
    """Test lazy initialization decorator."""

    def test_lazy_init_called_once(self):
        """Test that lazy_init only calls function once."""
        from core.decorators import lazy_init

        call_count = 0

        class MyClass:
            @property
            @lazy_init
            def expensive_property(self):
                nonlocal call_count
                call_count += 1
                return "computed"

        obj = MyClass()

        # First access
        assert obj.expensive_property == "computed"
        assert call_count == 1

        # Second access - should use cached value
        assert obj.expensive_property == "computed"
        assert call_count == 1

    def test_lazy_init_per_instance(self):
        """Test that each instance has its own cache."""
        from core.decorators import lazy_init

        class MyClass:
            def __init__(self, value):
                self._value = value

            @property
            @lazy_init
            def computed(self):
                return self._value * 2

        obj1 = MyClass(5)
        obj2 = MyClass(10)

        assert obj1.computed == 10
        assert obj2.computed == 20


class TestModuleIntegration:
    """Test integration with other modules."""

    def test_decorators_import(self):
        """Test that all decorators can be imported."""
        from core.decorators import (
            cached_computation,
            jit_if_available,
            jit_method,
            cached_mask,
            vectorize_batch,
            profile_execution,
            validate_shapes,
            lazy_init,
            checkpoint_gradients,
            get_causal_mask,
            get_causal_mask_jax,
            enable_profiling,
            get_profile_stats,
            clear_profile_stats,
            JAX_AVAILABLE,
        )

        # All imports should work
        assert callable(cached_computation)
        assert callable(jit_if_available)
        assert callable(profile_execution)
        assert isinstance(JAX_AVAILABLE, bool)

    def test_backends_utils_caching(self):
        """Test that backends/utils.py functions use caching."""
        from core.backends.utils import detect_available_hardware

        # Should have cache methods if properly decorated
        assert hasattr(detect_available_hardware, 'cache_info')
        assert hasattr(detect_available_hardware, 'clear_cache')

        # Call twice
        result1 = detect_available_hardware()
        result2 = detect_available_hardware()

        # Results should be identical
        assert result1 == result2
