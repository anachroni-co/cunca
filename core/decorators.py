"""
Performance Optimization Decorators for CapibaraGPT

This module provides a collection of decorators to improve performance:
- @cached_computation: LRU cache with optional TTL
- @jit_if_available: JIT compile if JAX is available
- @cached_mask: Cache attention masks by shape
- @vectorize_batch: Auto-vectorize over batch dimension
- @profile_execution: Optional execution profiling
"""

import functools
import time
import threading
from typing import Callable, Optional, Dict, Any, Tuple, TypeVar, Union
from dataclasses import dataclass
import numpy as np

# Type variable for generic return types
F = TypeVar('F', bound=Callable[..., Any])

# Thread-safe cache storage
_mask_cache: Dict[Tuple, np.ndarray] = {}
_mask_cache_lock = threading.Lock()

# Try to import JAX (verify jax.jit exists to avoid project's own jax shim)
try:
    import jax
    import jax.numpy as jnp
    if not hasattr(jax, 'jit'):
        raise ImportError("jax module found but missing jit — likely project shim")
    JAX_AVAILABLE = True
except (ImportError, Exception):
    JAX_AVAILABLE = False
    jax = None
    jnp = None


# =============================================================================
# 1. CACHED COMPUTATION WITH TTL
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with timestamp for TTL support."""
    value: Any
    timestamp: float


def cached_computation(
    maxsize: int = 128,
    ttl_seconds: Optional[float] = None,
    key_fn: Optional[Callable[..., tuple]] = None
) -> Callable[[F], F]:
    """
    LRU cache with optional time-to-live (TTL).

    Args:
        maxsize: Maximum cache size
        ttl_seconds: Optional TTL in seconds (None = no expiry)
        key_fn: Optional function to generate cache key from args

    Example:
        @cached_computation(maxsize=64, ttl_seconds=300)
        def detect_hardware():
            # Expensive hardware detection
            ...
    """
    def decorator(func: F) -> F:
        cache: Dict[tuple, CacheEntry] = {}
        cache_lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn is not None:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default: use args and sorted kwargs
                try:
                    cache_key = (args, tuple(sorted(kwargs.items())))
                except TypeError:
                    # Unhashable args - don't cache
                    return func(*args, **kwargs)

            current_time = time.time()

            with cache_lock:
                # Check if cached and not expired
                if cache_key in cache:
                    entry = cache[cache_key]
                    if ttl_seconds is None or (current_time - entry.timestamp) < ttl_seconds:
                        return entry.value
                    else:
                        # Expired - remove
                        del cache[cache_key]

                # Compute value
                result = func(*args, **kwargs)

                # Evict oldest if at capacity
                if len(cache) >= maxsize:
                    oldest_key = min(cache.keys(), key=lambda k: cache[k].timestamp)
                    del cache[oldest_key]

                # Store in cache
                cache[cache_key] = CacheEntry(value=result, timestamp=current_time)

            return result

        # Add cache management methods
        def clear_cache():
            with cache_lock:
                cache.clear()

        def cache_info():
            with cache_lock:
                return {"size": len(cache), "maxsize": maxsize, "ttl": ttl_seconds}

        wrapper.clear_cache = clear_cache
        wrapper.cache_info = cache_info

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# 2. JIT COMPILATION (JAX/Numba)
# =============================================================================

def jit_if_available(
    static_argnums: Optional[Tuple[int, ...]] = None,
    donate_argnums: Optional[Tuple[int, ...]] = None,
    backend: str = "auto"
) -> Callable[[F], F]:
    """
    JIT compile function if JAX is available, otherwise pass through.

    Args:
        static_argnums: Argument indices to treat as static (for JAX)
        donate_argnums: Argument indices to donate buffers (for JAX)
        backend: "jax", "numba", or "auto"

    Example:
        @jit_if_available(static_argnums=(1,))
        def compute_attention(x, num_heads):
            ...
    """
    def decorator(func: F) -> F:
        if backend in ("jax", "auto") and JAX_AVAILABLE:
            jit_kwargs = {}
            if static_argnums is not None:
                jit_kwargs["static_argnums"] = static_argnums
            if donate_argnums is not None:
                jit_kwargs["donate_argnums"] = donate_argnums
            return jax.jit(func, **jit_kwargs)  # type: ignore

        # Fallback: no JIT
        return func

    return decorator


def jit_method(
    static_argnums: Optional[Tuple[int, ...]] = None
) -> Callable[[F], F]:
    """
    JIT compile a method (handles 'self' automatically).

    The 'self' argument is automatically treated as static.

    Example:
        class MyModule:
            @jit_method()
            def forward(self, x):
                ...
    """
    def decorator(func: F) -> F:
        if not JAX_AVAILABLE:
            return func

        # Adjust static_argnums to account for 'self' at position 0
        adjusted_static = (0,) if static_argnums is None else (0,) + tuple(i + 1 for i in static_argnums)

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create JIT-compiled version on first call (per instance)
            cache_attr = f"_jit_cache_{func.__name__}"
            if not hasattr(self, cache_attr):
                # Create a closure that captures self
                def jit_fn(*args, **kwargs):
                    return func(self, *args, **kwargs)
                setattr(self, cache_attr, jax.jit(jit_fn))

            return getattr(self, cache_attr)(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# 3. CACHED ATTENTION MASKS
# =============================================================================

def cached_mask(mask_type: str = "causal") -> Callable[[F], F]:
    """
    Cache attention masks by sequence length to avoid recomputation.

    Args:
        mask_type: Type of mask ("causal", "padding", "custom")

    Example:
        @cached_mask("causal")
        def create_causal_mask(seq_len: int, dtype=np.float32):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(seq_len: int, *args, **kwargs):
            # Create cache key from mask type and sequence length
            dtype = kwargs.get('dtype', np.float32)
            cache_key = (mask_type, seq_len, str(dtype))

            with _mask_cache_lock:
                if cache_key in _mask_cache:
                    return _mask_cache[cache_key]

                # Compute mask
                result = func(seq_len, *args, **kwargs)

                # Cache (limit cache size to prevent memory issues)
                if len(_mask_cache) < 256:
                    _mask_cache[cache_key] = result

            return result

        def clear_mask_cache():
            with _mask_cache_lock:
                _mask_cache.clear()

        wrapper.clear_cache = clear_mask_cache

        return wrapper  # type: ignore

    return decorator


# Pre-built cached mask generators
@cached_mask("causal")
def get_causal_mask(seq_len: int, dtype=np.float32) -> np.ndarray:
    """Get or create a causal attention mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype=dtype))


@cached_mask("causal_jax")
def get_causal_mask_jax(seq_len: int, dtype=None):
    """Get or create a causal attention mask (JAX version)."""
    if not JAX_AVAILABLE:
        return get_causal_mask(seq_len, dtype or np.float32)
    dtype = dtype or jnp.float32
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))


# =============================================================================
# 4. VECTORIZATION HELPERS
# =============================================================================

def vectorize_batch(in_axes: Union[int, Tuple[int, ...]] = 0) -> Callable[[F], F]:
    """
    Automatically vectorize a function over the batch dimension using JAX vmap.

    Args:
        in_axes: Which axes to vectorize over (default: 0 for batch)

    Example:
        @vectorize_batch(in_axes=(0, 0, None))
        def process_single(x, y, config):
            # Process single example
            ...
    """
    def decorator(func: F) -> F:
        if not JAX_AVAILABLE:
            return func

        return jax.vmap(func, in_axes=in_axes)  # type: ignore

    return decorator


# =============================================================================
# 5. EXECUTION PROFILING
# =============================================================================

# Global profiling state
_profiling_enabled = False
_profile_stats: Dict[str, Dict[str, float]] = {}
_profile_lock = threading.Lock()


def enable_profiling(enabled: bool = True):
    """Enable or disable profiling globally."""
    global _profiling_enabled
    _profiling_enabled = enabled


def get_profile_stats() -> Dict[str, Dict[str, float]]:
    """Get profiling statistics."""
    with _profile_lock:
        return dict(_profile_stats)


def clear_profile_stats():
    """Clear profiling statistics."""
    global _profile_stats
    with _profile_lock:
        _profile_stats.clear()


def profile_execution(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Profile function execution time (when profiling is enabled).

    Args:
        name: Optional name for the profiled function

    Example:
        @profile_execution("attention_forward")
        def attention(q, k, v):
            ...
    """
    def decorator(func: F) -> F:
        profile_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _profiling_enabled:
                return func(*args, **kwargs)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start

                with _profile_lock:
                    if profile_name not in _profile_stats:
                        _profile_stats[profile_name] = {
                            "calls": 0,
                            "total_time": 0.0,
                            "min_time": float('inf'),
                            "max_time": 0.0
                        }

                    stats = _profile_stats[profile_name]
                    stats["calls"] += 1
                    stats["total_time"] += elapsed
                    stats["min_time"] = min(stats["min_time"], elapsed)
                    stats["max_time"] = max(stats["max_time"], elapsed)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# 6. SHAPE VALIDATION (Debug Helper)
# =============================================================================

def validate_shapes(**expected_shapes) -> Callable[[F], F]:
    """
    Validate input tensor shapes (useful for debugging).

    Args:
        **expected_shapes: Mapping of argument names to expected shape tuples
                          Use -1 for any dimension, None for variable rank

    Example:
        @validate_shapes(x=(-1, 768), mask=(-1, -1))
        def forward(x, mask):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for arg_name, expected in expected_shapes.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if hasattr(value, 'shape'):
                        actual = value.shape
                        if expected is not None:
                            if len(actual) != len(expected):
                                raise ValueError(
                                    f"{func.__name__}: {arg_name} has rank {len(actual)}, "
                                    f"expected {len(expected)}"
                                )
                            for i, (a, e) in enumerate(zip(actual, expected)):
                                if e != -1 and a != e:
                                    raise ValueError(
                                        f"{func.__name__}: {arg_name} dim {i} is {a}, "
                                        f"expected {e}"
                                    )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# 7. LAZY INITIALIZATION
# =============================================================================

def lazy_init(func: F) -> F:
    """
    Lazily initialize a value on first access and cache it.

    Useful for expensive module initialization.

    Example:
        class Model:
            @property
            @lazy_init
            def heavy_component(self):
                return create_expensive_thing()
    """
    attr_name = f"_lazy_{func.__name__}"

    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return wrapper  # type: ignore


# =============================================================================
# 8. GRADIENT CHECKPOINTING
# =============================================================================

def checkpoint_gradients(func: F) -> F:
    """
    Apply gradient checkpointing to reduce memory usage during backprop.

    Only works with JAX - passes through otherwise.

    Example:
        @checkpoint_gradients
        def transformer_block(x):
            # Memory-intensive computation
            ...
    """
    if not JAX_AVAILABLE:
        return func

    return jax.checkpoint(func)  # type: ignore


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core decorators
    "cached_computation",
    "jit_if_available",
    "jit_method",
    "cached_mask",
    "vectorize_batch",
    "profile_execution",
    "validate_shapes",
    "lazy_init",
    "checkpoint_gradients",
    # Profiling utilities
    "enable_profiling",
    "get_profile_stats",
    "clear_profile_stats",
    # Pre-built mask generators
    "get_causal_mask",
    "get_causal_mask_jax",
    # Constants
    "JAX_AVAILABLE",
]
