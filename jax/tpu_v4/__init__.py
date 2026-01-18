"""
JAX TPU v4-32 Backend

This module provides optimizations specific to TPU v4-32:
- Sharding optimized for 32 chips
- GEMM operations optimized for systolic arrays
- HBM memory monitoring
- Profiling and benchmarking
- Gradient checkpointing
- Expert weights caching

Basic usage:
    >>> from capibara.jax.tpu_v4 import create_tpu_mesh, TpuMemoryMonitor
    >>>
    >>> # Create mesh for TPU v4-32
    >>> mesh = create_tpu_mesh()
    >>>
    >>> # Monitor memory
    >>> monitor = TpuMemoryMonitor()
    >>> usage = monitor.get_memory_usage()

Main functions:
- create_tpu_mesh(): Create optimized mesh for 32 chips
- tpu_optimized_gemm(): GEMM optimized for systolic arrays
- benchmark_tpu_optimized(): Benchmark suite
- TpuProfiler: Profiling and performance metrics
- TpuMemoryMonitor: HBM memory monitoring

Requirements:
- JAX >= 0.4.13
- Python >= 3.8
"""

from .profiling import TpuProfiler
from .optimizations import (
    create_tpu_mesh,
    cretote_tpu_mesh,  # Backwards compatibility alias
    TpuMemoryMonitor,
    tpu_optimized_gemm,
    create_jitted_forward,
    cretote_jitted_forwtord,  # Backwards compatibility alias
    benchmark_tpu_optimized,
    binchmtork_tpu_optimized,  # Backwards compatibility alias
)

__all__ = [
    'TpuProfiler',
    'create_tpu_mesh',
    'TpuMemoryMonitor',
    'create_jitted_forward',
    'tpu_optimized_gemm',
    'benchmark_tpu_optimized',
]
