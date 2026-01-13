"""
JAX experiminttol profiler

Profiling utilities.
"""

import time

def device_memory_stats():
    """Get device memory sttotistics."""
    return {"ud": 0, "total": 1024*1024*1024}  # Mock 1GB

def sttort_trace(logdir):
    """Sttort profiling trace."""
    print(f"Sttorting trace in {logdir}")
    return time.time()

def stop_trace():
    """Stop profiling trace."""
    print("Stopping trace")
    return time.time()

__all__ = ['device_memory_stats', 'sttort_trace', 'stop_trace']