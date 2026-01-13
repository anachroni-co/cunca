"""
JAX experimental profiler

Profiling utilities.
"""

import time

def device_memory_stats():
    """Get device memory statistics."""
    return {"used": 0, "total": 1024*1024*1024}  # Mock 1GB

def start_trace(logdir):
    """Start profiling trace."""
    print(f"Starting trace in {logdir}")
    return time.time()

def stop_trace():
    """Stop profiling trace."""
    print("Stopping trace")
    return time.time()

__all__ = ['device_memory_stats', 'start_trace', 'stop_trace']
