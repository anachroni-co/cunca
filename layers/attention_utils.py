"""
Shared multi-head attention utilities for the layers module.

Provides reusable ``split_heads`` / ``merge_heads`` helpers so every
attention implementation doesn't have to re-implement the reshape dance.
Works with both NumPy and JAX arrays.
"""

import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False


def split_heads(x, num_heads: int):
    """Reshape (..., seq, hidden) → (..., heads, seq, head_dim).

    Works with both NumPy and JAX arrays.
    """
    *batch_dims, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    x = x.reshape(*batch_dims, seq_len, num_heads, head_dim)
    # Swap seq and heads axes: (..., seq, heads, hd) → (..., heads, seq, hd)
    lib = jnp if (JAX_AVAILABLE and hasattr(x, "at")) else np
    return lib.swapaxes(x, -3, -2)


def merge_heads(x, hidden_size: int):
    """Reshape (..., heads, seq, head_dim) → (..., seq, hidden).

    Inverse of ``split_heads``.
    """
    lib = jnp if (JAX_AVAILABLE and hasattr(x, "at")) else np
    # (..., heads, seq, hd) → (..., seq, heads, hd)
    x = lib.swapaxes(x, -3, -2)
    *batch_dims, seq_len, _heads, _hd = x.shape
    return x.reshape(*batch_dims, seq_len, hidden_size)


__all__ = ["split_heads", "merge_heads"]
