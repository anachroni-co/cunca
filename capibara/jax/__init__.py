"""
CapibaraGPT v3 JAX package shim.

This module aliases the top-level `jax` package so that both
`jax.*` and `capibara.jax.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_jax = importlib.import_module("jax")

# Expose as capibara.jax
sys.modules[__name__] = _jax
