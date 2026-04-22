"""
Contextual activation functions for CapibaraGPT core.

This module provides small, CPU-friendly activation helpers that optionally
adapt their response based on an external context vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jax import numpy as jnp

logger = logging.getLogger(__name__)


def _relu(x):
    return jnp.maximum(x, 0)


def _silu(x):
    return x / (1.0 + jnp.exp(-x))


def _gelu(x):
    # Approximate GELU (Hendrycks & Gimpel).
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * (x ** 3))))


_ACTIVATIONS: dict[str, Callable[[Any], Any]] = {
    "relu": _relu,
    "silu": _silu,
    "swish": _silu,
    "gelu": _gelu,
    "tanh": jnp.tanh,
}


def _context_scale(context: Optional[Any]) -> float:
    if context is None:
        return 1.0
    try:
        context_arr = jnp.asarray(context)
        # Use a bounded scale derived from mean context magnitude.
        return float(jnp.tanh(jnp.mean(jnp.abs(context_arr))) + 1.0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Context scaling failed: %s", exc)
        return 1.0


def apply(
    x: Any,
    *,
    context: Optional[Any] = None,
    activation: str = "gelu",
    context_gain: float = 0.1,
) -> Any:
    """Apply a contextual activation to ``x``.

    Args:
        x: Input tensor/array.
        context: Optional context tensor used to modulate activation strength.
        activation: Activation name ("relu", "gelu", "silu", "tanh").
        context_gain: How strongly to modulate the activation with context.
    """
    fn = _ACTIVATIONS.get(activation, _gelu)
    try:
        x_arr = jnp.asarray(x)
    except Exception:
        x_arr = x

    scale = _context_scale(context)
    if context_gain:
        x_arr = x_arr * (1.0 + context_gain * (scale - 1.0))
    return fn(x_arr)


@dataclass
class ContextualActivation:
    activation: str = "gelu"
    context_gain: float = 0.1

    def __call__(self, x: Any, *, context: Optional[Any] = None) -> Any:
        return apply(x, context=context, activation=self.activation, context_gain=self.context_gain)


class ContextualReLU(ContextualActivation):
    def __init__(self, context_gain: float = 0.1):
        super().__init__(activation="relu", context_gain=context_gain)


class ContextualGELU(ContextualActivation):
    def __init__(self, context_gain: float = 0.1):
        super().__init__(activation="gelu", context_gain=context_gain)


class ContextualSiLU(ContextualActivation):
    def __init__(self, context_gain: float = 0.1):
        super().__init__(activation="silu", context_gain=context_gain)


def main() -> bool:
    """Lightweight self-check used by docs/tests."""
    sample = jnp.asarray([[1.0, -0.5, 0.25]])
    _ = apply(sample)
    logger.info("Contextual activation module initialized")
    return True


__all__ = [
    "apply",
    "ContextualActivation",
    "ContextualReLU",
    "ContextualGELU",
    "ContextualSiLU",
    "main",
]
