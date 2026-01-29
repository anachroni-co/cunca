"""
Centralized JAX/Flax import guard and fallback factory for the layers module.

Instead of repeating try/except import blocks and fallback classes in every
layer file, import the pre-resolved symbols from here:

    from layers.jax_compat import jax, jnp, nn, JAX_AVAILABLE

When JAX/Flax is not installed, ``nn`` is a namespace whose attributes
(``Module``, ``Dense``, ``Conv``, etc.) raise ``ImportError`` on
instantiation, so class definitions that inherit from ``nn.Module`` still
parse but fail early at construction time with a helpful message.
"""

import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Attempt real JAX / Flax import
# ------------------------------------------------------------------
try:
    import jax as _jax                 # noqa: F401
    import jax.numpy as _jnp           # noqa: F401
    from flax import linen as _nn      # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    _jax = None
    _jnp = None
    _nn = None
    JAX_AVAILABLE = False

# ------------------------------------------------------------------
# Fallback module factory (used when JAX/Flax missing)
# ------------------------------------------------------------------

def _make_fallback(name: str):
    """Return a class that raises ImportError on instantiation."""
    class _Fallback:
        __qualname__ = name
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"JAX and Flax are required for {name}. "
                "Install with: pip install jax flax"
            )
        def __init_subclass__(cls, **kw):
            pass  # allow class body to be parsed
    _Fallback.__name__ = name
    return _Fallback


class _FallbackInitializers:
    """Stub for ``nn.initializers``."""
    @staticmethod
    def xavier_uniform():
        return None
    @staticmethod
    def zeros():
        return None
    @staticmethod
    def ones():
        return None
    @staticmethod
    def lecun_normal():
        return None


class _FallbackNN:
    """Namespace that mimics ``flax.linen`` for class definitions only."""
    Module = _make_fallback("nn.Module")
    Dense = _make_fallback("nn.Dense")
    Conv = _make_fallback("nn.Conv")
    LayerNorm = _make_fallback("nn.LayerNorm")
    Dropout = _make_fallback("nn.Dropout")
    Sequential = _make_fallback("nn.Sequential")
    Embed = _make_fallback("nn.Embed")
    initializers = _FallbackInitializers

    @staticmethod
    def compact(fn):
        return fn

    @staticmethod
    def gelu(x):
        raise ImportError("JAX/Flax required for nn.gelu")

    @staticmethod
    def softmax(x, axis=-1):
        raise ImportError("JAX/Flax required for nn.softmax")

    @staticmethod
    def relu(x):
        raise ImportError("JAX/Flax required for nn.relu")


class _FallbackJnpLinalg:
    @staticmethod
    def norm(*a, **kw):
        raise ImportError("JAX required for jnp.linalg.norm")


class _FallbackJnp:
    """Minimal stub so that ``jnp.xxx`` references parse at import time.

    Attribute access returns a sentinel string for dtype constants (float32,
    bfloat16, etc.) so that class-level annotations like
    ``dtype: Any = jnp.float32`` work without JAX installed.  Actual
    computation methods raise ``ImportError``.
    """
    linalg = _FallbackJnpLinalg

    # dtype / type constants used at class-definition time
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    int32 = "int32"
    int64 = "int64"
    bool_ = "bool_"
    ndarray = "ndarray"  # used as type annotation

    def __getattr__(self, name):
        # For anything else, return a placeholder string so class bodies
        # can be parsed.  Actual computation will fail at call time.
        return f"<jnp.{name} unavailable>"


# ------------------------------------------------------------------
# Public symbols — every layer file imports these
# ------------------------------------------------------------------
jax = _jax
jnp = _jnp if _jnp is not None else _FallbackJnp()
nn = _nn if _nn is not None else _FallbackNN()

__all__ = ["jax", "jnp", "nn", "JAX_AVAILABLE"]
