"""
Safe import utilities for CapibaraGPT.

Eliminates repetitive try/except import blocks across the codebase.

Usage::

    from core.import_utils import safe_import, safe_import_module

    # Single class import
    MyClass = safe_import("some.module", "MyClass")
    # MyClass is None if import failed

    # Multiple classes from one module
    Foo, Bar = safe_import("some.module", "Foo", "Bar")

    # With availability flag
    mod, available = safe_import_module("some.module")
    # mod is the module object (or None), available is bool
"""

import importlib
import logging
from typing import Any, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def safe_import(module_path: str, *class_names: str) -> Any:
    """Import one or more names from *module_path*, returning ``None`` for each on failure.

    Returns a single value when one name is requested, or a tuple when
    multiple names are requested.

    Examples::

        Router = safe_import("core.router", "Router")
        Foo, Bar = safe_import("mod", "Foo", "Bar")
    """
    try:
        mod = importlib.import_module(module_path)
    except (ImportError, Exception) as exc:
        logger.debug("safe_import: %s unavailable: %s", module_path, exc)
        results = tuple(None for _ in class_names)
        return results[0] if len(results) == 1 else results

    results = tuple(getattr(mod, name, None) for name in class_names)
    return results[0] if len(results) == 1 else results


def safe_import_module(module_path: str) -> Tuple[Any, bool]:
    """Import a module and return ``(module, True)`` or ``(None, False)``."""
    try:
        mod = importlib.import_module(module_path)
        return mod, True
    except (ImportError, Exception) as exc:
        logger.debug("safe_import_module: %s unavailable: %s", module_path, exc)
        return None, False


def require_or_none(*module_paths: str) -> bool:
    """Return ``True`` only if **all** modules can be imported."""
    for path in module_paths:
        try:
            importlib.import_module(path)
        except (ImportError, Exception):
            return False
    return True


# ── Pre-built bundles for the most common try/except patterns ────


class _JAXBundle:
    """Lazy bundle for the ``try: import jax …`` pattern repeated 190+ times.

    Usage::

        from core.import_utils import jax_bundle
        jax, jnp, nn, optax, JAX_AVAILABLE = (
            jax_bundle.jax, jax_bundle.jnp, jax_bundle.nn,
            jax_bundle.optax, jax_bundle.available,
        )
    """

    def __init__(self) -> None:
        self._loaded = False
        self.available: bool = False
        self.jax: Any = None
        self.jnp: Any = None
        self.nn: Any = None
        self.optax: Any = None
        self.flax: Any = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import jax as _jax
            import jax.numpy as _jnp
        except ImportError:
            return
        self.jax = _jax
        self.jnp = _jnp
        self.available = True
        try:
            from flax import linen as _nn
            import flax as _flax
            self.nn = _nn
            self.flax = _flax
        except ImportError:
            pass
        try:
            import optax as _optax
            self.optax = _optax
        except ImportError:
            pass

    def __getattr__(self, name: str) -> Any:
        # Triggered only for attributes not yet set
        if name.startswith("_"):
            raise AttributeError(name)
        self._load()
        return self.__dict__.get(name)


jax_bundle = _JAXBundle()


__all__ = [
    "safe_import", "safe_import_module", "require_or_none", "jax_bundle",
]
