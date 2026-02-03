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

    # Lazy imports (deferred until first access)
    from core.import_utils import LazyModule, lazy_import
    torch = LazyModule("torch")  # Not imported yet
    torch.tensor([1,2,3])  # Now it imports
"""

import importlib
import logging
import sys
import types
from typing import Any, Optional, Sequence, Tuple, Dict, List

logger = logging.getLogger(__name__)


class LazyModule(types.ModuleType):
    """A module that delays its import until first attribute access.

    This significantly reduces startup time for modules with heavy dependencies
    like torch, transformers, tensorflow, etc.

    Usage::

        # Instead of: import torch
        torch = LazyModule("torch")

        # The actual import happens only when you use it:
        x = torch.tensor([1, 2, 3])  # Import happens here

    You can also specify submodules::

        nn = LazyModule("torch.nn")
        functional = LazyModule("torch.nn.functional")
    """

    def __init__(self, module_name: str, package: str = None):
        super().__init__(module_name)
        self._module_name = module_name
        self._package = package
        self._module: Optional[types.ModuleType] = None
        self._loading = False
        self._failed = False
        self._error: Optional[Exception] = None

    def _load(self) -> types.ModuleType:
        """Load the actual module on first access."""
        if self._module is not None:
            return self._module

        if self._failed:
            raise ImportError(
                f"Previously failed to import '{self._module_name}': {self._error}"
            )

        if self._loading:
            # Prevent infinite recursion during import
            raise ImportError(f"Circular import detected for '{self._module_name}'")

        self._loading = True
        try:
            self._module = importlib.import_module(self._module_name, self._package)
            logger.debug(f"Lazy loaded module: {self._module_name}")
            return self._module
        except ImportError as e:
            self._failed = True
            self._error = e
            logger.debug(f"Failed to lazy load '{self._module_name}': {e}")
            raise
        finally:
            self._loading = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{self._module_name}' has no attribute '{name}'")
        module = self._load()
        return getattr(module, name)

    def __dir__(self) -> List[str]:
        try:
            module = self._load()
            return dir(module)
        except ImportError:
            return []

    def __repr__(self) -> str:
        if self._module is not None:
            return repr(self._module)
        return f"<LazyModule '{self._module_name}' (not loaded)>"

    @property
    def __loaded__(self) -> bool:
        """Check if the module has been loaded without triggering a load."""
        return self._module is not None


def lazy_import(module_name: str, package: str = None) -> LazyModule:
    """Create a lazy module that imports on first access.

    Args:
        module_name: Full module path (e.g., "torch.nn")
        package: Optional package for relative imports

    Returns:
        LazyModule instance that imports on first attribute access
    """
    return LazyModule(module_name, package)


def lazy_import_from(module_name: str, *names: str) -> Tuple[Any, ...]:
    """Create lazy attribute accessors for specific names from a module.

    Usage::

        tensor, nn = lazy_import_from("torch", "tensor", "nn")
        # Neither torch nor these attributes are imported yet

        x = tensor([1, 2, 3])  # Now torch is imported

    Returns:
        Tuple of LazyAttribute objects
    """
    class LazyAttribute:
        def __init__(self, module_name: str, attr_name: str):
            self._module_name = module_name
            self._attr_name = attr_name
            self._value: Any = None
            self._loaded = False

        def _load(self) -> Any:
            if not self._loaded:
                module = importlib.import_module(self._module_name)
                self._value = getattr(module, self._attr_name)
                self._loaded = True
            return self._value

        def __call__(self, *args, **kwargs):
            return self._load()(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._load(), name)

        def __repr__(self) -> str:
            if self._loaded:
                return repr(self._value)
            return f"<LazyAttribute '{self._module_name}.{self._attr_name}' (not loaded)>"

    return tuple(LazyAttribute(module_name, name) for name in names)


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


class _TransformersBundle:
    """Lazy bundle for transformers library.

    Usage::

        from core.import_utils import transformers_bundle
        AutoModel = transformers_bundle.AutoModel
        AutoTokenizer = transformers_bundle.AutoTokenizer
    """

    def __init__(self) -> None:
        self._loaded = False
        self._module: Any = None
        self.available: bool = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import transformers as _transformers
            self._module = _transformers
            self.available = True
        except ImportError:
            logger.debug("transformers not available")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self._load()
        if self._module is None:
            return None
        return getattr(self._module, name, None)


class _TorchBundle:
    """Lazy bundle for PyTorch.

    Usage::

        from core.import_utils import torch_bundle
        torch = torch_bundle.torch
        nn = torch_bundle.nn
    """

    def __init__(self) -> None:
        self._loaded = False
        self.available: bool = False
        self.torch: Any = None
        self.nn: Any = None
        self.optim: Any = None
        self.cuda_available: bool = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import torch as _torch
            self.torch = _torch
            self.nn = _torch.nn
            self.optim = _torch.optim
            self.available = True
            self.cuda_available = _torch.cuda.is_available()
        except ImportError:
            logger.debug("torch not available")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self._load()
        return self.__dict__.get(name)


# Pre-initialized lazy bundles for common heavy imports
transformers_bundle = _TransformersBundle()
torch_bundle = _TorchBundle()


__all__ = [
    # Safe imports (eager but exception-safe)
    "safe_import",
    "safe_import_module",
    "require_or_none",
    # Lazy imports (deferred until first access)
    "LazyModule",
    "lazy_import",
    "lazy_import_from",
    # Pre-built lazy bundles
    "jax_bundle",
    "transformers_bundle",
    "torch_bundle",
]
