"""
Backend Registry

Manages registration and selection of compute backends.
Provides automatic detection of best available backend.
"""

import logging
from typing import Dict, Optional, Type

from .base import BackendConfig, BackendType, ComputeBackend

logger = logging.getLogger(__name__)

# Global registry of backends
_BACKEND_REGISTRY: Dict[BackendType, Type[ComputeBackend]] = {}
_DEFAULT_BACKEND: Optional[BackendType] = None
_BACKEND_INSTANCES: Dict[BackendType, ComputeBackend] = {}


def register_backend(
    backend_type: BackendType,
    backend_class: Type[ComputeBackend],
) -> None:
    """Register a backend class."""
    _BACKEND_REGISTRY[backend_type] = backend_class
    logger.debug(f"Registered backend: {backend_type.name}")


def list_available_backends() -> Dict[str, bool]:
    """List all backends and their availability."""
    result = {}

    for backend_type, backend_class in _BACKEND_REGISTRY.items():
        try:
            instance = backend_class()
            result[backend_type.name.lower()] = instance.is_available
        except Exception:
            result[backend_type.name.lower()] = False

    return result


def detect_best_backend() -> BackendType:
    """Auto-detect the best available backend."""
    # Priority: TPU > GPU > CPU
    priority = [BackendType.TPU, BackendType.GPU, BackendType.CPU]

    for backend_type in priority:
        if backend_type in _BACKEND_REGISTRY:
            try:
                backend_class = _BACKEND_REGISTRY[backend_type]
                instance = backend_class()
                if instance.is_available:
                    logger.info(f"Auto-detected backend: {backend_type.name}")
                    return backend_type
            except Exception as e:
                logger.debug(f"Backend {backend_type.name} not available: {e}")

    # Fallback to CPU
    return BackendType.CPU


def get_backend(
    backend_type: Optional[BackendType] = None,
    config: Optional[BackendConfig] = None,
    initialize: bool = True,
) -> ComputeBackend:
    """
    Get a compute backend instance.

    Args:
        backend_type: Type of backend (TPU, GPU, CPU, or AUTO)
        config: Backend configuration
        initialize: Whether to initialize the backend

    Returns:
        Initialized ComputeBackend instance

    Example:
        >>> backend = get_backend(BackendType.GPU)
        >>> tensor = backend.randn((32, 512, 768))
    """
    # Auto-detect if not specified
    if backend_type is None or backend_type == BackendType.AUTO:
        backend_type = detect_best_backend()

    # Check if we already have an instance
    if backend_type in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[backend_type]

    # Get backend class
    if backend_type not in _BACKEND_REGISTRY:
        raise ValueError(f"Backend {backend_type.name} not registered")

    backend_class = _BACKEND_REGISTRY[backend_type]

    # Create config if not provided
    if config is None:
        config = BackendConfig(backend_type=backend_type)

    # Create instance
    backend = backend_class(config)

    # Initialize if requested
    if initialize:
        try:
            backend.initialize()
            logger.info(f"Initialized {backend_type.name} backend")
        except Exception as e:
            logger.error(f"Failed to initialize {backend_type.name} backend: {e}")
            # Fallback to CPU
            if backend_type != BackendType.CPU:
                logger.warning("Falling back to CPU backend")
                return get_backend(BackendType.CPU, config, initialize)
            raise

    # Cache instance
    _BACKEND_INSTANCES[backend_type] = backend

    return backend


def get_default_backend() -> BackendType:
    """Get the default backend type."""
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = detect_best_backend()
    return _DEFAULT_BACKEND


def set_default_backend(backend_type: BackendType) -> None:
    """Set the default backend type."""
    global _DEFAULT_BACKEND
    if backend_type not in _BACKEND_REGISTRY:
        raise ValueError(f"Backend {backend_type.name} not registered")
    _DEFAULT_BACKEND = backend_type
    logger.info(f"Default backend set to: {backend_type.name}")


def clear_backend_cache() -> None:
    """Clear cached backend instances."""
    global _BACKEND_INSTANCES
    for backend in _BACKEND_INSTANCES.values():
        try:
            backend.shutdown()
        except Exception:
            pass
    _BACKEND_INSTANCES.clear()


# ==================== Auto-register backends ====================

def _auto_register_backends():
    """Automatically register available backends."""
    # Register CPU backend (always available)
    try:
        from .cpu_backend import CPUBackend
        register_backend(BackendType.CPU, CPUBackend)
    except ImportError:
        pass

    # Register GPU backend
    try:
        from .gpu_backend import GPUBackend
        register_backend(BackendType.GPU, GPUBackend)
    except ImportError:
        logger.debug("GPU backend not available (PyTorch not installed)")

    # Register TPU backend
    try:
        from .tpu_backend import TPUBackend
        register_backend(BackendType.TPU, TPUBackend)
    except ImportError:
        logger.debug("TPU backend not available (JAX not installed)")


# Auto-register on import
_auto_register_backends()
