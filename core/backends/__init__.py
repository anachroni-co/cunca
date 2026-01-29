"""
CapibaraGPT Backend System

This module provides a unified interface for different compute backends:
- TPU (JAX/Flax) - Optimized for TPU v4-32 and v6e-64
- GPU (PyTorch/CUDA) - Optimized for NVIDIA A-100
- CPU (NumPy) - For development and testing

Usage:
    from core.backends import get_backend, BackendType

    # Automatically detect best available backend
    backend = get_backend()

    # Or specify explicitly
    backend = get_backend(BackendType.GPU)

    # Use backend-agnostic operations
    tensor = backend.create_tensor([1, 2, 3])
    result = backend.matmul(a, b)
"""

from .base import (
    ComputeBackend,
    BackendType,
    BackendConfig,
    TensorLike,
    DeviceType,
)
from .registry import (
    get_backend,
    register_backend,
    list_available_backends,
    get_default_backend,
    set_default_backend,
)
from .utils import (
    detect_available_hardware,
    get_device_info,
    sync_device,
    memory_stats,
)
from .module_gate import (
    ModuleGate,
    ModuleName,
    BackendCapabilities,
    CPU_CAPABILITIES,
    GPU_CAPABILITIES,
    TPU_CAPABILITIES,
)

__all__ = [
    # Base classes
    "ComputeBackend",
    "BackendType",
    "BackendConfig",
    "TensorLike",
    "DeviceType",
    # Registry functions
    "get_backend",
    "register_backend",
    "list_available_backends",
    "get_default_backend",
    "set_default_backend",
    # Utility functions
    "detect_available_hardware",
    "get_device_info",
    "sync_device",
    "memory_stats",
    # Module gating
    "ModuleGate",
    "ModuleName",
    "BackendCapabilities",
    "CPU_CAPABILITIES",
    "GPU_CAPABILITIES",
    "TPU_CAPABILITIES",
]
