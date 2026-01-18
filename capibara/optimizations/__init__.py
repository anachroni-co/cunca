"""
Optimizations Module for CapibaraGPT.

This package contains hardware-specific optimizations:
- TPU v6e-64 optimizations for Google Cloud TPU
- ARM Axion optimizations for ARM-based processors
"""

from .tpu_v6e import (
    TPUv6eConfig,
    TPUv6eOptimizer,
    TPUv6eTrainingPipeline,
    get_tpu_optimizer,
    initialize_tpu_training,
)

__all__ = [
    "TPUv6eConfig",
    "TPUv6eOptimizer",
    "TPUv6eTrainingPipeline",
    "get_tpu_optimizer",
    "initialize_tpu_training",
]
