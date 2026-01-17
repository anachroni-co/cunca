"""
Vector Quantization (VQ) Module for CapibaraGPT.

This package contains vector quantization implementations:
- VQ ARM Axion optimizer for ARM processors
- Efficient codebook operations
- Memory-optimized attention
"""

from .vq_arm_axion import (
    ARMAxionConfig,
    ARMAxionOptimizer,
    create_arm_axion_optimizer,
    get_arm_axion_optimizer,
)

__all__ = [
    "ARMAxionConfig",
    "ARMAxionOptimizer",
    "create_arm_axion_optimizer",
    "get_arm_axion_optimizer",
]
