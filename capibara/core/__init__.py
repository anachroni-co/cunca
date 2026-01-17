"""
Core Module for CapibaraGPT v3.

This package contains the core components of CapibaraGPT:

Submodules:
    - ssm: State Space Models (SSM, SpikeSSM, AdaptiveSpikeSSM)
    - routers: Routing mechanisms (SelfModifyingRouter with Nested Learning)
    - optimizations: Hardware optimizations (TPU v6e-64)
    - vq: Vector Quantization (ARM Axion optimizations)
"""

from . import ssm
from . import routers
from . import optimizations
from . import vq

__all__ = [
    "ssm",
    "routers",
    "optimizations",
    "vq",
]
