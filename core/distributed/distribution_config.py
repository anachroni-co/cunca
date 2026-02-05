import logging
import sys
import os
"""
distributed configuration module.
"""

from typing import Dict, Any

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from capibara.jax import jax # type: ignore
from capibara.jax import numpy as jnp # type: ignore
from capibara.jax.experimental import PartitionSpec as P # type: ignore
from capibara.jax.experimental.maps import Mesh # type: ignore
from capibara.jax.experimental.maps import mesh # type: ignore
from capibara.jax.experimental.maps import shard_map # type: ignore
from functools import wraps
from typing import Optional, Tuple, Dict, Any, Callable, List, Union, TypeVar, cast
import numpy as np # type: ignore
from capibara.jax.experimental.pjit import pjit # type: ignore

from capibara.jax.tpu_v4.backend import (
    TpuV4LinalgOps,
    TpuV4SparseOps,
    TpuV4NeuralOps,
    TpuV4RandomOps,
    TpuV4PerformanceUtils
)

logger = logging.getLogger(__name__)

# Generic types for functions
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')

# TPU v4-32 mesh setup
TPU_MESH_SHAPE = (32, 8)  # 32 chips, 8 cores per chip
TPU_MESH = None  # Will be initialized in setup_mesh()

# Sharding specifications optimized for TPU v4-32
BATCH_SHARDING = P('batch', None, None)  # (batch, seq_len, hidden)
MODEL_SHARDING = P(None, None, 'model')  # (batch, seq_len, hidden)
HYBRID_SHARDING = P('batch', None, 'model')  # (batch, seq_len, hidden)
REPLICATED = P(None, None, None)

# Data types optimized for TPU v4-32
DTYPE = jnp.float32
TPU_DTYPE = jnp.bfloat16  # better performance on TPU v4-32

def setup_mesh(shape: Tuple[int, ...] = TPU_MESH_SHAPE) -> Mesh:
    """Initializes the global TPU v4-32 mesh."""
    pass
    def __init__(self):
        self.settings = {}
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.settings[key] = value

# Global config instance
config = Config()
