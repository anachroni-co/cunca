"""
layers self_attention module.

# This module provides functionality for self_attention.
"""

import os
import sys

import logging
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

import time
import logging
import hashlib
from capibara.jax import jax
from flax import linen as nn
from functools import partial
from capibara.jax import numpy as jnp
from capibara.core.kernels import tpu_kernel
from capibara.interfaces.ilayer import ILayer
from typing import Optional, Dict, Any, Tuple
from capibara.layers.base import BaseLayer, LayerConfig

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module self_attention.py starting")
    return True

if __name__ == "__main__":
    main()
