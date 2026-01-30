"""
layers neurogenesis module.

# This module provides functionality for neurogenesis.
"""

import os
import sys

import logging
import hashlib

# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

from capibara.jax import jax
from flax import linen as nn
from functools import partial
from capibara.jax import numpy as jnp
from typing import Optional, Tuple, Dict, Any, Union

# Import correct de configuraciones
try:
    from capibara.interfaces.imodules import Imodule
except ImportError:
    # Fallback stubs if not existen
    class IModule:
        pass
    class LayerConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class BaseLayer(nn.Module):
        config: LayerConfig

from capibara.core.config import CapibaraConfig

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module neurogenesis.py starting")
    return True

if __name__ == "__main__":
    main()
