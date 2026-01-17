"""Aleph-TILDE implementation for CapibaraModel."""

import os
import sys
import logging
from typing import Tuple, Optional, Dict, Any, Union

from flax import linen as nn
from capibara.jax import jax
from capibara.jax.numpy import jnp
from capibara.interfaces.isub_models import ISubModel
from capibara.core.arm_optimizations import ARM_OPTIMIZATIONS, HARDWARE_INFO

logger = logging.getLogger(__name__)

# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation

class AlephTilde(nn.Module, ISubModel):
    """Neural implementation of Aleph-TILDE algorithm."""
    
    hidden_size: int
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize model parameters."""
        # Capas de transformation
        self.hypothesis = nn.Dense(self.hidden_size)
        self.background = nn.Dense(self.hidden_size)
        self.rules = nn.Dense(self.hidden_size)
        
        # Capas de integration
        self.combine = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.hidden_size)
        
        # Capas auxiliares
        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs: Any
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass optimizado for ARM Axion."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, dim), got shape {x.shape}"
            )

        # detect ARM Axion
        use_arm = not training and HARDWARE_INFO["is_axion"]

        # Generate hypothesis with optimizaciones ARM
        if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
            h = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                x,
                self.hypothesis.weight
            )
        else:
            h = self.hypothesis(x)
        h = self.norm(h)
        if training:
            h = self.dropout(h, deterministic=not training)
        h = jax.nn.gelu(h)

        # Apply background knowledge with optimizaciones ARM
        if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
            b = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                x,
                self.background.weight
            )
        else:
            b = self.background(x)
        b = self.norm(b)
        if training:
            b = self.dropout(b, deterministic=not training)
        
        # Combine with optimizaciones ARM
        combined = jnp.concatenate([h, b], axis=-1)
        if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
            combined = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                combined,
                self.combine.weight
            )
        else:
            combined = self.combine(combined)
        combined = jax.nn.gelu(combined)

        # Induce rules with optimizaciones ARM
        if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
            r = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                combined,
                self.rules.weight
            )
        else:
            r = self.rules(combined)
        r = self.norm(r)
        if training:
            r = self.dropout(r, deterministic=not training)

        # Output with optimizaciones ARM
        if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
            output = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                r,
                self.output.weight
            )
        else:
            output = self.output(r)

        return {
            "output": output,
            "metrics": {
                "arm_optimizations_used": use_arm,
                "arm_features": list(ARM_OPTIMIZATIONS.keys()) if use_arm else []
            }
        }
