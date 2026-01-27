"""
Meta-Learning Attention (Meta-LA) Module.

This module provides meta-learning enhanced attention mechanisms
for improved context understanding and few-shot adaptation.

Optimized with @nn.compact and JIT-compiled meta-learning computations.
"""

import functools
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from capibara.jax import jax
from capibara.jax import numpy as jnp
from flax import linen as nn

from capibara.interfaces.ilayer import ILayer
from capibara.layers.base import BaseLayer, LayerConfig

logger = logging.getLogger(__name__)


# JIT-compiled meta-learning computation
@functools.partial(jax.jit, static_argnums=())
def _meta_adaptation_kernel(
    x: jnp.ndarray,
    meta_weights: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled meta-learning context computation."""
    return jnp.dot(x, meta_weights)

@dataclass
class MetaLAConfig(LayerConfig):
    """Configuration for Meta-Learning Attention (Meta-LA) layer"""
    hidden_dim: int = 768  # Use main's larger default
    hidden_size: int = 768  # Compatibility alias
    num_heads: int = 12  # Use main's larger default
    num_attention_heads: int = 12  # Compatibility alias
    dropout_rate: float = 0.1
    meta_learning_rate: float = 0.001
    adaptation_steps: int = 5  # From main branch

class MetaLA(BaseLayer):
    """
    Meta-Learning Attention (Meta-LA) Layer

    Advanced attention mechanism that adapts based on meta-learning
    principles for improved context understanding.

    Uses @nn.compact for optimal JIT compilation and extracted
    meta-learning kernels for performance.
    """
    config: MetaLAConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 training: bool = False) -> jnp.ndarray:
        """Forward pass with enhanced meta-learning adaptation."""
        batch_size, seq_len, hidden_size = x.shape

        # Use the larger of the two head counts for better performance
        num_heads = max(getattr(self.config, 'num_heads', 8),
                       getattr(self.config, 'num_attention_heads', 12))
        hidden_dim = max(getattr(self.config, 'hidden_dim', 512),
                        getattr(self.config, 'hidden_size', 768))

        # Apply meta-adaptation
        adapted_x = nn.Dense(hidden_dim, name='meta_adapter')(x)

        # Apply attention with adaptation
        attended = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            qkv_features=hidden_dim,
            dropout_rate=self.config.dropout_rate,
            name='attention'
        )(adapted_x, adapted_x, mask=mask, deterministic=not training)

        # Meta-learning adaptation using JIT-compiled kernel
        meta_weights = self.param(
            'meta_weights',
            nn.initializers.normal(0.02),
            (hidden_dim, hidden_dim)
        )
        meta_context = _meta_adaptation_kernel(x, meta_weights)
        adapted_features = nn.Dense(hidden_dim, name='adaptation_layer')(meta_context)

        # Combine all features: attention + meta-adaptation + meta-features
        combined = attended + adapted_features
        output = nn.LayerNorm(name='layer_norm')(x + combined)

        return output
    
    def fast_adapt(self, support_x, support_y, query_x):
        """Fast adaptation for few-shot learning."""
        # Simplified meta-learning adaptation
        adapted_params = self.adapt_parameters(support_x, support_y)
        return self.apply_adapted_params(query_x, adapted_params)
    
    def adapt_parameters(self, support_x, support_y):
        """Adapt parameters based on support set."""
        # Placeholder for meta-learning parameter adaptation
        return self.meta_weights
    
    def apply_adapted_params(self, query_x, adapted_params):
        """Apply adapted parameters to query set."""
        adapted_features = jnp.dot(query_x, adapted_params)
        return adapted_features

# Alias for backward compatibility
class MetaLearningAttention(MetaLA):
    """Backward compatibility alias."""
    pass

def main():
    # Main function for this module.
    logger.info("Module meta_la.py starting")
    # Test MetaLA configuration with updated parameter names
    config = MetaLAConfig(hidden_dim=768, num_heads=12)
    logger.info(f"✅ MetaLA config created: {config}")
    return True

if __name__ == "__main__":
    main()
