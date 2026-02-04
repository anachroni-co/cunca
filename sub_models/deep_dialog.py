"""DeepDialog – Trainable Flax Version (Capibara)"""

from typing import Optional, Callable
from dataclasses import dataclass
import logging

from flax import linen as nn
from capibara.jax import jax
from capibara.jax.numpy import jnp

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class DeepDialogConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 8
    dropout_rate: float = 0.1
    activation_fn: Callable = jax.nn.gelu
    context_dim: Optional[int] = None
    residual_scale: float = 1.0

# ============================================================================
# MODEL
# ============================================================================

class DeepDialog(nn.Module):
    config: DeepDialogConfig

    def setup(self):
        self.input_proj = nn.Dense(self.config.hidden_size)

        if self.config.context_dim:
            self.context_proj = nn.Dense(self.config.hidden_size)
            self.cross_layers = [
                CrossAttentionLayer(self.config)
                for _ in range(self.config.num_layers)
            ]

        self.layers = [
            TransformerLayer(self.config)
            for _ in range(self.config.num_layers)
        ]

        self.norm = nn.LayerNorm()

    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:

        x = self.input_proj(x)

        if context is not None and self.config.context_dim:
            context = self.context_proj(context)
            for layer in self.cross_layers:
                x = layer(x, context, deterministic)

        for layer in self.layers:
            x = layer(x, deterministic)

        return self.norm(x)

# ============================================================================
# LAYERS
# ============================================================================

class CrossAttentionLayer(nn.Module):
    config: DeepDialogConfig

    @nn.compact
    def __call__(self, x, context, deterministic):
        h = nn.LayerNorm()(x)
        c = nn.LayerNorm()(context)

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate,
        )(h, c, deterministic=deterministic)

        x = x + self.config.residual_scale * attn

        y = nn.Dense(self.config.hidden_size * 4)(x)
        y = self.config.activation_fn(y)
        y = nn.Dropout(self.config.dropout_rate)(
            y, deterministic=deterministic
        )
        y = nn.Dense(self.config.hidden_size)(y)

        return x + self.config.residual_scale * y

class TransformerLayer(nn.Module):
    config: DeepDialogConfig

    @nn.compact
    def __call__(self, x, deterministic):
        h = nn.LayerNorm()(x)

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate,
        )(h, h, deterministic=deterministic)

        x = x + self.config.residual_scale * attn

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.config.hidden_size * 4)(y)
        y = self.config.activation_fn(y)
        y = nn.Dropout(self.config.dropout_rate)(
            y, deterministic=deterministic
        )
        y = nn.Dense(self.config.hidden_size)(y)

        return x + self.config.residual_scale * y