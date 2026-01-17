"""implementation optimizada de Capibara for TPUs."""

import os
import sys

# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation

from capibara.jax import jax
from capibara.jax.numpy import jnp
from flax import linen as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import partial

from .tpu_base_config import TPUBaseConfig
from capibara.interfaces.isub_models import nfigTPU, PrecisionMode
from capibara.core.arm_optimizations import ARM_OPTIMIZATIONS, HARDWARE_INFO

logger = logging.getLogger(__name__)

class CapibaraByteConfig(TPUBaseConfig):
    """setup específica for CapibaraByte."""
    context_window: int = 128
    cache_size: int = 1024
    use_hybrid_sharding: bool = True

class CapibaraByte(nn.Module, ISubModel):
    """implementation optimizada de Capibara for TPUs.
    
    Características:
    - Sharding híbrido for TPUs
    - precision mixta
    - cache JIT-compatible
    - Optimizaciones de memory
    """
    config: CapibaraByteConfig
    
    def setup(self):
        """Inicializa componentes optimizados for tpu."""
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.config.dtype,
            name="byte_dense"
        )
        self.norm = nn.LayerNorm(
            dtype=self.config.dtype,
            name="byte_norm"
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def _compute_context_weight(self, context_embedding: jnp.ndarray) -> jnp.ndarray:
        """Calcula peso del contexto de forma JIT-compatible."""
        return jnp.mean(context_embedding, axis=-1)
        
    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Forward pass optimizado for ARM Axion.
        
        Args:
            x: tensor de input
            context: tensor de contexto optional
            training: Modo entrenamiento
            **kwargs: Argumentos adicionales
            
        Returns:
            Dict with output and métricas
        """
        # detect ARM Axion
        use_arm = not training and HARDWARE_INFO["is_axion"]
        
        # Normalizar input
        x = self.norm(x)
        
        # process contexto if existe
        if context is not None:
            if use_arm and ARM_OPTIMIZATIONS.get("sve2_vectorization"):
                # use SVE2 for calculation de pesos
                context_weight = ARM_OPTIMIZATIONS["arm_optimization_suite"].sve2_context_weight(
                    context,
                    self.dense.weight
                )
            else:
                context_weight = self._compute_context_weight(context)
            x = x * context_weight[..., None]
        
        # Proyección densa with optimizaciones ARM
        if use_arm:
            if ARM_OPTIMIZATIONS.get("arm_kleidi"):
                # use Kleidi for matmul
                x = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                    x,
                    self.dense.weight
                )
            elif ARM_OPTIMIZATIONS.get("arm_quantization_available"):
                # use quantización ARM
                x = ARM_OPTIMIZATIONS["arm_optimization_suite"].quantized_matmul(
                    x,
                    self.dense.weight
                )
            else:
                x = self.dense(x)
        else:
            x = self.dense(x)
        
        # apply dropout if es necessary
        if training and self.config.dropout_rate > 0:
            x = nn.Dropout(self.config.dropout_rate)(
                x, deterministic=False
            )
        
        # Métricas
        metrics = {
            "input_norm": jnp.linalg.norm(x),
            "output_norm": jnp.linalg.norm(x),
            "context_weight": context_weight if context is not None else 0.0,
            "arm_optimizations_used": use_arm,
            "arm_features": list(ARM_OPTIMIZATIONS.keys()) if use_arm else []
        }
        
        return {
            "output": x,
            "metrics": metrics
        }

# setup extendida
config = ByteConfigTPU(
    input_dim=256,
    hidden_size=512,
    dtype=jnp.bfloat16,  # better for TPUs
    precision=PrecisionMode.BFLOAT16,
    update_rate=0.1,
    use_memory_efficient=True,
    gradient_checkpointing=True,
    residual_connections=True
)

# initialization del model
model = TPUCapibaraByte(config)
input_data = jnp.ones((1, 128, 256), dtype=config.dtype)  # (batch, seq_len, input_dim)
context_data = None  # if tu forward acepta contexto, pásalo here

# Inicializar parámetros
params = model.init(jax.random.PRNGKey(0), input_data)

# Forward pass with todas las características
final_state, all_states = model.apply(
    params,
    x=input_data,
    initial_state=None,  # or tu estado inicial if lo tienes
    training=True
)
print("Final state shape:", final_state.shape)
print("All states shape:", all_states.shape)