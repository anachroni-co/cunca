"""
Ultra Layer Integration - CapibaraGPT v2024
===========================================

Ultra-advanced orchestration layer connecting:
- Ultra Core System
- Training & optimization subsystems
- Dynamic layer composition
- Adaptive architecture evolution
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

from .base import BaseLayer, LayerConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Optional integrations (safe imports)
# ============================================================================

try:
    from ..core.ultra_core_integration import create_ultra_core_system
    ULTRA_CORE_AVAILABLE = True
except Exception:
    ULTRA_CORE_AVAILABLE = False
    create_ultra_core_system = None

try:
    from ..training.optimizations import UltraAdvancedTrainer
    ULTRA_TRAINING_INTEGRATION = True
except Exception:
    ULTRA_TRAINING_INTEGRATION = False
    UltraAdvancedTrainer = None

try:
    from .ssm_hybrid_layers import create_ssm_layer, create_ssm_config
    SSM_LAYERS_AVAILABLE = True
except Exception:
    SSM_LAYERS_AVAILABLE = False
    create_ssm_layer = None
    create_ssm_config = None

try:
    from .self_attention import TpuOptimizedSelfAttention, TpuSelfAttentionConfig
except Exception:
    TpuOptimizedSelfAttention = None
    TpuSelfAttentionConfig = None

try:
    from .neurogenesis import (
        TpuOptimizedNeurogenesisModule,
        TpuNeurogenesisModuleConfig,
    )
except Exception:
    TpuOptimizedNeurogenesisModule = None
    TpuNeurogenesisModuleConfig = None

try:
    from .meta_la import MetaLA, MetaLAConfig
    META_LA_AVAILABLE = True
except Exception:
    META_LA_AVAILABLE = False
    MetaLA = None
    MetaLAConfig = None

try:
    from .pasive.attention import DistributedAttention
    DISTRIBUTED_ATTENTION_AVAILABLE = True
except Exception:
    DISTRIBUTED_ATTENTION_AVAILABLE = False
    DistributedAttention = None

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UltraLayerIntegrationConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12

    enable_ssm_layers: bool = True
    ssm_ratio: float = 0.4

    enable_neurogenesis: bool = True
    neurogenesis_frequency: int = 2
    neurogenesis_sparsity: float = 0.1

    enable_meta_la: bool = True
    meta_la_frequency: int = 3
    meta_learning_rate: float = 0.01
    adaptation_steps: int = 5

    enable_distributed_attention: bool = True
    distributed_attention_frequency: int = 4

    enable_tpu_optimization: bool = True
    use_mixed_precision: bool = True

    auto_core_integration: bool = True
    auto_training_integration: bool = True

    graceful_degradation: bool = True


# ============================================================================
# Composite wrapper
# ============================================================================

class CompositeWrapper(BaseLayer):
    def __init__(
        self,
        base_layer: BaseLayer,
        component,
        label: str,
        use_residual: bool = True,
        pass_training: bool = True,
    ):
        super().__init__(getattr(base_layer, "config", LayerConfig()))
        self.base_layer = base_layer
        self.component = component
        self.use_residual = use_residual
        self.pass_training = pass_training
        self.name = f"{label}({type(base_layer).__name__})"

    def __call__(self, x, training: bool = False, **kwargs):
        y = self.base_layer(x, training=training, **kwargs)

        if not callable(self.component):
            return y

        kw = {"training": training} if self.pass_training else {}
        z = self.component(y, **kw)

        return y + z if self.use_residual else z


NeurogenesisWrapper = partial(
    CompositeWrapper, label="Neurogenesis", use_residual=False
)
MetaLAWrapper = partial(
    CompositeWrapper, label="MetaLA", use_residual=True
)
DistributedAttentionWrapper = partial(
    CompositeWrapper, label="DistAttn", use_residual=True
)

# ============================================================================
# Orchestrator
# ============================================================================

class UltraLayerOrchestrator:
    def __init__(self, config: UltraLayerIntegrationConfig):
        self.config = config
        self.layers: List[BaseLayer] = []
        self.core_orchestrator = None
        self.training_integration = None

        self._initialize()

    def _initialize(self):
        if self.config.auto_core_integration and ULTRA_CORE_AVAILABLE:
            try:
                self.core_orchestrator = create_ultra_core_system()
                logger.info("Ultra Core integrated")
            except Exception as e:
                logger.warning(f"Ultra Core integration failed: {e}")

        self._build_layers()

        if self.config.auto_training_integration and ULTRA_TRAINING_INTEGRATION:
            self.training_integration = True

    def _build_layers(self):
        for idx in range(self.config.num_layers):
            layer = self._create_base_layer(idx)
            if layer is not None:
                self.layers.append(layer)

    def _create_base_layer(self, idx: int) -> Optional[BaseLayer]:
        try:
            if (
                self.config.enable_ssm_layers
                and SSM_LAYERS_AVAILABLE
                and idx < int(self.config.num_layers * self.config.ssm_ratio)
            ):
                layer = self._create_ssm_layer()
            else:
                layer = self._create_attention_layer()

            return self._apply_wrappers(layer, idx)

        except Exception as e:
            logger.error(f"Layer {idx} failed: {e}")
            if self.config.graceful_degradation:
                return self._create_attention_layer()
            raise

    def _create_attention_layer(self) -> Optional[BaseLayer]:
        if TpuOptimizedSelfAttention is None:
            return None

        cfg = TpuSelfAttentionConfig(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            use_mixed_precision=self.config.use_mixed_precision,
        )
        return TpuOptimizedSelfAttention(cfg)

    def _create_ssm_layer(self) -> Optional[BaseLayer]:
        cfg = create_ssm_config(
            hidden_size=self.config.hidden_size,
            enable_all_optimizations=self.config.enable_tpu_optimization,
        )
        return create_ssm_layer("ultra", cfg)

    def _apply_wrappers(self, layer: BaseLayer, idx: int) -> BaseLayer:
        if (
            self.config.enable_neurogenesis
            and idx % self.config.neurogenesis_frequency == 0
            and TpuOptimizedNeurogenesisModule is not None
        ):
            ng_cfg = TpuNeurogenesisModuleConfig(
                hidden_size=self.config.hidden_size,
                sparsity=self.config.neurogenesis_sparsity,
            )
            layer = NeurogenesisWrapper(layer, TpuOptimizedNeurogenesisModule(ng_cfg))

        if (
            self.config.enable_meta_la
            and META_LA_AVAILABLE
            and idx % self.config.meta_la_frequency == 0
        ):
            ml_cfg = MetaLAConfig(
                hidden_size=self.config.hidden_size,
                meta_learning_rate=self.config.meta_learning_rate,
                adaptation_steps=self.config.adaptation_steps,
            )
            layer = MetaLAWrapper(layer, MetaLA(ml_cfg))

        if (
            self.config.enable_distributed_attention
            and DISTRIBUTED_ATTENTION_AVAILABLE
            and idx % self.config.distributed_attention_frequency == 0
        ):
            layer = DistributedAttentionWrapper(
                layer,
                DistributedAttention(
                    {
                        "hidden_size": self.config.hidden_size,
                        "num_heads": self.config.num_heads,
                    }
                ),
            )

        return layer

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "layers": len(self.layers),
            "ultra_core": self.core_orchestrator is not None,
            "training_integration": self.training_integration is not None,
        }


# ============================================================================
# Factory
# ============================================================================

def create_ultra_layer_system(
    config: Optional[UltraLayerIntegrationConfig] = None,
    **kwargs,
) -> UltraLayerOrchestrator:
    if config is None:
        config = UltraLayerIntegrationConfig(**kwargs)
    return UltraLayerOrchestrator(config)

