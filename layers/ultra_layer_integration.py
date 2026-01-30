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

import jax
import jax.numpy as jnp

from functools import partial

from .base import BaseLayer, LayerConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Safe optional imports
# ============================================================================

try:
    from ..core.ultra_core_integration import (
        create_ultra_core_system,
        ULTRA_TRAINING_AVAILABLE,
        SSM_AVAILABLE,
    )
    ULTRA_CORE_AVAILABLE = True
except Exception:
    ULTRA_CORE_AVAILABLE = False
    ULTRA_TRAINING_AVAILABLE = False
    SSM_AVAILABLE = False

try:
    from ..training.optimizations import (
        UltraAdvancedTrainer,
        ExpertSoupIntegration,
        ULTRA_OPTIMIZATIONS_AVAILABLE,
    )
    ULTRA_TRAINING_INTEGRATION = True
except Exception:
    ULTRA_TRAINING_INTEGRATION = False
    ULTRA_OPTIMIZATIONS_AVAILABLE = False

try:
    from .ssm_hybrid_layers import create_ssm_layer, create_ssm_config
    SSM_LAYERS_AVAILABLE = True
except Exception:
    SSM_LAYERS_AVAILABLE = False

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
    from .neuro_adaptive import NeuroAdaptiveLayer, NeuroAdaptiveLayerConfig
except Exception:
    NeuroAdaptiveLayer = None
    NeuroAdaptiveLayerConfig = None

try:
    from .meta_la import MetaLA, MetaLAConfig
    META_LA_AVAILABLE = True
except Exception:
    MetaLA = None
    MetaLAConfig = None
    META_LA_AVAILABLE = False

try:
    from .pasive.attention import DistributedAttention
    DISTRIBUTED_ATTENTION_AVAILABLE = True
except Exception:
    DistributedAttention = None
    DISTRIBUTED_ATTENTION_AVAILABLE = False

try:
    from .abstract_reasoning.platonic import Platonic
    PLATONIC_AVAILABLE = True
except Exception:
    Platonic = None
    PLATONIC_AVAILABLE = False

try:
    from .abstract_reasoning.game_theory import GameTheory
    GAME_THEORY_AVAILABLE = True
except Exception:
    GameTheory = None
    GAME_THEORY_AVAILABLE = False

REASONING_AVAILABLE = PLATONIC_AVAILABLE or GAME_THEORY_AVAILABLE

# ============================================================================
# Configurations
# ============================================================================

@dataclass
class UltraLayerIntegrationConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12

    layer_composition: str = "adaptive"

    enable_ssm_layers: bool = True
    ssm_ratio: float = 0.4
    mamba_s4_ratio: float = 0.6

    enable_neurogenesis: bool = True
    neurogenesis_sparsity: float = 0.1
    neurogenesis_frequency: int = 2

    enable_abstract_reasoning: bool = True
    reasoning_layer_frequency: int = 4
    reasoning_types: List[str] = field(default_factory=lambda: ["game_theory", "platonic"])

    enable_meta_la: bool = True
    meta_la_frequency: int = 3
    meta_learning_rate: float = 0.01
    adaptation_steps: int = 5

    enable_distributed_attention: bool = True
    distributed_attention_frequency: int = 4

    enable_tpu_optimization: bool = True
    use_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    use_expert_soup: bool = True

    auto_core_integration: bool = True
    auto_training_integration: bool = True

    graceful_degradation: bool = True


@dataclass
class LayerPerformanceMetrics:
    layer_id: str
    layer_type: str
    computation_time_ms: float
    memory_usage_mb: float
    throughput_tokens_per_sec: float
    efficiency_score: float = 0.0

    def __post_init__(self):
        if self.computation_time_ms > 0:
            self.efficiency_score = self.throughput_tokens_per_sec / self.computation_time_ms


# ============================================================================
# Composite Wrapper
# ============================================================================

class CompositeWrapper(BaseLayer):
    """Composable wrapper: base_layer → component (+ residual optional)."""

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

        comp_kwargs = {"training": training} if self.pass_training else {}
        z = self.component(y, **comp_kwargs)

        return y + z if self.use_residual else z

    def get_output_shape(self, input_shape):
        return self.base_layer.get_output_shape(input_shape)


NeurogenesisWrapper = partial(CompositeWrapper, label="Neurogenesis", use_residual=False)
ReasoningWrapper = partial(CompositeWrapper, label="Reasoning", use_residual=True, pass_training=False)
MetaLAWrapper = partial(CompositeWrapper, label="MetaLA", use_residual=True)
DistributedAttentionWrapper = partial(CompositeWrapper, label="DistAttn", use_residual=True)


# ============================================================================
# Orchestrator
# ============================================================================

class UltraLayerOrchestrator:
    """Ultra-advanced layer stack orchestrator."""

    def __init__(self, config: UltraLayerIntegrationConfig):
        self.config = config
        self.layers: List[BaseLayer] = []
        self.layer_metrics: Dict[str, LayerPerformanceMetrics] = {}
        self.core_orchestrator = None
        self.training_integration = None

        self.global_metrics = {
            "total_layers": 0,
            "ssm_layers": 0,
            "attention_layers": 0,
            "neurogenesis_layers": 0,
            "reasoning_layers": 0,
        }

        self._initialize()

    def _initialize(self):
        if self.config.auto_core_integration and ULTRA_CORE_AVAILABLE:
            try:
                self.core_orchestrator = create_ultra_core_system()
                logger.info("Ultra Core integrated")
            except Exception as e:
                logger.warning(f"Ultra Core failed: {e}")

        self._build_layers()

        if self.config.auto_training_integration and ULTRA_TRAINING_INTEGRATION:
            self.training_integration = True

    def _build_layers(self):
        for i in range(self.config.num_layers):
            layer = self._create_layer(i)
            if layer is not None:
                self.layers.append(layer)
                self.global_metrics["total_layers"] += 1

    def _create_layer(self, idx: int) -> Optional[BaseLayer]:
        try:
            if self.config.enable_ssm_layers and SSM_LAYERS_AVAILABLE and idx < self.config.num_layers * self.config.ssm_ratio:
                layer = self._create_ssm_layer()
                self.global_metrics["ssm_layers"] += 1
            else:
                layer = self._create_attention_layer()
                self.global_metrics["attention_layers"] += 1

            if layer is None:
                return None

            layer = self._apply_wrappers(layer, idx)
            return layer
        except Exception as e:
            logger.error(f"Layer {idx} failed: {e}")
            return None

    def _apply_wrappers(self, layer: BaseLayer, idx: int) -> BaseLayer:
        """Apply optional wrappers (neurogenesis, reasoning, meta-la, etc.)."""
        if (
            self.config.enable_neurogenesis
            and TpuOptimizedNeurogenesisModule is not None
            and idx % self.config.neurogenesis_frequency == 0
        ):
            ng_cfg = TpuNeurogenesisModuleConfig(
                hidden_size=self.config.hidden_size,
                sparsity=self.config.neurogenesis_sparsity,
            )
            layer = NeurogenesisWrapper(layer, TpuOptimizedNeurogenesisModule(ng_cfg))
            self.global_metrics["neurogenesis_layers"] += 1

        if (
            self.config.enable_abstract_reasoning
            and REASONING_AVAILABLE
            and idx % self.config.reasoning_layer_frequency == 0
        ):
            reasoning_component = self._create_reasoning_component()
            if reasoning_component is not None:
                layer = ReasoningWrapper(layer, reasoning_component)
                self.global_metrics["reasoning_layers"] += 1

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
                    {"hidden_size": self.config.hidden_size, "num_heads": self.config.num_heads}
                ),
            )

        return layer

    def _create_reasoning_component(self):
        """Create an abstract reasoning component from available modules."""
        reasoning_types = self.config.reasoning_types
        for rtype in reasoning_types:
            if rtype == "platonic" and Platonic is not None:
                return Platonic(features=self.config.hidden_size)
            if rtype == "game_theory" and GameTheory is not None:
                return GameTheory(features=self.config.hidden_size)
        return None

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

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "layers": len(self.layers),
            "metrics": self.global_metrics,
            "ultra_core": self.core_orchestrator is not None,
            "training_integration": self.training_integration is not None,
        }


# ============================================================================
# Factory helpers
# ============================================================================

def create_ultra_layer_system(
    config: Optional[UltraLayerIntegrationConfig] = None,
    **kwargs,
) -> UltraLayerOrchestrator:
    if config is None:
        config = UltraLayerIntegrationConfig(**kwargs)
    return UltraLayerOrchestrator(config)
