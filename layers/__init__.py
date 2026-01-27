"""
Layers Module - CapibaraGPT v3

Neural network layer implementations:
- ssm_hybrid_layers: State-space model hybrid layers (Mamba, S4)
- pasive: Synthetic embedding and distributed attention
- abstract_reasoning: Platonic, Quineana, GameTheory
- sparsity: BitNet, sparse layers, quantization
- ultra_layer_integration: Ecosystem orchestration

Note: Most layers require JAX/Flax or PyTorch. Check availability flags
before using specific layers.
"""

import logging

logger = logging.getLogger(__name__)

# SSM hybrid layers (requires JAX/Flax)
try:
    from .ssm_hybrid_layers import (
        UltraSSMLayer,
        MambaLayer,
        S4Layer,
        HybridSSMLayer,
        SSMHybridLayerConfig,
        create_ssm_layer,
        create_ssm_config,
        validate_ssm_system,
        SSM_COMPONENTS_AVAILABLE,
    )
    SSM_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"SSM hybrid layers not available: {e}")
    SSM_LAYERS_AVAILABLE = False
    SSM_COMPONENTS_AVAILABLE = False
    UltraSSMLayer = None
    MambaLayer = None
    S4Layer = None
    HybridSSMLayer = None
    SSMHybridLayerConfig = None
    create_ssm_layer = None
    create_ssm_config = None
    validate_ssm_system = None

# Passive learning layers (requires JAX/Flax)
try:
    from .pasive.synthetic_embedding import SyntheticEmbedding
    from .pasive.attention import DistributedAttention
    PASSIVE_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Passive layers not available: {e}")
    PASSIVE_LAYERS_AVAILABLE = False
    SyntheticEmbedding = None
    DistributedAttention = None

# Abstract reasoning layers (requires JAX/Flax)
try:
    from .abstract_reasoning.platonic import Platonic
    from .abstract_reasoning.quineana import Quineana
    from .abstract_reasoning.game_theory import GameTheory, BasicGameTheory
    ABSTRACT_REASONING_AVAILABLE = True
except Exception as e:
    logger.debug(f"Abstract reasoning layers not available: {e}")
    ABSTRACT_REASONING_AVAILABLE = False
    Platonic = None
    Quineana = None
    GameTheory = None
    BasicGameTheory = None

# Sparsity and quantization layers (requires JAX/Flax)
try:
    from .sparsity.bitnet import Conv1DBlock, BitNet158
    from .sparsity.sparse_capibara import SparseCapibara
    from .sparsity.affine_quantizer import AffineQuantizer
    from .sparsity.mixture_of_rookies import MixtureOfRookies
    SPARSITY_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Sparsity layers not available: {e}")
    SPARSITY_LAYERS_AVAILABLE = False
    Conv1DBlock = None
    BitNet158 = None
    SparseCapibara = None
    AffineQuantizer = None
    MixtureOfRookies = None

# Additional specialized layers
try:
    from .conv1d_block import Conv1DBlock as Conv1DBlockLayer
    from .meta_la import MetaLA, MetaLAConfig
    from .smb_layer import SMBLayer
    ADDITIONAL_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Additional layers not available: {e}")
    ADDITIONAL_LAYERS_AVAILABLE = False
    Conv1DBlockLayer = None
    MetaLA = None
    MetaLAConfig = None
    SMBLayer = None

# Ultra layer integration
try:
    from .ultra_layer_integration import (
        UltraLayerOrchestrator,
        UltraLayerIntegrationConfig,
        LayerPerformanceMetrics,
        create_ultra_layer_system,
        create_ultra_layer_config,
        demonstrate_ultra_layer_integration,
        ULTRA_CORE_AVAILABLE,
        ULTRA_TRAINING_INTEGRATION,
    )
    ULTRA_LAYER_INTEGRATION_AVAILABLE = True
except Exception as e:
    logger.debug(f"Ultra layer integration not available: {e}")
    ULTRA_LAYER_INTEGRATION_AVAILABLE = False
    ULTRA_CORE_AVAILABLE = False
    ULTRA_TRAINING_INTEGRATION = False
    UltraLayerOrchestrator = None
    UltraLayerIntegrationConfig = None
    LayerPerformanceMetrics = None
    create_ultra_layer_system = None
    create_ultra_layer_config = None
    demonstrate_ultra_layer_integration = None


__all__ = [
    # SSM layers
    "UltraSSMLayer",
    "MambaLayer",
    "S4Layer",
    "HybridSSMLayer",
    "SSMHybridLayerConfig",
    "create_ssm_layer",
    "create_ssm_config",
    "validate_ssm_system",
    # Passive layers
    "SyntheticEmbedding",
    "DistributedAttention",
    # Abstract reasoning
    "Platonic",
    "Quineana",
    "GameTheory",
    "BasicGameTheory",
    # Sparsity
    "Conv1DBlock",
    "BitNet158",
    "SparseCapibara",
    "AffineQuantizer",
    "MixtureOfRookies",
    # Additional
    "Conv1DBlockLayer",
    "MetaLA",
    "MetaLAConfig",
    "SMBLayer",
    # Ultra layer
    "UltraLayerOrchestrator",
    "UltraLayerIntegrationConfig",
    "LayerPerformanceMetrics",
    "create_ultra_layer_system",
    "create_ultra_layer_config",
    "demonstrate_ultra_layer_integration",
    # Availability flags
    "SSM_LAYERS_AVAILABLE",
    "SSM_COMPONENTS_AVAILABLE",
    "PASSIVE_LAYERS_AVAILABLE",
    "ABSTRACT_REASONING_AVAILABLE",
    "SPARSITY_LAYERS_AVAILABLE",
    "ADDITIONAL_LAYERS_AVAILABLE",
    "ULTRA_LAYER_INTEGRATION_AVAILABLE",
    "ULTRA_CORE_AVAILABLE",
    "ULTRA_TRAINING_INTEGRATION",
]
