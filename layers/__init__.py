"""
Layers Module - CapibaraGPT v3

Neural network layer implementations:
- pasive: Synthetic embedding and distributed attention
- abstract_reasoning: Platonic, Quineana, GameTheory
- sparsity: BitNet, sparse layers, quantization

Note: Most layers require JAX/Flax or PyTorch. Check availability flags
before using specific layers.
"""

import logging

logger = logging.getLogger(__name__)

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

SSM_LAYERS_AVAILABLE = False
SSM_COMPONENTS_AVAILABLE = False
ADDITIONAL_LAYERS_AVAILABLE = False
ULTRA_LAYER_INTEGRATION_AVAILABLE = False
ULTRA_CORE_AVAILABLE = False
ULTRA_TRAINING_INTEGRATION = False


__all__ = [
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
    # Availability flags
    "SSM_LAYERS_AVAILABLE",
    "SSM_COMPONENTS_AVAILABLE",
    "PASSIVE_LAYERS_AVAILABLE",
    "ABSTRACT_REASONING_AVAILABLE",
    "SPARSITY_LAYERS_AVAILABLE",
    "ADDITIONAL_LAYERS_AVAILABLE",
]
