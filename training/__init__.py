"""
Training Module - CapibaraGPT v3

Meta-consensus training system with HuggingFace integration:
- unified_consensus: Base consensus coordination
- enhanced_hf_consensus_strategy: HuggingFace-optimized consensus
- hybrid_expert_router: Multi-tier expert routing
- btx_training_system: Branch-Train-MiX training
- meta_consensus_system: Higher-order consensus coordination
"""

import logging

logger = logging.getLogger(__name__)

# Unified consensus
try:
    from .unified_consensus import UnifiedConsensusStrategy, ConsensusConfig
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False
    UnifiedConsensusStrategy = None
    ConsensusConfig = None

# Enhanced HuggingFace consensus
try:
    from .enhanced_hf_consensus_strategy import EnhancedHFConsensusStrategy, ServerlessExpertConfig
    HF_CONSENSUS_AVAILABLE = True
except ImportError:
    HF_CONSENSUS_AVAILABLE = False
    EnhancedHFConsensusStrategy = None
    ServerlessExpertConfig = None

# Hybrid expert router
try:
    from .hybrid_expert_router import HybridExpertRouter, ExpertTier, RoutingStrategy
    HYBRID_ROUTER_AVAILABLE = True
except ImportError:
    HYBRID_ROUTER_AVAILABLE = False
    HybridExpertRouter = None
    ExpertTier = None
    RoutingStrategy = None

# BTX training system
try:
    from .btx_training_system import BTXTrainingSystem, BTXExpertConfig
    BTX_AVAILABLE = True
except ImportError:
    BTX_AVAILABLE = False
    BTXTrainingSystem = None
    BTXExpertConfig = None

# Meta-consensus system
try:
    from .meta_consensus_system import MetaConsensusSystem, MetaConsensusConfig, create_meta_consensus_system
    META_CONSENSUS_AVAILABLE = True
except ImportError:
    META_CONSENSUS_AVAILABLE = False
    MetaConsensusSystem = None
    MetaConsensusConfig = None
    create_meta_consensus_system = None


__all__ = [
    # Unified consensus
    "UnifiedConsensusStrategy",
    "ConsensusConfig",
    # HuggingFace consensus
    "EnhancedHFConsensusStrategy",
    "ServerlessExpertConfig",
    # Hybrid router
    "HybridExpertRouter",
    "ExpertTier",
    "RoutingStrategy",
    # BTX training
    "BTXTrainingSystem",
    "BTXExpertConfig",
    # Meta-consensus
    "MetaConsensusSystem",
    "MetaConsensusConfig",
    "create_meta_consensus_system",
    # Availability flags
    "UNIFIED_AVAILABLE",
    "HF_CONSENSUS_AVAILABLE",
    "HYBRID_ROUTER_AVAILABLE",
    "BTX_AVAILABLE",
    "META_CONSENSUS_AVAILABLE",
]
