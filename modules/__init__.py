"""
Modules Package - CapibaraGPT v3

Provides modular neural network components including:
- Attention mechanisms (shared, multi-scale)
- Hierarchical reasoning
- Adaptive routing
- Specialized processors
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Core module imports with fallbacks
try:
    from .shared_attention import (
        OptimizedSharedAttention,
        MultiScaleSharedAttention,
        create_shared_attention,
    )
    ATTENTION_AVAILABLE = True
except Exception as e:
    ATTENTION_AVAILABLE = False
    OptimizedSharedAttention = None
    MultiScaleSharedAttention = None
    create_shared_attention = None
    logger.warning("Attention modules unavailable: %s", e)

try:
    from .hierarchical_reasoning import HierarchicalReasoning
    REASONING_AVAILABLE = True
except Exception as e:
    REASONING_AVAILABLE = False
    HierarchicalReasoning = None
    logger.warning("Hierarchical reasoning unavailable: %s", e)

try:
    from .capibara_adaptive_router import AdaptiveRouter
    ROUTER_AVAILABLE = True
except Exception as e:
    ROUTER_AVAILABLE = False
    AdaptiveRouter = None
    logger.warning("Adaptive router unavailable: %s", e)

try:
    from .specialized_processors import (
        SpecializedProcessorManager,
        create_processor_manager,
    )
    PROCESSORS_AVAILABLE = True
    SpecializedProcessor = SpecializedProcessorManager
except Exception as e:
    PROCESSORS_AVAILABLE = False
    SpecializedProcessor = None
    SpecializedProcessorManager = None
    create_processor_manager = None
    logger.warning("Specialized processors unavailable: %s", e)

try:
    from .ultra_module_orchestrator import (
        UltraModuleOrchestrator,
        create_ultra_module_system,
    )
    ORCHESTRATOR_AVAILABLE = True
except Exception as e:
    ORCHESTRATOR_AVAILABLE = False
    UltraModuleOrchestrator = None
    create_ultra_module_system = None
    logger.warning("Ultra module orchestrator unavailable: %s", e)


def get_module_status() -> Dict[str, bool]:
    """Get availability status of all modules."""
    return {
        "attention": ATTENTION_AVAILABLE,
        "reasoning": REASONING_AVAILABLE,
        "router": ROUTER_AVAILABLE,
        "processors": PROCESSORS_AVAILABLE,
        "orchestrator": ORCHESTRATOR_AVAILABLE,
    }


__all__ = [
    # Attention
    "OptimizedSharedAttention",
    "MultiScaleSharedAttention",
    "create_shared_attention",
    # Reasoning
    "HierarchicalReasoning",
    # Router
    "AdaptiveRouter",
    # Processors
    "SpecializedProcessor",
    "SpecializedProcessorManager",
    "create_processor_manager",
    # Orchestrator
    "UltraModuleOrchestrator",
    "create_ultra_module_system",
    # Status
    "get_module_status",
]
