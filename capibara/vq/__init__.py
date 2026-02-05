"""
Vector Quantization Module - CapibaraGPT v3

Advanced vector quantization system supporting:
- ultra_vq_orchestrator: Main VQ orchestration
- vqbit_layer: VQbit layer implementation
- multi_modal_vq_intelligence: Multi-modal VQ
- adaptive_vq_performance_intelligence: Adaptive optimization
- vq_arm_axion: ARM Axion optimizations
"""

import logging

logger = logging.getLogger(__name__)

# Ultra VQ orchestrator
try:
    from .ultra_vq_orchestrator import (
        UltraVQOrchestrator,
        create_ultra_vq_system,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    UltraVQOrchestrator = None
    create_ultra_vq_system = None

# VQbit layer
try:
    from .vqbit_layer import VQbitLayer
    VQBIT_AVAILABLE = True
except ImportError:
    VQBIT_AVAILABLE = False
    VQbitLayer = None

# Multi-modal VQ
try:
    from .multi_modal_vq_intelligence import MultiModalVQ
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    MultiModalVQ = None

# Adaptive VQ
try:
    from .adaptive_vq_performance_intelligence import AdaptiveVQ
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    AdaptiveVQ = None

# ARM Axion optimizations
try:
    from .vq_arm_axion import VQArmAxion
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False
    VQArmAxion = None


__all__ = [
    # Orchestrator
    "UltraVQOrchestrator",
    "create_ultra_vq_system",
    # VQbit
    "VQbitLayer",
    # Multi-modal
    "MultiModalVQ",
    # Adaptive
    "AdaptiveVQ",
    # ARM
    "VQArmAxion",
    # Flags
    "ORCHESTRATOR_AVAILABLE",
    "VQBIT_AVAILABLE",
    "MULTIMODAL_AVAILABLE",
    "ADAPTIVE_AVAILABLE",
    "ARM_AVAILABLE",
]
