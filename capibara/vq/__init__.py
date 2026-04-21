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

# VQbit package (submodule) availability
try:
    from . import vqbit  # noqa: F401
    VQBIT_PACKAGE_AVAILABLE = True
except Exception as e:
    vqbit = None  # type: ignore
    VQBIT_PACKAGE_AVAILABLE = False
    logger.debug("VQ subpackage 'vqbit' unavailable: %s", e)

# Ultra VQ orchestrator
try:
    from .ultra_vq_orchestrator import (
        UltraVQOrchestrator,
        create_ultra_vq_system,
    )
    ORCHESTRATOR_AVAILABLE = True
except Exception as e:
    ORCHESTRATOR_AVAILABLE = False
    UltraVQOrchestrator = None
    create_ultra_vq_system = None
    logger.debug("Ultra VQ orchestrator unavailable: %s", e)

# VQbit layer
try:
    from .vqbit_layer import VQbitLayer
    VQBIT_AVAILABLE = True
except Exception as e:
    VQBIT_AVAILABLE = False
    VQbitLayer = None
    logger.debug("VQbit layer unavailable: %s", e)

# Multi-modal VQ
try:
    from .multi_modal_vq_intelligence import MultiModalVQ
    MULTIMODAL_AVAILABLE = True
except Exception as e:
    MULTIMODAL_AVAILABLE = False
    MultiModalVQ = None
    logger.debug("Multi-modal VQ unavailable: %s", e)

# Adaptive VQ
try:
    from .adaptive_vq_performance_intelligence import AdaptiveVQ
    ADAPTIVE_AVAILABLE = True
except Exception as e:
    ADAPTIVE_AVAILABLE = False
    AdaptiveVQ = None
    logger.debug("Adaptive VQ unavailable: %s", e)

# ARM Axion optimizations
try:
    from .vq_arm_axion import VQArmAxion
    ARM_AVAILABLE = True
except Exception as e:
    ARM_AVAILABLE = False
    VQArmAxion = None
    logger.debug("ARM Axion VQ unavailable: %s", e)


__all__ = [
    # Subpackage
    "vqbit",
    "VQBIT_PACKAGE_AVAILABLE",
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
