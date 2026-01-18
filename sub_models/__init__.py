"""
Sub-Models Module - CapibaraGPT v3

Specialized sub-model implementations including:
- SSM_TPU: State Space Model for TPU
- Byte_TPU: Byte-level processing
- csa_expert: Cognitive Specialization Agent
- deep_dialog: Dialog system
- reasoning_enhancement: Reasoning capabilities
- ultra_submodel_orchestrator: Sub-model orchestration
"""

import logging

logger = logging.getLogger(__name__)

# SSM TPU
try:
    from .SSM_TPU import SSMTPU
    SSM_AVAILABLE = True
except ImportError:
    SSM_AVAILABLE = False
    SSMTPU = None

# Byte TPU
try:
    from .Byte_TPU import ByteTPU
    BYTE_AVAILABLE = True
except ImportError:
    BYTE_AVAILABLE = False
    ByteTPU = None

# CSA Expert
try:
    from .csa_expert import CSAExpert
    CSA_AVAILABLE = True
except ImportError:
    CSA_AVAILABLE = False
    CSAExpert = None

# Deep Dialog
try:
    from .deep_dialog import DeepDialog
    DIALOG_AVAILABLE = True
except ImportError:
    DIALOG_AVAILABLE = False
    DeepDialog = None

# Reasoning Enhancement
try:
    from .reasoning_enhancement import ReasoningEnhancement
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False
    ReasoningEnhancement = None

# Ultra Orchestrator
try:
    from .ultra_submodel_orchestrator import (
        UltraSubModelOrchestrator,
        create_ultra_submodel_system,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    UltraSubModelOrchestrator = None
    create_ultra_submodel_system = None


__all__ = [
    # Models
    "SSMTPU",
    "ByteTPU",
    "CSAExpert",
    "DeepDialog",
    "ReasoningEnhancement",
    # Orchestrator
    "UltraSubModelOrchestrator",
    "create_ultra_submodel_system",
    # Flags
    "SSM_AVAILABLE",
    "BYTE_AVAILABLE",
    "CSA_AVAILABLE",
    "DIALOG_AVAILABLE",
    "REASONING_AVAILABLE",
    "ORCHESTRATOR_AVAILABLE",
]
