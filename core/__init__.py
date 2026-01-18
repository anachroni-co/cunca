"""
Core Module - CapibaraGPT v3

Central integration point for model construction, training, and inference:
- config: Model and training configuration
- modular_model: Dynamic model composition
- router/routing: Intelligent module routing
- optimization: Training state and metrics
- encoders: Vision, video, multimodal encoders
- cot: Chain of Thought reasoning
- ultra_core_integration: SSM-hybrid layers and orchestration
"""

import logging

logger = logging.getLogger(__name__)

# Configuration
try:
    from .config import Config, ModelConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    ModelConfig = None

# Modular model
try:
    from .modular_model import ModularCapibaraModel, ModularConfig
    MODULAR_AVAILABLE = True
except ImportError:
    MODULAR_AVAILABLE = False
    ModularCapibaraModel = None
    ModularConfig = None

# Enhanced router
try:
    from .router import (
        EnhancedRouter,
        Router as AdvancedRouter,
        create_enhanced_router,
        create_core_integrated_router,
        RouterConfig as RouterCoreConfig,
        RouterType,
    )
    ENHANCED_ROUTER_AVAILABLE = True
except ImportError:
    ENHANCED_ROUTER_AVAILABLE = False
    EnhancedRouter = None
    AdvancedRouter = None
    create_enhanced_router = None
    create_core_integrated_router = None
    RouterCoreConfig = None
    RouterType = None

# Simple router
try:
    from .routing import Router as SimpleRouter, create_router
    SIMPLE_ROUTER_AVAILABLE = True
except ImportError:
    SIMPLE_ROUTER_AVAILABLE = False
    SimpleRouter = None
    create_router = None

# Optimization
try:
    from .optimization import TrainingMetrics, TrainingState
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    TrainingMetrics = None
    TrainingState = None

# Legacy routers
try:
    from .routers.base import BaseRouter, BaseRouterV2
    from .routers.adaptive_router import AdaptiveRouter
    LEGACY_ROUTERS_AVAILABLE = True
except ImportError:
    LEGACY_ROUTERS_AVAILABLE = False
    BaseRouter = None
    BaseRouterV2 = None
    AdaptiveRouter = None

# TTS Router (optional)
try:
    from .routers.tts_router import TTSRouter
    TTS_ROUTER_AVAILABLE = True
except ImportError:
    TTS_ROUTER_AVAILABLE = False
    TTSRouter = None

# Optimizers
try:
    from .optimizers.optimizer import OptimizerConfig, create_optimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    OptimizerConfig = None
    create_optimizer = None

# Encoders
try:
    from .encoders import (
        VisionEncoder,
        VisionEncoderConfig,
        VideoEncoder,
        VideoEncoderConfig,
        MultimodalCombiner,
        CombinerConfig,
    )
    ENCODERS_AVAILABLE = True
except ImportError:
    ENCODERS_AVAILABLE = False
    VisionEncoder = None
    VisionEncoderConfig = None
    VideoEncoder = None
    VideoEncoderConfig = None
    MultimodalCombiner = None
    CombinerConfig = None

# Chain of Thought
try:
    from .cot import ChainOfThought, create_cot_handler
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False
    ChainOfThought = None
    create_cot_handler = None

# Model core
try:
    from ._model import ModelCore, create_model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    ModelCore = None
    create_model = None

# Ultra Core Integration (advanced SSM-hybrid components)
try:
    from .ultra_core_integration import (
        UltraIntelligentCoreRouter,
        SSMHybridLayer,
        UltraCoreOrchestrator,
        create_ultra_core_system,
        create_ultra_config,
        integrate_with_training,
        validate_ultra_core_system,
        ULTRA_TRAINING_AVAILABLE,
        SSM_AVAILABLE,
    )
    ULTRA_CORE_AVAILABLE = True
except ImportError:
    ULTRA_CORE_AVAILABLE = False
    ULTRA_TRAINING_AVAILABLE = False
    SSM_AVAILABLE = False
    UltraIntelligentCoreRouter = None
    SSMHybridLayer = None
    UltraCoreOrchestrator = None
    create_ultra_core_system = None
    create_ultra_config = None
    integrate_with_training = None
    validate_ultra_core_system = None


__all__ = [
    # Config
    "Config",
    "ModelConfig",
    "ModularConfig",
    # Enhanced router
    "EnhancedRouter",
    "AdvancedRouter",
    "create_enhanced_router",
    "create_core_integrated_router",
    "RouterCoreConfig",
    "RouterType",
    # Simple router
    "SimpleRouter",
    "create_router",
    # Optimization
    "TrainingMetrics",
    "TrainingState",
    # Legacy routers
    "BaseRouter",
    "BaseRouterV2",
    "AdaptiveRouter",
    "TTSRouter",
    # Optimizers
    "OptimizerConfig",
    "create_optimizer",
    # Encoders
    "VisionEncoder",
    "VisionEncoderConfig",
    "VideoEncoder",
    "VideoEncoderConfig",
    "MultimodalCombiner",
    "CombinerConfig",
    # Chain of Thought
    "ChainOfThought",
    "create_cot_handler",
    # Model
    "ModelCore",
    "create_model",
    "ModularCapibaraModel",
    # Ultra Core
    "UltraIntelligentCoreRouter",
    "SSMHybridLayer",
    "UltraCoreOrchestrator",
    "create_ultra_core_system",
    "create_ultra_config",
    "integrate_with_training",
    "validate_ultra_core_system",
    # Availability flags
    "CONFIG_AVAILABLE",
    "MODULAR_AVAILABLE",
    "ENHANCED_ROUTER_AVAILABLE",
    "SIMPLE_ROUTER_AVAILABLE",
    "OPTIMIZATION_AVAILABLE",
    "LEGACY_ROUTERS_AVAILABLE",
    "TTS_ROUTER_AVAILABLE",
    "OPTIMIZER_AVAILABLE",
    "ENCODERS_AVAILABLE",
    "COT_AVAILABLE",
    "MODEL_AVAILABLE",
    "ULTRA_CORE_AVAILABLE",
    "ULTRA_TRAINING_AVAILABLE",
    "SSM_AVAILABLE",
]
