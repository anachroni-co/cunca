"""
Hierarchical Training Strategy Module - CapibaraGPT v3

Complete implementation of hierarchical training strategy:
- Transfer Learning Pipeline: 300M -> 600M -> 1.2B -> 3B -> 7B -> 13B
- Hierarchical MoE with 3 levels and 2.6B router
- Selective Ensemble for complex queries
- Cost optimization: $0.27/1K tokens (vs industry $0.50+)

Components:
- HierarchicalTrainingPipeline: Main training pipeline
- HierarchicalMoERouter: Intelligent routing system
- EnsembleManager: Selective ensemble management
- TransferLearningManager: Transfer learning management
"""

from .training_pipeline import (
    ModelTier,
    ModelConfig,
    DistillationConfig,
    create_training_pipeline,
    validate_training_strategy,
    HierarchicalTrainingPipeline,
)

from .moe_router import (
    ExpertDomain,
    QueryAnalysis,
    RoutingDecision,
    QueryComplexity,
    HierarchicalMoERouter,
    create_hierarchical_router,
    estimate_routing_efficiency,
)

from .ensemble_manager import (
    EnsembleResult,
    EnsembleManager,
    EnsembleStrategy,
    create_ensemble_manager,
)

from .transfer_learning_manager import (
    TransferConfig,
    TransferLearningManager,
    create_transfer_manager,
)

__version__ = "1.0.0"
__author__ = "CapibaraGPT Team"

__all__ = [
    # Main Pipeline
    'HierarchicalTrainingPipeline',
    'ModelConfig',
    'DistillationConfig',
    'ModelTier',
    'create_training_pipeline',
    'validate_training_strategy',

    # MoE Router
    'HierarchicalMoERouter',
    'QueryAnalysis',
    'RoutingDecision',
    'QueryComplexity',
    'ExpertDomain',
    'create_hierarchical_router',
    'estimate_routing_efficiency',

    # Ensemble Manager
    'EnsembleManager',
    'EnsembleStrategy',
    'EnsembleResult',
    'create_ensemble_manager',

    # Transfer Learning
    'TransferLearningManager',
    'TransferConfig',
    'create_transfer_manager'
]
