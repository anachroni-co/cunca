"""
TPU utilities for CapibaraGPT core.
"""

import logging

from .tpu_config import TpuConfig, config

logger = logging.getLogger(__name__)

try:
    from .tpu_v6e import (
        TPUv6eConfig,
        TPUv6eOptimizer,
        TPUv6eTrainingPipeline,
        get_tpu_optimizer,
        initialize_tpu_training,
    )
except Exception as exc:
    logger.warning("TPU v6e utilities unavailable: %s", exc)
    TPUv6eConfig = None  # type: ignore
    TPUv6eOptimizer = None  # type: ignore
    TPUv6eTrainingPipeline = None  # type: ignore
    get_tpu_optimizer = None  # type: ignore
    initialize_tpu_training = None  # type: ignore

__all__ = [
    "TpuConfig",
    "config",
    "TPUv6eConfig",
    "TPUv6eOptimizer",
    "TPUv6eTrainingPipeline",
    "get_tpu_optimizer",
    "initialize_tpu_training",
]
