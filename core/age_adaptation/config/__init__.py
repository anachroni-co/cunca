"""
Configuration for the age adaptation system.
"""

from .age_config import (
    AgeAdaptationConfig,
    HardwareType,
    DEFAULT_TPU_CONFIG,
    DEFAULT_ARM_CONFIG
)

__all__ = [
    "AgeAdaptationConfig",
    "HardwareType",
    "DEFAULT_TPU_CONFIG",
    "DEFAULT_ARM_CONFIG"
]
