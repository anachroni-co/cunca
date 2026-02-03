"""
Age Adaptation Module

Provides age-appropriate content adaptation and configuration.
"""

try:
    from .age_config import AgeAdaptationConfig
    from .metrics import AgeMetrics
    AGE_ADAPTATION_AVAILABLE = True
except ImportError as e:
    AGE_ADAPTATION_AVAILABLE = False
    AgeAdaptationConfig = None
    AgeMetrics = None

__all__ = [
    'AgeAdaptationConfig',
    'AgeMetrics',
    'AGE_ADAPTATION_AVAILABLE',
]
