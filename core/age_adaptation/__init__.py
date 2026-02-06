"""
Age Adaptation Module

Provides age-appropriate content adaptation and configuration.
"""

try:
    from .age_config import AgeAdaptationConfig
    from .metrics import AgeMetrics
    from .models import DataSegment, AdaptiveContentVariant
    AGE_ADAPTATION_AVAILABLE = True
except ImportError:
    AGE_ADAPTATION_AVAILABLE = False
    AgeAdaptationConfig = None
    AgeMetrics = None
    DataSegment = None
    AdaptiveContentVariant = None

__all__ = [
    'AgeAdaptationConfig',
    'AgeMetrics',
    'DataSegment',
    'AdaptiveContentVariant',
    'AGE_ADAPTATION_AVAILABLE',
]
