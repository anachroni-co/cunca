"""
TPU monitoring module for CapibaraGPT-v2.
"""

from .tpu_monitor import TPUMonitor, TPUMetrics, tpu_logger
from .tpu_decorators import (
    register_fallback,
    monitor_tpu_fallback,
    monitor_tpu_operation,
)
from .tpu_alerts import (
    AlertConfig,
    alerts_logger,
    TPUAlertManager,
    AlertThresholds,
)

__all__ = [
    # Main monitor
    'TPUMonitor',
    'TPUMetrics',
    'tpu_logger',

    # Decorators
    'monitor_tpu_operation',
    'monitor_tpu_fallback',
    'register_fallback',

    # Alert system
    'TPUAlertManager',
    'AlertConfig',
    'AlertThresholds',
    'alerts_logger'
]
