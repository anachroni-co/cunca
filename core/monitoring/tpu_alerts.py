"""
Alert system for TPU operations.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

# configure logger specific for TPU alerts
alerts_logger = logging.getLogger('capibara.tpu.alerts')
alerts_logger.setLevel(logging.WARNING)

@dataclass
class AlertThresholds:
    """Thresholds for TPU alerts."""
    max_memory_usage_gb: float = 80.0  # 80% of the total
    min_tflops: float = 100.0  # minimum expected
    max_latency_ms: float = 100.0  # maximum acceptable
    min_utilization: float = 0.5  # 50% minimum
    max_temperature_c: float = 85.0  # maximum safe
    max_power_watts: float = 450.0  # maximum safe
    max_fallbacks_per_hour: int = 10  # maximum acceptable

@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = True
    thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    alert_handlers: List[Callable] = None
    cooldown_seconds: int = 300  # 5 minutes between similar alerts

class TPUAlertManager:
    """TPU alert manager."""

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize the TPU alert manager.

        Args:
            config: Alert configuration. If None, uses default configuration.
        """
        self.config = config or AlertConfig()
        self.recent_alerts: deque = deque(maxlen=1000)
        self.alert_timestamps: Dict[str, float] = {}

    def check_metrics(self, metrics: Dict) -> None:
        """
        Verify metrics and generate alerts if necessary.

        Args:
            metrics: Dictionary with current metrics
        """
        if not self.config.enabled:
            return

        current_time = time.time()

        # verify memory
        if metrics['memory_gb']['current'] > self.config.thresholds.max_memory_usage_gb:
            self._emit_alert(
                'high_memory',
                f"High memory usage: {metrics['memory_gb']['current']:.1f}GB "
                f"(max: {self.config.thresholds.max_memory_usage_gb}GB)",
                current_time
            )

        # verify TFLOPS
        if metrics['tflops']['current'] < self.config.thresholds.min_tflops:
            self._emit_alert(
                'low_tflops',
                f"Low TFLOPS: {metrics['tflops']['current']:.1f} "
                f"(min: {self.config.thresholds.min_tflops})",
                current_time
            )

        # verify latency
        if metrics['latency_ms']['current'] > self.config.thresholds.max_latency_ms:
            self._emit_alert(
                'high_latency',
                f"High latency: {metrics['latency_ms']['current']:.1f}ms "
                f"(max: {self.config.thresholds.max_latency_ms}ms)",
                current_time
            )

        # verify utilization
        if metrics['utilization']['current'] < self.config.thresholds.min_utilization:
            self._emit_alert(
                'low_utilization',
                f"Low utilization: {metrics['utilization']['current']*100:.1f}% "
                f"(min: {self.config.thresholds.min_utilization*100}%)",
                current_time
            )

        # verify fallbacks
        total_fallbacks = sum(metrics['fallbacks'].values())
        if total_fallbacks > self.config.thresholds.max_fallbacks_per_hour:
            self._emit_alert(
                'high_fallbacks',
                f"Too many fallbacks: {total_fallbacks} in the last hour "
                f"(max: {self.config.thresholds.max_fallbacks_per_hour})",
                current_time
            )

    def _emit_alert(self, alert_type: str, message: str, current_time: float) -> None:
        """
        Emit an alert if cooldown has passed.

        Args:
            alert_type: Type of alert
            message: Alert message
            current_time: Current time
        """
        # verify cooldown
        last_alert = self.alert_timestamps.get(alert_type, 0)
        if current_time - last_alert < self.config.cooldown_seconds:
            return

        # Register alert
        self.alert_timestamps[alert_type] = current_time
        self.recent_alerts.append({
            'type': alert_type,
            'message': message,
            'timestamp': current_time
        })

        # Log the alert
        alerts_logger.warning(f"TPU Alert - {message}")

        # execute custom handlers
        if self.config.alert_handlers:
            for handler in self.config.alert_handlers:
                try:
                    handler(alert_type, message)
                except Exception as e:
                    alerts_logger.error(f"Error in alert handler: {e}")

    def get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts."""
        return list(self.recent_alerts)

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self.recent_alerts.clear()
        self.alert_timestamps.clear()

# usage example:
"""
# configure alert configuration
alert_config = AlertConfig(
    thresholds=AlertThresholds(
        max_memory_usage_gb=90.0,
        min_tflops=200.0
    ),
    alert_handlers=[
        lambda type, msg: print(f"Alert: {msg}")
    ]
)

# create manager
alert_manager = TPUAlertManager(config)

# verify metrics
metrics = {
    'memory_gb': {'current': 95.0},
    'tflops': {'current': 150.0},
    'latency_ms': {'current': 50.0},
    'utilization': {'current': 0.8},
    'fallbacks': {'memory': 5, 'computation': 3}
}

alert_manager.check_metrics(metrics)
"""
