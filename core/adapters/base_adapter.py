"""
Base Adapter Interface

Defines the base interface that all adapters must implement,
providing common functionality and implementation standards.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class AdapterStatus(Enum):
    """Possible adapter states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class AdapterConfig:
    """Base configuration for adapters."""
    name: str = ""
    enabled: bool = True
    priority: int = 50
    max_retries: int = 3
    timeout_seconds: float = 30.0
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    fallback_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdapterMetrics:
    """Performance metrics for an adapter."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_execution_time: float = 0.0
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)

class BaseAdapter(ABC):
    """
    Abstract base class for all system adapters.

    Provides common functionality such as:
    - Configuration management
    - Performance metrics
    - Error handling and retries
    - Basic caching
    - Standardized logging
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.status = AdapterStatus.UNINITIALIZED
        self.metrics = AdapterMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._cache: Dict[str, Any] = {}
        self._initialization_time: Optional[float] = None
        
        # Auto-configure name if not set
        if not self.config.name:
            self.config.name = self.__class__.__name__
        
        self.logger.info(f"Adapter {self.config.name} created")
    
    def initialize(self) -> bool:
        """
        Initialize the adapter. Must be called before first use.

        Returns:
            True if initialization was successful
        """
        if self.status == AdapterStatus.READY:
            return True
        
        self.status = AdapterStatus.INITIALIZING
        start_time = time.time()
        
        try:
            self.logger.info(f"Initializing adapter {self.config.name}")
            
            # Call adapter-specific initialization
            success = self._initialize_impl()
            
            if success:
                self.status = AdapterStatus.READY
                self._initialization_time = time.time() - start_time
                self.logger.info(f"Adapter {self.config.name} initialized successfully in {self._initialization_time:.3f}s")
                return True
            else:
                self.status = AdapterStatus.ERROR
                self.logger.error(f"Failed to initialize adapter {self.config.name}")
                return False
                
        except Exception as e:
            self.status = AdapterStatus.ERROR
            self.logger.error(f"Exception during initialization of {self.config.name}: {e}")
            return False
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the adapter's main operation with error handling and metrics.

        Returns:
            Execution result
        """
        if not self.config.enabled:
            raise RuntimeError(f"Adapter {self.config.name} is disabled")
        
        if self.status != AdapterStatus.READY:
            if not self.initialize():
                raise RuntimeError(f"Adapter {self.config.name} is not ready")
        
        # Try to get result from cache
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(*args, **kwargs)
            if cache_key in self._cache:
                self.metrics.cache_hits += 1
                self.logger.debug(f"Cache hit for {self.config.name}")
                return self._cache[cache_key]
            else:
                self.metrics.cache_misses += 1
        
        # Execute with retries
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                self.status = AdapterStatus.BUSY
                
                # Execute specific implementation
                result = self._execute_impl(*args, **kwargs)
                
                # Update success metrics
                execution_time = (time.time() - start_time) * 1000
                self._update_success_metrics(execution_time)
                
                # Save to cache if enabled
                if self.config.enable_caching and cache_key:
                    self._update_cache(cache_key, result)
                
                self.status = AdapterStatus.READY
                return result
                
            except Exception as e:
                last_exception = e
                execution_time = (time.time() - start_time) * 1000
                self._update_failure_metrics(execution_time, str(e))
                
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {self.config.name}: {e}. Retrying...")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"All {self.config.max_retries + 1} attempts failed for {self.config.name}")
        
        self.status = AdapterStatus.ERROR
        raise RuntimeError(f"Adapter {self.config.name} failed after {self.config.max_retries + 1} attempts") from last_exception
    
    def get_metrics(self) -> AdapterMetrics:
        """Get the adapter's current metrics."""
        return self.metrics

    def get_status(self) -> AdapterStatus:
        """Get the adapter's current state."""
        return self.status

    def reset_metrics(self):
        """Reset the adapter's metrics."""
        self.metrics = AdapterMetrics()
        self.logger.info(f"Metrics reset for adapter {self.config.name}")

    def clear_cache(self):
        """Clear the adapter's cache."""
        self._cache.clear()
        self.logger.info(f"Cache cleared for adapter {self.config.name}")

    def disable(self):
        """Disable the adapter."""
        self.config.enabled = False
        self.status = AdapterStatus.DISABLED
        self.logger.info(f"Adapter {self.config.name} disabled")

    def enable(self):
        """Enable the adapter."""
        self.config.enabled = True
        if self.status == AdapterStatus.DISABLED:
            self.status = AdapterStatus.UNINITIALIZED
        self.logger.info(f"Adapter {self.config.name} enabled")

    def get_info(self) -> Dict[str, Any]:
        """Get complete adapter information."""
        return {
            'name': self.config.name,
            'class': self.__class__.__name__,
            'status': self.status.value,
            'enabled': self.config.enabled,
            'priority': self.config.priority,
            'initialization_time': self._initialization_time,
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'successful_calls': self.metrics.successful_calls,
                'failed_calls': self.metrics.failed_calls,
                'success_rate': self.get_success_rate(),
                'average_execution_time_ms': self.metrics.average_execution_time_ms,
                'cache_hit_rate': self.get_cache_hit_rate(),
                'uptime_seconds': time.time() - self.metrics.created_at
            },
            'config': {
                'max_retries': self.config.max_retries,
                'timeout_seconds': self.config.timeout_seconds,
                'cache_enabled': self.config.enable_caching,
                'cache_size': len(self._cache),
                'fallback_enabled': self.config.fallback_enabled
            }
        }
    
    def get_success_rate(self) -> float:
        """Calculate the adapter's success rate."""
        if self.metrics.total_calls == 0:
            return 1.0
        return self.metrics.successful_calls / self.metrics.total_calls

    def get_cache_hit_rate(self) -> float:
        """Calculate the cache hit rate."""
        total_cache_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.metrics.cache_hits / total_cache_requests

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _initialize_impl(self) -> bool:
        """
        Specific initialization implementation.

        Returns:
            True if initialization was successful
        """
        pass

    @abstractmethod
    def _execute_impl(self, *args, **kwargs) -> Any:
        """
        Specific execution implementation.

        Returns:
            Operation result
        """
        pass

    # Optional methods that can be overridden

    def _generate_cache_key(self, *args, **kwargs) -> Optional[str]:
        """
        Generate a cache key for the given arguments.

        Returns:
            Cache key or None if it should not be cached
        """
        try:
            # Basic implementation using argument hash
            import hashlib
            key_data = str(args) + str(sorted(kwargs.items()))
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return None

    def _update_cache(self, key: str, value: Any):
        """Update the cache with a new value."""
        if len(self._cache) >= self.config.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value

    def _update_success_metrics(self, execution_time_ms: float):
        """Update metrics after a successful execution."""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.metrics.total_execution_time_ms += execution_time_ms
        self.metrics.average_execution_time_ms = (
            self.metrics.total_execution_time_ms / self.metrics.total_calls
        )
        self.metrics.last_execution_time = time.time()

    def _update_failure_metrics(self, execution_time_ms: float, error_message: str):
        """Update metrics after a failed execution."""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.metrics.total_execution_time_ms += execution_time_ms
        self.metrics.average_execution_time_ms = (
            self.metrics.total_execution_time_ms / self.metrics.total_calls
        )
        self.metrics.last_error = error_message
        self.metrics.last_execution_time = time.time()


class FallbackAdapter(BaseAdapter):
    """
    Fallback adapter that can use multiple implementations.

    Useful when robustness against specific adapter failures is needed.
    """
    
    def __init__(self, 
                 primary_adapter: BaseAdapter,
                 fallback_adapters: List[BaseAdapter],
                 config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.primary_adapter = primary_adapter
        self.fallback_adapters = fallback_adapters
        self.current_adapter = primary_adapter
        self.fallback_count = 0
    
    def _initialize_impl(self) -> bool:
        """Initialize the primary adapter and fallbacks."""
        success = self.primary_adapter.initialize()

        # Try to initialize fallbacks
        for adapter in self.fallback_adapters:
            try:
                adapter.initialize()
            except Exception as e:
                self.logger.warning(f"Failed to initialize fallback adapter {adapter.config.name}: {e}")
        
        return success
    
    def _execute_impl(self, *args, **kwargs) -> Any:
        """Execute with automatic fallback on failure."""
        # Try with primary adapter
        try:
            return self.primary_adapter.execute(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary adapter {self.primary_adapter.config.name} failed: {e}")

        # Try with fallbacks
        for adapter in self.fallback_adapters:
            try:
                self.logger.info(f"Trying fallback adapter {adapter.config.name}")
                result = adapter.execute(*args, **kwargs)
                self.fallback_count += 1
                return result
            except Exception as e:
                self.logger.warning(f"Fallback adapter {adapter.config.name} failed: {e}")
        
        raise RuntimeError("All adapters (primary and fallbacks) failed")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information including fallback statistics."""
        info = super().get_info()
        info['fallback_count'] = self.fallback_count
        info['primary_adapter'] = self.primary_adapter.config.name
        info['fallback_adapters'] = [a.config.name for a in self.fallback_adapters]
        return info