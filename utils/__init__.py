"""
🚀 CAPIBARA UTILS - ECOSISTEMA ULTRA-advanced
============================================

El sistema de utilidades more advanced del mundo with:
- 🧠 Ultra Utils Orchestrator with IA integrada
- 🔧 Smart Contracts System with auto-healing
- ⚡ Performance ultra-optimizado
- 📊 Analytics predictivo en tiempo real
- 🤖 Auto-optimization and self-healing
- 🔄 Smart routing automático

Versión: 3.3.0 Ultra
Autor: CapibaraGPT Ultra Team
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# =============================================================================
# IMPORTS ULTRA-AVANZADOS
# =============================================================================

# Ultra Utils Orchestrator - El Cerebro Central
try:
    from .ultra_utils_orchestrator import (
        UltraUtilsOrchestrator,
        UltraUtilsConfig,
        UtilsMetrics,
        get_ultra_orchestrator,
        ultra_route,
        ultra_status,
        ultra_cache,
        ultra_monitor
    )
    ULTRA_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Ultra Orchestrator no disponible: {e}")
    ULTRA_ORCHESTRATOR_AVAILABLE = False

# Smart Contracts System
try:
    from .smart_utils_contracts import (
        SmartContractsManager,
        PerformanceContract,
        ResourceContract,
        QualityContract,
        SmartContract,
        ContractType,
        ContractStatus,
        ViolationSeverity,
        get_contracts_manager,
        create_performance_contract,
        create_resource_contract,
        create_quality_contract
    )
    SMART_CONTRACTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Smart Contracts no disponible: {e}")
    SMART_CONTRACTS_AVAILABLE = False

# =============================================================================
# IMPORTS TRADICIONALES (with fallbacks seguros)
# =============================================================================

# cache & Performance
try:
    from .cache_standalone import (
        create_cache as create_tpu_cache,
        CacheDecorator as TpuCacheDecorator,
        StandaloneCache as TpuOptimizedCache,
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Checkpoint Management
try:
    from .checkpoint_manager import (
        CheckpointManager as CapibaraCheckpointManager,
    )
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False

# Monitoring & System Info
try:
    from .monitoring import (
        RealTimeMonitor,
        ResourceMonitor,
        setup_monitoring,
    )
    from .system_info import (
        SystemMonitor,
        get_system_info,
        check_tpu_availability,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Logging Utils
try:
    from .logging_utils import (
        setup_logger,
        create_structured_logger,
        log_execution_time,
        log_function_call,
        create_rotating_logger,
        create_timed_rotating_logger,
        create_context_logger,
        log_memory_usage
    )
    LOGGING_UTILS_AVAILABLE = True
except ImportError:
    LOGGING_UTILS_AVAILABLE = False

# Config Utils
try:
    from .config_utils import (
        load_config_file,
        save_config_file,
        merge_configs,
        get_config_value,
        set_config_value,
        validate_config_schema,
        expand_config_variables,
        compare_configs,
        normalize_config_keys
    )
    CONFIG_UTILS_AVAILABLE = True
except ImportError:
    CONFIG_UTILS_AVAILABLE = False

# Data Utils
try:
    from .data_utils import (
        flatten_dict,
        unflatten_dict,
        merge_dicts,
        filter_dict,
        deep_get,
        deep_set,
        chunk_list,
        deduplicate_list,
        transpose_list_of_dicts,
        group_by,
        safe_json_load,
        safe_json_save,
        batch_process
    )
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False

# Validation Utils
try:
    from .validation_utils import (
        is_valid_url,
        is_valid_ipv4,
        is_valid_port,
        validate_file_path,
        is_valid_json_structure,
        validate_password_strength,
        is_valid_username,
        validate_model_config,
        is_safe_filename,
        validate_batch_size,
        is_valid_tensor_shape,
        validate_hyperparameters
    )
    VALIDATION_UTILS_AVAILABLE = True
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False

# String & Format Utils
try:
    from .string_utils import *
    from .format_utils import *
    STRING_FORMAT_AVAILABLE = True
except ImportError:
    STRING_FORMAT_AVAILABLE = False

# Math Utils
try:
    from .math_utils import *
    MATH_UTILS_AVAILABLE = True
except ImportError:
    MATH_UTILS_AVAILABLE = False

# error Handling
try:
    from .error_handling import *
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False

# Metrics
try:
    from .metrics import *
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# =============================================================================
# ULTRA UTILS FACADE - INTERFAZ UNIFICADA
# =============================================================================

class UltraUtils:
    """
    🚀 INTERFAZ UNIFICADA for all el ecosistema de utilities.
    
    Proporciona acceso simple and elegante a todas las funcionalidades:
    - Smart routing automático
    - Caché inteligente with IA
    - Contratos inteligentes
    - Performance analytics
    - Auto-healing
    """
    
    def __init__(self):
        self._orchestrator = None
        self._contracts_manager = None
        self._logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Configura logger for UltraUtils."""
        logger = logging.getLogger("UltraUtils")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @property
    def orchestrator(self) -> Optional['UltraUtilsOrchestrator']:
        """Obtiene el orchestrator de forma lazy."""
        if ULTRA_ORCHESTRATOR_AVAILABLE and self._orchestrator is None:
            self._orchestrator = get_ultra_orchestrator()
        return self._orchestrator
    
    @property
    def contracts(self) -> Optional['SmartContractsManager']:
        """Obtiene el contracts manager de forma lazy."""
        if SMART_CONTRACTS_AVAILABLE and self._contracts_manager is None:
            self._contracts_manager = get_contracts_manager()
        return self._contracts_manager
    
    def route(self, utility_type: str, operation: str, **kwargs) -> Any:
        """🧠 Smart routing - Auto-selecciona la better utilidad."""
        if self.orchestrator:
            return self.orchestrator.smart_route(utility_type, operation, **kwargs)
        else:
            # Fallback manual
            return self._manual_route(utility_type, operation, **kwargs)
    
    def _manual_route(self, utility_type: str, operation: str, **kwargs) -> Any:
        """Routing manual when orchestrator not está available."""
        routing_map = {
            'data': {
                'flatten_dict': flatten_dict if DATA_UTILS_AVAILABLE else None,
                'merge_dicts': merge_dicts if DATA_UTILS_AVAILABLE else None,
                'chunk_list': chunk_list if DATA_UTILS_AVAILABLE else None,
            },
            'config': {
                'load_config': load_config_file if CONFIG_UTILS_AVAILABLE else None,
                'merge_configs': merge_configs if CONFIG_UTILS_AVAILABLE else None,
            },
            'validation': {
                'validate_model_config': validate_model_config if VALIDATION_UTILS_AVAILABLE else None,
                'is_valid_url': is_valid_url if VALIDATION_UTILS_AVAILABLE else None,
            }
        }
        
        if utility_type in routing_map and operation in routing_map[utility_type]:
            func = routing_map[utility_type][operation]
            if func:
                return func(**kwargs)
        
        raise ValueError(f"Operación no soportada: {utility_type}.{operation}")
    
    def cache(self, ttl_seconds: int = 3600):
        """🤖 Decorador de caché inteligente."""
        if self.orchestrator:
            return self.orchestrator._smart_cache_decorator(ttl_seconds)
        else:
            # Fallback basic
            return lambda func: func
    
    def monitor(self, func):
        """📊 Decorador de monitoreo."""
        if self.orchestrator:
            return self.orchestrator._monitor_execution(func)
        else:
            # Fallback basic
            return func
    
    def status(self) -> Dict[str, Any]:
        """📈 Estado complete del ecosistema."""
        status = {
            'timestamp': datetime.now().isoformat() if 'datetime' in globals() else 'unknown',
            'ultra_orchestrator': ULTRA_ORCHESTRATOR_AVAILABLE,
            'smart_contracts': SMART_CONTRACTS_AVAILABLE,
            'components': {
                'cache': CACHE_AVAILABLE,
                'monitoring': MONITORING_AVAILABLE,
                'logging_utils': LOGGING_UTILS_AVAILABLE,
                'config_utils': CONFIG_UTILS_AVAILABLE,
                'data_utils': DATA_UTILS_AVAILABLE,
                'validation_utils': VALIDATION_UTILS_AVAILABLE,
                'string_format': STRING_FORMAT_AVAILABLE,
                'math_utils': MATH_UTILS_AVAILABLE,
                'error_handling': ERROR_HANDLING_AVAILABLE,
                'metrics': METRICS_AVAILABLE
            }
        }
        
        if self.orchestrator:
            status['orchestrator_status'] = self.orchestrator.get_status()
        
        if self.contracts:
            status['contracts_status'] = self.contracts.get_global_status()
        
        return status
    
    def setup_contracts(self, performance: bool = True, resources: bool = True, quality: bool = True) -> None:
        """🔧 setup rápida de contratos inteligentes."""
        if not self.contracts:
            self._logger.warning("Smart Contracts no disponible")
            return
        
        if performance:
            contract = create_performance_contract(
                "default_performance",
                max_execution_time_ms=100.0,
                max_memory_mb=512.0
            )
            self.contracts.register_contract(contract)
            self.contracts.activate_contract("default_performance")
            self._logger.info("✅ Contrato de performance activado")
        
        if resources:
            contract = create_resource_contract(
                "default_resources",
                max_memory_allocation_mb=1024.0,
                max_cpu_cores=4
            )
            self.contracts.register_contract(contract)
            self.contracts.activate_contract("default_resources")
            self._logger.info("✅ Contrato de recursos activado")
        
        if quality:
            contract = create_quality_contract(
                "default_quality",
                min_accuracy=95.0,
                max_error_rate=1.0
            )
            self.contracts.register_contract(contract)
            self.contracts.activate_contract("default_quality")
            self._logger.info("✅ Contrato de calidad activado")
        
        # start monitoreo
        self.contracts.start_monitoring()
        self._logger.info("🚀 Monitoreo de contratos iniciado")


# =============================================================================
# instance GLOBAL ULTRA
# =============================================================================

# instance global del Ultra Utils
Ultra = UltraUtils()

# =============================================================================
# FUNCIONES DE CONVENIENCIA GLOBAL
# =============================================================================

def ultra_route(utility_type: str, operation: str, **kwargs) -> Any:
    """🧠 function global de smart routing."""
    return Ultra.route(utility_type, operation, **kwargs)

def ultra_cache(ttl_seconds: int = 3600):
    """🤖 function global de caché inteligente."""
    return Ultra.cache(ttl_seconds)

def ultra_monitor(func):
    """📊 function global de monitoreo."""
    return Ultra.monitor(func)

def ultra_status() -> Dict[str, Any]:
    """📈 function global de estado."""
    return Ultra.status()

def setup_ultra_environment(
    enable_orchestrator: bool = True,
    enable_contracts: bool = True,
    enable_monitoring: bool = True
) -> Dict[str, bool]:
    """
    🚀 setup ultra-rápida del entorno complete.
    
    Args:
        enable_orchestrator: enable Ultra Orchestrator
        enable_contracts: enable Smart Contracts
        enable_monitoring: enable monitoreo automático
        
    Returns:
        Estado de initialization
    """
    results = {
        'orchestrator_ready': False,
        'contracts_ready': False,
        'monitoring_active': False
    }
    
    try:
        if enable_orchestrator and ULTRA_ORCHESTRATOR_AVAILABLE:
            # Inicializar orchestrator
            orchestrator = Ultra.orchestrator
            if orchestrator:
                results['orchestrator_ready'] = True
                print("🚀 Ultra Orchestrator iniciado")
        
        if enable_contracts and SMART_CONTRACTS_AVAILABLE:
            # configure contratos by defect
            Ultra.setup_contracts()
            results['contracts_ready'] = True
            print("🔧 Smart Contracts configurados")
        
        if enable_monitoring and results['contracts_ready']:
            results['monitoring_active'] = True
            print("📊 Monitoreo automático activado")
            
    except Exception as e:
        print(f"⚠️ Error configurando entorno ultra: {e}")
    
    return results

# =============================================================================
# EXPORTS PRINCIPALES
# =============================================================================

# Ultra Core Exports
__all__: List[str] = [
    # Ultra Core
    'Ultra',
    'UltraUtils',
    'ultra_route',
    'ultra_cache', 
    'ultra_monitor',
    'ultra_status',
    'setup_ultra_environment',
]

# add exports condicionales
if ULTRA_ORCHESTRATOR_AVAILABLE:
    __all__.extend([
        'UltraUtilsOrchestrator',
        'UltraUtilsConfig',
        'UtilsMetrics',
        'get_ultra_orchestrator'
    ])

if SMART_CONTRACTS_AVAILABLE:
    __all__.extend([
        'SmartContractsManager',
        'PerformanceContract',
        'ResourceContract', 
        'QualityContract',
        'get_contracts_manager',
        'create_performance_contract',
        'create_resource_contract',
        'create_quality_contract'
    ])

if CACHE_AVAILABLE:
    __all__.extend([
        'TpuOptimizedCache',
        'TpuCacheDecorator', 
        'create_tpu_cache'
    ])

if CHECKPOINT_AVAILABLE:
    __all__.extend(['CapibaraCheckpointManager'])

if MONITORING_AVAILABLE:
    __all__.extend([
        'RealTimeMonitor',
        'ResourceMonitor',
        'setup_monitoring',
        'SystemMonitor',
        'get_system_info',
        'check_tpu_availability'
    ])

if LOGGING_UTILS_AVAILABLE:
    __all__.extend([
        'setup_logger',
        'create_structured_logger',
        'log_execution_time'
    ])

if CONFIG_UTILS_AVAILABLE:
    __all__.extend([
        'load_config_file',
        'merge_configs',
        'get_config_value'
    ])

if DATA_UTILS_AVAILABLE:
    __all__.extend([
        'flatten_dict',
        'merge_dicts',
        'chunk_list',
        'batch_process'
    ])

if VALIDATION_UTILS_AVAILABLE:
    __all__.extend([
        'validate_model_config',
        'is_valid_url',
        'is_valid_json_structure'
    ])

# =============================================================================
# initialization AUTOMÁTICA
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

# Logger del módulo utils
_utils_logger = logging.getLogger("capibara.utils")

def _log_initialization_status():
    """Registra el estado de initialization."""
    _utils_logger.info("🚀 CapibaraGPT Ultra Utils inicializados")
    _utils_logger.info(f"📦 Componentes disponibles:")
    _utils_logger.info(f"   🧠 Ultra Orchestrator: {'✅' if ULTRA_ORCHESTRATOR_AVAILABLE else '❌'}")
    _utils_logger.info(f"   🔧 Smart Contracts: {'✅' if SMART_CONTRACTS_AVAILABLE else '❌'}")
    _utils_logger.info(f"   💾 Cache System: {'✅' if CACHE_AVAILABLE else '❌'}")
    _utils_logger.info(f"   📊 Monitoring: {'✅' if MONITORING_AVAILABLE else '❌'}")
    _utils_logger.info(f"   🔍 Validation: {'✅' if VALIDATION_UTILS_AVAILABLE else '❌'}")

# Registrar estado al import
try:
    _log_initialization_status()
except:
    pass  # Silencio if hay problemas with logging

# =============================================================================
# setup RECOMENDADA
# =============================================================================

def get_recommended_config() -> Dict[str, Any]:
    """
    📋 Obtiene setup recomendada for production.
    
    Returns:
        setup optimizada
    """
    return {
        'orchestrator': {
            'max_memory_mb': 2048,
            'max_threads': 8,
            'cache_size_mb': 512,
            'enable_ai_optimization': True,
            'enable_analytics': True,
            'enable_auto_healing': True
        },
        'contracts': {
            'performance': {
                'max_execution_time_ms': 100.0,
                'max_memory_mb': 512.0,
                'min_success_rate': 95.0
            },
            'resources': {
                'max_memory_allocation_mb': 1024.0,
                'max_cpu_cores': 4
            },
            'quality': {
                'min_accuracy': 95.0,
                'max_error_rate': 1.0
            }
        },
        'monitoring': {
            'check_interval_seconds': 30,
            'enable_real_time': True
        }
    }

# =============================================================================
# MENSAJE DE BIENVENIDA
# =============================================================================

if ULTRA_ORCHESTRATOR_AVAILABLE and SMART_CONTRACTS_AVAILABLE:
    welcome_message = """
🚀 ¡BIENVENIDO AL ECOSISTEMA ULTRA-advanced DE CAPIBARA UTILS!

Funcionalidades disponibles:
├── 🧠 Ultra Orchestrator: IA integrada + Smart routing
├── 🔧 Smart Contracts: Auto-healing + SLA management  
├── ⚡ Performance: Caché inteligente + Optimization
├── 📊 Analytics: Monitoreo predictivo en tiempo real
└── 🤖 Auto-optimization: Self-healing systems

Uso fast:
>>> from capibara.utils import Ultra
>>> Ultra.route('data', 'flatten_dict', data={'a': {'b': 1}})
>>> @Ultra.cache(3600)
... def mi_funcion(): pass
>>> Ultra.setup_contracts()
>>> status = Ultra.status()

¡Disfruta del poder ultra! 🚀
"""
    # print(welcome_message)  # Comentado for evitar spam en imports
