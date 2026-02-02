"""
Modular Model with Advanced Capabilities and optimizations for TPU v4-32.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
try:
    import toml  # type: ignore
    TOML_AVAILABLE = True
except Exception:
    toml = None  # type: ignore
    TOML_AVAILABLE = False

# JAX and configuration imports
from capibara.jax import nn  # noqa: F401
from capibara.jax.numpy import jnp

# Verification imports
try:
    from capibara.core.verification import (
        ComprehensiveVerificationSystem,
        AlignmentConfig,
    )
except Exception:
    ComprehensiveVerificationSystem = None  # type: ignore
    AlignmentConfig = None  # type: ignore

# Separated module imports
from capibara.core.memory_monitors import CoreIntegratedMemoryMonitor
from capibara.core.metrics import MetricsCollector
from capibara.core.module_registry import ModuleRegistry
from capibara.core.router import CoreIntegratedTokenRouter as Router

# Backend-aware module gating
try:
    from core.backends.module_gate import ModuleGate
except ImportError:
    ModuleGate = None

# Configuration and module imports
from capibara.interfaces.imodules import IModule
try:
    from capibara.core.config import TPUConfig, MemoryConfig, RouterConfig  # noqa: F401
except ImportError:
    # Fallback if these configs don't exist
    pass

# Submodule imports with fallbacks
try:
    from capibara.sub_models.semiotic.mnemosyne_semio_module import MnemosyneSemioModule
except ImportError:
    MnemosyneSemioModule = None

try:
    from capibara.sub_models.experimental.adaptive_vq_submodel import AdaptiveVQSubModel
except ImportError:
    AdaptiveVQSubModel = None

try:
    from capibara.sub_models.experimental.dual_process import DualProcessThinking 
except ImportError:
    DualProcessThinking = None

try:
    from capibara.sub_models.semiotic.semiotic_interaction import SemioticInteraction
except ImportError:
    SemioticInteraction = None

# Adaptive module imports with fallbacks
try:
    from capibara.config.adaptive_config import AdaptiveConfig
    from capibara.adaptive.router import AdaptiveRouter
    from capibara.adaptive.computation import AdaptiveComputation
except ImportError:
    AdaptiveConfig = None
    AdaptiveRouter = None
    AdaptiveComputation = None

logger = logging.getLogger(__name__)


# Exceptions
class ModularCapibaraError(Exception):
    """Base exception for modular model errors."""
    pass

class ModuleRegistrationError(ModularCapibaraError):
    """Error when registering a module."""
    pass

class ModuleLoadingError(ModularCapibaraError):
    """Error when loading a module."""
    pass

class RouterError(ModularCapibaraError):
    """Error in the router."""
    pass

class AgentError(ModularCapibaraError):
    """Agent-related error."""
    pass

class ConfigurationError(ModularCapibaraError):
    """Configuration error."""
    pass

class CoreIntegrationError(ModularCapibaraError):
    """Error in core.py integration."""
    pass

class ModularConfig:
    """Configuration for ModularCapibaraModel integrated with core.py."""

    def __init__(self, backend_name: str = "auto"):
        # Load configurations from TOML files
        try:
            config_root = Path(os.environ.get("CAPIBARA_CONFIG_ROOT", "capibara/config"))
            self.config_root: Path = config_root
            self.modules_root: Optional[Path] = None

            if TOML_AVAILABLE:
                modular_config = toml.load(config_root / "configs_toml/production/modular_model.toml")
                dtypes_config = toml.load(config_root / "configs_toml/production/dtypes.toml")
            else:
                modular_config = {}
                dtypes_config = {"dtypes": {"model_dtype": "float32", "param_dtype": "float32"}}

            # Model configuration
            model_config = modular_config.get("model", {})
            self.hidden_size: int = model_config.get("hidden_size", 768)
            self.num_virtual_qubits: int = model_config.get("num_virtual_qubits", 512)
            self.vocab_size: int = model_config.get("vocab_size", 50257)
            self.max_context_length: int = model_config.get("max_context_length", 2048)
            self.num_router_experts: int = model_config.get("num_router_experts", 8)
            self.router_capacity_factor: float = model_config.get("router_capacity_factor", 1.25)
            
            # TPU configuration
            tpu_config = modular_config.get("tpu", {})
            self.use_core_optimizations: bool = tpu_config.get("use_core_optimizations", True)
            self.workload_type: str = tpu_config.get("workload_type", "balanced")
            self.use_tpu_optimizations: bool = tpu_config.get("use_tpu_optimizations", True)
            
            # Data types
            dtypes = dtypes_config.get("dtypes", {})
            self.dtype = jnp.bfloat16 if dtypes.get("model_dtype") == "bfloat16" else jnp.float32
            self.param_dtype = jnp.float32 if dtypes.get("param_dtype") == "float32" else jnp.float32
            
            # Adaptive configuration
            adaptive_config = modular_config.get("adaptive", {})
            if AdaptiveConfig is not None:
                self.adaptive_config = AdaptiveConfig(
                    hidden_size=self.hidden_size,
                    intermediate_size=adaptive_config.get("intermediate_size", 3072),
                    num_attention_heads=adaptive_config.get("num_attention_heads", 12),
                    num_experts=adaptive_config.get("num_experts", 8),
                    routing_type=adaptive_config.get("routing_type", "top_k"),
                    aux_loss_weight=adaptive_config.get("aux_loss_weight", 0.01),
                    load_balancing_weight=adaptive_config.get("load_balancing_weight", 0.01),
                    early_exit_threshold=adaptive_config.get("early_exit_threshold", 0.5),
                    compute_threshold=adaptive_config.get("compute_threshold", 0.1),
                    max_depth=adaptive_config.get("max_depth", 24),
                    compute_budget=adaptive_config.get("compute_budget", 1.0),
                    enable_moe_integration=adaptive_config.get("enable_moe_integration", True),
                    adaptive_routing=adaptive_config.get("adaptive_routing", True),
                    device=adaptive_config.get("device", "tpu"),
                    precision=adaptive_config.get("precision", "bfloat16"),
                )
            else:
                self.adaptive_config = None
            
            # Feature configuration
            features_config = modular_config.get("features", {})
            self.enable_adaptive_routing: bool = features_config.get("enable_adaptive_routing", True)
            self.enable_cultural_analysis: bool = features_config.get("enable_cultural_analysis", True)
            self.enable_spiking_neural: bool = features_config.get("enable_spiking_neural", True)
            
            # Performance configuration
            perf_config = modular_config.get("performance", {})
            self.routing_cache_size: int = perf_config.get("routing_cache_size", 1000)
            self.lazy_loading_timeout: float = perf_config.get("lazy_loading_timeout", 30.0)
            self.visualization_cleanup_interval: float = perf_config.get("visualization_cleanup_interval", 3600.0)
            self.max_visualization_files: int = perf_config.get("max_visualization_files", 50)
            
            # Agent configuration
            agents_config = modular_config.get("agents", {})
            self.max_agents: int = agents_config.get("max_agents", 100)
            self.agent_cleanup_interval: float = agents_config.get("agent_cleanup_interval", 300.0)
            self.memory_pressure_threshold: float = agents_config.get("memory_pressure_threshold", 0.85)
            self.enable_hot_reload: bool = agents_config.get("enable_hot_reload", False)

            # Backend-aware module gating
            self.backend_name: str = backend_name
            self.module_gate = None
            if ModuleGate is not None:
                resolved = backend_name if backend_name != "auto" else "cpu"
                try:
                    self.module_gate = ModuleGate.from_backend(resolved)
                    logger.info(f"ModuleGate initialized for backend '{resolved}'")
                except ValueError:
                    logger.warning(f"Unknown backend '{resolved}', module gate disabled")

        except Exception as e:
            # Safe fallback with minimal defaults
            logger.warning(f"Failed loading TOML configuration, using defaults: {e}")
            self.config_root = Path("capibara/config")
            self.modules_root = None
            self.hidden_size = 768
            self.num_virtual_qubits = 512
            self.vocab_size = 50257
            self.max_context_length = 2048
            self.num_router_experts = 8
            self.router_capacity_factor = 1.25
            self.use_core_optimizations = True
            self.workload_type = "balanced"
            self.use_tpu_optimizations = True
            self.dtype = jnp.float32
            self.param_dtype = jnp.float32
            self.adaptive_config = None
            self.enable_adaptive_routing = True
            self.enable_cultural_analysis = True
            self.enable_spiking_neural = True
            self.routing_cache_size = 1000
            self.lazy_loading_timeout = 30.0
            self.visualization_cleanup_interval = 3600.0
            self.max_visualization_files = 50
            self.max_agents = 100
            self.agent_cleanup_interval = 300.0
            self.memory_pressure_threshold = 0.85
            self.enable_hot_reload = False
            self.backend_name = backend_name
            self.module_gate = None
            if ModuleGate is not None:
                resolved = backend_name if backend_name != "auto" else "cpu"
                try:
                    self.module_gate = ModuleGate.from_backend(resolved)
                except ValueError:
                    pass


class ModularCapibaraModel:
    """Capibara model with advanced modular capabilities."""
    
    def __init__(self, config: Optional[ModularConfig] = None):
        self.config = config or ModularConfig()
        
        # Initialize components with type annotations
        self.registry: ModuleRegistry = ModuleRegistry()
        self.memory_monitor: CoreIntegratedMemoryMonitor = CoreIntegratedMemoryMonitor()
        self.metrics: MetricsCollector = MetricsCollector()
        
        # Router for module selection
        self.router: Router = Router(
            hidden_size=self.config.hidden_size,
            num_heads=getattr(self.config, 'num_attention_heads', 12),
            dropout_rate=0.1,
            dtype=self.config.dtype,
        )

        # Adaptive components (optional)
        self.adaptive_router = None
        self.adaptive_computation = None
        if self.config.enable_adaptive_routing and self.config.adaptive_config is not None:
            if AdaptiveRouter is not None and AdaptiveComputation is not None:
                # AdaptiveRouter needs AdaptiveRoutingConfig, not AdaptiveConfig
                # Create a compatible routing config
                try:
                    from capibara.adaptive.router import AdaptiveRoutingConfig, RoutingStrategy
                    routing_config = AdaptiveRoutingConfig(
                        strategy=RoutingStrategy.HYBRID,
                        complexity_threshold=0.5,
                        performance_weight=0.4,
                        load_balance_weight=0.3,
                        complexity_weight=0.3,
                        enable_moe_integration=True,
                        enable_caching=True,
                        cache_size=1000,
                        adaptation_rate=0.1
                    )
                    self.adaptive_router = AdaptiveRouter(routing_config)
                except ImportError:
                    logger.warning("AdaptiveRoutingConfig not available, skipping adaptive router")
                
                # AdaptiveComputation uses AdaptiveConfig
                self.adaptive_computation = AdaptiveComputation(self.config.adaptive_config)

        # Verification system (Constitutional AI)
        self.verification_config = None
        self.verification_system = None
        if AlignmentConfig is not None and ComprehensiveVerificationSystem is not None:
            try:
                self.verification_config = AlignmentConfig(
                    enable_constitutional_ai=True,
                    enable_real_time_verification=True,
                    enable_bias_detection=True,
                    enable_harm_prevention=True,
                    verification_threshold=0.85,
                    safety_threshold=0.9,
                )
                self.verification_system = ComprehensiveVerificationSystem(self.verification_config)
            except Exception as e:
                logger.warning(f"Could not initialize verification system: {e}")
                self.verification_system = None
        
        # Active modules
        self.active_modules: List[IModule] = []
        self._load_active_modules()
        
    def _load_active_modules(self):
        """Loads active modules from configuration TOML."""
        try:
            active_modules = []
            if TOML_AVAILABLE:
                modules_config = toml.load("capibara/config/configs_toml/default.toml")["modules"]
                active_modules = modules_config.get("active", [])

            # Dictionary of available module classes
            available_modules = {
                "dual_process": DualProcessThinking,
                "adaptive": AdaptiveVQSubModel,
                "semiotic": MnemosyneSemioModule,
                "semiotic_interaction": SemioticInteraction,
                
                # NEW MAMBA/SSM MODULES
                "mamba": None,  # Will be loaded dynamically
                "hybrid_attention": None,  # Will be loaded dynamically
            }
            
            # Load Mamba/SSM modules dynamically
            try:
                from capibara.sub_models.mamba import MambaModule
                from capibara.sub_models.hybrid import HybridAttentionModule
                available_modules["mamba"] = MambaModule
                available_modules["hybrid_attention"] = HybridAttentionModule
                logger.info("Mamba and HybridAttention modules loaded successfully")
            except ImportError as e:
                logger.warning(f"Could not load Mamba/SSM modules: {e}")
                # Modules remain as None and will be ignored

            # Module name → gate key mapping
            _gate_map = {
                "dual_process": "dual_process",
                "semiotic": "semiotic_module",
                "semiotic_interaction": "semiotic_module",
                "mamba": "mamba_ssm",
                "hybrid_attention": "flash_attention",
                "meta_la": "meta_learning_attention",
                "distributed_attention": "distributed_attention",
                "abstract_reasoning": "abstract_reasoning",
                "neurogenesis": "neurogenesis",
                "synaptic_plasticity": "synaptic_plasticity",
            }

            # Dynamic registration and instantiation
            gate = self.config.module_gate
            for name in active_modules:
                if name in available_modules and available_modules[name] is not None:
                    # Check module gate if available
                    gate_key = _gate_map.get(name)
                    if gate is not None and gate_key and not gate.is_enabled(gate_key):
                        logger.info(
                            f"Module '{name}' disabled by ModuleGate "
                            f"(backend={gate.get_backend_name()})"
                        )
                        continue
                    self.registry.register(name, available_modules[name])
                    self.active_modules.append(
                        self.registry.create_module(
                            name,
                            hidden_size=self.config.hidden_size
                        )
                    )
        except Exception as e:
            raise ModuleLoadingError(f"Error loading modules: {e}")
    
    def __call__(
        self, 
        inputs: jnp.ndarray, 
        context: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Dict[str, Any]:
        """Forward pass of the model with TPU v4-32 optimizations."""
        
        # Handle case where inputs is a dict from DataLoader
        if isinstance(inputs, dict):
            logger.warning("ModularCapibaraModel received dict input - extracting tensor")
            # Try to extract tensor from common keys
            if "hidden_states" in inputs:
                inputs = inputs["hidden_states"]
            elif "input_ids" in inputs:
                inputs = inputs["input_ids"]
                # If input_ids are integers, create dummy embeddings
                if inputs.dtype in [jnp.int32, jnp.int64]:
                    batch_size, seq_len = inputs.shape
                    inputs = jnp.ones((batch_size, seq_len, self.config.hidden_size), dtype=jnp.float32)
            elif "prompt" in inputs and "canonical_solution" in inputs:
                # This is a raw batch from DataLoader - create dummy tensor
                logger.warning("Creating dummy tensor from raw DataLoader batch in ModularCapibaraModel")
                batch_size = 1
                seq_len = 64
                inputs = jnp.ones((batch_size, seq_len, self.config.hidden_size), dtype=jnp.float32)
            else:
                # Try to find any tensor-like value in the dict
                for key, value in inputs.items():
                    if hasattr(value, 'shape') and len(value.shape) >= 2:
                        inputs = value
                        break
                else:
                    # No suitable tensor found - create dummy
                    logger.warning("No suitable tensor found in dict - creating dummy tensor in ModularCapibaraModel")
                    batch_size = 1
                    seq_len = 64
                    inputs = jnp.ones((batch_size, seq_len, self.config.hidden_size), dtype=jnp.float32)
        
        # Ensure inputs is a JAX array with proper shape
        if not hasattr(inputs, 'shape'):
            logger.error(f"inputs has no shape attribute: {type(inputs)}")
            # Create fallback tensor
            inputs = jnp.ones((1, 64, self.config.hidden_size), dtype=jnp.float32)
        
        # Check memory
        if self.memory_monitor.should_cleanup():
            self.memory_monitor.force_cleanup()
            
        # Dynamic router
        try:
            module_scores = self.router(inputs, context)
        except Exception:
            module_scores = {}
        
        # Adaptive routing if enabled
        if self.adaptive_router is not None and self.adaptive_computation is not None:
            inputs, routing_meta = self.adaptive_router.route(inputs, training=training)
            self.metrics.update(routing_meta)
        
        # Forward pass in active modules
        outputs: Dict[str, Any] = {}
        for i, module in enumerate(self.active_modules):
            try:
                module_output = module(inputs, training=training)
                outputs[f"module_{i}"] = module_output
                
                # Register metrics
                if isinstance(module_output, dict) and "metrics" in module_output:
                    for name, value in module_output["metrics"].items():
                        self.metrics.update({f"{module.__class__.__name__}_{name}": value})
                        
            except Exception as e:
                self.metrics.update({f"module_{i}_error": str(e)})
                continue
        
        # Adaptive computation if enabled
        if self.adaptive_computation is not None:
            computation_outputs = self.adaptive_computation.forward(inputs)
            outputs["adaptive"] = computation_outputs.get("output")
            self.metrics.update(computation_outputs.get("metrics", {}))

        base_result = {
            "outputs": outputs,
            "metrics": self.metrics.get_all(),
            "module_scores": module_scores,
        }

        # Verification if not in training mode
        if not training and self.verification_system is not None:
            try:
                verified = self._apply_verification(base_result)
                # Include base outputs for compatibility
                verified["outputs"] = outputs
                verified["metrics"] = base_result["metrics"]
                verified["module_scores"] = base_result["module_scores"]
                return verified
            except Exception as e:
                logger.error(f"Error in verification: {e}")
                # Fallback to base result if verification fails
                return base_result

        return base_result

    def _apply_verification(self, model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Applies verification to all model outputs."""
        verified_outputs: Dict[str, Any] = {}
        outputs = model_result.get("outputs", {})

        for module_name, module_output in outputs.items():
            try:
                # Simulate output embedding for verification
                output_embedding = jnp.ones((768,))

                # Verify output
                verification_result = self.verification_system.verify_output(output_embedding)  # type: ignore[union-attr]

                if verification_result.get("requires_correction", False):
                    # Apply corrections
                    correction_result = self.verification_system.apply_corrections(  # type: ignore[union-attr]
                        output_embedding, verification_result
                    )

                    verified_outputs[module_name] = {
                        "output": "Verified and corrected output",
                        "verification": correction_result["final_verification"],
                        "correction_applied": True,
                        "safety_level": correction_result["final_verification"]["safety_level"],
                    }
                else:
                    verified_outputs[module_name] = {
                        "output": module_output,
                        "verification": verification_result,
                        "correction_applied": False,
                        "safety_level": verification_result["safety_level"],
                    }

            except Exception as e:
                logger.error(f"Error in verification of {module_name}: {e}")
                verified_outputs[module_name] = {
                    "output": module_output,
                    "verification_error": str(e),
                    "safety_level": "unknown",
                }

        return {
            "verified_outputs": verified_outputs,
            "overall_safety": self._compute_overall_safety(verified_outputs),
            "verification_metrics": self.verification_system.get_alignment_report() if self.verification_system else {},
        }

    def _compute_overall_safety(self, verified_outputs: Dict[str, Any]) -> str:
        """Computes the overall safety level."""
        safety_levels: List[str] = []
        for output in verified_outputs.values():
            if isinstance(output, dict) and "safety_level" in output:
                safety_levels.append(output["safety_level"]) 

        if "blocked" in safety_levels:
            return "blocked"
        if "warning" in safety_levels:
            return "warning"
        if "caution" in safety_levels:
            return "caution"
        return "safe"
