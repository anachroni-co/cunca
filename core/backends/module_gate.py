"""
Backend-aware Module Gating System for CapibaraGPT.

Enables/disables model modules based on the active compute backend,
saving unnecessary computation on backends that cannot benefit from
certain modules or where they add overhead without value.

Usage:
    from core.backends.module_gate import ModuleGate, BackendCapabilities

    gate = ModuleGate.from_backend("cpu")
    if gate.is_enabled("flash_attention"):
        # run flash attention path
    else:
        # run standard attention path

    # Get all disabled modules for logging
    disabled = gate.get_disabled_modules()
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class ModuleName(str, Enum):
    """Identifiers for gatable model modules."""
    # Attention variants
    FLASH_ATTENTION = "flash_attention"
    SPARSE_ATTENTION = "sparse_attention"
    CAUSAL_MASK_CACHE = "causal_mask_cache"

    # SSM / Mamba
    SPIKE_SSM = "spike_ssm"
    MAMBA_SSM = "mamba_ssm"
    S4_SSM = "s4_ssm"

    # Mixture of Experts
    MOE_ROUTING = "moe_routing"
    MOE_LOAD_BALANCING = "moe_load_balancing"

    # Quantization / Compression
    VQ_COMPRESSION = "vq_compression"
    BITNET_QUANTIZATION = "bitnet_quantization"
    DYNAMIC_QUANTIZATION = "dynamic_quantization"

    # Advanced features
    ADAPTIVE_ROUTING = "adaptive_routing"
    ADAPTIVE_COMPUTATION = "adaptive_computation"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    MIXED_PRECISION = "mixed_precision"

    # Compilation
    JIT_COMPILATION = "jit_compilation"
    XLA_OPTIMIZATION = "xla_optimization"
    TORCH_COMPILE = "torch_compile"

    # Distributed
    DATA_PARALLELISM = "data_parallelism"
    MODEL_PARALLELISM = "model_parallelism"
    PIPELINE_PARALLELISM = "pipeline_parallelism"

    # Experimental
    SEMIOTIC_MODULE = "semiotic_module"
    DUAL_PROCESS = "dual_process"
    CULTURAL_ANALYSIS = "cultural_analysis"
    NEUROGENESIS = "neurogenesis"


@dataclass(frozen=True)
class BackendCapabilities:
    """Declares what a backend supports natively."""
    name: str
    supports_jit: bool = False
    supports_autograd: bool = False
    supports_distributed: bool = False
    supports_mixed_precision: bool = False
    supports_flash_attention: bool = False
    supports_gradient_checkpointing: bool = False
    supports_xla: bool = False
    supports_torch_compile: bool = False
    supports_scan: bool = False          # lax.scan / sequential ops
    supports_vmap: bool = False           # vectorized map
    supports_pmap: bool = False           # parallel map
    max_efficient_seq_len: int = 0        # 0 = no limit
    memory_gb: float = 0.0               # 0 = unknown


# Pre-defined backend capability profiles
CPU_CAPABILITIES = BackendCapabilities(
    name="cpu",
    supports_jit=False,
    supports_autograd=False,
    supports_distributed=False,
    supports_mixed_precision=False,
    supports_flash_attention=False,
    supports_gradient_checkpointing=False,
    supports_xla=False,
    supports_torch_compile=False,
    supports_scan=False,
    supports_vmap=False,
    supports_pmap=False,
    max_efficient_seq_len=512,
    memory_gb=0.0,
)

GPU_CAPABILITIES = BackendCapabilities(
    name="gpu",
    supports_jit=True,
    supports_autograd=True,
    supports_distributed=True,
    supports_mixed_precision=True,
    supports_flash_attention=True,
    supports_gradient_checkpointing=True,
    supports_xla=False,
    supports_torch_compile=True,
    supports_scan=False,
    supports_vmap=False,
    supports_pmap=False,
    max_efficient_seq_len=0,
    memory_gb=80.0,
)

TPU_CAPABILITIES = BackendCapabilities(
    name="tpu",
    supports_jit=True,
    supports_autograd=True,
    supports_distributed=True,
    supports_mixed_precision=True,
    supports_flash_attention=False,   # uses XLA-optimized attention instead
    supports_gradient_checkpointing=True,
    supports_xla=True,
    supports_torch_compile=False,
    supports_scan=True,
    supports_vmap=True,
    supports_pmap=True,
    max_efficient_seq_len=0,
    memory_gb=64.0,
)

_BACKEND_PROFILES: Dict[str, BackendCapabilities] = {
    "cpu": CPU_CAPABILITIES,
    "gpu": GPU_CAPABILITIES,
    "tpu": TPU_CAPABILITIES,
}


# Module requirements: which capabilities each module needs
_MODULE_REQUIREMENTS: Dict[ModuleName, Set[str]] = {
    # Attention
    ModuleName.FLASH_ATTENTION: {"supports_flash_attention"},
    ModuleName.SPARSE_ATTENTION: {"supports_jit"},
    ModuleName.CAUSAL_MASK_CACHE: set(),  # always available

    # SSM
    ModuleName.SPIKE_SSM: {"supports_jit"},
    ModuleName.MAMBA_SSM: {"supports_jit"},
    ModuleName.S4_SSM: {"supports_jit"},

    # MoE
    ModuleName.MOE_ROUTING: set(),  # works on all backends
    ModuleName.MOE_LOAD_BALANCING: set(),

    # Quantization
    ModuleName.VQ_COMPRESSION: set(),
    ModuleName.BITNET_QUANTIZATION: set(),
    ModuleName.DYNAMIC_QUANTIZATION: set(),

    # Advanced
    ModuleName.ADAPTIVE_ROUTING: set(),
    ModuleName.ADAPTIVE_COMPUTATION: set(),
    ModuleName.GRADIENT_CHECKPOINTING: {"supports_gradient_checkpointing"},
    ModuleName.MIXED_PRECISION: {"supports_mixed_precision"},

    # Compilation
    ModuleName.JIT_COMPILATION: {"supports_jit"},
    ModuleName.XLA_OPTIMIZATION: {"supports_xla"},
    ModuleName.TORCH_COMPILE: {"supports_torch_compile"},

    # Distributed
    ModuleName.DATA_PARALLELISM: {"supports_distributed"},
    ModuleName.MODEL_PARALLELISM: {"supports_distributed"},
    ModuleName.PIPELINE_PARALLELISM: {"supports_distributed"},

    # Experimental
    ModuleName.SEMIOTIC_MODULE: set(),
    ModuleName.DUAL_PROCESS: set(),
    ModuleName.CULTURAL_ANALYSIS: set(),
    ModuleName.NEUROGENESIS: {"supports_jit"},
}


class ModuleGate:
    """
    Central gate that decides which modules are enabled for the active backend.

    Checks backend capabilities against module requirements and allows
    manual overrides for testing or special configurations.
    """

    def __init__(
        self,
        capabilities: BackendCapabilities,
        overrides: Optional[Dict[str, bool]] = None,
    ):
        self._capabilities = capabilities
        self._overrides: Dict[str, bool] = overrides or {}
        self._resolved: Dict[str, bool] = {}
        self._resolve_all()

    @classmethod
    def from_backend(
        cls,
        backend_name: str,
        overrides: Optional[Dict[str, bool]] = None,
    ) -> "ModuleGate":
        """Create a gate from a backend name ('cpu', 'gpu', 'tpu')."""
        name = backend_name.lower()
        capabilities = _BACKEND_PROFILES.get(name)
        if capabilities is None:
            raise ValueError(
                f"Unknown backend '{backend_name}'. "
                f"Available: {list(_BACKEND_PROFILES.keys())}"
            )
        return cls(capabilities, overrides)

    @classmethod
    def from_backend_instance(
        cls,
        backend,
        overrides: Optional[Dict[str, bool]] = None,
    ) -> "ModuleGate":
        """Create a gate from a ComputeBackend instance."""
        name = getattr(backend, "name", "cpu")
        if callable(name):
            name = name()
        return cls.from_backend(str(name), overrides)

    def _resolve_all(self) -> None:
        """Resolve enabled/disabled state for every module."""
        caps = self._capabilities
        cap_dict = {
            "supports_jit": caps.supports_jit,
            "supports_autograd": caps.supports_autograd,
            "supports_distributed": caps.supports_distributed,
            "supports_mixed_precision": caps.supports_mixed_precision,
            "supports_flash_attention": caps.supports_flash_attention,
            "supports_gradient_checkpointing": caps.supports_gradient_checkpointing,
            "supports_xla": caps.supports_xla,
            "supports_torch_compile": caps.supports_torch_compile,
            "supports_scan": caps.supports_scan,
            "supports_vmap": caps.supports_vmap,
            "supports_pmap": caps.supports_pmap,
        }

        for module, requirements in _MODULE_REQUIREMENTS.items():
            key = module.value
            # Manual override takes priority
            if key in self._overrides:
                self._resolved[key] = self._overrides[key]
            else:
                # All required capabilities must be present
                self._resolved[key] = all(
                    cap_dict.get(req, False) for req in requirements
                )

    def is_enabled(self, module: str) -> bool:
        """Check if a module is enabled for the current backend."""
        return self._resolved.get(module, False)

    def get_enabled_modules(self) -> List[str]:
        """Return list of all enabled module names."""
        return sorted(k for k, v in self._resolved.items() if v)

    def get_disabled_modules(self) -> List[str]:
        """Return list of all disabled module names."""
        return sorted(k for k, v in self._resolved.items() if not v)

    def get_status(self) -> Dict[str, bool]:
        """Return full module status dict."""
        return dict(self._resolved)

    def get_backend_name(self) -> str:
        """Return the backend name."""
        return self._capabilities.name

    def summary(self) -> str:
        """Human-readable summary of module status."""
        enabled = self.get_enabled_modules()
        disabled = self.get_disabled_modules()
        lines = [
            f"ModuleGate [{self._capabilities.name.upper()}]",
            f"  Enabled  ({len(enabled)}): {', '.join(enabled) or 'none'}",
            f"  Disabled ({len(disabled)}): {', '.join(disabled) or 'none'}",
        ]
        if self._overrides:
            lines.append(f"  Overrides: {self._overrides}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ModuleGate(backend={self._capabilities.name!r}, "
            f"enabled={len(self.get_enabled_modules())}, "
            f"disabled={len(self.get_disabled_modules())})"
        )
