"""
Tests for Backend-aware Module Gating System.

Verifies that modules are correctly enabled/disabled per backend
and that overrides work as expected.
"""

import pytest
from core.backends.module_gate import (
    ModuleGate,
    ModuleName,
    BackendCapabilities,
    CPU_CAPABILITIES,
    GPU_CAPABILITIES,
    TPU_CAPABILITIES,
)


class TestBackendCapabilities:
    """Test the pre-defined capability profiles."""

    def test_cpu_has_no_jit(self):
        assert CPU_CAPABILITIES.supports_jit is False

    def test_cpu_has_no_flash_attention(self):
        assert CPU_CAPABILITIES.supports_flash_attention is False

    def test_cpu_has_no_distributed(self):
        assert CPU_CAPABILITIES.supports_distributed is False

    def test_gpu_has_jit(self):
        assert GPU_CAPABILITIES.supports_jit is True

    def test_gpu_has_flash_attention(self):
        assert GPU_CAPABILITIES.supports_flash_attention is True

    def test_gpu_has_torch_compile(self):
        assert GPU_CAPABILITIES.supports_torch_compile is True

    def test_gpu_no_xla(self):
        assert GPU_CAPABILITIES.supports_xla is False

    def test_tpu_has_xla(self):
        assert TPU_CAPABILITIES.supports_xla is True

    def test_tpu_has_scan(self):
        assert TPU_CAPABILITIES.supports_scan is True

    def test_tpu_no_flash_attention(self):
        assert TPU_CAPABILITIES.supports_flash_attention is False

    def test_tpu_no_torch_compile(self):
        assert TPU_CAPABILITIES.supports_torch_compile is False


class TestModuleGateCPU:
    """Test module gating for CPU backend."""

    @pytest.fixture
    def gate(self):
        return ModuleGate.from_backend("cpu")

    def test_flash_attention_disabled(self, gate):
        assert gate.is_enabled("flash_attention") is False

    def test_jit_disabled(self, gate):
        assert gate.is_enabled("jit_compilation") is False

    def test_gradient_checkpointing_disabled(self, gate):
        assert gate.is_enabled("gradient_checkpointing") is False

    def test_mixed_precision_disabled(self, gate):
        assert gate.is_enabled("mixed_precision") is False

    def test_distributed_disabled(self, gate):
        assert gate.is_enabled("data_parallelism") is False

    def test_moe_routing_enabled(self, gate):
        """MoE routing has no special requirements, works on CPU."""
        assert gate.is_enabled("moe_routing") is True

    def test_vq_compression_enabled(self, gate):
        assert gate.is_enabled("vq_compression") is True

    def test_adaptive_routing_enabled(self, gate):
        assert gate.is_enabled("adaptive_routing") is True

    def test_semiotic_enabled(self, gate):
        assert gate.is_enabled("semiotic_module") is True

    def test_ssm_disabled(self, gate):
        """SSM modules require JIT, disabled on CPU."""
        assert gate.is_enabled("spike_ssm") is False
        assert gate.is_enabled("mamba_ssm") is False

    def test_many_modules_disabled(self, gate):
        disabled = gate.get_disabled_modules()
        assert len(disabled) > 5

    def test_backend_name(self, gate):
        assert gate.get_backend_name() == "cpu"


class TestModuleGateGPU:
    """Test module gating for GPU backend."""

    @pytest.fixture
    def gate(self):
        return ModuleGate.from_backend("gpu")

    def test_flash_attention_enabled(self, gate):
        assert gate.is_enabled("flash_attention") is True

    def test_torch_compile_enabled(self, gate):
        assert gate.is_enabled("torch_compile") is True

    def test_jit_enabled(self, gate):
        assert gate.is_enabled("jit_compilation") is True

    def test_mixed_precision_enabled(self, gate):
        assert gate.is_enabled("mixed_precision") is True

    def test_gradient_checkpointing_enabled(self, gate):
        assert gate.is_enabled("gradient_checkpointing") is True

    def test_distributed_enabled(self, gate):
        assert gate.is_enabled("data_parallelism") is True

    def test_xla_disabled(self, gate):
        """XLA is TPU-only."""
        assert gate.is_enabled("xla_optimization") is False

    def test_more_enabled_than_cpu(self, gate):
        cpu_gate = ModuleGate.from_backend("cpu")
        assert len(gate.get_enabled_modules()) > len(cpu_gate.get_enabled_modules())


class TestModuleGateTPU:
    """Test module gating for TPU backend."""

    @pytest.fixture
    def gate(self):
        return ModuleGate.from_backend("tpu")

    def test_xla_enabled(self, gate):
        assert gate.is_enabled("xla_optimization") is True

    def test_jit_enabled(self, gate):
        assert gate.is_enabled("jit_compilation") is True

    def test_flash_attention_disabled(self, gate):
        """TPU uses XLA-optimized attention, not Flash Attention."""
        assert gate.is_enabled("flash_attention") is False

    def test_torch_compile_disabled(self, gate):
        assert gate.is_enabled("torch_compile") is False

    def test_ssm_enabled(self, gate):
        assert gate.is_enabled("spike_ssm") is True
        assert gate.is_enabled("mamba_ssm") is True


class TestModuleGateOverrides:
    """Test manual override functionality."""

    def test_force_enable_on_cpu(self):
        gate = ModuleGate.from_backend("cpu", overrides={"flash_attention": True})
        assert gate.is_enabled("flash_attention") is True

    def test_force_disable_on_gpu(self):
        gate = ModuleGate.from_backend("gpu", overrides={"flash_attention": False})
        assert gate.is_enabled("flash_attention") is False

    def test_override_does_not_affect_others(self):
        gate = ModuleGate.from_backend("cpu", overrides={"flash_attention": True})
        # JIT still disabled (no override for it)
        assert gate.is_enabled("jit_compilation") is False

    def test_multiple_overrides(self):
        gate = ModuleGate.from_backend("cpu", overrides={
            "flash_attention": True,
            "jit_compilation": True,
            "moe_routing": False,
        })
        assert gate.is_enabled("flash_attention") is True
        assert gate.is_enabled("jit_compilation") is True
        assert gate.is_enabled("moe_routing") is False


class TestModuleGateAPI:
    """Test the ModuleGate API."""

    def test_from_backend_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            ModuleGate.from_backend("quantum")

    def test_get_status_returns_dict(self):
        gate = ModuleGate.from_backend("cpu")
        status = gate.get_status()
        assert isinstance(status, dict)
        assert len(status) == len(ModuleName)

    def test_summary_contains_backend(self):
        gate = ModuleGate.from_backend("gpu")
        summary = gate.summary()
        assert "GPU" in summary

    def test_repr(self):
        gate = ModuleGate.from_backend("cpu")
        r = repr(gate)
        assert "cpu" in r
        assert "enabled=" in r

    def test_get_enabled_and_disabled_partition(self):
        gate = ModuleGate.from_backend("gpu")
        enabled = set(gate.get_enabled_modules())
        disabled = set(gate.get_disabled_modules())
        assert len(enabled & disabled) == 0
        assert len(enabled) + len(disabled) == len(ModuleName)

    def test_from_backend_instance(self):
        """Test creating gate from a mock backend object."""
        class MockBackend:
            name = "cpu"
        gate = ModuleGate.from_backend_instance(MockBackend())
        assert gate.get_backend_name() == "cpu"

    def test_from_backend_instance_callable_name(self):
        class MockBackend:
            def name(self):
                return "gpu"
        gate = ModuleGate.from_backend_instance(MockBackend())
        assert gate.get_backend_name() == "gpu"


class TestCustomCapabilities:
    """Test with custom BackendCapabilities."""

    def test_minimal_capabilities(self):
        caps = BackendCapabilities(name="minimal")
        gate = ModuleGate(caps)
        # Only no-requirement modules should be enabled
        assert gate.is_enabled("moe_routing") is True
        assert gate.is_enabled("flash_attention") is False

    def test_full_capabilities(self):
        caps = BackendCapabilities(
            name="supercomputer",
            supports_jit=True,
            supports_autograd=True,
            supports_distributed=True,
            supports_mixed_precision=True,
            supports_flash_attention=True,
            supports_gradient_checkpointing=True,
            supports_xla=True,
            supports_torch_compile=True,
            supports_scan=True,
            supports_vmap=True,
            supports_pmap=True,
        )
        gate = ModuleGate(caps)
        # Everything should be enabled
        assert len(gate.get_disabled_modules()) == 0


class TestBackendComparison:
    """Compare module counts across backends."""

    def test_gpu_enables_more_than_cpu(self):
        cpu = ModuleGate.from_backend("cpu")
        gpu = ModuleGate.from_backend("gpu")
        assert len(gpu.get_enabled_modules()) > len(cpu.get_enabled_modules())

    def test_tpu_enables_more_than_cpu(self):
        cpu = ModuleGate.from_backend("cpu")
        tpu = ModuleGate.from_backend("tpu")
        assert len(tpu.get_enabled_modules()) > len(cpu.get_enabled_modules())

    def test_gpu_and_tpu_differ(self):
        gpu = ModuleGate.from_backend("gpu")
        tpu = ModuleGate.from_backend("tpu")
        gpu_set = set(gpu.get_enabled_modules())
        tpu_set = set(tpu.get_enabled_modules())
        # They should have different sets (flash_attention vs xla, etc.)
        assert gpu_set != tpu_set

    def test_cpu_is_subset_of_gpu(self):
        """All modules enabled on CPU should also be enabled on GPU."""
        cpu = ModuleGate.from_backend("cpu")
        gpu = ModuleGate.from_backend("gpu")
        cpu_enabled = set(cpu.get_enabled_modules())
        gpu_enabled = set(gpu.get_enabled_modules())
        assert cpu_enabled.issubset(gpu_enabled)
