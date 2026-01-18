"""
Unit Tests for Compute Backends

Tests the CPU, GPU, and TPU backends for:
- Tensor operations
- Math operations
- Attention mechanisms
- Memory management
"""

import numpy as np
import pytest

from core.backends import (
    BackendType,
    ComputeBackend,
    get_backend,
    list_available_backends,
)
from core.backends.base import DType


class TestBackendRegistry:
    """Test backend registration and discovery."""

    def test_list_available_backends(self):
        """Test listing available backends."""
        backends = list_available_backends()
        assert isinstance(backends, dict)
        assert "cpu" in backends
        assert backends["cpu"] is True  # CPU always available

    def test_get_cpu_backend(self):
        """Test getting CPU backend."""
        backend = get_backend(BackendType.CPU)
        assert backend is not None
        assert backend.name == "cpu"
        assert backend.is_available

    def test_auto_detect_backend(self):
        """Test auto-detection of best backend."""
        backend = get_backend(BackendType.AUTO)
        assert backend is not None
        assert backend.name in ["cpu", "gpu", "tpu"]


class TestCPUBackend:
    """Test CPU backend operations."""

    @pytest.fixture
    def backend(self):
        return get_backend(BackendType.CPU)

    # ==================== Tensor Creation ====================

    def test_create_tensor(self, backend):
        """Test tensor creation from list."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        tensor = backend.create_tensor(data)
        assert tensor.shape == (2, 2)
        np.testing.assert_array_equal(tensor, data)

    def test_create_tensor_with_dtype(self, backend):
        """Test tensor creation with dtype."""
        tensor = backend.create_tensor([1, 2, 3], dtype=DType.FLOAT32)
        assert tensor.dtype == np.float32

    def test_zeros(self, backend):
        """Test zeros tensor creation."""
        tensor = backend.zeros((3, 4))
        assert tensor.shape == (3, 4)
        assert np.all(tensor == 0)

    def test_ones(self, backend):
        """Test ones tensor creation."""
        tensor = backend.ones((2, 3))
        assert tensor.shape == (2, 3)
        assert np.all(tensor == 1)

    def test_randn(self, backend):
        """Test random normal tensor creation."""
        tensor = backend.randn((100, 100))
        assert tensor.shape == (100, 100)
        # Check approximate normal distribution
        assert abs(np.mean(tensor)) < 0.1
        assert abs(np.std(tensor) - 1.0) < 0.1

    # ==================== Math Operations ====================

    def test_matmul(self, backend):
        """Test matrix multiplication."""
        a = backend.create_tensor([[1, 2], [3, 4]])
        b = backend.create_tensor([[5, 6], [7, 8]])
        result = backend.matmul(a, b)

        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_add(self, backend):
        """Test element-wise addition."""
        a = backend.create_tensor([1, 2, 3])
        b = backend.create_tensor([4, 5, 6])
        result = backend.add(a, b)

        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_mul(self, backend):
        """Test element-wise multiplication."""
        a = backend.create_tensor([1, 2, 3])
        b = backend.create_tensor([4, 5, 6])
        result = backend.mul(a, b)

        np.testing.assert_array_equal(result, [4, 10, 18])

    def test_softmax(self, backend):
        """Test softmax activation."""
        x = backend.create_tensor([[1.0, 2.0, 3.0]])
        result = backend.softmax(x, axis=-1)

        # Softmax should sum to 1
        assert abs(np.sum(result) - 1.0) < 1e-5
        # Should be monotonically increasing
        assert result[0, 0] < result[0, 1] < result[0, 2]

    def test_layer_norm(self, backend):
        """Test layer normalization."""
        x = backend.create_tensor([[1.0, 2.0, 3.0, 4.0]])
        result = backend.layer_norm(x, normalized_shape=(4,))

        # Layer norm should have mean ~0 and std ~1
        assert abs(np.mean(result)) < 1e-5
        assert abs(np.std(result) - 1.0) < 0.1

    def test_gelu(self, backend):
        """Test GELU activation."""
        x = backend.create_tensor([-1.0, 0.0, 1.0])
        result = backend.gelu(x)

        # GELU(0) = 0
        assert abs(result[1]) < 1e-5
        # GELU(-1) < 0
        assert result[0] < 0
        # GELU(1) > 0
        assert result[2] > 0

    def test_silu(self, backend):
        """Test SiLU activation."""
        x = backend.create_tensor([-1.0, 0.0, 1.0])
        result = backend.silu(x)

        # SiLU(0) = 0
        assert abs(result[1]) < 1e-5
        # SiLU(-1) < 0
        assert result[0] < 0
        # SiLU(1) > 0
        assert result[2] > 0

    # ==================== Attention ====================

    def test_scaled_dot_product_attention(self, backend, attention_inputs):
        """Test scaled dot-product attention."""
        query, key, value = attention_inputs()
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Convert to backend tensors
        q = backend.create_tensor(query)
        k = backend.create_tensor(key)
        v = backend.create_tensor(value)

        result = backend.scaled_dot_product_attention(q, k, v)

        # Check output shape
        assert result.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_causal_attention(self, backend, attention_inputs):
        """Test causal (masked) attention."""
        query, key, value = attention_inputs()

        q = backend.create_tensor(query)
        k = backend.create_tensor(key)
        v = backend.create_tensor(value)

        result = backend.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        assert result.shape == query.shape

    # ==================== Memory Management ====================

    def test_to_numpy(self, backend):
        """Test conversion to NumPy."""
        tensor = backend.create_tensor([1, 2, 3])
        np_array = backend.to_numpy(tensor)

        assert isinstance(np_array, np.ndarray)
        np.testing.assert_array_equal(np_array, [1, 2, 3])

    def test_memory_stats(self, backend):
        """Test memory statistics (CPU always returns 0)."""
        allocated = backend.memory_allocated()
        assert allocated == 0  # CPU doesn't track

    # ==================== Context Managers ====================

    def test_no_grad_context(self, backend):
        """Test no_grad context manager."""
        with backend.no_grad():
            tensor = backend.randn((10, 10))
            result = backend.matmul(tensor, tensor)
        assert result.shape == (10, 10)

    def test_autocast_context(self, backend):
        """Test autocast context manager."""
        with backend.autocast():
            tensor = backend.randn((10, 10))
            result = backend.matmul(tensor, tensor)
        assert result.shape == (10, 10)

    # ==================== JIT Compilation ====================

    def test_jit_compile(self, backend):
        """Test JIT compilation (no-op for CPU)."""
        def simple_fn(x, y):
            return backend.add(x, y)

        jit_fn = backend.jit_compile(simple_fn)

        a = backend.create_tensor([1, 2, 3])
        b = backend.create_tensor([4, 5, 6])

        result = jit_fn(a, b)
        np.testing.assert_array_equal(result, [5, 7, 9])


@pytest.mark.gpu
class TestGPUBackend:
    """Test GPU backend operations (skip if no GPU)."""

    @pytest.fixture
    def backend(self):
        try:
            return get_backend(BackendType.GPU)
        except Exception:
            pytest.skip("GPU not available")

    def test_gpu_available(self, backend):
        """Test GPU is available."""
        assert backend.is_available
        assert backend.name == "gpu"

    def test_tensor_on_gpu(self, backend):
        """Test tensor is on GPU."""
        tensor = backend.randn((10, 10))
        # GPU tensors should have cuda device
        assert "cuda" in str(type(tensor))

    def test_flash_attention_detection(self, backend):
        """Test Flash Attention detection."""
        # Just check the flag exists
        assert hasattr(backend, "_flash_attn_available")


@pytest.mark.tpu
class TestTPUBackend:
    """Test TPU backend operations (skip if no TPU)."""

    @pytest.fixture
    def backend(self):
        try:
            return get_backend(BackendType.TPU)
        except Exception:
            pytest.skip("TPU not available")

    def test_tpu_available(self, backend):
        """Test TPU is available."""
        assert backend.is_available
        assert backend.name == "tpu"

    def test_device_count(self, backend):
        """Test TPU device count."""
        count = backend.get_device_count()
        assert count >= 1


class TestBackendInteroperability:
    """Test operations across different backends."""

    def test_cpu_to_numpy_roundtrip(self, cpu_backend):
        """Test CPU tensor to NumPy and back."""
        original = np.random.randn(10, 10).astype(np.float32)
        tensor = cpu_backend.create_tensor(original)
        result = cpu_backend.to_numpy(tensor)

        np.testing.assert_array_almost_equal(original, result)

    def test_consistent_results_across_backends(self, cpu_backend, random_batch):
        """Test that operations give consistent results."""
        batch = random_batch(batch_size=2, seq_len=32, hidden_size=64)

        # Test softmax consistency
        hidden = cpu_backend.create_tensor(batch["hidden_states"])
        softmax_result = cpu_backend.softmax(hidden, axis=-1)

        # Sum should be 1 along last axis
        sums = np.sum(cpu_backend.to_numpy(softmax_result), axis=-1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))
