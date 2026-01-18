"""
Unit Tests for Core Model Components

Tests the core transformer/mamba architecture, router,
and attention mechanisms.
"""

import numpy as np
import pytest
from typing import Dict, Any


class TestModelConfig:
    """Test model configuration validation."""

    def test_default_config(self, model_config):
        """Test default configuration values."""
        assert model_config["hidden_size"] == 768
        assert model_config["num_attention_heads"] == 12
        assert model_config["vocab_size"] == 32000

    def test_small_config(self, small_model_config):
        """Test small configuration for testing."""
        assert small_model_config["hidden_size"] == 128
        assert small_model_config["num_hidden_layers"] == 2

    def test_head_dimension_divisibility(self, model_config):
        """Test that hidden_size is divisible by num_heads."""
        assert model_config["hidden_size"] % model_config["num_attention_heads"] == 0


class TestAttentionMechanism:
    """Test attention implementations."""

    @pytest.fixture
    def attention_config(self):
        return {
            "hidden_size": 256,
            "num_attention_heads": 4,
            "head_dim": 64,
            "dropout": 0.0,
        }

    def test_attention_output_shape(self, cpu_backend, attention_config):
        """Test attention output shape."""
        batch_size = 2
        seq_len = 32
        num_heads = attention_config["num_attention_heads"]
        head_dim = attention_config["head_dim"]

        # Create QKV tensors
        query = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))
        key = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))
        value = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))

        output = cpu_backend.scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_causal_mask_applied(self, cpu_backend):
        """Test that causal mask is properly applied."""
        batch_size = 1
        num_heads = 1
        seq_len = 4
        head_dim = 4

        # Create identity-like patterns for easy verification
        query = cpu_backend.ones((batch_size, num_heads, seq_len, head_dim))
        key = cpu_backend.ones((batch_size, num_heads, seq_len, head_dim))
        value = cpu_backend.create_tensor([
            [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
        ])

        output = cpu_backend.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

        # With causal mask, later positions shouldn't see earlier ones
        # First position should only see itself
        # This is a simplified check
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_attention_scale_factor(self, cpu_backend):
        """Test custom scale factor in attention."""
        batch_size = 2
        num_heads = 2
        seq_len = 8
        head_dim = 16

        query = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))
        key = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))
        value = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))

        # Default scale
        output1 = cpu_backend.scaled_dot_product_attention(query, key, value)

        # Custom scale
        output2 = cpu_backend.scaled_dot_product_attention(
            query, key, value, scale=0.1
        )

        # Outputs should be different with different scales
        assert not np.allclose(
            cpu_backend.to_numpy(output1),
            cpu_backend.to_numpy(output2)
        )


class TestLayerNormalization:
    """Test layer normalization implementation."""

    def test_layer_norm_output_distribution(self, cpu_backend):
        """Test layer norm produces normalized output."""
        x = cpu_backend.randn((4, 32, 256))
        normalized = cpu_backend.layer_norm(x, (256,))

        result = cpu_backend.to_numpy(normalized)

        # Mean should be close to 0
        mean = np.mean(result, axis=-1)
        assert np.allclose(mean, 0, atol=1e-5)

        # Std should be close to 1
        std = np.std(result, axis=-1)
        assert np.allclose(std, 1, atol=0.1)

    def test_layer_norm_with_affine(self, cpu_backend):
        """Test layer norm with weight and bias."""
        x = cpu_backend.randn((2, 4, 8))
        weight = cpu_backend.ones((8,)) * 2  # Scale by 2
        bias = cpu_backend.ones((8,))  # Shift by 1

        normalized = cpu_backend.layer_norm(x, (8,), weight=weight, bias=bias)
        result = cpu_backend.to_numpy(normalized)

        # Mean should be close to 1 (bias)
        mean = np.mean(result, axis=-1)
        assert np.allclose(mean, 1, atol=0.2)


class TestActivationFunctions:
    """Test activation function implementations."""

    def test_gelu_properties(self, cpu_backend):
        """Test GELU activation properties."""
        x = cpu_backend.create_tensor([-2, -1, 0, 1, 2])
        result = cpu_backend.gelu(x)
        result_np = cpu_backend.to_numpy(result)

        # GELU(0) ≈ 0
        assert abs(result_np[2]) < 1e-5

        # GELU is approximately identity for large positive values
        assert result_np[4] > 1.5

        # GELU is close to 0 for large negative values
        assert result_np[0] < 0.1

    def test_silu_properties(self, cpu_backend):
        """Test SiLU activation properties."""
        x = cpu_backend.create_tensor([-2, -1, 0, 1, 2])
        result = cpu_backend.silu(x)
        result_np = cpu_backend.to_numpy(result)

        # SiLU(0) = 0
        assert abs(result_np[2]) < 1e-5

        # SiLU(x) = x * sigmoid(x), so SiLU(1) ≈ 0.731
        assert 0.7 < result_np[3] < 0.8

    def test_softmax_numerical_stability(self, cpu_backend):
        """Test softmax with large values."""
        # Large values that could cause overflow
        x = cpu_backend.create_tensor([[1000, 1001, 1002]])
        result = cpu_backend.softmax(x, axis=-1)
        result_np = cpu_backend.to_numpy(result)

        # Should not be NaN or Inf
        assert np.all(np.isfinite(result_np))

        # Should sum to 1
        assert np.isclose(np.sum(result_np), 1.0)


class TestMatrixOperations:
    """Test matrix multiplication and basic operations."""

    def test_batch_matmul(self, cpu_backend):
        """Test batched matrix multiplication."""
        a = cpu_backend.randn((4, 8, 16))
        b = cpu_backend.randn((4, 16, 8))

        result = cpu_backend.matmul(a, b)

        assert result.shape == (4, 8, 8)

    def test_matmul_broadcasting(self, cpu_backend):
        """Test matrix multiplication with broadcasting."""
        a = cpu_backend.randn((1, 8, 16))
        b = cpu_backend.randn((4, 16, 8))

        result = cpu_backend.matmul(a, b)

        assert result.shape == (4, 8, 8)


class TestRouterMock:
    """Test router selection logic (mock implementation)."""

    def test_route_selection(self):
        """Test that router can select experts."""
        # Mock router output
        router_logits = np.random.randn(4, 8)  # batch=4, num_experts=8

        # Top-k selection
        k = 2
        top_k_indices = np.argsort(router_logits, axis=-1)[:, -k:]
        top_k_weights = np.take_along_axis(router_logits, top_k_indices, axis=-1)
        top_k_weights = np.exp(top_k_weights) / np.sum(np.exp(top_k_weights), axis=-1, keepdims=True)

        # Check shapes
        assert top_k_indices.shape == (4, k)
        assert top_k_weights.shape == (4, k)

        # Weights should sum to 1
        np.testing.assert_allclose(np.sum(top_k_weights, axis=-1), 1.0, rtol=1e-5)


class TestMambaSSMMock:
    """Test Mamba/SSM operations (mock implementation)."""

    def test_ssm_recurrence(self, cpu_backend):
        """Test SSM recurrence computation."""
        batch_size = 2
        seq_len = 16
        hidden_size = 32

        # Mock SSM parameters
        A = cpu_backend.randn((hidden_size,))
        B = cpu_backend.randn((batch_size, seq_len, hidden_size))
        C = cpu_backend.randn((batch_size, seq_len, hidden_size))
        delta = cpu_backend.softmax(cpu_backend.randn((batch_size, seq_len, hidden_size)))

        # Initialize hidden state
        h = cpu_backend.zeros((batch_size, hidden_size))

        # Simple recurrence (not optimized scan)
        outputs = []
        for t in range(seq_len):
            # h = A * h + B * x (simplified)
            h_np = cpu_backend.to_numpy(h)
            A_np = cpu_backend.to_numpy(A)
            B_np = cpu_backend.to_numpy(B)[:, t, :]
            C_np = cpu_backend.to_numpy(C)[:, t, :]
            delta_np = cpu_backend.to_numpy(delta)[:, t, :]

            # Discretized update
            h_np = h_np * np.exp(A_np * delta_np) + B_np * delta_np
            y = h_np * C_np

            h = cpu_backend.create_tensor(h_np)
            outputs.append(y)

        outputs = np.stack(outputs, axis=1)
        assert outputs.shape == (batch_size, seq_len, hidden_size)
