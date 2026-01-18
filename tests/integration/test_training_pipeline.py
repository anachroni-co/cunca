"""
Integration Tests for Training Pipeline

Tests end-to-end training with:
- Synthetic data
- Backend abstraction
- Gradient computation
- Checkpointing
"""

import numpy as np
import pytest
from pathlib import Path

# Import fixtures
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrainingIntegration:
    """Test training pipeline integration."""

    @pytest.fixture
    def training_config(self):
        """Training configuration for testing."""
        return {
            "batch_size": 4,
            "seq_len": 64,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 4,
            "learning_rate": 1e-4,
            "max_steps": 10,
            "vocab_size": 1000,
        }

    def test_forward_pass(self, cpu_backend, training_config):
        """Test model forward pass."""
        batch_size = training_config["batch_size"]
        seq_len = training_config["seq_len"]
        hidden_size = training_config["hidden_size"]

        # Simulate input
        hidden_states = cpu_backend.randn((batch_size, seq_len, hidden_size))

        # Simulate attention
        num_heads = training_config["num_heads"]
        head_dim = hidden_size // num_heads

        # Reshape for attention
        query = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))
        key = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))
        value = cpu_backend.randn((batch_size, num_heads, seq_len, head_dim))

        output = cpu_backend.scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_loss_computation(self, cpu_backend, training_config):
        """Test loss computation."""
        batch_size = training_config["batch_size"]
        seq_len = training_config["seq_len"]
        vocab_size = training_config["vocab_size"]

        # Simulated logits and labels
        logits = cpu_backend.randn((batch_size, seq_len, vocab_size))
        labels = np.random.randint(0, vocab_size, (batch_size, seq_len))

        # Cross entropy loss (manual computation)
        logits_np = cpu_backend.to_numpy(logits)
        probs = np.exp(logits_np) / np.sum(np.exp(logits_np), axis=-1, keepdims=True)

        # Compute loss
        losses = []
        for b in range(batch_size):
            for s in range(seq_len):
                losses.append(-np.log(probs[b, s, labels[b, s]] + 1e-10))

        loss = np.mean(losses)

        assert np.isfinite(loss)
        assert loss > 0  # Loss should be positive

    def test_gradient_flow(self, cpu_backend, training_config):
        """Test gradient computation (simulated)."""
        hidden_size = training_config["hidden_size"]

        # Simulated parameters
        weight = cpu_backend.randn((hidden_size, hidden_size))
        bias = cpu_backend.zeros((hidden_size,))

        # Simulated input
        x = cpu_backend.randn((4, 32, hidden_size))

        # Forward pass
        output = cpu_backend.matmul(x, weight)
        output = cpu_backend.add(output, bias)
        output = cpu_backend.gelu(output)

        # Loss (mean squared error to zero)
        loss = np.mean(cpu_backend.to_numpy(output) ** 2)

        assert np.isfinite(loss)

    def test_checkpoint_save_load(self, cpu_backend, temp_checkpoint_path, training_config):
        """Test checkpoint saving and loading."""
        hidden_size = training_config["hidden_size"]

        # Create some state
        state = {
            "model_weights": cpu_backend.to_numpy(cpu_backend.randn((hidden_size, hidden_size))),
            "optimizer_state": {"step": 100, "lr": 1e-4},
            "epoch": 5,
        }

        # Save
        checkpoint_path = str(temp_checkpoint_path)
        if checkpoint_path.endswith(".pt"):
            checkpoint_path = checkpoint_path.replace(".pt", ".npz")

        cpu_backend.save_checkpoint(state, checkpoint_path)

        # Load
        loaded_state = cpu_backend.load_checkpoint(checkpoint_path)

        # Verify
        np.testing.assert_array_almost_equal(
            state["model_weights"],
            loaded_state["model_weights"]
        )


class TestBatchProcessing:
    """Test batch data processing."""

    def test_synthetic_data_shapes(self):
        """Test synthetic data generator produces correct shapes."""
        from tests.fixtures import generate_synthetic_batch

        batch = generate_synthetic_batch(
            batch_size=8,
            seq_len=256,
            hidden_size=512,
            vocab_size=10000,
        )

        assert batch["input_ids"].shape == (8, 256)
        assert batch["attention_mask"].shape == (8, 256)
        assert batch["labels"].shape == (8, 256)
        assert batch["hidden_states"].shape == (8, 256, 512)

    def test_batch_iteration(self):
        """Test iterating over batches."""
        from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig

        config = SyntheticDataConfig(
            vocab_size=1000,
            max_seq_len=64,
            hidden_size=128,
        )
        generator = SyntheticDataGenerator(config)

        batches = list(generator.iterate_batches(
            total_samples=100,
            batch_size=10,
            seq_len=64,
        ))

        assert len(batches) == 10

        for batch in batches:
            assert batch["input_ids"].shape[0] == 10
            assert batch["input_ids"].shape[1] == 64


class TestExpertRouting:
    """Test Mixture of Experts routing."""

    def test_expert_selection(self):
        """Test expert selection from router logits."""
        from tests.fixtures import generate_expert_routing_data

        data = generate_expert_routing_data(
            batch_size=4,
            seq_len=32,
            hidden_size=128,
            num_experts=8,
            top_k=2,
        )

        # Check shapes
        assert data["hidden_states"].shape == (4, 32, 128)
        assert data["router_logits"].shape == (4, 32, 8)
        assert data["expert_indices"].shape == (4, 32, 2)
        assert data["expert_weights"].shape == (4, 32, 2)

        # Check weights sum to 1
        weight_sums = np.sum(data["expert_weights"], axis=-1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-5)

    def test_load_balancing(self):
        """Test expert load balancing."""
        from tests.fixtures import generate_expert_routing_data

        data = generate_expert_routing_data(
            batch_size=32,
            seq_len=128,
            num_experts=8,
        )

        # Count expert usage
        expert_counts = np.zeros(8)
        for expert_idx in data["expert_indices"].flatten():
            expert_counts[expert_idx] += 1

        # No expert should be completely unused
        assert np.all(expert_counts > 0)

        # Compute load balance coefficient
        mean_count = np.mean(expert_counts)
        max_count = np.max(expert_counts)
        balance = mean_count / max_count

        # Should be reasonably balanced (> 0.3)
        assert balance > 0.3


class TestMultimodalIntegration:
    """Test multimodal data processing."""

    def test_multimodal_batch(self):
        """Test multimodal batch generation."""
        from tests.fixtures.synthetic_data import SyntheticDataGenerator

        generator = SyntheticDataGenerator()
        batch = generator.generate_multimodal_batch(
            batch_size=4,
            seq_len=128,
            image_size=(224, 224),
        )

        assert "input_ids" in batch
        assert "pixel_values" in batch
        assert batch["pixel_values"].shape == (4, 3, 224, 224)

        # Check image values are normalized
        assert np.all(batch["pixel_values"] >= -1)
        assert np.all(batch["pixel_values"] <= 1)


@pytest.mark.integration
class TestEndToEndTraining:
    """End-to-end training integration tests."""

    def test_mini_training_loop(self, cpu_backend):
        """Test a minimal training loop."""
        from tests.fixtures import generate_synthetic_batch

        # Configuration
        batch_size = 2
        seq_len = 32
        hidden_size = 64
        num_steps = 5

        # Simulated model weights
        w1 = cpu_backend.randn((hidden_size, hidden_size))
        w2 = cpu_backend.randn((hidden_size, hidden_size))

        losses = []

        for step in range(num_steps):
            # Generate batch
            batch = generate_synthetic_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=hidden_size,
                seed=42 + step,
            )

            # Forward pass (simplified)
            x = cpu_backend.create_tensor(batch["hidden_states"])
            h = cpu_backend.gelu(cpu_backend.matmul(x, w1))
            out = cpu_backend.matmul(h, w2)

            # Compute loss (MSE to target)
            loss = np.mean(cpu_backend.to_numpy(out) ** 2)
            losses.append(loss)

        # Losses should be computed without errors
        assert len(losses) == num_steps
        assert all(np.isfinite(l) for l in losses)
