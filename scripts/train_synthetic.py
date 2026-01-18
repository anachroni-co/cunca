#!/usr/bin/env python3
"""
Synthetic Data Training Script

Trains CapibaraGPT on synthetic data for validation before real training.

Usage:
    python scripts/train_synthetic.py --backend=cpu --steps=100
    python scripts/train_synthetic.py --backend=gpu --steps=1000 --batch-size=8
    python scripts/train_synthetic.py --backend=tpu --config=configs/production/tpu_v4.toml
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_synthetic")


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 1024
    vocab_size: int = 32000

    # Training
    batch_size: int = 4
    seq_len: int = 128
    learning_rate: float = 1e-4
    max_steps: int = 100
    warmup_steps: int = 10
    grad_clip: float = 1.0

    # Backend
    backend: str = "cpu"
    dtype: str = "float32"
    mixed_precision: bool = False

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100

    # Checkpointing
    output_dir: str = "outputs/synthetic"
    save_checkpoints: bool = False


@dataclass
class TrainingState:
    """Training state."""
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)


class MockModel:
    """
    Mock model for synthetic training validation.

    Simulates forward pass and loss computation without actual model.
    """

    def __init__(self, config: TrainingConfig, backend):
        self.config = config
        self.backend = backend

        # Initialize mock weights
        self.weights = {
            "embedding": backend.randn((config.vocab_size, config.hidden_size)),
            "layers": [
                {
                    "attention": {
                        "q": backend.randn((config.hidden_size, config.hidden_size)),
                        "k": backend.randn((config.hidden_size, config.hidden_size)),
                        "v": backend.randn((config.hidden_size, config.hidden_size)),
                        "o": backend.randn((config.hidden_size, config.hidden_size)),
                    },
                    "ffn": {
                        "w1": backend.randn((config.hidden_size, config.intermediate_size)),
                        "w2": backend.randn((config.intermediate_size, config.hidden_size)),
                    },
                }
                for _ in range(config.num_layers)
            ],
            "lm_head": backend.randn((config.hidden_size, config.vocab_size)),
        }

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass returning logits."""
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_heads
        head_dim = hidden_size // num_heads

        # Embedding lookup (simulated)
        hidden = self.backend.randn((batch_size, seq_len, hidden_size))

        # Process through layers
        for layer in self.weights["layers"]:
            # Self-attention (simplified)
            q = self.backend.matmul(hidden, layer["attention"]["q"])
            k = self.backend.matmul(hidden, layer["attention"]["k"])
            v = self.backend.matmul(hidden, layer["attention"]["v"])

            # Reshape for attention
            q = self.backend.to_numpy(q).reshape(batch_size, seq_len, num_heads, head_dim)
            k = self.backend.to_numpy(k).reshape(batch_size, seq_len, num_heads, head_dim)
            v = self.backend.to_numpy(v).reshape(batch_size, seq_len, num_heads, head_dim)

            # Transpose to (batch, heads, seq, dim)
            q = self.backend.create_tensor(q.transpose(0, 2, 1, 3))
            k = self.backend.create_tensor(k.transpose(0, 2, 1, 3))
            v = self.backend.create_tensor(v.transpose(0, 2, 1, 3))

            # Attention
            attn_out = self.backend.scaled_dot_product_attention(q, k, v, is_causal=True)

            # Reshape back
            attn_out = self.backend.to_numpy(attn_out).transpose(0, 2, 1, 3)
            attn_out = attn_out.reshape(batch_size, seq_len, hidden_size)
            attn_out = self.backend.create_tensor(attn_out)

            # Output projection
            attn_out = self.backend.matmul(attn_out, layer["attention"]["o"])

            # Residual
            hidden = self.backend.add(hidden, attn_out)

            # Layer norm (simplified)
            hidden = self.backend.layer_norm(hidden, (hidden_size,))

            # FFN
            ffn_hidden = self.backend.matmul(hidden, layer["ffn"]["w1"])
            ffn_hidden = self.backend.gelu(ffn_hidden)
            ffn_hidden = self.backend.matmul(ffn_hidden, layer["ffn"]["w2"])

            # Residual
            hidden = self.backend.add(hidden, ffn_hidden)
            hidden = self.backend.layer_norm(hidden, (hidden_size,))

        # LM head
        logits = self.backend.matmul(hidden, self.weights["lm_head"])

        return logits

    def compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        logits_np = self.backend.to_numpy(logits)
        batch_size, seq_len, vocab_size = logits_np.shape

        # Softmax
        max_logits = np.max(logits_np, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_np - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Cross entropy loss
        loss = 0.0
        count = 0
        for b in range(batch_size):
            for s in range(seq_len):
                if labels[b, s] >= 0:  # Ignore -100 labels
                    loss += -np.log(probs[b, s, labels[b, s]] + 1e-10)
                    count += 1

        return loss / max(count, 1)


def get_lr_scheduler(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train(config: TrainingConfig) -> TrainingState:
    """
    Run synthetic training loop.

    Args:
        config: Training configuration

    Returns:
        Final training state
    """
    logger.info("=" * 60)
    logger.info("Synthetic Training for CapibaraGPT")
    logger.info("=" * 60)

    # Get backend
    from core.backends import get_backend, BackendType

    backend_map = {
        "cpu": BackendType.CPU,
        "gpu": BackendType.GPU,
        "tpu": BackendType.TPU,
    }
    backend_type = backend_map.get(config.backend, BackendType.CPU)

    try:
        backend = get_backend(backend_type)
        logger.info(f"Backend: {backend.name.upper()}")
    except Exception as e:
        logger.warning(f"Failed to get {config.backend} backend: {e}")
        logger.info("Falling back to CPU")
        backend = get_backend(BackendType.CPU)

    # Get device info
    from core.backends.utils import get_device_info, memory_stats
    device_info = get_device_info(backend.name)
    logger.info(f"Device info: {device_info}")

    # Create data generator
    from tests.fixtures.synthetic_data import SyntheticDataGenerator, SyntheticDataConfig

    data_config = SyntheticDataConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        hidden_size=config.hidden_size,
    )
    data_generator = SyntheticDataGenerator(data_config)

    # Create mock model
    model = MockModel(config, backend)

    # Training state
    state = TrainingState()

    logger.info(f"Training configuration:")
    logger.info(f"  - Hidden size: {config.hidden_size}")
    logger.info(f"  - Num layers: {config.num_layers}")
    logger.info(f"  - Num heads: {config.num_heads}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Sequence length: {config.seq_len}")
    logger.info(f"  - Max steps: {config.max_steps}")
    logger.info("")
    logger.info("Starting training...")

    start_time = time.time()

    for step in range(1, config.max_steps + 1):
        step_start = time.time()

        # Get batch
        batch = data_generator.generate_batch(config.batch_size, config.seq_len)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Get learning rate
        lr = get_lr_scheduler(
            step, config.warmup_steps, config.max_steps, config.learning_rate
        )

        # Forward pass
        logits = model.forward(input_ids)

        # Compute loss
        loss = model.compute_loss(logits, labels)

        # Update state
        state.step = step
        state.total_loss += loss
        state.losses.append(loss)
        state.learning_rates.append(lr)

        step_time = time.time() - step_start
        state.step_times.append(step_time)

        # Logging
        if step % config.log_interval == 0:
            avg_loss = np.mean(state.losses[-config.log_interval:])
            avg_time = np.mean(state.step_times[-config.log_interval:])
            samples_per_sec = config.batch_size / avg_time

            logger.info(
                f"Step {step:5d}/{config.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {avg_time*1000:.1f}ms | "
                f"Samples/s: {samples_per_sec:.1f}"
            )

            # Memory stats
            if backend.name in ["gpu", "tpu"]:
                mem = memory_stats(backend.name)
                logger.info(f"  Memory: {mem['allocated_gb']:.2f}GB / {mem['total_gb']:.2f}GB")

    # Training complete
    total_time = time.time() - start_time
    avg_loss = np.mean(state.losses)
    final_loss = np.mean(state.losses[-10:])

    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Final loss: {final_loss:.4f}")
    logger.info(f"Total steps: {state.step}")
    logger.info(f"Samples processed: {state.step * config.batch_size}")
    logger.info(f"Throughput: {state.step * config.batch_size / total_time:.1f} samples/s")

    return state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Synthetic Data Training for CapibaraGPT")

    # Model config
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")

    # Training config
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")

    # Backend
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="Compute backend",
    )
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        backend=args.backend,
        mixed_precision=args.mixed_precision,
        log_interval=args.log_interval,
    )

    # Run training
    state = train(config)

    # Return success if loss decreased
    if len(state.losses) > 10:
        initial_loss = np.mean(state.losses[:10])
        final_loss = np.mean(state.losses[-10:])
        if final_loss < initial_loss:
            logger.info("SUCCESS: Loss decreased during training")
            return 0
        else:
            logger.warning("WARNING: Loss did not decrease")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
