"""
Synthetic Data Generators for Testing

Generates realistic-looking synthetic data for:
- Training data simulation
- Integration testing
- Performance benchmarking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    vocab_size: int = 32000
    max_seq_len: int = 2048
    hidden_size: int = 768
    num_experts: int = 8
    seed: int = 42


class SyntheticDataGenerator:
    """
    Generator for synthetic training/testing data.

    Produces:
    - Token sequences with realistic patterns
    - Attention masks
    - Expert routing targets
    - Loss labels
    """

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.config = config or SyntheticDataConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Vocabulary simulation: create word frequency distribution
        # Zipf's law: frequency ∝ 1/rank
        self._word_probs = self._create_word_distribution()

    def _create_word_distribution(self) -> np.ndarray:
        """Create Zipfian word distribution."""
        ranks = np.arange(1, self.config.vocab_size + 1)
        probs = 1.0 / ranks
        probs = probs / probs.sum()
        return probs

    def generate_tokens(
        self,
        batch_size: int,
        seq_len: Optional[int] = None,
        include_special_tokens: bool = True,
    ) -> np.ndarray:
        """
        Generate synthetic token sequences.

        Args:
            batch_size: Number of sequences
            seq_len: Sequence length (default: max_seq_len)
            include_special_tokens: Add BOS/EOS tokens

        Returns:
            Token IDs array of shape (batch_size, seq_len)
        """
        seq_len = seq_len or self.config.max_seq_len

        # Sample from Zipfian distribution for realistic token distribution
        tokens = self.rng.choice(
            self.config.vocab_size,
            size=(batch_size, seq_len),
            p=self._word_probs,
        )

        if include_special_tokens:
            # BOS = 1, EOS = 2
            tokens[:, 0] = 1  # BOS
            # Add EOS at random positions (simulating variable length)
            for i in range(batch_size):
                eos_pos = self.rng.integers(seq_len // 2, seq_len)
                tokens[i, eos_pos] = 2  # EOS

        return tokens.astype(np.int64)

    def generate_attention_mask(
        self,
        input_ids: np.ndarray,
        pad_token_id: int = 0,
    ) -> np.ndarray:
        """Generate attention mask (1 for real tokens, 0 for padding)."""
        return (input_ids != pad_token_id).astype(np.float32)

    def generate_batch(
        self,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete training batch.

        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Target labels (shifted input_ids)
            - hidden_states: Simulated hidden states
        """
        seq_len = seq_len or self.config.max_seq_len

        input_ids = self.generate_tokens(batch_size, seq_len)
        attention_mask = self.generate_attention_mask(input_ids)

        # Labels are shifted input_ids (for causal LM)
        labels = np.roll(input_ids, -1, axis=1)
        labels[:, -1] = -100  # Ignore last position

        # Simulated hidden states (random but normalized)
        hidden_states = self.rng.standard_normal(
            (batch_size, seq_len, self.config.hidden_size)
        ).astype(np.float32)

        # Normalize to unit variance
        hidden_states = hidden_states / np.std(hidden_states)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "hidden_states": hidden_states,
        }

    def generate_expert_data(
        self,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate data for Mixture of Experts testing.

        Returns:
            Dictionary with:
            - hidden_states: Input to router
            - expert_indices: Ground truth expert assignments
            - expert_weights: Ground truth expert weights
        """
        seq_len = seq_len or self.config.max_seq_len
        num_experts = self.config.num_experts

        # Hidden states
        hidden_states = self.rng.standard_normal(
            (batch_size, seq_len, self.config.hidden_size)
        ).astype(np.float32)

        # Expert indices (top-2 selection)
        expert_indices = self.rng.integers(
            0, num_experts, size=(batch_size, seq_len, 2)
        ).astype(np.int64)

        # Expert weights (normalized)
        raw_weights = self.rng.random((batch_size, seq_len, 2)).astype(np.float32)
        expert_weights = raw_weights / raw_weights.sum(axis=-1, keepdims=True)

        return {
            "hidden_states": hidden_states,
            "expert_indices": expert_indices,
            "expert_weights": expert_weights,
        }

    def generate_multimodal_batch(
        self,
        batch_size: int,
        seq_len: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        num_channels: int = 3,
    ) -> Dict[str, np.ndarray]:
        """
        Generate multimodal batch with text and images.

        Returns:
            Dictionary with text and image data.
        """
        seq_len = seq_len or self.config.max_seq_len

        # Text data
        text_batch = self.generate_batch(batch_size, seq_len)

        # Image data (normalized to [-1, 1])
        images = self.rng.standard_normal(
            (batch_size, num_channels, *image_size)
        ).astype(np.float32)
        images = np.clip(images, -3, 3) / 3  # Normalize

        return {
            **text_batch,
            "pixel_values": images,
        }

    def iterate_batches(
        self,
        total_samples: int,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        Iterate over synthetic batches.

        Args:
            total_samples: Total number of samples to generate
            batch_size: Batch size
            seq_len: Sequence length

        Yields:
            Training batches
        """
        num_batches = (total_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            yield self.generate_batch(batch_size, seq_len)


# ==================== Convenience Functions ====================

def generate_random_tokens(
    batch_size: int = 4,
    seq_len: int = 128,
    vocab_size: int = 32000,
    seed: int = 42,
) -> np.ndarray:
    """Generate random token IDs."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)


def generate_synthetic_batch(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 768,
    vocab_size: int = 32000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a complete synthetic batch."""
    config = SyntheticDataConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        hidden_size=hidden_size,
        seed=seed,
    )
    generator = SyntheticDataGenerator(config)
    return generator.generate_batch(batch_size, seq_len)


def generate_attention_patterns(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 64,
    pattern: str = "random",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate attention pattern matrices.

    Args:
        pattern: Type of pattern ('random', 'causal', 'local', 'global')
    """
    rng = np.random.default_rng(seed)

    if pattern == "random":
        # Random attention scores
        scores = rng.standard_normal((batch_size, num_heads, seq_len, seq_len))

    elif pattern == "causal":
        # Causal (lower triangular)
        scores = np.tril(np.ones((seq_len, seq_len)))
        scores = np.broadcast_to(scores, (batch_size, num_heads, seq_len, seq_len))
        scores = scores * rng.standard_normal((batch_size, num_heads, seq_len, seq_len))

    elif pattern == "local":
        # Local attention (band matrix)
        window = seq_len // 8
        scores = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            scores[i, start:end] = 1
        scores = np.broadcast_to(scores, (batch_size, num_heads, seq_len, seq_len))
        scores = scores * rng.standard_normal((batch_size, num_heads, seq_len, seq_len))

    elif pattern == "global":
        # Global + local attention
        local_window = seq_len // 8
        global_tokens = seq_len // 16

        scores = np.zeros((seq_len, seq_len))
        # Local attention
        for i in range(seq_len):
            start = max(0, i - local_window)
            end = min(seq_len, i + local_window + 1)
            scores[i, start:end] = 1
        # Global tokens (first few tokens attend to all)
        scores[:global_tokens, :] = 1
        scores[:, :global_tokens] = 1

        scores = np.broadcast_to(scores, (batch_size, num_heads, seq_len, seq_len))
        scores = scores * rng.standard_normal((batch_size, num_heads, seq_len, seq_len))

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return scores.astype(np.float32)


def generate_expert_routing_data(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 768,
    num_experts: int = 8,
    top_k: int = 2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data for testing expert routing."""
    rng = np.random.default_rng(seed)

    # Input hidden states
    hidden_states = rng.standard_normal((batch_size, seq_len, hidden_size)).astype(np.float32)

    # Router logits
    router_logits = rng.standard_normal((batch_size, seq_len, num_experts)).astype(np.float32)

    # Top-k expert indices
    expert_indices = np.argsort(router_logits, axis=-1)[:, :, -top_k:]

    # Expert weights (softmax over top-k)
    top_k_logits = np.take_along_axis(router_logits, expert_indices, axis=-1)
    expert_weights = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits), axis=-1, keepdims=True)

    return {
        "hidden_states": hidden_states,
        "router_logits": router_logits,
        "expert_indices": expert_indices,
        "expert_weights": expert_weights,
    }
