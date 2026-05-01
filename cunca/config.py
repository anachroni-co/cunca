"""CUNCA-Hybrid model configuration.

Extends the Capibara Slim base with:
  - Grouped Query Attention (GQA) — reduces KV cache memory
  - Sliding-Window Attention — bounded local context O(L·w) vs O(L²)
  - Configurable SSM/attention ratio (ssm_ratio)
  - 1.3B / 3B / 7B presets tuned for Galician + Romance languages

Presets match the CUNCA Memoria Técnica (Anexo VIII) parameter targets:
  1.3B  hidden=2048  layers=24  heads=16  kv_heads=4   window=512
  3b    hidden=2560  layers=32  heads=20  kv_heads=4   window=1024
  7b    hidden=4096  layers=32  heads=32  kv_heads=8   window=2048
"""
from __future__ import annotations

from dataclasses import dataclass


_PRESETS: dict[str, dict] = {
    "1.3b": dict(
        hidden_size=2048,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5504,
        vocab_size=32000,
        max_seq_len=4096,
        window_size=512,
        ssm_ratio=0.5,
    ),
    "3b": dict(
        hidden_size=2560,
        num_layers=32,
        num_heads=20,
        num_kv_heads=4,
        intermediate_size=6912,
        vocab_size=32000,
        max_seq_len=4096,
        window_size=1024,
        ssm_ratio=0.5,
    ),
    "7b": dict(
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=11008,
        vocab_size=32000,
        max_seq_len=8192,
        window_size=2048,
        ssm_ratio=0.5,
    ),
}


@dataclass
class CUNCAConfig:
    """Configuration for CUNCA-Hybrid model.

    Args:
        hidden_size: Model hidden dimension.
        num_layers: Total number of blocks (attention + SSM combined).
        num_heads: Number of query heads in attention.
        num_kv_heads: Number of key/value heads (GQA). Must divide num_heads.
        intermediate_size: SwiGLU inner dimension.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length for RoPE.
        rms_norm_eps: RMSNorm epsilon.
        dropout: Dropout probability (0 = disabled).
        window_size: Sliding-window attention radius (tokens). 0 = full attention.
        ssm_ratio: Fraction of blocks that are Mamba SSM (vs Transformer).
                   0.0 = all transformer, 1.0 = all mamba, 0.5 = alternating.
        mamba_d_state: Mamba SSM state dimension.
        mamba_d_conv: Mamba depthwise conv width.
        mamba_expand: Mamba inner-dim expansion factor.
        tie_embeddings: Tie input/output embedding weights.
    """
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_size: int = 5504
    vocab_size: int = 32000
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    window_size: int = 512
    ssm_ratio: float = 0.5
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be > 0")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if not 0.0 <= self.ssm_ratio <= 1.0:
            raise ValueError(f"ssm_ratio must be in [0, 1], got {self.ssm_ratio}")

    @property
    def kv_groups(self) -> int:
        """Number of query heads per KV head (GQA group factor)."""
        return self.num_heads // self.num_kv_heads

    @classmethod
    def preset(cls, name: str) -> "CUNCAConfig":
        key = name.lower().replace("-", "").replace("_", "")
        if key not in _PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(_PRESETS)}")
        return cls(**_PRESETS[key])

    def estimate_params(self) -> int:
        """Rough parameter count."""
        embed = self.vocab_size * self.hidden_size
        h = self.hidden_size
        # GQA: Q is full, K/V use kv_heads
        attn = h * h + 2 * (self.num_kv_heads * (h // self.num_heads)) * h + h * h
        mlp = 3 * h * self.intermediate_size
        n_ssm = round(self.num_layers * self.ssm_ratio)
        n_attn = self.num_layers - n_ssm
        ssm_params = (
            h * (self.mamba_expand * h * 2)       # in_proj
            + self.mamba_expand * h * h            # out_proj (approx)
        )
        return embed + n_attn * (attn + mlp) + n_ssm * ssm_params

    def block_types(self) -> list[str]:
        """Return list of block types ('attn' or 'ssm') for all layers."""
        n_ssm = round(self.num_layers * self.ssm_ratio)
        types = []
        ssm_assigned = 0
        for i in range(self.num_layers):
            remaining = self.num_layers - i
            ssm_remaining = n_ssm - ssm_assigned
            if ssm_remaining > 0 and (i % 2 == 1 or ssm_remaining >= remaining):
                types.append("ssm")
                ssm_assigned += 1
            else:
                types.append("attn")
        return types
