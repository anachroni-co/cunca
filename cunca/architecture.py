"""CUNCA-Hybrid architecture.

Extends the Capibara Slim base with:
  - Grouped Query Attention (GQA): num_kv_heads < num_heads
  - Sliding-Window Attention: local context of window_size tokens
  - Configurable SSM/attention ratio (ssm_ratio)

Block layout is determined by CUNCAConfig.block_types() which distributes
SSM blocks at odd layer positions when ssm_ratio=0.5 (alternating pattern).
"""
from __future__ import annotations

import math
from typing import Optional

from cunca.config import CUNCAConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch(name: str) -> None:
    if not _TORCH:
        raise ImportError(f"{name} requires PyTorch")


# ---------------------------------------------------------------------------
# Shared primitives (reuse Slim's RMSNorm / RoPE pattern)
# ---------------------------------------------------------------------------

if _TORCH:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            return self.weight * x / rms

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq_len: int = 8192) -> None:
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len).float()
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos", emb.cos()[None, None, :, :])
            self.register_buffer("sin", emb.sin()[None, None, :, :])

        def forward(self, seq_len: int):
            return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]

    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(q, k, cos, sin):
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

else:
    class RMSNorm:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("RMSNorm")

    class RotaryEmbedding:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("RotaryEmbedding")


# ---------------------------------------------------------------------------
# GQA + Sliding-Window Attention
# ---------------------------------------------------------------------------

if _TORCH:
    class CUNCAAttention(nn.Module):
        """Grouped Query Attention with optional sliding-window mask.

        With num_kv_heads < num_heads, K and V projections are shared across
        (num_heads // num_kv_heads) query heads, reducing KV cache size.

        window_size > 0 restricts each token's attention span to the
        window_size most recent tokens (causal sliding window).
        """

        def __init__(self, cfg: CUNCAConfig) -> None:
            super().__init__()
            assert cfg.hidden_size % cfg.num_heads == 0
            self.num_heads = cfg.num_heads
            self.num_kv_heads = cfg.num_kv_heads
            self.kv_groups = cfg.kv_groups
            self.head_dim = cfg.hidden_size // cfg.num_heads
            self.scale = self.head_dim ** -0.5
            self.window_size = cfg.window_size

            h = cfg.hidden_size
            kv_dim = self.num_kv_heads * self.head_dim

            self.q_proj = nn.Linear(h, h, bias=False)
            self.k_proj = nn.Linear(h, kv_dim, bias=False)
            self.v_proj = nn.Linear(h, kv_dim, bias=False)
            self.o_proj = nn.Linear(h, h, bias=False)

            self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)
            self.norm = RMSNorm(h, cfg.rms_norm_eps)
            self.dropout = nn.Dropout(cfg.dropout)

        def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
            """Repeat KV heads to match query head count (GQA expand)."""
            # kv: (B, num_kv_heads, L, head_dim)
            return kv.repeat_interleave(self.kv_groups, dim=1)

        def _sliding_window_mask(self, L: int, device) -> torch.Tensor:
            """Return bool mask (1 = masked) with causal + window constraint."""
            idx = torch.arange(L, device=device)
            # position j is visible to position i if: i-window < j <= i
            diff = idx.unsqueeze(0) - idx.unsqueeze(1)   # (L, L)  row=query, col=key
            causal = diff < 0                             # future tokens
            too_far = diff < -self.window_size            # beyond window
            return (causal | too_far).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

        def forward(self, x, mask=None):
            residual = x
            x = self.norm(x)
            B, L, _ = x.shape
            H, KV, D = self.num_heads, self.num_kv_heads, self.head_dim

            q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)   # (B,H,L,D)
            k = self.k_proj(x).view(B, L, KV, D).transpose(1, 2)  # (B,KV,L,D)
            v = self.v_proj(x).view(B, L, KV, D).transpose(1, 2)

            cos, sin = self.rope(L)
            # cos/sin shape: (1, 1, L, head_dim) — broadcast over all heads
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

            # Expand KV heads → query heads
            k = self._expand_kv(k)
            v = self._expand_kv(v)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Build combined mask: external + sliding-window causal
            if self.window_size > 0:
                sw_mask = self._sliding_window_mask(L, x.device)
                attn = attn.masked_fill(sw_mask, float("-inf"))
            elif mask is not None:
                attn = attn.masked_fill(mask, float("-inf"))
            else:
                causal = torch.triu(
                    torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
                ).unsqueeze(0).unsqueeze(0)
                attn = attn.masked_fill(causal, float("-inf"))

            attn = self.dropout(F.softmax(attn, dim=-1))
            out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
            return self.o_proj(out) + residual

else:
    class CUNCAAttention:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("CUNCAAttention")


# ---------------------------------------------------------------------------
# SwiGLU MLP (same as Slim, uses CUNCAConfig)
# ---------------------------------------------------------------------------

if _TORCH:
    class CUNCAMLP(nn.Module):
        def __init__(self, cfg: CUNCAConfig) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) + residual

else:
    class CUNCAMLP:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("CUNCAMLP")


# ---------------------------------------------------------------------------
# Transformer block (GQA + sliding-window)
# ---------------------------------------------------------------------------

if _TORCH:
    class CUNCATransformerBlock(nn.Module):
        def __init__(self, cfg: CUNCAConfig) -> None:
            super().__init__()
            self.attn = CUNCAAttention(cfg)
            self.mlp  = CUNCAMLP(cfg)

        def forward(self, x, mask=None):
            return self.mlp(self.attn(x, mask))

else:
    class CUNCATransformerBlock:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("CUNCATransformerBlock")


# ---------------------------------------------------------------------------
# Mamba SSM Block (reuse Slim's selective SSM, adapted for CUNCAConfig)
# ---------------------------------------------------------------------------

if _TORCH:
    class CUNCAMambaBlock(nn.Module):
        """Selective SSM block (Mamba-style) for CUNCA-Hybrid."""

        def __init__(self, cfg: CUNCAConfig) -> None:
            super().__init__()
            d = cfg.hidden_size
            self.d_state = cfg.mamba_d_state
            d_inner = cfg.mamba_expand * d

            self.norm = RMSNorm(d, cfg.rms_norm_eps)
            self.in_proj = nn.Linear(d, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, cfg.mamba_d_conv,
                padding=cfg.mamba_d_conv - 1, groups=d_inner, bias=True,
            )
            self.x_proj = nn.Linear(d_inner, self.d_state * 2 + 1, bias=False)
            self.dt_proj = nn.Linear(1, d_inner)
            self.out_proj = nn.Linear(d_inner, d, bias=False)

            A = torch.arange(1, self.d_state + 1, dtype=torch.float).repeat(d_inner, 1)
            self.A_log = nn.Parameter(A.log())
            self.D = nn.Parameter(torch.ones(d_inner))

        def forward(self, x, mask=None):
            B_batch, L, _ = x.shape
            residual = x
            x = self.norm(x)

            xz = self.in_proj(x)
            x_part, z = xz.chunk(2, dim=-1)

            x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
            x_conv = F.silu(x_conv)

            x_dbl = self.x_proj(x_conv)
            dt_raw, B_mat, C_mat = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt_raw))

            A = -torch.exp(self.A_log.float())
            dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
            dB = torch.einsum("bld,bln->bldn", dt, B_mat)

            # Zero-out padded positions so they don't contaminate the SSM state.
            # The sequential scan is inherently causal; mask only needed for padding.
            if mask is not None:
                # mask shape expected: (B, 1, L, L) bool or (B, L) bool
                # Derive per-token padding mask: True = padded (should be zeroed)
                if mask.dim() == 4:
                    pad_mask = mask[:, 0, :, 0]  # (B, L)
                else:
                    pad_mask = ~mask.bool()  # (B, L), True = padding
                # Zero dt for padded positions to prevent state update
                dt = dt * (~pad_mask).unsqueeze(-1).float()

            h = torch.zeros(B_batch, x_part.shape[-1], self.d_state,
                            device=x.device, dtype=x.dtype)
            ys = []
            for t in range(L):
                h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
                y = torch.einsum("bdn,bn->bd", h, C_mat[:, t])
                ys.append(y)

            y = torch.stack(ys, dim=1)
            y = y + x_conv * self.D
            y = y * F.silu(z)
            return self.out_proj(y) + residual

else:
    class CUNCAMambaBlock:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("CUNCAMambaBlock")


# ---------------------------------------------------------------------------
# CUNCA-Hybrid Model
# ---------------------------------------------------------------------------

if _TORCH:
    class CUNCAModel(nn.Module):
        """CUNCA-Hybrid — GQA + sliding-window + Mamba SSM language model.

        Block layout driven by CUNCAConfig.block_types():
          ssm_ratio=0.0  → all transformer (pure attention)
          ssm_ratio=0.5  → alternating attn/ssm (default CUNCA pattern)
          ssm_ratio=1.0  → all Mamba SSM
        """

        def __init__(self, cfg: CUNCAConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

            block_types = cfg.block_types()
            blocks: list[nn.Module] = []
            for bt in block_types:
                if bt == "ssm":
                    blocks.append(CUNCAMambaBlock(cfg))
                else:
                    blocks.append(CUNCATransformerBlock(cfg))
            self.blocks = nn.ModuleList(blocks)

            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

            if cfg.tie_embeddings:
                self.lm_head.weight = self.embed.weight

            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, input_ids, attention_mask=None, return_hidden_states: bool = False):
            x = self.embed(input_ids)
            for block in self.blocks:
                x = block(x, mask=attention_mask)
            x = self.norm(x)
            logits = self.lm_head(x)
            if return_hidden_states:
                return logits, x
            return logits

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def block_summary(self) -> list[dict]:
            """Return per-block type info for inspection."""
            return [
                {"index": i, "type": type(b).__name__, "params": sum(p.numel() for p in b.parameters())}
                for i, b in enumerate(self.blocks)
            ]

else:
    class CUNCAModel:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch("CUNCAModel")


__all__ = [
    "CUNCAModel",
    "CUNCATransformerBlock",
    "CUNCAMambaBlock",
    "CUNCAAttention",
    "CUNCAMLP",
    "RMSNorm",
]
