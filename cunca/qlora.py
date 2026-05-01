"""CUNCA QLoRA — parameter-efficient fine-tuning with NF4 quantized base.

Implements:
  - LoRALinear: frozen Linear layer + low-rank trainable adapter (A·B)
  - nf4_quantize_weight: NF4-style INT4 quantization (2-bit exponent,
    2-bit mantissa per weight group, implemented as scaled INT4 via numpy)
  - apply_qlora(model, rank, alpha, target_modules): replaces named Linear
    layers with LoRALinear wrappers

Usage:
    from cunca.qlora import apply_qlora, LoRALinear
    from cunca.architecture import CUNCAModel
    from cunca.config import CUNCAConfig

    model = CUNCAModel(CUNCAConfig.preset("1.3b"))
    apply_qlora(model, rank=16, alpha=32, target_modules=["q_proj", "v_proj"])

    # Only LoRA parameters are trainable; base is frozen + quantized
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# NF4 quantization (numpy, no bitsandbytes required)
# ---------------------------------------------------------------------------

# The 16 NF4 values are the quantiles of N(0,1) mapped to [-1, 1]
# (following the QLoRA paper, Dettmers et al. 2023)
_NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float32)


def nf4_quantize_weight(w: np.ndarray, block_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a weight matrix to NF4 (4-bit) using block-wise absmax scaling.

    Args:
        w: float32 weight array of any shape.
        block_size: number of elements per quantization block.

    Returns:
        (codes, scales):
            codes  — uint8 array, same shape as w, values in [0, 15]
            scales — float32 array of per-block absmax values
    """
    flat = w.flatten().astype(np.float32)
    n = len(flat)
    # Pad to multiple of block_size
    pad = (-n) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)

    normalized = blocks / scales                   # in [-1, 1]
    # Map each value to the closest NF4 level
    diffs = np.abs(normalized[:, :, None] - _NF4_LEVELS[None, None, :])
    codes = diffs.argmin(axis=2).astype(np.uint8)  # (n_blocks, block_size)

    return codes.flatten()[:n].reshape(w.shape), scales.flatten()


def nf4_dequantize_weight(
    codes: np.ndarray, scales: np.ndarray, original_shape, block_size: int = 64
) -> np.ndarray:
    """Inverse of nf4_quantize_weight."""
    flat_codes = codes.flatten().astype(np.int32)
    n = len(flat_codes)
    pad = (-n) % block_size
    if pad:
        flat_codes = np.concatenate([flat_codes, np.zeros(pad, dtype=np.int32)])

    blocks = flat_codes.reshape(-1, block_size)
    vals = _NF4_LEVELS[blocks]                     # dequantized in [-1, 1]
    vals = vals * scales[:, np.newaxis]             # rescale
    return vals.flatten()[:n].reshape(original_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

if _TORCH:
    class LoRALinear(nn.Module):
        """Drop-in replacement for nn.Linear with a frozen NF4-quantized base
        and trainable low-rank LoRA adapter.

        Forward:
            y = x @ W_frozen.T + (x @ A.T) @ B.T * (alpha / rank)
                ^^^^^^^^^^^^^^^^^^^^  base (frozen, NF4-quantized)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^  LoRA delta

        Only A and B are trainable.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 16,
            alpha: float = 32.0,
            bias: bool = False,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.scaling = alpha / rank

            # Frozen base weight stored in float32 (NF4 dequantized on construction)
            self.register_buffer(
                "weight",
                torch.zeros(out_features, in_features),
            )
            # NF4 metadata
            self._nf4_codes: Optional[np.ndarray] = None
            self._nf4_scales: Optional[np.ndarray] = None

            # Trainable LoRA adapters
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias = None

            self.reset_lora_parameters()

        def reset_lora_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        def load_base_weight(self, w: torch.Tensor) -> None:
            """Load and NF4-quantize a weight tensor into this layer."""
            w_np = w.detach().cpu().float().numpy()
            codes, scales = nf4_quantize_weight(w_np)
            # Store dequantized approximation in the buffer
            w_deq = nf4_dequantize_weight(codes, scales, w_np.shape)
            self.weight.copy_(torch.from_numpy(w_deq))
            self._nf4_codes = codes
            self._nf4_scales = scales

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = F.linear(x, self.weight, self.bias)
            lora_out = self.lora_B(self.lora_A(x)) * self.scaling
            return base_out + lora_out

        def extra_repr(self) -> str:
            return (
                f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.3f}"
            )

    # Keep F.linear accessible
    import torch.nn.functional as F

else:
    class LoRALinear:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("LoRALinear requires PyTorch")


# ---------------------------------------------------------------------------
# apply_qlora
# ---------------------------------------------------------------------------

def apply_qlora(
    model,
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[list[str]] = None,
) -> int:
    """Replace target Linear layers in model with LoRALinear (QLoRA).

    Freezes all existing parameters, then replaces Linear layers whose
    names end with any of the target_module suffixes.  The LoRA adapters
    (lora_A, lora_B) are left trainable.

    Args:
        model:          nn.Module to modify in-place.
        rank:           LoRA rank r.
        alpha:          LoRA scaling alpha (effective LR scale = alpha/rank).
        target_modules: List of Linear layer name suffixes to replace.
                        Defaults to ["q_proj", "v_proj"].

    Returns:
        Number of layers replaced.
    """
    if not _TORCH:
        raise ImportError("apply_qlora requires PyTorch")

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_modules):
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else _get_submodule(model, parent_name)

        lora = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            bias=module.bias is not None,
        )
        lora.load_base_weight(module.weight)
        if module.bias is not None and lora.bias is not None:
            lora.bias.data.copy_(module.bias.data)

        setattr(parent, child_name, lora)
        replaced += 1

    return replaced


def _get_submodule(model, dotted_name: str):
    parts = dotted_name.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


@dataclass
class QLoRAStats:
    """Summary of a QLoRA application."""
    total_params: int = 0
    trainable_params: int = 0
    frozen_params: int = 0
    replaced_layers: int = 0

    @property
    def trainable_ratio(self) -> float:
        return self.trainable_params / max(self.total_params, 1)


def qlora_stats(model) -> "QLoRAStats":
    """Compute QLoRA parameter statistics for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    replaced = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
    return QLoRAStats(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        replaced_layers=replaced,
    )


__all__ = [
    "LoRALinear",
    "apply_qlora",
    "qlora_stats",
    "QLoRAStats",
    "nf4_quantize_weight",
    "nf4_dequantize_weight",
]
