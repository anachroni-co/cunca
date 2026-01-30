"""
GPU Backend Implementation (PyTorch/CUDA)

Optimized for NVIDIA A-100 GPUs with:
- Flash Attention 2
- Tensor Cores
- Mixed Precision (BF16/FP16)
- DeepSpeed ZeRO
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .base import BackendConfig, ComputeBackend, DType, TensorLike
from .lazy_import import ensure_torch

# Module-level aliases populated on first _ensure_torch() call
torch = None
nn = None
F = None


def _ensure_torch():
    global torch, nn, F
    if torch is None:
        torch, nn, F = ensure_torch()


# Robust dtype mapping
DTYPE_MAP = {
    DType.FLOAT32: lambda: torch.float32,
    DType.FLOAT16: lambda: torch.float16,
    DType.BFLOAT16: lambda: torch.bfloat16,
    DType.INT32: lambda: torch.int32,
    DType.INT64: lambda: torch.int64,
    DType.BOOL: lambda: torch.bool,
}


class GPUBackend(ComputeBackend):
    """
    GPU Backend using PyTorch/CUDA.

    Optimized for NVIDIA A-100:
    - Tensor Cores
    - Flash Attention 2
    - AMP (BF16 preferred)
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._scaler = None
        self._flash_attn_available = False
        self._device = None
        self._training = True

    # ==================== Properties ====================

    @property
    def name(self) -> str:
        return "gpu"

    @property
    def training(self) -> bool:
        return self._training

    def set_training(self, mode: bool) -> None:
        self._training = mode

    @property
    def is_available(self) -> bool:
        try:
            _ensure_torch()
            return torch.cuda.is_available()
        except Exception:
            return False

    # ==================== Initialization ====================

    def initialize(self) -> None:
        _ensure_torch()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        device_id = self.config.cuda_device_ids[0] if self.config.cuda_device_ids else 0
        self._device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)

        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if self.config.tensor_cores:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.use_mixed_precision
        )

        self._flash_attn_available = self._check_flash_attention()
        self._initialized = True

    def shutdown(self) -> None:
        _ensure_torch()
        torch.cuda.empty_cache()
        self._initialized = False

    def _check_flash_attention(self) -> bool:
        try:
            if hasattr(F, "scaled_dot_product_attention"):
                return torch.backends.cuda.flash_sdp_enabled()
        except Exception:
            pass

        try:
            import flash_attn  # noqa: F401
            return True
        except Exception:
            return False

    # ==================== Tensor Creation ====================

    def create_tensor(
        self,
        data: Any,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> TensorLike:
        _ensure_torch()
        torch_dtype = DTYPE_MAP[dtype]() if dtype else None
        return torch.tensor(
            data,
            dtype=torch_dtype,
            device=device or self._device,
            requires_grad=requires_grad,
        )

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_torch()
        return torch.zeros(
            shape,
            dtype=DTYPE_MAP[dtype]() if dtype else None,
            device=self._device,
        )

    def ones(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_torch()
        return torch.ones(
            shape,
            dtype=DTYPE_MAP[dtype]() if dtype else None,
            device=self._device,
        )

    def randn(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_torch()
        return torch.randn(
            shape,
            dtype=DTYPE_MAP[dtype]() if dtype else None,
            device=self._device,
        )

    def to_device(self, tensor: TensorLike, device: str) -> TensorLike:
        _ensure_torch()
        return tensor.to(device)

    def to_numpy(self, tensor: TensorLike) -> np.ndarray:
        _ensure_torch()
        return tensor.detach().cpu().numpy()

    # ==================== Math ====================

    def matmul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        _ensure_torch()
        return torch.matmul(a, b)

    def add(self, a: TensorLike, b: TensorLike) -> TensorLike:
        _ensure_torch()
        return torch.add(a, b)

    def mul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        _ensure_torch()
        return torch.mul(a, b)

    def softmax(self, x: TensorLike, axis: int = -1) -> TensorLike:
        _ensure_torch()
        return F.softmax(x, dim=axis)

    def layer_norm(
        self,
        x: TensorLike,
        normalized_shape: Tuple[int, ...],
        weight: Optional[TensorLike] = None,
        bias: Optional[TensorLike] = None,
        eps: float = 1e-5,
    ) -> TensorLike:
        _ensure_torch()
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    def gelu(self, x: TensorLike) -> TensorLike:
        _ensure_torch()
        return F.gelu(x, approximate="tanh")

    def silu(self, x: TensorLike) -> TensorLike:
        _ensure_torch()
        return F.silu(x)

    # ==================== Attention ====================

    def scaled_dot_product_attention(
        self,
        query: TensorLike,
        key: TensorLike,
        value: TensorLike,
        mask: Optional[TensorLike] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> TensorLike:
        _ensure_torch()

        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout_p if self.training else 0.0,
                is_causal=is_causal,
                scale=scale,
            )

        head_dim = query.size(-1)
        scale = scale or head_dim ** -0.5

        attn = torch.matmul(query, key.transpose(-2, -1)) * scale

        if is_causal:
            seq_len = query.size(-2)
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            attn.masked_fill_(causal, float("-inf"))

        if mask is not None:
            attn.masked_fill_(~mask.bool(), float("-inf"))

        attn = F.softmax(attn, dim=-1)

        if self.training and dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        return torch.matmul(attn, value)

    # ==================== Gradients ====================

    def backward(self, loss: TensorLike) -> None:
        _ensure_torch()
        if self._scaler and self._scaler.is_enabled():
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def zero_grad(self, parameters: Sequence[TensorLike]) -> None:
        _ensure_torch()
        for p in parameters:
            if p.grad is not None:
                p.grad.zero_()

    def clip_grad_norm(
        self,
        parameters: Sequence[TensorLike],
        max_norm: float,
        optimizer: Optional[Any] = None,
    ) -> TensorLike:
        _ensure_torch()
        if self._scaler and optimizer is not None and self._scaler.is_enabled():
            self._scaler.unscale_(optimizer)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    # ==================== JIT ====================

    def jit_compile(self, fn: Callable) -> Callable:
        _ensure_torch()
        if hasattr(torch, "compile"):
            return torch.compile(fn, mode="reduce-overhead", backend="inductor")
        return fn

    # ==================== Distributed ====================

    def all_reduce(self, tensor: TensorLike, op: str = "sum") -> TensorLike:
        _ensure_torch()
        if not torch.distributed.is_initialized():
            return tensor

        ops = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.SUM,
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }

        torch.distributed.all_reduce(tensor, op=ops.get(op, torch.distributed.ReduceOp.SUM))

        if op == "mean":
            tensor /= torch.distributed.get_world_size()

        return tensor

    def broadcast(self, tensor: TensorLike, src: int = 0) -> TensorLike:
        _ensure_torch()
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(tensor, src=src)
        return tensor

    # ==================== Memory ====================

    def memory_allocated(self) -> int:
        _ensure_torch()
        return torch.cuda.memory_allocated(self._device)

    def memory_reserved(self) -> int:
        _ensure_torch()
        return torch.cuda.memory_reserved(self._device)

    def empty_cache(self) -> None:
        _ensure_torch()
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        _ensure_torch()
        torch.cuda.synchronize(self._device)

    # ==================== Contexts ====================

    @contextmanager
    def no_grad(self):
        _ensure_torch()
        with torch.no_grad():
            yield

    @contextmanager
    def autocast(self, dtype: Optional[DType] = None):
        _ensure_torch()

        if not self.config.use_mixed_precision:
            yield
            return

        torch_dtype = (
            torch.bfloat16
            if dtype in (None, DType.BFLOAT16)
            else torch.float16
        )

        with torch.cuda.amp.autocast(dtype=torch_dtype):
            yield

    # ==================== Checkpoints ====================

    def save_checkpoint(self, state: Dict[str, Any], path: str) -> None:
        _ensure_torch()
        torch.save(state, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ensure_torch()
        loc = map_location or self._device
        try:
            return torch.load(path, map_location=loc, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=loc)

    # ==================== Device Info ====================

    def get_device_count(self) -> int:
        _ensure_torch()
        return torch.cuda.device_count()

    def get_device_properties(self, device_id: int = 0) -> Dict[str, Any]:
        _ensure_torch()
        p = torch.cuda.get_device_properties(device_id)
        return {
            "name": p.name,
            "total_memory": p.total_memory,
            "major": p.major,
            "minor": p.minor,
            "multi_processor_count": p.multi_processor_count,
            "is_a100": "A100" in p.name,
        }

    # ==================== Model Helpers ====================

    def enable_gradient_checkpointing(self, model: Any) -> None:
        _ensure_torch()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    def compile_model(self, model: Any, mode: str = "reduce-overhead") -> Any:
        _ensure_torch()
        if hasattr(torch, "compile"):
            return torch.compile(model, mode=mode, backend="inductor")
        return model


