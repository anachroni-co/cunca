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

# Lazy imports for PyTorch
torch = None
nn = None
F = None


def _ensure_torch():
    """Lazy import PyTorch modules."""
    global torch, nn, F
    if torch is None:
        try:
            import torch as _torch
            import torch.nn as _nn
            import torch.nn.functional as _F

            torch = _torch
            nn = _nn
            F = _F
        except ImportError as e:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            ) from e


# Dtype mapping
DTYPE_MAP = {
    DType.FLOAT32: "float32",
    DType.FLOAT16: "float16",
    DType.BFLOAT16: "bfloat16",
    DType.INT32: "int32",
    DType.INT64: "int64",
    DType.BOOL: "bool",
}


class GPUBackend(ComputeBackend):
    """
    GPU Backend using PyTorch/CUDA.

    Optimized for NVIDIA A-100:
    - 80GB HBM2e memory
    - Tensor Cores for mixed precision
    - Flash Attention 2 for efficient attention
    - DeepSpeed ZeRO for distributed training

    Features:
    - torch.compile for automatic kernel optimization
    - Flash Attention 2 integration
    - Automatic mixed precision (AMP)
    - Gradient checkpointing
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._scaler = None  # For AMP
        self._flash_attn_available = False

    @property
    def name(self) -> str:
        return "gpu"

    @property
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            _ensure_torch()
            return torch.cuda.is_available()
        except Exception:
            return False

    def initialize(self) -> None:
        """Initialize PyTorch for GPU."""
        _ensure_torch()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Set device
        device_id = self.config.cuda_device_ids[0] if self.config.cuda_device_ids else 0
        self._device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(self._device)

        # Enable optimizations for A-100
        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if self.config.tensor_cores:
            # Enable TF32 for Tensor Cores (A-100 optimization)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Initialize AMP scaler
        if self.config.use_mixed_precision:
            self._scaler = torch.cuda.amp.GradScaler()

        # Check for Flash Attention
        self._flash_attn_available = self._check_flash_attention()

        self._initialized = True

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            # Check for PyTorch 2.0+ native Flash Attention
            if hasattr(F, "scaled_dot_product_attention"):
                # Check if flash attention backend is available
                return torch.backends.cuda.flash_sdp_enabled()
        except Exception:
            pass

        try:
            # Check for flash-attn package
            import flash_attn
            return True
        except ImportError:
            pass

        return False

    def shutdown(self) -> None:
        """Clean up CUDA resources."""
        _ensure_torch()
        torch.cuda.empty_cache()
        self._initialized = False

    # ==================== Tensor Operations ====================

    def create_tensor(
        self,
        data: Any,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> TensorLike:
        _ensure_torch()
        torch_dtype = getattr(torch, DTYPE_MAP.get(dtype, "float32")) if dtype else None
        device = device or self._device
        return torch.tensor(data, dtype=torch_dtype, device=device, requires_grad=requires_grad)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_torch()
        torch_dtype = getattr(torch, DTYPE_MAP.get(dtype, "float32")) if dtype else None
        return torch.zeros(shape, dtype=torch_dtype, device=self._device)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_torch()
        torch_dtype = getattr(torch, DTYPE_MAP.get(dtype, "float32")) if dtype else None
        return torch.ones(shape, dtype=torch_dtype, device=self._device)

    def randn(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        _ensure_torch()
        torch_dtype = getattr(torch, DTYPE_MAP.get(dtype, "float32")) if dtype else None
        return torch.randn(shape, dtype=torch_dtype, device=self._device)

    def to_device(self, tensor: TensorLike, device: str) -> TensorLike:
        _ensure_torch()
        return tensor.to(device)

    def to_numpy(self, tensor: TensorLike) -> np.ndarray:
        _ensure_torch()
        return tensor.detach().cpu().numpy()

    # ==================== Math Operations ====================

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
        return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)

    def gelu(self, x: TensorLike) -> TensorLike:
        _ensure_torch()
        return F.gelu(x, approximate="tanh")

    def silu(self, x: TensorLike) -> TensorLike:
        _ensure_torch()
        return F.silu(x)

    # ==================== Attention Operations ====================

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
        """
        A-100 optimized attention using Flash Attention 2.

        Uses PyTorch 2.0+ scaled_dot_product_attention with
        automatic Flash Attention backend selection.
        """
        _ensure_torch()

        # Use PyTorch 2.0+ native scaled_dot_product_attention
        # Automatically uses Flash Attention on A-100
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

        # Fallback to manual implementation
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = scale or (head_dim ** -0.5)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        if mask is not None:
            attn_weights.masked_fill_(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        return torch.matmul(attn_weights, value)

    # ==================== Gradient Operations ====================

    def backward(self, loss: TensorLike) -> None:
        """Compute gradients with optional AMP scaling."""
        _ensure_torch()
        if self._scaler:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def zero_grad(self, parameters: Sequence[TensorLike]) -> None:
        """Zero out gradients."""
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
        """Clip gradients with AMP support.

        Args:
            parameters: Model parameters to clip gradients for
            max_norm: Maximum gradient norm
            optimizer: Optimizer for AMP unscaling (required when using GradScaler)

        Note:
            When using mixed precision training with GradScaler, pass the
            optimizer to properly unscale gradients before clipping.
        """
        _ensure_torch()
        if self._scaler and optimizer is not None:
            self._scaler.unscale_(optimizer)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    # ==================== JIT Compilation ====================

    def jit_compile(
        self,
        fn: Callable,
        static_argnums: Optional[Tuple[int, ...]] = None,
    ) -> Callable:
        """JIT compile using torch.compile (PyTorch 2.0+)."""
        _ensure_torch()
        if hasattr(torch, "compile"):
            # Use inductor backend for A-100 optimization
            return torch.compile(fn, mode="reduce-overhead", backend="inductor")
        return fn  # Fallback for older PyTorch

    # ==================== Distributed Operations ====================

    def all_reduce(
        self,
        tensor: TensorLike,
        op: str = "sum",
    ) -> TensorLike:
        """All-reduce using torch.distributed."""
        _ensure_torch()
        if not torch.distributed.is_initialized():
            return tensor

        op_map = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.SUM,  # Divide after
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }
        reduce_op = op_map.get(op, torch.distributed.ReduceOp.SUM)

        torch.distributed.all_reduce(tensor, op=reduce_op)

        if op == "mean":
            tensor /= torch.distributed.get_world_size()

        return tensor

    def broadcast(
        self,
        tensor: TensorLike,
        src: int = 0,
    ) -> TensorLike:
        """Broadcast using torch.distributed."""
        _ensure_torch()
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(tensor, src=src)
        return tensor

    # ==================== Memory Management ====================

    def memory_allocated(self) -> int:
        """Return currently allocated CUDA memory in bytes."""
        _ensure_torch()
        return torch.cuda.memory_allocated(self._device)

    def memory_reserved(self) -> int:
        """Return currently reserved CUDA memory in bytes."""
        _ensure_torch()
        return torch.cuda.memory_reserved(self._device)

    def empty_cache(self) -> None:
        """Free cached CUDA memory."""
        _ensure_torch()
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        _ensure_torch()
        torch.cuda.synchronize(self._device)

    # ==================== Context Managers ====================

    @contextmanager
    def no_grad(self):
        """Disable gradient computation."""
        _ensure_torch()
        with torch.no_grad():
            yield

    @contextmanager
    def autocast(self, dtype: Optional[DType] = None):
        """Mixed precision context for A-100."""
        _ensure_torch()
        # A-100 supports both BF16 and FP16
        # BF16 is preferred for training stability
        torch_dtype = torch.bfloat16 if dtype == DType.BFLOAT16 else torch.float16
        with torch.cuda.amp.autocast(dtype=torch_dtype):
            yield

    # ==================== Checkpoint Operations ====================

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        path: str,
    ) -> None:
        """Save PyTorch checkpoint."""
        _ensure_torch()
        torch.save(state, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load PyTorch checkpoint.

        Warning: Only load checkpoints from trusted sources.
        Uses weights_only=True by default for safety.
        """
        _ensure_torch()
        map_loc = map_location or self._device
        try:
            return torch.load(path, map_location=map_loc, weights_only=True)
        except TypeError:
            # PyTorch < 1.13 doesn't support weights_only
            return torch.load(path, map_location=map_loc)  # nosec B614

    # ==================== GPU-Specific Methods ====================

    def get_device_count(self) -> int:
        """Get number of GPUs."""
        _ensure_torch()
        return torch.cuda.device_count()

    def get_device_properties(self, device_id: int = 0) -> Dict[str, Any]:
        """Get GPU properties (useful for A-100 detection)."""
        _ensure_torch()
        props = torch.cuda.get_device_properties(device_id)
        return {
            "name": props.name,
            "total_memory": props.total_memory,
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
            "is_a100": "A100" in props.name,
        }

    @property
    def training(self) -> bool:
        """Check if in training mode."""
        return True  # Override in model

    def enable_gradient_checkpointing(self, model: Any) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        _ensure_torch()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    def compile_model(
        self,
        model: Any,
        mode: str = "reduce-overhead",
    ) -> Any:
        """Compile model with torch.compile for A-100 optimization."""
        _ensure_torch()
        if hasattr(torch, "compile"):
            return torch.compile(model, mode=mode, backend="inductor")
        return model
