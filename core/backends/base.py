"""
Base Backend Interface for CapibaraGPT

Provides abstract interface that all backends must implement,
enabling seamless switching between TPU, GPU, and CPU.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

# Type aliases for backend-agnostic tensor operations
TensorLike = TypeVar("TensorLike")
ArrayLike = Union[List, Tuple, "TensorLike"]


class BackendType(Enum):
    """Available compute backend types."""
    TPU = auto()      # JAX/Flax for TPU
    GPU = auto()      # PyTorch/CUDA for GPU
    CPU = auto()      # NumPy/basic for CPU
    AUTO = auto()     # Auto-detect best available


class DeviceType(Enum):
    """Device placement types."""
    CPU = "cpu"
    GPU = "cuda"
    TPU = "tpu"
    XLA = "xla"


class DType(Enum):
    """Data types supported across backends."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


@dataclass
class BackendConfig:
    """Configuration for compute backend."""
    backend_type: BackendType = BackendType.AUTO
    device: str = "auto"
    dtype: DType = DType.BFLOAT16

    # Memory management
    memory_fraction: float = 0.9
    preallocate_memory: bool = True

    # Distributed settings
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0

    # Optimization flags
    use_mixed_precision: bool = True
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    compile_model: bool = True  # torch.compile / jax.jit

    # TPU-specific
    tpu_topology: str = "v4-32"  # v4-32, v6e-64, etc.
    tpu_cores: int = 32

    # GPU-specific (A-100)
    cuda_device_ids: List[int] = field(default_factory=lambda: [0])
    tensor_cores: bool = True
    cudnn_benchmark: bool = True

    # Debugging
    debug_mode: bool = False
    log_memory: bool = False


@dataclass
class TensorSpec:
    """Specification for tensor creation."""
    shape: Tuple[int, ...]
    dtype: DType = DType.FLOAT32
    device: Optional[str] = None
    requires_grad: bool = False
    name: Optional[str] = None


class ComputeBackend(ABC):
    """
    Abstract base class for compute backends.

    All backends (TPU, GPU, CPU) must implement these methods
    to ensure interoperability.
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._initialized = False
        self._device = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name (e.g., 'tpu', 'gpu', 'cpu')."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the system."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend and set up devices."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up backend resources."""
        pass

    # ==================== Tensor Operations ====================

    @abstractmethod
    def create_tensor(
        self,
        data: ArrayLike,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> TensorLike:
        """Create a tensor from data."""
        pass

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        """Create a tensor filled with zeros."""
        pass

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        """Create a tensor filled with ones."""
        pass

    @abstractmethod
    def randn(self, shape: Tuple[int, ...], dtype: Optional[DType] = None) -> TensorLike:
        """Create a tensor with random normal values."""
        pass

    @abstractmethod
    def to_device(self, tensor: TensorLike, device: str) -> TensorLike:
        """Move tensor to specified device."""
        pass

    @abstractmethod
    def to_numpy(self, tensor: TensorLike) -> Any:
        """Convert tensor to NumPy array."""
        pass

    # ==================== Math Operations ====================

    @abstractmethod
    def matmul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        """Matrix multiplication."""
        pass

    @abstractmethod
    def add(self, a: TensorLike, b: TensorLike) -> TensorLike:
        """Element-wise addition."""
        pass

    @abstractmethod
    def mul(self, a: TensorLike, b: TensorLike) -> TensorLike:
        """Element-wise multiplication."""
        pass

    @abstractmethod
    def softmax(self, x: TensorLike, axis: int = -1) -> TensorLike:
        """Softmax activation."""
        pass

    @abstractmethod
    def layer_norm(
        self,
        x: TensorLike,
        normalized_shape: Tuple[int, ...],
        weight: Optional[TensorLike] = None,
        bias: Optional[TensorLike] = None,
        eps: float = 1e-5,
    ) -> TensorLike:
        """Layer normalization."""
        pass

    @abstractmethod
    def gelu(self, x: TensorLike) -> TensorLike:
        """GELU activation function."""
        pass

    @abstractmethod
    def silu(self, x: TensorLike) -> TensorLike:
        """SiLU/Swish activation function."""
        pass

    # ==================== Attention Operations ====================

    @abstractmethod
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
        Scaled dot-product attention.
        Uses Flash Attention on GPU if available.
        """
        pass

    # ==================== Gradient Operations ====================

    @abstractmethod
    def backward(self, loss: TensorLike) -> None:
        """Compute gradients via backpropagation."""
        pass

    @abstractmethod
    def zero_grad(self, parameters: Sequence[TensorLike]) -> None:
        """Zero out gradients of parameters."""
        pass

    @abstractmethod
    def clip_grad_norm(
        self,
        parameters: Sequence[TensorLike],
        max_norm: float,
    ) -> TensorLike:
        """Clip gradient norms."""
        pass

    # ==================== JIT Compilation ====================

    @abstractmethod
    def jit_compile(
        self,
        fn: Callable,
        static_argnums: Optional[Tuple[int, ...]] = None,
    ) -> Callable:
        """JIT compile a function (jax.jit / torch.compile)."""
        pass

    # ==================== Distributed Operations ====================

    @abstractmethod
    def all_reduce(
        self,
        tensor: TensorLike,
        op: str = "sum",
    ) -> TensorLike:
        """All-reduce operation across devices."""
        pass

    @abstractmethod
    def broadcast(
        self,
        tensor: TensorLike,
        src: int = 0,
    ) -> TensorLike:
        """Broadcast tensor from source to all devices."""
        pass

    # ==================== Memory Management ====================

    @abstractmethod
    def memory_allocated(self) -> int:
        """Return currently allocated memory in bytes."""
        pass

    @abstractmethod
    def memory_reserved(self) -> int:
        """Return currently reserved memory in bytes."""
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        """Free cached memory."""
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize device operations."""
        pass

    # ==================== Context Managers ====================

    @abstractmethod
    def no_grad(self):
        """Context manager for disabling gradient computation."""
        pass

    @abstractmethod
    def autocast(self, dtype: Optional[DType] = None):
        """Context manager for mixed precision."""
        pass

    # ==================== Model Operations ====================

    @abstractmethod
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        path: str,
    ) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        pass

    # ==================== Utility Methods ====================

    def get_device_count(self) -> int:
        """Get number of available devices."""
        return 1

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about current device."""
        return {
            "backend": self.name,
            "device": str(self._device),
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self._device}, initialized={self._initialized})"
