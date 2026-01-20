# Core Backends

**Hardware Abstraction Layer for Multi-Platform AI Computation**

```
┌─────────────────────────────────────────────────────────────────┐
│                    CapibaraGPT Application                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Backend Abstraction Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ get_backend │  │  registry   │  │    utils    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐      ┌────────────┐
   │ CPUBackend │      │ GPUBackend │      │ TPUBackend │
   │  (NumPy)   │      │ (PyTorch)  │      │ (JAX/Flax) │
   └────────────┘      └────────────┘      └────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐      ┌────────────┐
   │    CPU     │      │   NVIDIA   │      │  Google    │
   │            │      │   A-100    │      │  TPU v4/v6 │
   └────────────┘      └────────────┘      └────────────┘
```

## Overview

The backends module provides a unified interface for running CapibaraGPT across different hardware platforms. This abstraction allows the same model code to run seamlessly on CPU, GPU, or TPU without modification.

### Design Principles

1. **Hardware Agnostic**: Write code once, run anywhere
2. **Automatic Detection**: Automatically selects the best available hardware
3. **Graceful Fallback**: Falls back to CPU if accelerators unavailable
4. **Lazy Loading**: Only loads dependencies when needed (no torch/jax required for CPU)
5. **Consistent API**: Same method signatures across all backends

## Module Structure

```
backends/
├── __init__.py       # Public API exports
├── base.py           # Abstract base class and types
├── registry.py       # Backend registration and discovery
├── cpu_backend.py    # NumPy-based CPU implementation
├── gpu_backend.py    # PyTorch/CUDA GPU implementation
├── tpu_backend.py    # JAX/Flax TPU implementation
└── utils.py          # Hardware detection utilities
```

## Quick Start

### Basic Usage

```python
from core.backends import get_backend, BackendType

# Automatic detection (recommended)
backend = get_backend()
print(f"Using: {backend.name}")  # cpu, gpu, or tpu

# Explicit backend selection
cpu_backend = get_backend(BackendType.CPU)
gpu_backend = get_backend(BackendType.GPU)
tpu_backend = get_backend(BackendType.TPU)
```

### Tensor Operations

```python
from core.backends import get_backend

backend = get_backend()

# Create tensors
a = backend.create_tensor([[1, 2], [3, 4]])
b = backend.randn((2, 2))
zeros = backend.zeros((3, 3))
ones = backend.ones((3, 3))

# Matrix operations
c = backend.matmul(a, b)
d = backend.add(a, b)
e = backend.mul(a, b)

# Convert back to NumPy
result = backend.to_numpy(c)
```

### Neural Network Operations

```python
# Activation functions
x = backend.randn((batch_size, hidden_dim))
gelu_out = backend.gelu(x)
silu_out = backend.silu(x)
softmax_out = backend.softmax(x, axis=-1)

# Normalization
normalized = backend.layer_norm(x, normalized_shape=(hidden_dim,))

# Attention
query = backend.randn((batch, heads, seq_len, head_dim))
key = backend.randn((batch, heads, seq_len, head_dim))
value = backend.randn((batch, heads, seq_len, head_dim))

# Standard attention
attn_out = backend.scaled_dot_product_attention(query, key, value)

# Causal attention (for autoregressive models)
causal_out = backend.scaled_dot_product_attention(query, key, value, is_causal=True)
```

## Backend Details

### CPU Backend (NumPy)

The CPU backend serves as the reference implementation and development fallback.

```python
from core.backends import get_backend, BackendType

backend = get_backend(BackendType.CPU)

# Always available, no special dependencies
assert backend.is_available == True
```

**Features:**
- Pure NumPy implementation
- No GPU/TPU dependencies required
- Ideal for testing and development
- Supports all operations (may be slower)

**Use Cases:**
- Local development and testing
- CI/CD pipelines
- Systems without GPU/TPU
- Debugging and validation

### GPU Backend (PyTorch/CUDA)

Optimized for NVIDIA GPUs, particularly A-100.

```python
from core.backends import get_backend, BackendType

backend = get_backend(BackendType.GPU)

if backend.name == "gpu":
    # GPU-specific features
    print(f"CUDA available: {backend.is_available}")
    print(f"Device: {backend._device}")

    # Flash Attention (if available)
    if backend._flash_attn_available:
        print("Flash Attention 2 enabled")
```

**Features:**
- Flash Attention 2 support
- Automatic Mixed Precision (AMP)
- torch.compile optimization
- Tensor Core utilization
- Gradient checkpointing

**Requirements:**
- PyTorch >= 2.0.0
- CUDA-capable GPU
- Optional: flash-attn package

### TPU Backend (JAX/Flax)

Optimized for Google TPU v4 and v6e pods.

```python
from core.backends import get_backend, BackendType

backend = get_backend(BackendType.TPU)

if backend.name == "tpu":
    # TPU-specific features
    device_count = backend.get_device_count()
    print(f"TPU cores: {device_count}")
```

**Features:**
- XLA compilation
- Automatic sharding
- bfloat16 precision
- Multi-host support
- Pallas kernel integration

**Requirements:**
- JAX >= 0.4.0
- Flax >= 0.7.0
- TPU access (Google Cloud)

## Advanced Usage

### Custom Backend Configuration

```python
from core.backends import get_backend, BackendConfig, BackendType

# Configure backend options
config = BackendConfig(
    backend_type=BackendType.GPU,
    precision="bfloat16",          # float32, float16, bfloat16
    enable_amp=True,               # Automatic Mixed Precision
    memory_fraction=0.9,           # GPU memory limit
    cuda_device_ids=[0, 1],        # Multi-GPU
)

backend = get_backend(BackendType.GPU, config=config)
```

### Context Managers

```python
backend = get_backend()

# Disable gradients for inference
with backend.no_grad():
    output = model(input)

# Mixed precision training
with backend.autocast():
    output = model(input)
    loss = criterion(output, target)
```

### JIT Compilation

```python
backend = get_backend()

def my_function(x, y):
    return backend.add(backend.matmul(x, y), x)

# JIT compile for performance
jit_function = backend.jit_compile(my_function)

# Use compiled function
result = jit_function(a, b)
```

### Memory Management

```python
backend = get_backend()

# Get memory statistics
stats = backend.get_memory_stats()
print(f"Memory used: {stats.get('allocated', 'N/A')}")

# Synchronize device (wait for async operations)
backend.synchronize()
```

## Hardware Detection

```python
from core.backends import list_available_backends, detect_available_hardware

# Check what's available
backends = list_available_backends()
# {'cpu': True, 'gpu': False, 'tpu': False}

# Detailed hardware info
hardware = detect_available_hardware()
# {
#     'cpu': {'cores': 8, 'memory_gb': 32},
#     'gpu': {'available': True, 'count': 2, 'name': 'NVIDIA A100'},
#     'tpu': {'available': False}
# }
```

## Data Type Support

```python
from core.backends.base import DType

# Supported data types
backend.create_tensor([1, 2, 3], dtype=DType.FLOAT32)
backend.create_tensor([1, 2, 3], dtype=DType.FLOAT16)
backend.create_tensor([1, 2, 3], dtype=DType.BFLOAT16)
backend.create_tensor([1, 2, 3], dtype=DType.INT32)
backend.create_tensor([1, 2, 3], dtype=DType.INT64)
backend.create_tensor([True, False], dtype=DType.BOOL)
```

## Performance Comparison

| Operation | CPU (NumPy) | GPU (A100) | TPU (v4-32) |
|-----------|-------------|------------|-------------|
| MatMul 1024x1024 | 50ms | 0.5ms | 0.3ms |
| Attention (seq=512) | 200ms | 2ms | 1.5ms |
| Layer Norm | 10ms | 0.1ms | 0.08ms |
| Softmax | 5ms | 0.05ms | 0.04ms |

*Approximate values for reference. Actual performance varies by hardware.*

## Error Handling

```python
from core.backends import get_backend, BackendType

try:
    backend = get_backend(BackendType.GPU)
except ImportError as e:
    print(f"GPU backend unavailable: {e}")
    backend = get_backend(BackendType.CPU)  # Fallback

# Or use AUTO for automatic fallback
backend = get_backend(BackendType.AUTO)  # Never fails
```

## Extending with Custom Backends

```python
from core.backends.base import ComputeBackend
from core.backends.registry import register_backend, BackendType

class MyCustomBackend(ComputeBackend):
    @property
    def name(self) -> str:
        return "custom"

    @property
    def is_available(self) -> bool:
        return True

    def initialize(self) -> None:
        # Setup code
        pass

    def create_tensor(self, data, dtype=None):
        # Implementation
        pass

    # ... implement other methods

# Register the backend
# register_backend(BackendType.CUSTOM, MyCustomBackend)
```

## Testing

```python
import pytest
from core.backends import get_backend, BackendType

@pytest.fixture
def backend():
    """Use best available backend for tests."""
    return get_backend(BackendType.AUTO)

def test_matmul(backend):
    a = backend.create_tensor([[1, 2], [3, 4]])
    b = backend.create_tensor([[5, 6], [7, 8]])
    result = backend.to_numpy(backend.matmul(a, b))
    expected = [[19, 22], [43, 50]]
    assert result.tolist() == expected
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `get_backend(type)` | Get or create a backend instance |
| `list_available_backends()` | List all available backends |
| `detect_available_hardware()` | Detect hardware capabilities |
| `register_backend(type, cls)` | Register a custom backend |

### Backend Methods

| Method | Description |
|--------|-------------|
| `create_tensor(data, dtype)` | Create tensor from data |
| `zeros(shape, dtype)` | Create zero tensor |
| `ones(shape, dtype)` | Create ones tensor |
| `randn(shape, dtype)` | Create random normal tensor |
| `matmul(a, b)` | Matrix multiplication |
| `add(a, b)` | Element-wise addition |
| `mul(a, b)` | Element-wise multiplication |
| `softmax(x, axis)` | Softmax activation |
| `gelu(x)` | GELU activation |
| `silu(x)` | SiLU/Swish activation |
| `layer_norm(x, shape)` | Layer normalization |
| `scaled_dot_product_attention(q, k, v)` | Attention mechanism |
| `to_numpy(tensor)` | Convert to NumPy array |
| `jit_compile(fn)` | JIT compile function |
| `no_grad()` | Context for inference |
| `autocast()` | Context for mixed precision |

## See Also

- [Core Module README](../README.md)
- [TPU Configuration](../tpu/README.md)
- [Distributed Training](../distributed/README.md)
