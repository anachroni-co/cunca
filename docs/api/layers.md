# Layers API

## `layers.jax_compat`

Centralized JAX/Flax import guard. Provides fallback objects when
JAX is not installed so that class bodies can be parsed without errors.

```python
from layers.jax_compat import jax, jnp, nn, JAX_AVAILABLE
```

## `layers.attention_utils`

Shared multi-head attention reshape utilities.

### `split_heads(x, num_heads)`

Reshape `(..., seq, hidden)` to `(..., heads, seq, head_dim)`.

### `merge_heads(x, hidden_size)`

Inverse of `split_heads`.

## `layers.ultra_layer_integration`

### `UltraLayerIntegrationConfig`

Dataclass controlling the layer stack composition. See
[Configuration](../guides/configuration.md) for all fields.

### `UltraLayerOrchestrator`

Builds and manages the full layer stack.

```python
from layers.ultra_layer_integration import create_ultra_layer_system

system = create_ultra_layer_system(hidden_size=768, num_layers=12)
status = system.get_system_status()
```

### `CompositeWrapper`

Generic wrapper with `use_residual` and `pass_training` flags.
Pre-configured aliases:

| Alias | `use_residual` | `pass_training` |
|-------|---------------|----------------|
| `NeurogenesisWrapper` | `False` | `True` |
| `ReasoningWrapper` | `True` | `False` |
| `MetaLAWrapper` | `True` | `True` |
| `DistributedAttentionWrapper` | `True` | `True` |

## `layers.meta_la`

### `MetaLA`

MAML-style meta-learning layer with inner-loop gradient adaptation.

```python
from layers.meta_la import MetaLA, MetaLAConfig

config = MetaLAConfig(hidden_size=768, meta_learning_rate=0.01)
meta = MetaLA(config)
output = meta.fast_adapt(support_x, support_y, query_x)
```

## `layers.sparsity`

### `BitNet` — 1-bit quantization with straight-through estimator
### `MixtureOfRookies` — Sparse expert routing
### `SparseCapibara` — Sparse multi-head attention
### `AffineQuantizer` — Affine weight quantization
