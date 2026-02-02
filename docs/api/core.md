# Core API

## `core.import_utils`

::: core.import_utils

### `safe_import(module_path, *class_names)`

Import one or more names from a module, returning `None` on failure.

```python
from core.import_utils import safe_import

Router = safe_import("core.router", "Router")
Foo, Bar = safe_import("some.module", "Foo", "Bar")
```

### `safe_import_module(module_path)`

Returns `(module, True)` or `(None, False)`.

### `jax_bundle`

Lazy singleton for JAX/Flax/optax imports:

```python
from core.import_utils import jax_bundle

jax = jax_bundle.jax       # None if JAX not installed
jnp = jax_bundle.jnp
nn = jax_bundle.nn          # flax.linen
available = jax_bundle.available  # bool
```

## `core.backends`

### `ComputeBackend`

Abstract base class for all backends. Key methods:

| Method | Description |
|--------|-------------|
| `create_tensor(data, dtype)` | Create a tensor |
| `matmul(a, b)` | Matrix multiplication |
| `softmax(x, axis)` | Softmax activation |
| `layer_norm(x, weight, bias)` | Layer normalization |

### `ModuleGate`

Runtime capability checker:

```python
gate = ModuleGate(backend)
gate.is_available(ModuleName.FLASH_ATTENTION)  # bool
```

### `lazy_import`

Thread-safe lazy import helpers:

```python
from core.backends.lazy_import import ensure_torch, ensure_jax

torch, nn, F = ensure_torch()
jax, jnp, flax, optax = ensure_jax()
```

## `core.router`

### `CoreIntegratedTokenRouter`

Routes tokens to expert sub-networks. Supports both sync and async calling:

```python
router = CoreIntegratedTokenRouter(config)
output = router(input_tensor)  # sync via __call__
```
