# Activations Module (Core)

This subpackage provides **real, minimal contextual activations** implemented in
`contextual_activation.py`. It is CPU-friendly and works with the local
`capibara.jax` fallback.

## Available API

- `contextual_activation.apply(x, context=None, activation="gelu", context_gain=0.1)`
- `ContextualActivation`, `ContextualReLU`, `ContextualGELU`, `ContextualSiLU`

## Example

```python
from capibara.core.activations import contextual_activation as ca
import capibara.jax.numpy as jnp

x = jnp.array([[1.0, -0.5, 0.25]])
ctx = jnp.array([0.2, 0.1, 0.0])

out = ca.apply(x, context=ctx, activation="gelu")
print(out)
```

## Notes

- This module intentionally stays small and dependency-light.
- If you need TPU/GPU-specific fused activations, they are **not implemented**
  here yet; add them in `contextual_activation.py` with explicit tests.
