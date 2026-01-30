# Adding Layers

This guide shows how to add a new layer to CapibaraGPT v3 while following
the project's factorization standards.

## 1. Create the layer

Create a new file under `layers/`. Use the centralized JAX import:

```python
# layers/my_layer.py
from layers.jax_compat import jax, jnp, nn, JAX_AVAILABLE
from layers.base import BaseLayer, LayerConfig

class MyLayerConfig(LayerConfig):
    hidden_size: int = 768

class MyLayer(BaseLayer):
    def __init__(self, config: MyLayerConfig):
        super().__init__(config)
        self.hidden_size = config.hidden_size

    def __call__(self, x, training=False, **kwargs):
        # Your implementation
        return x

    def get_output_shape(self, input_shape):
        return input_shape
```

!!! warning "Do NOT use inline JAX imports"
    The factorization tests will reject `try: import jax` blocks in
    `layers/`. Always use `from layers.jax_compat import ...`.

## 2. Register with ModuleGate (optional)

If your layer has hardware requirements, add it to `core/backends/module_gate.py`:

```python
class ModuleName(str, Enum):
    # ...existing entries...
    MY_LAYER = "my_layer"

_MODULE_REQUIREMENTS = {
    # ...
    ModuleName.MY_LAYER: {"autograd"},  # or empty set
}
```

Then add it to the gate map in `core/modular_model.py`:

```python
_gate_map = {
    # ...
    "my_layer": ModuleName.MY_LAYER,
}
```

## 3. Integrate with the orchestrator (optional)

To apply your layer as a wrapper, use `CompositeWrapper`:

```python
from functools import partial
from layers.ultra_layer_integration import CompositeWrapper

MyLayerWrapper = partial(CompositeWrapper, label="MyLayer", use_residual=True)
```

## 4. Run the tests

```bash
# Factorization guardrails must pass
pytest tests/unit/test_factorization.py -v

# Full suite
pytest
```

The guardrail tests will catch:

- Inline `try: import jax/flax` blocks in `layers/`
- Manual `.reshape(…num_heads…head_dim)` instead of `attention_utils`
- Full wrapper class definitions instead of `CompositeWrapper` + `partial`
