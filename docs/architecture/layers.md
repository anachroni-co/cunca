# Layers

All neural network layers inherit from `BaseLayer` and follow a consistent
interface.

## Layer lifecycle

```python
class BaseLayer:
    def __init__(self, config: LayerConfig): ...
    def __call__(self, x, training=False, **kwargs): ...
    def get_output_shape(self, input_shape): ...
```

## CompositeWrapper

Layers can be wrapped with optional components using `CompositeWrapper`:

```python
from functools import partial
from layers.ultra_layer_integration import CompositeWrapper

# Pre-configured aliases
NeurogenesisWrapper = partial(CompositeWrapper, label="Neurogenesis", use_residual=False)
ReasoningWrapper    = partial(CompositeWrapper, label="Reasoning", use_residual=True, pass_training=False)
MetaLAWrapper       = partial(CompositeWrapper, label="MetaLA", use_residual=True)
```

The orchestrator applies wrappers via a data-driven pipeline:

```python
# In UltraLayerOrchestrator._apply_wrappers()
if config.enable_neurogenesis and idx % config.neurogenesis_frequency == 0:
    layer = NeurogenesisWrapper(layer, neurogenesis_module)
if config.enable_abstract_reasoning and idx % config.reasoning_layer_frequency == 0:
    layer = ReasoningWrapper(layer, reasoning_component)
```

## Sparsity layers

### BitNet

1-bit weight quantization with straight-through estimator:

```python
# Forward: use quantized weights
w_quant = self.quantize_weights(w)
# Backward: gradients flow through full-precision weights
w_ste = w + jax.lax.stop_gradient(w_quant - w)
```

### Mixture of Rookies (MoR)

Sparse expert routing with corrected einsum: `'bsfe,bse->bsf'`

## Abstract reasoning

- **Platonic** — Projects inputs onto an idealized unit-sphere manifold
- **GameTheory** — Multi-agent payoff computation with Nash equilibrium approximation

## Meta-learning

**MetaLA** implements MAML-style fast adaptation:

```python
def fast_adapt(self, support_x, support_y, query_x):
    adapted = self.adapt_parameters(support_x, support_y)
    return self.apply_adapted_params(query_x, adapted)
```
