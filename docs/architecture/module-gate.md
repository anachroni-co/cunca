# Module Gate

The module gate system provides runtime capability detection — it enables
or disables modules based on available hardware and libraries.

## How it works

```python
from core.backends.module_gate import ModuleGate, ModuleName

gate = ModuleGate(backend)

if gate.is_available(ModuleName.META_LEARNING_ATTENTION):
    # MetaLA requires autograd support
    layer = MetaLA(config)
```

## Available modules

| Module | Requirements | Description |
|--------|-------------|-------------|
| `SSM_HYBRID` | tpu_optimized | SSM/Mamba hybrid layers |
| `FLASH_ATTENTION` | flash_attention | Flash Attention 2 |
| `SPARSE_COMPUTATION` | — | Sparse capibara, BitNet |
| `META_LEARNING_ATTENTION` | autograd | MAML-style MetaLA |
| `DISTRIBUTED_ATTENTION` | — | Multi-head distributed attention |
| `ABSTRACT_REASONING` | — | Platonic, GameTheory |
| `SYNAPTIC_PLASTICITY` | autograd | Neuroplasticity adaptation |

## Gate map

The `ModularModel` maps component names to `ModuleName` entries:

```python
_gate_map = {
    "ssm": ModuleName.SSM_HYBRID,
    "flash_attention": ModuleName.FLASH_ATTENTION,
    "meta_la": ModuleName.META_LEARNING_ATTENTION,
    "distributed_attention": ModuleName.DISTRIBUTED_ATTENTION,
    "abstract_reasoning": ModuleName.ABSTRACT_REASONING,
    "neurogenesis": ModuleName.NEUROGENESIS,
    "synaptic_plasticity": ModuleName.SYNAPTIC_PLASTICITY,
}
```

Modules that fail the gate check are silently skipped, enabling graceful
degradation across different environments.
