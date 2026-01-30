# Configuration

## Model configuration

The main model is configured via `NeuroAdaptiveConfig` in
`config/model_config.py`:

```python
from config.model_config import NeuroAdaptiveConfig

config = NeuroAdaptiveConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_layers=6,
    plasticity_threshold=0.5,
    adaptation_rate=0.01,
)
```

## Ultra Layer Integration config

The layer orchestrator is configured via `UltraLayerIntegrationConfig`:

```python
from layers.ultra_layer_integration import UltraLayerIntegrationConfig

config = UltraLayerIntegrationConfig(
    hidden_size=768,
    num_layers=12,
    num_heads=12,

    # SSM
    enable_ssm_layers=True,
    ssm_ratio=0.4,

    # Neurogenesis
    enable_neurogenesis=True,
    neurogenesis_frequency=2,
    neurogenesis_sparsity=0.1,

    # Abstract reasoning
    enable_abstract_reasoning=True,
    reasoning_layer_frequency=4,
    reasoning_types=["game_theory", "platonic"],

    # Meta-learning
    enable_meta_la=True,
    meta_la_frequency=3,
    meta_learning_rate=0.01,
    adaptation_steps=5,

    # Distributed attention
    enable_distributed_attention=True,
    distributed_attention_frequency=4,
)
```

## Backend configuration

Each backend accepts a `BackendConfig`:

```python
from core.backends.base import BackendConfig

config = BackendConfig(
    device="auto",           # "cpu", "gpu", "tpu", or "auto"
    mixed_precision=True,
    memory_limit_gb=40,
)
```
