# Training API

## Overview

The training subsystem provides distributed training pipelines, consensus
algorithms, and TPU-optimized trainers.

## Key modules

### `training.unified_consensus`

Unified consensus training across distributed nodes.

### `training.tpu_v6e_trainer`

TPU v6e-optimized training pipeline with SPMD sharding.

### `training.optimizations`

Advanced training optimizations:

- **Expert Soup Integration** — Merges specialist models
- **Gradient checkpointing** — Memory-efficient backprop
- **Mixed precision** — BF16/FP16 training

### `training.data_preprocessing`

Data pipeline components:

- **Semantic deduplicator** — Removes near-duplicate training samples
- **Quality filter** — Filters low-quality data
- **TPU-optimized processor** — Preprocessing optimized for TPU data pipelines

## Configuration

Training is configured via dataclasses. Use `field_helpers` for clean defaults:

```python
from dataclasses import dataclass
from utils.field_helpers import dict_field, list_field

@dataclass
class MyTrainingConfig:
    learning_rate: float = 1e-4
    metadata: dict = dict_field()
    metrics: list = list_field()
```
