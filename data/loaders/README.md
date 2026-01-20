# Data Loaders

**Efficient Data Loading Pipeline for Training and Inference**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA LOADING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Raw Data Sources                                                           │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                       │
│   │  Text   │  │  JSON   │  │ Parquet │  │ HF Hub  │                       │
│   │  Files  │  │  Files  │  │  Files  │  │Datasets │                       │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                       │
│        │            │            │            │                             │
│        └────────────┴────────────┴────────────┘                             │
│                          │                                                   │
│                          ▼                                                   │
│              ┌───────────────────────┐                                      │
│              │     DataLoader        │                                      │
│              │  ┌─────────────────┐  │                                      │
│              │  │   Tokenization  │  │                                      │
│              │  │   Batching      │  │                                      │
│              │  │   Shuffling     │  │                                      │
│              │  │   Prefetching   │  │                                      │
│              │  └─────────────────┘  │                                      │
│              └───────────┬───────────┘                                      │
│                          │                                                   │
│                          ▼                                                   │
│              ┌───────────────────────┐                                      │
│              │    Training Loop      │                                      │
│              └───────────────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

The data loaders module provides efficient, hardware-optimized data loading for training and inference. It handles tokenization, batching, shuffling, and prefetching with support for multiple data formats and distributed training.

## Module Structure

```
data/loaders/
└── __init__.py    # DataLoader exports and utilities
```

## Quick Start

### Basic Usage

```python
from data.loaders import DataLoader, Dataset
from data.core import create_dataset

# Create dataset
dataset = create_dataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    max_length=512
)

# Create data loader
loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2
)

# Training loop
for batch in loader:
    outputs = model(batch["input_ids"])
    loss = compute_loss(outputs, batch["labels"])
```

### Multi-Dataset Loading

```python
from data.loaders import MultiDatasetLoader

# Load multiple datasets
loader = MultiDatasetLoader(
    datasets={
        "wikipedia": ("data/wiki.jsonl", 0.5),    # 50% weight
        "books": ("data/books.jsonl", 0.3),       # 30% weight
        "code": ("data/code.jsonl", 0.2),         # 20% weight
    },
    batch_size=32,
    tokenizer=tokenizer
)

for batch in loader:
    # Batch contains mixed samples from all datasets
    pass
```

## Features

### Efficient Batching

```
┌──────────────────────────────────────────────────────────────┐
│                     Batching Strategy                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Dynamic Batching (by tokens):                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Batch 1: [seq1: 128 tok] [seq2: 256 tok] [seq3: 128 tok]│ │
│  │          Total: 512 tokens                               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Static Batching (by samples):                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Batch 1: [seq1] [seq2] [seq3] [seq4]                    │ │
│  │          4 samples, padded to max_length                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Bucket Batching (by length similarity):                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Bucket 0-128:   [short sequences grouped together]      │ │
│  │ Bucket 128-256: [medium sequences grouped together]     │ │
│  │ Bucket 256-512: [long sequences grouped together]       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Data Prefetching

```python
# Async prefetching for GPU/TPU
loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    prefetch_factor=4,      # Prefetch 4 batches
    pin_memory=True,        # Pin to GPU memory
    persistent_workers=True # Keep workers alive
)
```

### Distributed Loading

```python
from data.loaders import DistributedDataLoader

# For multi-GPU/TPU training
loader = DistributedDataLoader(
    dataset=dataset,
    batch_size=32,
    num_replicas=8,         # Number of devices
    rank=device_id,         # Current device
    shuffle=True,
    drop_last=True
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Samples per batch |
| `shuffle` | bool | True | Shuffle data each epoch |
| `num_workers` | int | 4 | Parallel data loading workers |
| `prefetch_factor` | int | 2 | Batches to prefetch per worker |
| `pin_memory` | bool | True | Pin memory for GPU transfer |
| `drop_last` | bool | False | Drop incomplete final batch |
| `persistent_workers` | bool | True | Keep workers alive between epochs |

## Integration with Backends

```python
from core.backends import get_backend
from data.loaders import DataLoader

backend = get_backend()  # CPU, GPU, or TPU

loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    device=backend.device,
    collate_fn=lambda batch: {
        k: backend.create_tensor(v)
        for k, v in default_collate(batch).items()
    }
)
```

## See Also

- [Data Core](../core/README.md)
- [Data Processors](../processors/README.md)
- [Training Module](../../training/README.md)
