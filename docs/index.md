# CapibaraGPT v3

Advanced multi-backend LLM framework with neuroplasticity, meta-learning,
abstract reasoning, and modular layer orchestration.

## Features

- **Multi-backend support** — CPU, GPU (PyTorch/CUDA), TPU (JAX/Flax) via a unified `ComputeBackend` interface
- **Dynamic layer system** — SSM hybrids, sparse attention, neurogenesis, and meta-learning layers
- **Module gating** — Runtime capability detection enables or disables features based on available hardware and libraries
- **Abstract reasoning** — Platonic ideal forms and game-theory layers for higher-order reasoning
- **Neuroplasticity** — Synaptic plasticity thresholds and adaptation rates configurable per layer
- **Security hardened** — 44 security tests covering input validation, path traversal, injection, and resource limits

## Quick start

```bash
# Clone
git clone https://github.com/anachroni-co/capibaraGPT_v3.git
cd capibaraGPT_v3

# Install
pip install -r requirements.txt

# Run tests
pytest
```

## Project structure

```
capibaraGPT_v3/
├── core/               # Backend system, router, model, optimizers
│   ├── backends/       # CPU, GPU, TPU backends + module gate
│   └── import_utils.py # Centralized safe-import helpers
├── layers/             # All neural network layers
│   ├── abstract_reasoning/  # Platonic, GameTheory, Quineana
│   ├── pasive/              # Attention, embeddings
│   ├── sparsity/            # BitNet, MoR, sparse capibara
│   ├── jax_compat.py        # Centralized JAX/Flax fallback
│   └── ultra_layer_integration.py  # Layer orchestrator
├── training/           # Training pipelines and optimizations
├── inference/          # Inference engines and quantization
├── tests/              # Unit, integration, security, benchmarks
└── docs/               # This documentation
```
