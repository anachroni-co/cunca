# Getting Started

## Prerequisites

- Python 3.10+
- NumPy (required)
- Optional: JAX + Flax (for TPU layers), PyTorch (for GPU backend)

## Installation

```bash
git clone https://github.com/anachroni-co/capibaraGPT_v3.git
cd capibaraGPT_v3
pip install -r requirements.txt
```

## Verify installation

```bash
# Run the full test suite
pytest

# Run only factorization guardrails
pytest tests/unit/test_factorization.py -v
```

You should see 250+ tests passing with ~35 skipped (hardware-dependent).

## Basic usage

```python
from core.backends import create_backend
from core.modular_model import ModularModel

# Auto-select best backend
backend = create_backend("auto")

# Create model
model = ModularModel(backend=backend)
```

## Next steps

- [Configuration](configuration.md) — Customize model and layer parameters
- [Architecture Overview](../architecture/overview.md) — Understand the system design
- [Adding Layers](adding-layers.md) — Create custom layers
