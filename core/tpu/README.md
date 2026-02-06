# capibara/core/tpu

TPU helpers for CapibaraGPT v3. This folder currently exposes a small, honest set of utilities:

- `tpu_config.py` provides a lightweight `TpuConfig` dataclass for generic TPU settings.
- `tpu_v6e.py` contains **experimental** JAX/Flax-based utilities for TPU v6e-64 only.
- `__init__.py` exports the public API and gracefully degrades when TPU deps are missing.

## What exists today

- Generic config container: `TpuConfig` (version, device count, mesh shape, memory fraction, bf16 flag).
- TPU v6e helpers (optional):
  - `TPUv6eConfig`, `TPUv6eOptimizer`, `TPUv6eTrainingPipeline`
  - Optimized attention + Mamba-style SSM layers (Flax modules)
  - A simple benchmark helper

There are **no** dedicated v4/v5e implementations in this folder beyond the generic config object.

## Dependencies

`core/tpu` works without JAX, but the v6e utilities require:

- `jax`
- `flax`
- `optax`

If these libraries are missing, the v6e classes are not available and will raise at runtime.

## Quick usage

```python
from capibara.core.tpu import TpuConfig, TPUv6eOptimizer

config = TpuConfig(tpu_version="v6e", num_devices=64)

optimizer = TPUv6eOptimizer()  # Requires JAX/Flax/optax
attention = optimizer.create_optimized_attention(
    hidden_size=1024,
    num_heads=16,
    max_seq_length=2048,
)
```

## Notes and scope

- The v6e module focuses on TPU-side compute kernels and training helpers, not TPU VM setup.
- Environment variables, TPU VM provisioning, and multi-host orchestration are **out of scope** here.
- Treat `tpu_v6e.py` as experimental; it is designed to be extended as the project matures.

## Related

- `core/distributed` for distributed training scaffolding
- `training` for training loops and configs
