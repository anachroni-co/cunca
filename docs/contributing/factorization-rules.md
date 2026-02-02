# Factorization Rules

These rules are **enforced by automated tests** in
`tests/unit/test_factorization.py`. CI will fail if any rule is violated.

## Rules

### 1. No inline JAX/Flax imports in `layers/`

Files under `layers/` must not contain `try: import jax` or
`try: from flax` blocks. Use `layers.jax_compat` instead.

**Allowed exceptions:** `jax_compat.py` itself, `core/backends/lazy_import.py`,
`core/backends/tpu_backend.py`.

### 2. No inline torch imports in `core/backends/`

Backend files must not contain `try: import torch` blocks. Use
`core.backends.lazy_import.ensure_torch()`.

**Allowed exceptions:** `lazy_import.py`, `gpu_backend.py`, `utils.py`.

### 3. No try/except imports in `core/__init__.py`

All optional imports must use `safe_import` from `core.import_utils`.

### 4. No manual head reshapes in `layers/`

Do not write `.reshape(..., num_heads, head_dim)` manually. Use
`layers.attention_utils.split_heads()` and `merge_heads()`.

**Allowed exceptions:** `attention_utils.py`, `sparsity/sparse_capibara.py`
(reshape after `jnp.split`, not from fused dimension).

### 5. No redundant Wrapper classes

`layers/ultra_layer_integration.py` must not define full class bodies for
wrappers. Use `functools.partial(CompositeWrapper, ...)`.

### 6. Utilities must be importable

All centralized utility modules must exist and export expected symbols:

- `layers.jax_compat` — `jax`, `jnp`, `nn`, `JAX_AVAILABLE`
- `layers.attention_utils` — `split_heads`, `merge_heads`
- `core.import_utils` — `safe_import`, `safe_import_module`, `jax_bundle`
- `core.backends.lazy_import` — `ensure_torch`, `ensure_jax`
- `utils.field_helpers` — `dict_field`, `list_field`, `set_field`

## Adding exceptions

If a file legitimately needs to bypass a rule, add its path to the
corresponding `ALLOWED_*_FILES` set in `test_factorization.py` with a comment
explaining why.
