# Inference TTC (Test-Time Compute)

Minimal test-time compute scaling for inference. This module is **real code** with
CPU-safe heuristics; it does not claim advanced TPU integrations.

## Architecture

```
core/inference_ttc/
+-- __init__.py
+-- test_time_scaling.py   # Difficulty heuristic + scaling policy
+-- test_time_api.py       # API wrapper for generation
+-- README.md
```

## Quick Start

```python
from capibara.core.inference_ttc import TestTimeComputeAPI, ComputeStrategy

# Basic generate function (hook to your inference engine)
def generate_fn(prompt: str, **kwargs):
    return f"{prompt} :: {kwargs}"

api = TestTimeComputeAPI(generate_fn=generate_fn)

print(api.generate("What is gravity?", strategy=ComputeStrategy.AUTO))
print(api.get_metrics())
```

## How It Works

- `TestTimeComputeScaler` estimates difficulty from prompt length and symbols.
- It scales `max_new_tokens` and `temperature` based on the chosen strategy.
- Strategies: `fast`, `balanced`, `deep`, or `auto`.

## Notes

- This is a lightweight utility intended for integration with the real
  inference engine (`core/inference.py`).
- No TPU, GPU, or distributed logic is required to use it.
