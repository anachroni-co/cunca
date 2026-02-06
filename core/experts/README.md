# Experts Module

Minimal, working expert utilities for CapibaraGPT v3.
Includes MoE control/training **stubs** and a functional nested expert hierarchy.

## Architecture

```
core/experts/
+-- __init__.py          # Exports
+-- moe_control_api.py   # MoE control + health reporting (simulated metrics)
+-- moe_training.py      # MoE training loop (simulated metrics)
+-- nested_experts.py    # Nested expert hierarchy (real logic, simulated weights)
+-- README.md
```

## What Is Real Here

- `NestedExpertHierarchy` provides a real selection/update flow, but uses
  simulated parameters (no neural weights).
- `MoEControlAPI` and `MoETrainingSystem` are **stubs** that return synthetic
  metrics unless you pass a real model with `moe_layers` and `moe_manager`.

## MoE Control API

```python
from capibara.core.experts import MoEControlAPI

# The control API expects a model that exposes `moe_layers`.
class DummyMoEModel:
    moe_layers = {}
    moe_manager = None

control_api = MoEControlAPI(DummyMoEModel())
health = control_api.get_system_health()
print(health["status"])  # "unavailable" with dummy model
```

## MoE Training System

```python
from capibara.core.experts import MoETrainingSystem

class DummyMoEModel:
    moe_layers = {}
    moe_manager = None

trainer = MoETrainingSystem(DummyMoEModel())
results = trainer.train_expert_specialization(
    training_data={"general": ["example 1", "example 2"]}
)
print(results.get("error"))  # MoE system not available
```

## Nested Expert Hierarchy

```python
from capibara.core.experts import create_nested_expert_hierarchy

hierarchy = create_nested_expert_hierarchy()
output = hierarchy.forward(inputs={"text": "Explain gravity"})
print(output["num_experts_used"], output["outputs"][0]["expert_type"])
```

## Notes

- The MoE files are intentionally conservative and avoid claiming full training.
- If you wire in a real MoE model, the control/training APIs can be extended to
  surface real metrics.
