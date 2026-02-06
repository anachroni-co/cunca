# Age Adaptation Module

Age-based content adaptation utilities with optional JAX acceleration.

## Architecture

```
core/age_adaptation/
+-- age_config.py        # Hardware-specific configuration
+-- models.py            # DataSegment + AdaptiveContentVariant
+-- metrics.py           # Adaptation metrics (JAX optional)
+-- __init__.py          # Main exports
+-- README.md
```

Related module:

```
core/dataset_registry.py # DatasetRegistry + re-exports of models
```

## Main Components

### 1) Hardware Configuration (`core/age_adaptation/age_config.py`)

```python
from capibara.core.age_adaptation.age_config import AgeAdaptationConfig, HardwareType

config = AgeAdaptationConfig(
    hardware_type=HardwareType.TPU_V4,
    use_tpu=True,
    use_arm_optimization=False,
)

config = AgeAdaptationConfig(
    hardware_type=HardwareType.ARM_AXION,
    use_tpu=False,
    use_arm_optimization=True,
)

config.validate()
```

### 2) Dataset Registry (`core/dataset_registry.py`)

```python
from capibara.core.dataset_registry import DatasetRegistry, DataSegment, AdaptiveContentVariant

registry = DatasetRegistry()

segment = DataSegment(
    segment_id="seg-001",
    content="Photosynthesis is the biochemical process...",
    complexity_level=0.6,
    educational_value=0.8,
    maturity_themes=["science"],
    adaptation_strategies=["vocabulary", "examples"],
)
registry.register_segment(segment)

variant = AdaptiveContentVariant(
    variant_id="var-001",
    target_age_range=(8, 10),
    adaptation_type="simplify",
    adapted_content="Plants use sunlight to make their food.",
    adaptation_metadata={"strategy": "simplify"},
)
registry.register_variant(segment.segment_id, variant)

segments = registry.list_segments()
variants = registry.get_variants(segment.segment_id)
```

### 3) Metrics (`core/age_adaptation/metrics.py`)

Metrics work without JAX; advanced metrics are enabled only when JAX is available.

```python
from capibara.core.age_adaptation.metrics import (
    ADVANCED_METRICS_AVAILABLE,
    evaluate_age_appropriateness,
    evaluate_batch_adaptations,
)

metrics = evaluate_age_appropriateness(segment, variant)

if ADVANCED_METRICS_AVAILABLE:
    batch_metrics = evaluate_batch_adaptations(
        original_embeddings=[...],
        adapted_embeddings=[...],
        target_ages=[...],
    )
```

## Notes

- `AdaptationPipeline`, `ContentAppropriatenessScorer`, and monitoring hooks are not implemented in this repo yet.
- The module is intentionally optional: it loads without JAX and exposes CPU-safe metrics.
