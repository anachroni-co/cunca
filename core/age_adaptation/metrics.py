"""
Utilities for metrics and evaluation of the age adaptation system.
Optimized for TPU v4-32 and ARM Axion.
"""

from functools import partial
from typing import Dict, List, Optional, Tuple, Any

try:
    import jax as cjax
    import jax.numpy as jnp
    from jax import jit, vmap
    ADVANCED_METRICS_AVAILABLE = bool(getattr(cjax, "HAS_JAX", False))
except Exception:
    ADVANCED_METRICS_AVAILABLE = False
    try:
        import numpy as jnp  # type: ignore
    except Exception:
        jnp = None  # type: ignore
    def jit(fn=None, **kwargs):
        return fn
    def vmap(fn=None, **kwargs):
        return fn

from .models import DataSegment, AdaptiveContentVariant


class AgeMetrics:
    """Simple container for batch metric summaries."""

    def __init__(self, mean_quality: float, min_quality: float, max_quality: float, std_quality: float):
        self.mean_quality = mean_quality
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.std_quality = std_quality

def _compute_adaptation_quality(
    original_embedding: Any,
    adapted_embedding: Any
) -> float:
    """Calculate adaptation quality using cosine similarity."""
    if jnp is None:
        raise RuntimeError("Metrics backend unavailable (missing numpy/jax).")
    return jnp.dot(original_embedding, adapted_embedding) / (
        jnp.linalg.norm(original_embedding) * jnp.linalg.norm(adapted_embedding)
    )

if ADVANCED_METRICS_AVAILABLE:
    compute_adaptation_quality = jit(_compute_adaptation_quality)
else:
    compute_adaptation_quality = _compute_adaptation_quality


def evaluate_batch_adaptations(
    original_embeddings: Any,
    adapted_embeddings: Any,
    target_ages: Any,
    batch_size: int = 128
) -> Dict[str, Any]:
    """Evaluate batch of adaptations (JAX if available, otherwise CPU fallback)."""

    if jnp is None:
        raise RuntimeError("Metrics backend unavailable (missing numpy/jax).")

    if ADVANCED_METRICS_AVAILABLE:
        qualities = vmap(compute_adaptation_quality)(
            original_embeddings,
            adapted_embeddings
        )
    else:
        qualities = [compute_adaptation_quality(o, a) for o, a in zip(original_embeddings, adapted_embeddings)]

    return {
        "mean_quality": float(jnp.mean(qualities)),
        "min_quality": float(jnp.min(qualities)),
        "max_quality": float(jnp.max(qualities)),
        "std_quality": float(jnp.std(qualities))
    }

def evaluate_age_appropriateness(
    segment: DataSegment,
    variant: AdaptiveContentVariant
) -> Dict[str, float]:
    """Evaluate how appropriate the content is for the target age"""

    metrics = {}

    # Preservation of information
    if segment._content_embedding is not None and variant._adapted_embedding is not None:
        metrics["information_preservation"] = float(compute_adaptation_quality(
            segment._content_embedding,
            variant._adapted_embedding
        ))

    # Adaptation metrics
    strategies_count = len(segment.adaptation_strategies)
    coverage = len(variant.adaptation_metadata) / strategies_count if strategies_count else 0.0

    metrics.update({
        "age_appropriateness": float(variant.age_appropriateness_score),
        "educational_value": float(variant.educational_effectiveness),
        "adaptation_coverage": float(coverage)
    })

    return metrics

def generate_adaptation_report(
    segment: DataSegment,
    variant: AdaptiveContentVariant
) -> Dict[str, Any]:
    """Generate detailed adaptation report"""

    return {
        "segment_info": {
            "id": segment.segment_id,
            "original_complexity": segment.complexity_level,
            "educational_value": segment.educational_value,
            "maturity_themes": list(segment.maturity_themes)
        },
        "adaptation_info": {
            "target_age_range": variant.target_age_range,
            "strategy_used": variant.adaptation_type,
            "metadata": variant.adaptation_metadata
        },
        "metrics": evaluate_age_appropriateness(segment, variant),
        "recommendations": _generate_improvement_recommendations(segment, variant)
    }

def _generate_improvement_recommendations(
    segment: DataSegment,
    variant: AdaptiveContentVariant
) -> List[str]:
    """Generate recommendations to improve the adaptation"""

    recommendations = []

    # analyze information preservation
    if variant.information_preservation < 0.85:
        recommendations.append(
            "Consider strategies to preserve more original information"
        )

    # analyze age appropriateness
    if variant.age_appropriateness_score < 0.9:
        recommendations.append(
            "Review content for better adaptation to target age"
        )

    # analyze educational value
    if variant.educational_effectiveness < segment.educational_value:
        recommendations.append(
            "Explore ways to maintain/improve educational value in adaptation"
        )

    return recommendations
