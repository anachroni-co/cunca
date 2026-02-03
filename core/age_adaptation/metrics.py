"""
Utilities for metrics and evaluation of the age adaptation system.
Optimized for TPU v4-32 and ARM Axion.
"""

from functools import partial
import capibara.jax.numpy as jnp
from capibara.jax import jit, vmap
from typing import Dict, List, Optional, Tuple, Any

from ..core.dataset_registry import DataSegment, AdaptiveContentVariant

@jit
def compute_adaptation_quality(
    original_embedding: jnp.ndarray,
    adapted_embedding: jnp.ndarray
) -> float:
    """Calculate adaptation quality using cosine similarity"""
    return jnp.dot(original_embedding, adapted_embedding) / (
        jnp.linalg.norm(original_embedding) * jnp.linalg.norm(adapted_embedding)
    )

@partial(jit, static_argnums=(3,))
def evaluate_batch_adaptations(
    original_embeddings: jnp.ndarray,
    adapted_embeddings: jnp.ndarray,
    target_ages: jnp.ndarray,
    batch_size: int = 128
) -> Dict[str, jnp.ndarray]:
    """Evaluate batch of adaptations"""

    # compute metrics in parallel
    qualities = vmap(compute_adaptation_quality)(
        original_embeddings,
        adapted_embeddings
    )

    # calculate statistics
    return {
        "mean_quality": jnp.mean(qualities),
        "min_quality": jnp.min(qualities),
        "max_quality": jnp.max(qualities),
        "std_quality": jnp.std(qualities)
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
    metrics.update({
        "age_appropriateness": variant.age_appropriateness_score,
        "educational_value": variant.educational_effectiveness,
        "adaptation_coverage": len(variant.adaptation_metadata) / len(segment.adaptation_strategies)
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
