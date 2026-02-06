"""
Age adaptation data models.

Defines core dataclasses used by the age adaptation system and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataSegment:
    """Represents a content segment that can be adapted for a target age."""

    segment_id: str
    content: str
    complexity_level: float = 0.0
    educational_value: float = 0.0
    maturity_themes: List[str] = field(default_factory=list)
    adaptation_strategies: List[str] = field(default_factory=list)
    _content_embedding: Optional[Any] = None

    def set_content_embedding(self, embedding: Any) -> None:
        self._content_embedding = embedding


@dataclass
class AdaptiveContentVariant:
    """Represents an adapted variant of a content segment."""

    variant_id: str
    target_age_range: Tuple[int, int]
    adaptation_type: str
    adapted_content: str
    adaptation_metadata: Dict[str, Any] = field(default_factory=dict)
    age_appropriateness_score: float = 0.0
    educational_effectiveness: float = 0.0
    information_preservation: float = 0.0
    _adapted_embedding: Optional[Any] = None

    def set_adapted_embedding(self, embedding: Any) -> None:
        self._adapted_embedding = embedding

    @property
    def target_age_label(self) -> str:
        return f"{self.target_age_range[0]}-{self.target_age_range[1]}"
