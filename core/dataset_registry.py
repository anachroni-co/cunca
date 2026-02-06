"""
Dataset registry for age adaptation segments and variants.

This module provides a minimal registry and re-exports the core dataclasses.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .age_adaptation.models import DataSegment, AdaptiveContentVariant


class DatasetRegistry:
    """Simple in-memory registry for data segments and their variants."""

    def __init__(self) -> None:
        self._segments: Dict[str, DataSegment] = {}
        self._variants: Dict[str, List[AdaptiveContentVariant]] = {}

    def register_segment(self, segment: DataSegment) -> None:
        self._segments[segment.segment_id] = segment

    def get_segment(self, segment_id: str) -> Optional[DataSegment]:
        return self._segments.get(segment_id)

    def list_segments(self) -> List[DataSegment]:
        return list(self._segments.values())

    def register_variant(self, segment_id: str, variant: AdaptiveContentVariant) -> None:
        self._variants.setdefault(segment_id, []).append(variant)

    def get_variants(self, segment_id: str) -> List[AdaptiveContentVariant]:
        return list(self._variants.get(segment_id, []))


__all__ = [
    "DataSegment",
    "AdaptiveContentVariant",
    "DatasetRegistry",
]
