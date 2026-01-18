"""
Core components of the age adaptation system.
"""

try:
    from .dataset_registry import SegmentedDatasetRegistry, SegmentData
except ImportError:
    SegmentedDatasetRegistry = None
    SegmentData = None

__all__ = [
    "SegmentedDatasetRegistry",
    "SegmentData"
]
