"""Shorthand helpers for common ``dataclasses.field(default_factory=…)`` calls.

Instead of writing::

    metadata: Dict[str, Any] = field(default_factory=dict)
    items: List[str] = field(default_factory=list)
    tags: Set[int] = field(default_factory=set)

you can write::

    from utils.field_helpers import dict_field, list_field, set_field

    metadata: Dict[str, Any] = dict_field()
    items: List[str] = list_field()
    tags: Set[int] = set_field()
"""

from __future__ import annotations

from dataclasses import field
from typing import Any


def dict_field(**kwargs: Any) -> Any:
    """``field(default_factory=dict)``."""
    return field(default_factory=dict, **kwargs)


def list_field(**kwargs: Any) -> Any:
    """``field(default_factory=list)``."""
    return field(default_factory=list, **kwargs)


def set_field(**kwargs: Any) -> Any:
    """``field(default_factory=set)``."""
    return field(default_factory=set, **kwargs)


__all__ = ["dict_field", "list_field", "set_field"]
