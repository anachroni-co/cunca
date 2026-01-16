"""
Compatibility: stub for SemioticConfig if any import requires it.
"""
from dataclasses import dataclass

@dataclass
class SemioticConfig:
    enabled: bool = False
    max_rules: int = 0
    temperature: float = 0.7

__all__ = ["SemioticConfig"]


