"""
Test-Time Compute (TTC) scaling utilities.

Provides lightweight heuristics to scale generation parameters based on
prompt difficulty. Uses CPU-safe logic and has no hard dependency on JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ComputeStrategy(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
    AUTO = "auto"


@dataclass
class TestTimeConfig:
    base_max_new_tokens: int = 128
    min_new_tokens: int = 16
    max_new_tokens: int = 512
    base_temperature: float = 0.7
    min_temperature: float = 0.2
    max_temperature: float = 1.0
    difficulty_thresholds: tuple[float, float] = (0.35, 0.7)
    strategy_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class TestTimeComputeScaler:
    """Compute scaler for test-time generation."""

    def __init__(self, config: TestTimeConfig | None = None) -> None:
        self.config = config or TestTimeConfig()
        self.strategy_counts: Dict[str, int] = {}

    def estimate_difficulty(self, prompt: str) -> float:
        """Heuristic difficulty in [0, 1] based on length and symbols."""
        if not prompt:
            return 0.0
        length_factor = min(len(prompt) / 400.0, 1.0)
        digit_factor = min(sum(ch.isdigit() for ch in prompt) / 20.0, 1.0)
        symbol_factor = min(sum(ch in "{}[]=><+-*/" for ch in prompt) / 20.0, 1.0)
        return min(1.0, 0.6 * length_factor + 0.2 * digit_factor + 0.2 * symbol_factor)

    def select_strategy(self, prompt: str, strategy: ComputeStrategy) -> ComputeStrategy:
        if strategy != ComputeStrategy.AUTO:
            return strategy
        difficulty = self.estimate_difficulty(prompt)
        low, high = self.config.difficulty_thresholds
        if difficulty < low:
            return ComputeStrategy.FAST
        if difficulty < high:
            return ComputeStrategy.BALANCED
        return ComputeStrategy.DEEP

    def scale_parameters(self, prompt: str, strategy: ComputeStrategy = ComputeStrategy.AUTO) -> Dict[str, Any]:
        selected = self.select_strategy(prompt, strategy)
        self.strategy_counts[selected.value] = self.strategy_counts.get(selected.value, 0) + 1

        # Base parameters
        max_new_tokens = self.config.base_max_new_tokens
        temperature = self.config.base_temperature

        if selected == ComputeStrategy.FAST:
            max_new_tokens = int(max(self.config.min_new_tokens, max_new_tokens * 0.5))
            temperature = min(self.config.max_temperature, temperature + 0.1)
        elif selected == ComputeStrategy.DEEP:
            max_new_tokens = int(min(self.config.max_new_tokens, max_new_tokens * 1.5))
            temperature = max(self.config.min_temperature, temperature - 0.1)

        # Allow overrides per strategy
        overrides = self.config.strategy_overrides.get(selected.value, {})
        if "max_new_tokens" in overrides:
            max_new_tokens = int(overrides["max_new_tokens"])
        if "temperature" in overrides:
            temperature = float(overrides["temperature"])

        return {
            "strategy": selected,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }


class CapibaraTestTimeScaler(TestTimeComputeScaler):
    """Capibara-specific TTC scaler (currently same as base)."""

    def scale(self, prompt: str, strategy: ComputeStrategy = ComputeStrategy.AUTO) -> Dict[str, Any]:
        return self.scale_parameters(prompt, strategy)


__all__ = [
    "ComputeStrategy",
    "TestTimeConfig",
    "TestTimeComputeScaler",
    "CapibaraTestTimeScaler",
]
