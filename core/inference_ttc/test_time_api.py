"""
Test-Time Compute (TTC) API for inference.

Provides a small wrapper that adjusts generation parameters based on a
heuristic difficulty estimate and a selected compute strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .test_time_scaling import (
    ComputeStrategy,
    TestTimeConfig,
    TestTimeComputeScaler,
)


@dataclass
class TestTimeMetrics:
    total_requests: int = 0
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    avg_difficulty: float = 0.0

    def update(self, difficulty: float, strategy: ComputeStrategy) -> None:
        self.total_requests += 1
        self.strategy_counts[strategy.value] = self.strategy_counts.get(strategy.value, 0) + 1
        # moving average
        n = self.total_requests
        self.avg_difficulty = (self.avg_difficulty * (n - 1) + difficulty) / n


class TestTimeComputeAPI:
    """Simple TTC API for routing test-time compute."""

    def __init__(
        self,
        generate_fn: Optional[Callable[..., Any]] = None,
        config: Optional[TestTimeConfig] = None,
        scaler: Optional[TestTimeComputeScaler] = None,
        router: Optional[Any] = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.config = config or TestTimeConfig()
        self.scaler = scaler or TestTimeComputeScaler(self.config)
        self.router = router
        self.metrics = TestTimeMetrics()

    def _resolve_strategy(self, prompt: str, strategy: ComputeStrategy) -> ComputeStrategy:
        if strategy != ComputeStrategy.AUTO:
            return strategy
        if self.router is not None and hasattr(self.router, "choose_strategy"):
            try:
                routed = self.router.choose_strategy(prompt)
                if isinstance(routed, ComputeStrategy):
                    return routed
                if isinstance(routed, str):
                    return ComputeStrategy(routed)
            except Exception:
                pass
        return strategy

    def generate(self, prompt: str, strategy: ComputeStrategy = ComputeStrategy.AUTO, **kwargs) -> Any:
        if self.generate_fn is None:
            raise RuntimeError("generate_fn is required to use TestTimeComputeAPI")

        chosen = self._resolve_strategy(prompt, strategy)
        difficulty = self.scaler.estimate_difficulty(prompt)
        scaled = self.scaler.scale_parameters(prompt, chosen)

        self.metrics.update(difficulty, scaled["strategy"])

        gen_kwargs = dict(kwargs)
        gen_kwargs.setdefault("max_new_tokens", scaled["max_new_tokens"])
        gen_kwargs.setdefault("temperature", scaled["temperature"])

        return self.generate_fn(prompt, **gen_kwargs)

    def generate_batch(
        self,
        prompts: Iterable[str],
        strategy: ComputeStrategy = ComputeStrategy.AUTO,
        **kwargs,
    ) -> List[Any]:
        return [self.generate(prompt, strategy=strategy, **kwargs) for prompt in prompts]

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_requests": self.metrics.total_requests,
            "strategy_counts": dict(self.metrics.strategy_counts),
            "avg_difficulty": self.metrics.avg_difficulty,
        }


__all__ = [
    "TestTimeComputeAPI",
    "TestTimeMetrics",
]
