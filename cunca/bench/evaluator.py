"""CUNCA-Bench evaluator — runs all 12 tasks against a model."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cunca.bench.tasks import BenchResult, BenchTask, REGISTRY, list_tasks

logger = logging.getLogger(__name__)


@dataclass
class BenchReport:
    """Aggregated results across all tasks."""
    results: list[BenchResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        valid = [r for r in self.results if r.passed]
        if not valid:
            return 0.0
        return sum(r.score for r in valid) / len(valid)

    def by_category(self) -> dict[str, float]:
        cats: dict[str, list[float]] = {}
        for r in self.results:
            task = REGISTRY.get(r.task_id)
            cat = task.category if task else "unknown"
            cats.setdefault(cat, []).append(r.score)
        return {cat: sum(scores) / len(scores) for cat, scores in cats.items()}

    def summary(self) -> str:
        lines = [f"CUNCA-Bench  overall={self.overall_score:.3f}"]
        for cat, score in sorted(self.by_category().items()):
            lines.append(f"  {cat:<16} {score:.3f}")
        for r in self.results:
            status = "OK" if r.passed else f"ERR({r.error})"
            lines.append(f"  [{r.task_id}] {r.task_name:<25} score={r.score:.3f}  n={r.n_samples}  {status}")
        return "\n".join(lines)


class CUNCABench:
    """Runs the CUNCA-Bench evaluation suite.

    The evaluator is model-agnostic: it calls a user-supplied generate()
    function for each prompt and scores the outputs.

    Args:
        generate_fn: Callable[str, str] — takes a prompt, returns a string.
        tasks: Optional list of task IDs to run. Default: all 12 tasks.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        tasks: Optional[list[str]] = None,
    ) -> None:
        self.generate_fn = generate_fn
        if tasks is None:
            self._tasks = list(REGISTRY.values())
        else:
            self._tasks = [REGISTRY[tid] for tid in tasks if tid in REGISTRY]

    def run(self) -> BenchReport:
        """Execute all configured tasks and return a BenchReport."""
        report = BenchReport()
        for task in self._tasks:
            result = self._run_task(task)
            report.results.append(result)
            logger.info("[%s] %s  score=%.3f", task.id, task.name, result.score)
        return report

    def _run_task(self, task: BenchTask) -> BenchResult:
        if not task.samples:
            return BenchResult(
                task_id=task.id,
                task_name=task.name,
                score=0.0,
                n_samples=0,
                error="no samples",
            )
        try:
            predictions = [self.generate_fn(s.prompt) for s in task.samples]
            return task.evaluate(predictions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Task %s failed: %s", task.id, exc)
            return BenchResult(
                task_id=task.id,
                task_name=task.name,
                score=0.0,
                n_samples=len(task.samples),
                error=str(exc),
            )

    @staticmethod
    def available_tasks() -> list[str]:
        return list(REGISTRY.keys())


__all__ = ["CUNCABench", "BenchReport"]
