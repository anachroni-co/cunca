"""CUNCA-Bench — 12-task evaluation framework for Galician LLMs."""
from __future__ import annotations

from cunca.bench.tasks import (
    BenchTask,
    BenchResult,
    REGISTRY,
    list_tasks,
)
from cunca.bench.evaluator import CUNCABench

__all__ = ["BenchTask", "BenchResult", "CUNCABench", "REGISTRY", "list_tasks"]
