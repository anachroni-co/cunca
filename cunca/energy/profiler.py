"""CUNCA energy profiler — Joules/token via NVIDIA SMI.

Measures GPU power draw at ~100 ms intervals during a generation call
and integrates to compute total Joules consumed.

Usage:
    from cunca.energy.profiler import EnergyProfiler

    profiler = EnergyProfiler()
    with profiler.measure() as ctx:
        tokens = model.generate(...)
    result = ctx.result(n_tokens=len(tokens))
    print(f"{result.joules_per_token:.4f} J/token")

Falls back gracefully when nvidia-smi is not available (returns zeros).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

logger = logging.getLogger(__name__)

_SAMPLE_INTERVAL_S = 0.1   # 100 ms polling interval


def _nvidia_smi_available() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, timeout=3,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _read_power_watts() -> list[float]:
    """Return current power draw (W) for all GPUs. Returns [] on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        watts = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line and line != "N/A":
                try:
                    watts.append(float(line))
                except ValueError:
                    pass
        return watts
    except Exception:  # noqa: BLE001
        return []


@dataclass
class EnergyResult:
    """Energy measurement outcome."""
    total_joules: float        # GPU energy consumed during measurement
    wall_time_s: float         # wall-clock seconds elapsed
    n_tokens: int              # tokens generated
    n_samples: int             # number of power samples taken
    gpu_available: bool        # whether nvidia-smi was available

    @property
    def joules_per_token(self) -> float:
        if self.n_tokens <= 0:
            return 0.0
        return self.total_joules / self.n_tokens

    @property
    def tokens_per_joule(self) -> float:
        if self.total_joules <= 0:
            return 0.0
        return self.n_tokens / self.total_joules

    @property
    def average_power_watts(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_joules / self.wall_time_s

    def __str__(self) -> str:
        if self.gpu_available:
            return (
                f"EnergyResult: {self.total_joules:.3f} J total, "
                f"{self.joules_per_token:.4f} J/token, "
                f"{self.average_power_watts:.1f} W avg, "
                f"{self.wall_time_s:.2f} s, "
                f"{self.n_tokens} tokens"
            )
        return f"EnergyResult: GPU unavailable, {self.wall_time_s:.2f} s elapsed"


class _MeasurementContext:
    """Internal context that collects power samples in a background thread."""

    def __init__(self, gpu_available: bool) -> None:
        self._gpu_available = gpu_available
        self._samples: list[float] = []    # watts, one sample per interval
        self._start: float = 0.0
        self._end: float = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._start = time.perf_counter()
        if self._gpu_available:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._end = time.perf_counter()
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            watts = _read_power_watts()
            if watts:
                self._samples.append(sum(watts))   # total across all GPUs
            self._stop_event.wait(timeout=_SAMPLE_INTERVAL_S)

    def result(self, n_tokens: int = 0) -> EnergyResult:
        wall = self._end - self._start
        # Integrate: each sample represents _SAMPLE_INTERVAL_S seconds of power
        if self._samples:
            total_j = sum(self._samples) * _SAMPLE_INTERVAL_S
        else:
            total_j = 0.0
        return EnergyResult(
            total_joules=total_j,
            wall_time_s=wall,
            n_tokens=n_tokens,
            n_samples=len(self._samples),
            gpu_available=self._gpu_available,
        )


class EnergyProfiler:
    """Measures GPU energy consumption during model generation.

    Automatically detects nvidia-smi availability; falls back to
    wall-clock-only mode when GPU tools are unavailable (CI, CPU machines).
    """

    def __init__(self) -> None:
        self._gpu_available = _nvidia_smi_available()
        if not self._gpu_available:
            logger.info("nvidia-smi not found — energy profiler in CPU-only mode")

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    @contextmanager
    def measure(self) -> Generator["_MeasurementContext", None, None]:
        """Context manager: yields a _MeasurementContext; call .result(n_tokens) after."""
        ctx = _MeasurementContext(self._gpu_available)
        ctx.start()
        try:
            yield ctx
        finally:
            ctx.stop()

    def profile_callable(self, fn, n_tokens: int = 0) -> EnergyResult:
        """Run fn() inside a measurement context and return EnergyResult."""
        with self.measure() as ctx:
            fn()
        return ctx.result(n_tokens=n_tokens)


@contextmanager
def energy_context() -> Generator["_MeasurementContext", None, None]:
    """Module-level convenience wrapper for EnergyProfiler.measure()."""
    profiler = EnergyProfiler()
    with profiler.measure() as ctx:
        yield ctx


__all__ = ["EnergyProfiler", "EnergyResult", "energy_context"]
