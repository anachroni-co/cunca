"""
Benchmark Comparator Module
===========================

Compare benchmark results across runs and detect performance regressions.
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""
    benchmark_name: str
    baseline_mean_ms: float
    current_mean_ms: float
    absolute_diff_ms: float
    relative_diff_percent: float
    is_regression: bool
    is_improvement: bool
    baseline_iterations: int
    current_iterations: int
    significance: str  # "high", "medium", "low", "noise"


@dataclass
class RegressionReport:
    """Report of detected regressions."""
    timestamp: str
    baseline_file: str
    current_file: str
    total_benchmarks: int
    regressions: List[ComparisonResult]
    improvements: List[ComparisonResult]
    unchanged: List[ComparisonResult]
    regression_threshold_percent: float
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    @property
    def regression_count(self) -> int:
        return len(self.regressions)

    @property
    def improvement_count(self) -> int:
        return len(self.improvements)


class BenchmarkComparator:
    """
    Compare benchmark results and detect regressions.

    Example:
        >>> comparator = BenchmarkComparator()
        >>> report = comparator.compare(
        ...     baseline="results/baseline.json",
        ...     current="results/current.json"
        ... )
        >>> if report.has_regressions:
        ...     print("Performance regressions detected!")
    """

    def __init__(
        self,
        regression_threshold_percent: float = 10.0,
        improvement_threshold_percent: float = 10.0,
        noise_threshold_percent: float = 5.0
    ):
        """
        Initialize the comparator.

        Args:
            regression_threshold_percent: % slower to count as regression
            improvement_threshold_percent: % faster to count as improvement
            noise_threshold_percent: Ignore differences below this %
        """
        self.regression_threshold = regression_threshold_percent
        self.improvement_threshold = improvement_threshold_percent
        self.noise_threshold = noise_threshold_percent

    def load_results(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load benchmark results from a JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath) as f:
            return json.load(f)

    def compare(
        self,
        baseline: Union[str, Path, Dict],
        current: Union[str, Path, Dict]
    ) -> RegressionReport:
        """
        Compare two benchmark result sets.

        Args:
            baseline: Baseline results (file path or dict)
            current: Current results (file path or dict)

        Returns:
            RegressionReport with comparison results
        """
        # Load data
        if isinstance(baseline, (str, Path)):
            baseline_data = self.load_results(baseline)
            baseline_file = str(baseline)
        else:
            baseline_data = baseline
            baseline_file = "in-memory"

        if isinstance(current, (str, Path)):
            current_data = self.load_results(current)
            current_file = str(current)
        else:
            current_data = current
            current_file = "in-memory"

        # Extract results
        baseline_results = {
            r['name']: r for r in baseline_data.get('results', [])
        }
        current_results = {
            r['name']: r for r in current_data.get('results', [])
        }

        # Compare each benchmark
        regressions = []
        improvements = []
        unchanged = []

        for name, current_result in current_results.items():
            if name not in baseline_results:
                logger.debug(f"New benchmark (no baseline): {name}")
                continue

            baseline_result = baseline_results[name]
            comparison = self._compare_single(
                name, baseline_result, current_result
            )

            if comparison.is_regression:
                regressions.append(comparison)
            elif comparison.is_improvement:
                improvements.append(comparison)
            else:
                unchanged.append(comparison)

        # Sort by impact
        regressions.sort(key=lambda x: x.relative_diff_percent, reverse=True)
        improvements.sort(key=lambda x: x.relative_diff_percent)

        # Generate summary
        summary = self._generate_summary(
            regressions, improvements, unchanged, baseline_data, current_data
        )

        return RegressionReport(
            timestamp=datetime.now().isoformat(),
            baseline_file=baseline_file,
            current_file=current_file,
            total_benchmarks=len(current_results),
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            regression_threshold_percent=self.regression_threshold,
            summary=summary
        )

    def _compare_single(
        self,
        name: str,
        baseline: Dict,
        current: Dict
    ) -> ComparisonResult:
        """Compare a single benchmark."""
        baseline_mean = baseline.get('timing', {}).get('mean_ms', 0)
        current_mean = current.get('timing', {}).get('mean_ms', 0)

        # Handle zero baseline
        if baseline_mean == 0:
            relative_diff = 0.0
        else:
            relative_diff = ((current_mean - baseline_mean) / baseline_mean) * 100

        absolute_diff = current_mean - baseline_mean

        # Determine significance
        abs_relative = abs(relative_diff)
        if abs_relative < self.noise_threshold:
            significance = "noise"
        elif abs_relative < self.regression_threshold:
            significance = "low"
        elif abs_relative < self.regression_threshold * 2:
            significance = "medium"
        else:
            significance = "high"

        # Determine if regression or improvement
        is_regression = relative_diff > self.regression_threshold
        is_improvement = relative_diff < -self.improvement_threshold

        return ComparisonResult(
            benchmark_name=name,
            baseline_mean_ms=baseline_mean,
            current_mean_ms=current_mean,
            absolute_diff_ms=absolute_diff,
            relative_diff_percent=relative_diff,
            is_regression=is_regression,
            is_improvement=is_improvement,
            baseline_iterations=baseline.get('timing', {}).get('iterations', 0),
            current_iterations=current.get('timing', {}).get('iterations', 0),
            significance=significance
        )

    def _generate_summary(
        self,
        regressions: List[ComparisonResult],
        improvements: List[ComparisonResult],
        unchanged: List[ComparisonResult],
        baseline_data: Dict,
        current_data: Dict
    ) -> Dict[str, Any]:
        """Generate comparison summary statistics."""
        all_comparisons = regressions + improvements + unchanged

        if not all_comparisons:
            return {}

        relative_diffs = [c.relative_diff_percent for c in all_comparisons]

        summary = {
            "total_compared": len(all_comparisons),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "unchanged": len(unchanged),
            "mean_change_percent": statistics.mean(relative_diffs),
            "median_change_percent": statistics.median(relative_diffs),
            "worst_regression_percent": max(
                (c.relative_diff_percent for c in regressions),
                default=0
            ),
            "best_improvement_percent": min(
                (c.relative_diff_percent for c in improvements),
                default=0
            ),
            "baseline_timestamp": baseline_data.get('timestamp', ''),
            "current_timestamp": current_data.get('timestamp', ''),
        }

        # Hardware comparison
        baseline_hw = baseline_data.get('hardware_info', {})
        current_hw = current_data.get('hardware_info', {})

        if baseline_hw.get('git_commit') != current_hw.get('git_commit'):
            summary['git_commits_differ'] = True
            summary['baseline_commit'] = baseline_hw.get('git_commit', 'unknown')
            summary['current_commit'] = current_hw.get('git_commit', 'unknown')

        return summary


def compare_results(
    baseline: Union[str, Path, Dict],
    current: Union[str, Path, Dict],
    threshold_percent: float = 10.0
) -> RegressionReport:
    """
    Convenience function to compare benchmark results.

    Args:
        baseline: Baseline results
        current: Current results
        threshold_percent: Regression threshold

    Returns:
        RegressionReport
    """
    comparator = BenchmarkComparator(
        regression_threshold_percent=threshold_percent
    )
    return comparator.compare(baseline, current)


def detect_regressions(
    baseline: Union[str, Path, Dict],
    current: Union[str, Path, Dict],
    threshold_percent: float = 10.0,
    fail_on_regression: bool = True
) -> Tuple[bool, RegressionReport]:
    """
    Detect regressions and optionally fail if found.

    Args:
        baseline: Baseline results
        current: Current results
        threshold_percent: Regression threshold
        fail_on_regression: Raise exception if regressions found

    Returns:
        Tuple of (has_regressions, report)

    Raises:
        RuntimeError: If fail_on_regression=True and regressions detected
    """
    report = compare_results(baseline, current, threshold_percent)

    if report.has_regressions:
        logger.warning(
            f"Performance regressions detected: {report.regression_count} "
            f"benchmarks are >{threshold_percent}% slower"
        )

        for reg in report.regressions[:5]:
            logger.warning(
                f"  - {reg.benchmark_name}: "
                f"{reg.baseline_mean_ms:.2f}ms -> {reg.current_mean_ms:.2f}ms "
                f"(+{reg.relative_diff_percent:.1f}%)"
            )

        if fail_on_regression:
            raise RuntimeError(
                f"Performance regression detected: "
                f"{report.regression_count} benchmarks regressed"
            )

    return report.has_regressions, report


def load_baseline(
    results_dir: Union[str, Path],
    baseline_name: str = "baseline.json"
) -> Optional[Dict]:
    """
    Load baseline results from a directory.

    Args:
        results_dir: Directory containing results
        baseline_name: Name of baseline file

    Returns:
        Baseline data or None if not found
    """
    baseline_path = Path(results_dir) / baseline_name
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return None


def update_baseline(
    current_results: Union[str, Path, Dict],
    results_dir: Union[str, Path],
    baseline_name: str = "baseline.json"
) -> Path:
    """
    Update baseline with current results.

    Args:
        current_results: Current results to use as new baseline
        results_dir: Directory to store baseline
        baseline_name: Name for baseline file

    Returns:
        Path to saved baseline
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(current_results, (str, Path)):
        with open(current_results) as f:
            data = json.load(f)
    else:
        data = current_results

    baseline_path = results_dir / baseline_name
    with open(baseline_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Baseline updated: {baseline_path}")
    return baseline_path
