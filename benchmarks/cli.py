#!/usr/bin/env python3
"""
Benchmark CLI
=============

Command-line interface for running benchmarks and generating reports.
Designed for CI/CD pipeline integration.

Usage:
    python -m benchmarks.cli run [options]
    python -m benchmarks.cli compare --baseline FILE --current FILE
    python -m benchmarks.cli report --input FILE --output FILE
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from .runner import BenchmarkRunner, run_benchmarks
from .comparator import compare_results, detect_regressions, update_baseline
from .reporter import generate_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="CapibaraGPT Benchmark Suite CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks
    python -m benchmarks.cli run

    # Run with specific groups
    python -m benchmarks.cli run --groups core attention

    # Compare against baseline
    python -m benchmarks.cli compare --baseline baseline.json --current results.json

    # Generate HTML report
    python -m benchmarks.cli report --input results.json --output report.html

    # CI/CD mode (fail on regressions)
    python -m benchmarks.cli run --ci --baseline baseline.json --threshold 10
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run benchmarks')
    run_parser.add_argument(
        '--output', '-o',
        type=str,
        default='benchmark_results',
        help='Output directory for results'
    )
    run_parser.add_argument(
        '--groups', '-g',
        nargs='+',
        help='Only run benchmarks in these groups'
    )
    run_parser.add_argument(
        '--tags', '-t',
        nargs='+',
        help='Only run benchmarks with these tags'
    )
    run_parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='Warmup iterations (default: 3)'
    )
    run_parser.add_argument(
        '--min-iterations',
        type=int,
        default=5,
        help='Minimum timed iterations (default: 5)'
    )
    run_parser.add_argument(
        '--max-time',
        type=float,
        default=10.0,
        help='Maximum time per benchmark in seconds (default: 10.0)'
    )
    run_parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: compare with baseline and fail on regressions'
    )
    run_parser.add_argument(
        '--baseline',
        type=str,
        help='Baseline file for comparison (required for --ci)'
    )
    run_parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help='Regression threshold percentage (default: 10.0)'
    )
    run_parser.add_argument(
        '--update-baseline',
        action='store_true',
        help='Update baseline with current results'
    )
    run_parser.add_argument(
        '--report',
        choices=['html', 'markdown', 'json', 'none'],
        default='html',
        help='Report format (default: html)'
    )

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare benchmark results')
    compare_parser.add_argument(
        '--baseline', '-b',
        type=str,
        required=True,
        help='Baseline results file'
    )
    compare_parser.add_argument(
        '--current', '-c',
        type=str,
        required=True,
        help='Current results file'
    )
    compare_parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help='Regression threshold percentage'
    )
    compare_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for comparison report'
    )
    compare_parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Exit with error code if regressions detected'
    )

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate benchmark report')
    report_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input results file'
    )
    report_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output report file'
    )
    report_parser.add_argument(
        '--format', '-f',
        choices=['html', 'markdown', 'json'],
        default='html',
        help='Report format'
    )
    report_parser.add_argument(
        '--baseline',
        type=str,
        help='Baseline file for comparison in report'
    )
    report_parser.add_argument(
        '--title',
        type=str,
        default='CapibaraGPT Benchmark Report',
        help='Report title'
    )

    # Update-baseline command
    baseline_parser = subparsers.add_parser(
        'update-baseline',
        help='Update baseline from results'
    )
    baseline_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Results file to use as new baseline'
    )
    baseline_parser.add_argument(
        '--output', '-o',
        type=str,
        default='baseline.json',
        help='Output baseline file'
    )

    return parser


def cmd_run(args) -> int:
    """Run benchmarks command."""
    logger.info("Starting benchmark run...")

    # Create runner
    runner = BenchmarkRunner(
        warmup_iterations=args.warmup,
        min_iterations=args.min_iterations,
        max_time_seconds=args.max_time,
        results_dir=args.output
    )

    # Register default benchmarks
    _register_default_benchmarks(runner)

    # Run benchmarks
    results = runner.run(
        filter_groups=args.groups,
        filter_tags=args.tags
    )

    # Save results
    results_path = runner.save_results()
    runner.print_summary()

    # Generate report
    if args.report != 'none':
        comparison = None

        # CI mode comparison
        if args.ci and args.baseline:
            baseline_path = Path(args.baseline)
            if baseline_path.exists():
                logger.info(f"Comparing with baseline: {baseline_path}")
                has_regression, comparison = detect_regressions(
                    baseline=args.baseline,
                    current=results_path,
                    threshold_percent=args.threshold,
                    fail_on_regression=False
                )

        # Generate report
        report_ext = {'html': '.html', 'markdown': '.md', 'json': '.json'}
        report_path = Path(args.output) / f"report{report_ext.get(args.report, '.html')}"
        generate_report(
            results=results_path,
            output_path=report_path,
            format=args.report,
            comparison=comparison
        )
        logger.info(f"Report generated: {report_path}")

    # Update baseline if requested
    if args.update_baseline:
        baseline_path = Path(args.output) / "baseline.json"
        update_baseline(results_path, args.output)
        logger.info(f"Baseline updated: {baseline_path}")

    # CI mode: check for regressions
    if args.ci and args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            has_regression, report = detect_regressions(
                baseline=args.baseline,
                current=results_path,
                threshold_percent=args.threshold,
                fail_on_regression=False
            )

            if has_regression:
                logger.error(
                    f"CI FAILURE: {report.regression_count} performance regressions detected"
                )
                for reg in report.regressions[:5]:
                    logger.error(
                        f"  - {reg.benchmark_name}: "
                        f"+{reg.relative_diff_percent:.1f}% slower"
                    )
                return 1
            else:
                logger.info("CI PASS: No performance regressions detected")
        else:
            logger.warning(f"Baseline not found: {baseline_path}")

    # Check for failures
    failed = sum(1 for r in results if not r.passed)
    if failed > 0:
        logger.error(f"{failed} benchmark(s) failed")
        return 1

    return 0


def cmd_compare(args) -> int:
    """Compare benchmarks command."""
    logger.info(f"Comparing {args.current} against {args.baseline}")

    has_regression, report = detect_regressions(
        baseline=args.baseline,
        current=args.current,
        threshold_percent=args.threshold,
        fail_on_regression=False
    )

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Baseline: {report.baseline_file}")
    print(f"Current:  {report.current_file}")
    print(f"Threshold: {args.threshold}%")
    print()
    print(f"Regressions:  {report.regression_count}")
    print(f"Improvements: {report.improvement_count}")
    print(f"Unchanged:    {len(report.unchanged)}")

    if report.regressions:
        print("\nRegressions:")
        for reg in report.regressions:
            print(
                f"  - {reg.benchmark_name}: "
                f"{reg.baseline_mean_ms:.2f}ms -> {reg.current_mean_ms:.2f}ms "
                f"(+{reg.relative_diff_percent:.1f}%)"
            )

    if report.improvements:
        print("\nImprovements:")
        for imp in report.improvements[:5]:
            print(
                f"  + {imp.benchmark_name}: "
                f"{imp.baseline_mean_ms:.2f}ms -> {imp.current_mean_ms:.2f}ms "
                f"({imp.relative_diff_percent:.1f}%)"
            )

    print("=" * 60 + "\n")

    # Save report if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "baseline": report.baseline_file,
                "current": report.current_file,
                "threshold": args.threshold,
                "regressions": [
                    {
                        "name": r.benchmark_name,
                        "baseline_ms": r.baseline_mean_ms,
                        "current_ms": r.current_mean_ms,
                        "change_percent": r.relative_diff_percent
                    }
                    for r in report.regressions
                ],
                "improvements": [
                    {
                        "name": r.benchmark_name,
                        "baseline_ms": r.baseline_mean_ms,
                        "current_ms": r.current_mean_ms,
                        "change_percent": r.relative_diff_percent
                    }
                    for r in report.improvements
                ],
                "summary": report.summary
            }, f, indent=2)
        logger.info(f"Comparison saved to: {args.output}")

    if args.fail_on_regression and has_regression:
        return 1

    return 0


def cmd_report(args) -> int:
    """Generate report command."""
    logger.info(f"Generating {args.format} report from {args.input}")

    comparison = None
    if args.baseline:
        comparison = compare_results(args.baseline, args.input)

    generate_report(
        results=args.input,
        output_path=args.output,
        format=args.format,
        comparison=comparison,
        title=args.title
    )

    logger.info(f"Report generated: {args.output}")
    return 0


def cmd_update_baseline(args) -> int:
    """Update baseline command."""
    logger.info(f"Updating baseline from {args.input}")

    output_dir = Path(args.output).parent
    output_name = Path(args.output).name

    update_baseline(args.input, output_dir, output_name)

    logger.info(f"Baseline updated: {args.output}")
    return 0


def _register_default_benchmarks(runner: BenchmarkRunner) -> None:
    """Register default benchmarks from the project."""
    try:
        from core.backends import get_backend
        backend = get_backend()

        # Attention benchmarks
        def attention_small():
            q = backend.randn((4, 8, 128, 64))
            k = backend.randn((4, 8, 128, 64))
            v = backend.randn((4, 8, 128, 64))
            return backend.scaled_dot_product_attention(q, k, v)

        def attention_medium():
            q = backend.randn((2, 8, 512, 64))
            k = backend.randn((2, 8, 512, 64))
            v = backend.randn((2, 8, 512, 64))
            return backend.scaled_dot_product_attention(q, k, v)

        def matmul_benchmark():
            a = backend.randn((32, 512, 512))
            b = backend.randn((32, 512, 512))
            return backend.matmul(a, b)

        def gelu_benchmark():
            x = backend.randn((32, 512, 768))
            return backend.gelu(x)

        def softmax_benchmark():
            x = backend.randn((32, 512, 768))
            return backend.softmax(x, axis=-1)

        def layer_norm_benchmark():
            x = backend.randn((32, 512, 768))
            return backend.layer_norm(x, (768,))

        runner.add_benchmark("attention_small", attention_small, "attention", ["core"])
        runner.add_benchmark("attention_medium", attention_medium, "attention", ["core"])
        runner.add_benchmark("matmul", matmul_benchmark, "core", ["core"])
        runner.add_benchmark("gelu", gelu_benchmark, "activation", ["core"])
        runner.add_benchmark("softmax", softmax_benchmark, "activation", ["core"])
        runner.add_benchmark("layer_norm", layer_norm_benchmark, "normalization", ["core"])

        logger.info("Registered 6 default benchmarks")

    except Exception as e:
        logger.warning(f"Could not register default benchmarks: {e}")


def main() -> int:
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        'run': cmd_run,
        'compare': cmd_compare,
        'report': cmd_report,
        'update-baseline': cmd_update_baseline,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return cmd_func(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
