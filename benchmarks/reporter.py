"""
Benchmark Reporter Module
=========================

Generate reports from benchmark results in various formats (HTML, JSON, Markdown).
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

from .comparator import RegressionReport, ComparisonResult

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """
    Generate benchmark reports in various formats.

    Example:
        >>> reporter = BenchmarkReporter()
        >>> reporter.generate_html(results, "report.html")
        >>> reporter.generate_markdown(results, "report.md")
    """

    def __init__(self, title: str = "CapibaraGPT Benchmark Report"):
        """
        Initialize reporter.

        Args:
            title: Report title
        """
        self.title = title

    def generate_html(
        self,
        results: Union[Dict, Path, str],
        output_path: Union[str, Path],
        comparison: Optional[RegressionReport] = None
    ) -> Path:
        """
        Generate an HTML benchmark report.

        Args:
            results: Benchmark results data or path to JSON
            output_path: Output file path
            comparison: Optional comparison report to include

        Returns:
            Path to generated report
        """
        if isinstance(results, (str, Path)):
            with open(results) as f:
                results = json.load(f)

        output_path = Path(output_path)
        html = self._render_html(results, comparison)

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"HTML report generated: {output_path}")
        return output_path

    def _render_html(
        self,
        results: Dict,
        comparison: Optional[RegressionReport] = None
    ) -> str:
        """Render HTML report content."""
        timestamp = results.get('timestamp', datetime.now().isoformat())
        hardware = results.get('hardware_info', {})
        config = results.get('config', {})
        benchmark_results = results.get('results', [])
        summary = results.get('summary', {})

        # Group results
        groups: Dict[str, List] = {}
        for r in benchmark_results:
            group = r.get('group', 'default')
            if group not in groups:
                groups[group] = []
            groups[group].append(r)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        :root {{
            --primary: #4CAF50;
            --danger: #f44336;
            --warning: #ff9800;
            --success: #4CAF50;
            --bg: #f5f5f5;
            --card: #ffffff;
            --text: #333333;
            --border: #e0e0e0;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg);
            color: var(--text);
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: var(--primary); border-bottom: 3px solid var(--primary); padding-bottom: 15px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .card {{
            background: var(--card);
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat {{
            background: var(--card);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: var(--primary);
        }}
        .stat-label {{ color: #666; font-size: 14px; margin-top: 5px; }}
        .stat.danger .stat-value {{ color: var(--danger); }}
        .stat.warning .stat-value {{ color: var(--warning); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .pass {{ color: var(--success); }}
        .fail {{ color: var(--danger); }}
        .regression {{ background: #ffebee; }}
        .improvement {{ background: #e8f5e9; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-success {{ background: #c8e6c9; color: #2e7d32; }}
        .badge-danger {{ background: #ffcdd2; color: #c62828; }}
        .badge-warning {{ background: #fff3e0; color: #e65100; }}
        .chart-container {{ height: 400px; margin: 20px 0; }}
        .hardware-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
        }}
        .hardware-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }}
        .hardware-label {{ font-weight: 500; color: #666; }}
        .collapsible {{
            cursor: pointer;
            padding: 15px;
            background: #f8f9fa;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 16px;
            font-weight: 600;
            border-radius: 4px;
            margin-top: 10px;
        }}
        .collapsible:hover {{ background: #e9ecef; }}
        .collapsible:after {{ content: '+'; float: right; }}
        .collapsible.active:after {{ content: '-'; }}
        .content {{
            display: none;
            padding: 15px;
            background: white;
            border: 1px solid var(--border);
            border-top: none;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p>Generated: {timestamp}</p>

        <div class="stats-grid">
            <div class="stat">
                <div class="stat-value">{summary.get('total_benchmarks', len(benchmark_results))}</div>
                <div class="stat-label">Total Benchmarks</div>
            </div>
            <div class="stat">
                <div class="stat-value">{summary.get('passed', sum(1 for r in benchmark_results if r.get('passed', True)))}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat {'danger' if summary.get('failed', 0) > 0 else ''}">
                <div class="stat-value">{summary.get('failed', sum(1 for r in benchmark_results if not r.get('passed', True)))}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(groups)}</div>
                <div class="stat-label">Groups</div>
            </div>
        </div>

        {self._render_comparison_section(comparison) if comparison else ''}

        <div class="card">
            <h2>Benchmark Results</h2>
            <div class="chart-container">
                <canvas id="benchmarkChart"></canvas>
            </div>
        </div>

        {self._render_groups(groups)}

        <div class="card">
            <h2>Hardware Information</h2>
            <div class="hardware-info">
                {self._render_hardware_info(hardware)}
            </div>
        </div>

        <div class="card">
            <h2>Configuration</h2>
            <div class="hardware-info">
                {self._render_config(config)}
            </div>
        </div>
    </div>

    <script>
        // Chart.js visualization
        const ctx = document.getElementById('benchmarkChart').getContext('2d');
        const results = {json.dumps([{'name': r['name'], 'mean_ms': r.get('timing', {}).get('mean_ms', 0)} for r in benchmark_results[:20]])};

        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: results.map(r => r.name.split('/').pop()),
                datasets: [{{
                    label: 'Mean Time (ms)',
                    data: results.map(r => r.mean_ms),
                    backgroundColor: '#4CAF50',
                    borderColor: '#388E3C',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: true, text: 'Top 20 Benchmarks by Time' }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Time (ms)' }} }}
                }}
            }}
        }});

        // Collapsible sections
        document.querySelectorAll('.collapsible').forEach(btn => {{
            btn.addEventListener('click', function() {{
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                content.style.display = content.style.display === 'block' ? 'none' : 'block';
            }});
        }});
    </script>
</body>
</html>"""

        return html

    def _render_comparison_section(self, comparison: RegressionReport) -> str:
        """Render comparison/regression section."""
        if not comparison:
            return ""

        html = f"""
        <div class="card {'regression' if comparison.has_regressions else 'improvement' if comparison.improvements else ''}">
            <h2>Performance Comparison</h2>
            <p>Comparing against: {comparison.baseline_file}</p>

            <div class="stats-grid">
                <div class="stat {'danger' if comparison.has_regressions else ''}">
                    <div class="stat-value">{comparison.regression_count}</div>
                    <div class="stat-label">Regressions</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{comparison.improvement_count}</div>
                    <div class="stat-label">Improvements</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(comparison.unchanged)}</div>
                    <div class="stat-label">Unchanged</div>
                </div>
            </div>
        """

        if comparison.regressions:
            html += """
            <h3>Regressions</h3>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Change</th>
                    <th>Significance</th>
                </tr>
            """
            for reg in comparison.regressions:
                html += f"""
                <tr class="regression">
                    <td>{reg.benchmark_name}</td>
                    <td>{reg.baseline_mean_ms:.3f}ms</td>
                    <td>{reg.current_mean_ms:.3f}ms</td>
                    <td class="fail">+{reg.relative_diff_percent:.1f}%</td>
                    <td><span class="badge badge-danger">{reg.significance}</span></td>
                </tr>
                """
            html += "</table>"

        if comparison.improvements:
            html += """
            <h3>Improvements</h3>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Change</th>
                </tr>
            """
            for imp in comparison.improvements[:10]:
                html += f"""
                <tr class="improvement">
                    <td>{imp.benchmark_name}</td>
                    <td>{imp.baseline_mean_ms:.3f}ms</td>
                    <td>{imp.current_mean_ms:.3f}ms</td>
                    <td class="pass">{imp.relative_diff_percent:.1f}%</td>
                </tr>
                """
            html += "</table>"

        html += "</div>"
        return html

    def _render_groups(self, groups: Dict[str, List]) -> str:
        """Render benchmark groups."""
        html = ""
        for group_name, benchmarks in groups.items():
            html += f"""
            <button class="collapsible">{group_name} ({len(benchmarks)} benchmarks)</button>
            <div class="content">
                <table>
                    <tr>
                        <th>Benchmark</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Iterations</th>
                        <th>Status</th>
                    </tr>
            """
            for b in benchmarks:
                timing = b.get('timing', {})
                passed = b.get('passed', True)
                status_class = 'pass' if passed else 'fail'
                status_text = 'PASS' if passed else 'FAIL'

                html += f"""
                    <tr>
                        <td>{b['name']}</td>
                        <td>{timing.get('mean_ms', 0):.3f}ms</td>
                        <td>{timing.get('median_ms', 0):.3f}ms</td>
                        <td>{timing.get('std_ms', 0):.3f}ms</td>
                        <td>{timing.get('min_ms', 0):.3f}ms</td>
                        <td>{timing.get('max_ms', 0):.3f}ms</td>
                        <td>{timing.get('iterations', 0)}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
                """
            html += "</table></div>"
        return html

    def _render_hardware_info(self, hardware: Dict) -> str:
        """Render hardware information."""
        html = ""
        for key, value in hardware.items():
            html += f"""
            <div class="hardware-item">
                <span class="hardware-label">{key.replace('_', ' ').title()}</span>
                <span>{value}</span>
            </div>
            """
        return html

    def _render_config(self, config: Dict) -> str:
        """Render configuration."""
        html = ""
        for key, value in config.items():
            html += f"""
            <div class="hardware-item">
                <span class="hardware-label">{key.replace('_', ' ').title()}</span>
                <span>{value}</span>
            </div>
            """
        return html

    def generate_markdown(
        self,
        results: Union[Dict, Path, str],
        output_path: Union[str, Path],
        comparison: Optional[RegressionReport] = None
    ) -> Path:
        """
        Generate a Markdown benchmark report.

        Args:
            results: Benchmark results data or path to JSON
            output_path: Output file path
            comparison: Optional comparison report

        Returns:
            Path to generated report
        """
        if isinstance(results, (str, Path)):
            with open(results) as f:
                results = json.load(f)

        output_path = Path(output_path)
        md = self._render_markdown(results, comparison)

        with open(output_path, 'w') as f:
            f.write(md)

        logger.info(f"Markdown report generated: {output_path}")
        return output_path

    def _render_markdown(
        self,
        results: Dict,
        comparison: Optional[RegressionReport] = None
    ) -> str:
        """Render Markdown report content."""
        timestamp = results.get('timestamp', datetime.now().isoformat())
        benchmark_results = results.get('results', [])
        summary = results.get('summary', {})
        hardware = results.get('hardware_info', {})

        md = f"""# {self.title}

**Generated:** {timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Benchmarks | {summary.get('total_benchmarks', len(benchmark_results))} |
| Passed | {summary.get('passed', sum(1 for r in benchmark_results if r.get('passed', True)))} |
| Failed | {summary.get('failed', sum(1 for r in benchmark_results if not r.get('passed', True)))} |
| Pass Rate | {summary.get('pass_rate', 1.0):.1%} |

"""

        if comparison:
            md += f"""## Performance Comparison

Comparing against: `{comparison.baseline_file}`

| Category | Count |
|----------|-------|
| Regressions | {comparison.regression_count} |
| Improvements | {comparison.improvement_count} |
| Unchanged | {len(comparison.unchanged)} |

"""
            if comparison.regressions:
                md += "### Regressions\n\n"
                md += "| Benchmark | Baseline | Current | Change |\n"
                md += "|-----------|----------|---------|--------|\n"
                for reg in comparison.regressions:
                    md += f"| {reg.benchmark_name} | {reg.baseline_mean_ms:.3f}ms | {reg.current_mean_ms:.3f}ms | +{reg.relative_diff_percent:.1f}% |\n"
                md += "\n"

        md += "## Results\n\n"
        md += "| Benchmark | Mean | Std | Iterations | Status |\n"
        md += "|-----------|------|-----|------------|--------|\n"

        for r in benchmark_results:
            timing = r.get('timing', {})
            status = "PASS" if r.get('passed', True) else "FAIL"
            md += f"| {r['name']} | {timing.get('mean_ms', 0):.3f}ms | {timing.get('std_ms', 0):.3f}ms | {timing.get('iterations', 0)} | {status} |\n"

        md += f"""
## Hardware

| Property | Value |
|----------|-------|
"""
        for key, value in hardware.items():
            md += f"| {key.replace('_', ' ').title()} | {value} |\n"

        return md

    def generate_json(
        self,
        results: Union[Dict, Path, str],
        output_path: Union[str, Path],
        comparison: Optional[RegressionReport] = None
    ) -> Path:
        """
        Generate a JSON report (with optional comparison data).

        Args:
            results: Benchmark results data or path
            output_path: Output file path
            comparison: Optional comparison report

        Returns:
            Path to generated report
        """
        if isinstance(results, (str, Path)):
            with open(results) as f:
                results = json.load(f)

        output_path = Path(output_path)

        report_data = {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "results": results,
        }

        if comparison:
            report_data["comparison"] = {
                "baseline_file": comparison.baseline_file,
                "current_file": comparison.current_file,
                "has_regressions": comparison.has_regressions,
                "regression_count": comparison.regression_count,
                "improvement_count": comparison.improvement_count,
                "regressions": [
                    {
                        "name": r.benchmark_name,
                        "baseline_ms": r.baseline_mean_ms,
                        "current_ms": r.current_mean_ms,
                        "change_percent": r.relative_diff_percent
                    }
                    for r in comparison.regressions
                ],
                "summary": comparison.summary
            }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"JSON report generated: {output_path}")
        return output_path


def generate_report(
    results: Union[Dict, Path, str],
    output_path: Union[str, Path],
    format: str = "html",
    comparison: Optional[RegressionReport] = None,
    title: str = "Benchmark Report"
) -> Path:
    """
    Convenience function to generate a benchmark report.

    Args:
        results: Benchmark results
        output_path: Output file path
        format: Report format ("html", "markdown", "json")
        comparison: Optional comparison report
        title: Report title

    Returns:
        Path to generated report
    """
    reporter = BenchmarkReporter(title=title)

    if format == "html":
        return reporter.generate_html(results, output_path, comparison)
    elif format == "markdown" or format == "md":
        return reporter.generate_markdown(results, output_path, comparison)
    elif format == "json":
        return reporter.generate_json(results, output_path, comparison)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'html', 'markdown', or 'json'")
