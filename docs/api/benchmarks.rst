Benchmarks API
==============

The benchmarks module provides a comprehensive automated benchmarking system
for performance tracking, CI/CD integration, and regression detection.

benchmarks
----------

.. automodule:: benchmarks
   :members:
   :undoc-members:
   :show-inheritance:


Benchmark Runner
----------------

The runner module provides the core benchmark execution engine.

.. automodule:: benchmarks.runner
   :members:
   :undoc-members:
   :show-inheritance:

Classes
~~~~~~~

.. autoclass:: benchmarks.runner.BenchmarkRunner
   :members:
   :special-members: __init__

.. autoclass:: benchmarks.runner.BenchmarkResult
   :members:

.. autoclass:: benchmarks.runner.BenchmarkSuite
   :members:

.. autoclass:: benchmarks.runner.TimingStats
   :members:

Functions
~~~~~~~~~

.. autofunction:: benchmarks.runner.run_benchmarks

.. autofunction:: benchmarks.runner.benchmark

.. autofunction:: benchmarks.runner.benchmark_context


Benchmark Comparator
--------------------

The comparator module enables comparison of benchmark results and regression detection.

.. automodule:: benchmarks.comparator
   :members:
   :undoc-members:
   :show-inheritance:

Classes
~~~~~~~

.. autoclass:: benchmarks.comparator.BenchmarkComparator
   :members:
   :special-members: __init__

.. autoclass:: benchmarks.comparator.ComparisonResult
   :members:

.. autoclass:: benchmarks.comparator.RegressionReport
   :members:

Functions
~~~~~~~~~

.. autofunction:: benchmarks.comparator.compare_results

.. autofunction:: benchmarks.comparator.detect_regressions

.. autofunction:: benchmarks.comparator.update_baseline

.. autofunction:: benchmarks.comparator.load_baseline


Benchmark Reporter
------------------

The reporter module generates benchmark reports in various formats.

.. automodule:: benchmarks.reporter
   :members:
   :undoc-members:
   :show-inheritance:

Classes
~~~~~~~

.. autoclass:: benchmarks.reporter.BenchmarkReporter
   :members:
   :special-members: __init__

Functions
~~~~~~~~~

.. autofunction:: benchmarks.reporter.generate_report


CLI Usage
---------

The benchmark system can be run from the command line:

.. code-block:: bash

   # Run all benchmarks
   python -m benchmarks run

   # Run with specific groups
   python -m benchmarks run --groups core attention

   # Compare against baseline
   python -m benchmarks compare --baseline baseline.json --current results.json

   # Generate HTML report
   python -m benchmarks report --input results.json --output report.html

   # CI/CD mode (fail on regressions)
   python -m benchmarks run --ci --baseline baseline.json --threshold 10


Examples
--------

Basic Benchmark Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from benchmarks import BenchmarkRunner, run_benchmarks

   # Create a runner
   runner = BenchmarkRunner()

   # Add benchmarks
   runner.add_benchmark("matrix_multiply", my_matmul_func, group="core")
   runner.add_benchmark("attention", my_attention_func, group="attention")

   # Run benchmarks
   results = runner.run()

   # Save results
   runner.save_results("benchmark_results.json")


Using the Decorator
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from benchmarks import benchmark

   @benchmark(group="core", tags=["fast"])
   def my_benchmark():
       return compute_something()


Comparing Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from benchmarks import compare_results, detect_regressions

   # Compare two result files
   report = compare_results("baseline.json", "current.json")

   if report.has_regressions:
       print(f"Found {report.regression_count} regressions!")

   # Or use detect_regressions for CI/CD
   has_regression, report = detect_regressions(
       baseline="baseline.json",
       current="current.json",
       threshold_percent=10.0
   )


Generating Reports
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from benchmarks import generate_report

   # Generate HTML report
   generate_report(
       results="benchmark_results.json",
       output_path="report.html",
       format="html"
   )

   # Generate Markdown report
   generate_report(
       results="benchmark_results.json",
       output_path="report.md",
       format="markdown"
   )
