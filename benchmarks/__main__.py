"""
Allow running benchmarks as a module.

Usage:
    python -m benchmarks run
    python -m benchmarks compare --baseline FILE --current FILE
    python -m benchmarks report --input FILE --output FILE
"""

from .cli import main

if __name__ == '__main__':
    exit(main())
