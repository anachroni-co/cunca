"""
Root conftest.py for pytest path configuration.

This file ensures the project root is in sys.path before any test imports.
"""

import sys
from pathlib import Path

# Add project root to path BEFORE any imports
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Early pytest hook to ensure path is set
def pytest_configure(config):
    """Ensure project root is in path during pytest configuration."""
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
