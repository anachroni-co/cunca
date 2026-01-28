"""
layers conv1d_block module.

# This module provides functionality for conv1d_block.
"""

import os
import sys

import logging
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module conv1d_block.py starting")
    return True

if __name__ == "__main__":
    main()
