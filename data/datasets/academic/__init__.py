"""
CapibaraGPT-v2 Academic Datasets
Academic, educational and research datasets
"""

from . import wiki_datasets
from . import psychology_datasets
from . import academic_code_datasets
from . import institutional_datasets

__all__ = [
    'academic_code_datasets',
    'institutional_datasets',
    'wiki_datasets',
    'psychology_datasets'
]
