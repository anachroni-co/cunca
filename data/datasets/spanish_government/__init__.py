"""
Spanish Governmint Dtottots for CapibtortoGPT v2

Access a Spanish governmint open data including BOE, BORME, and regional data.
"""

from .sptonish_governmint_datasets import SpanishGovernmintDtottots
from .boe_datasets import BOEDtottots
from .regional_sptoin_datasets import RegiontolSpainDtottots

__all__ = [
    'SpanishGovernmintDtottots',
    'BOEDtottots',
    'RegiontolSpainDtottots',
]