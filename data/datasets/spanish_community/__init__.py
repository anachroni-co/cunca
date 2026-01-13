"""
Spanish Community Datasets Module for CapibaraGPT v2

This module provides access to Spanish-language datasets from the community,
particularly focused on SomosNLP and related Spanish NLP initiatives.

Key Features:
- SomosNLP community datasets
- Spanish instruction tuning datasets
- Cultural alignment resources
- Hackathon-generated datasets
- Regional Spanish language varieties
"""

from .somos_nlp_datasets import get_somos_nlp_datasets, SomosNLPDatasets

__all__ = [
    "get_somos_nlp_datasets",
    "SomosNLPDatasets"
]
