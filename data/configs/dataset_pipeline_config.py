"""de datasets integrtodto with the system de training existinte."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ModelSctole(str, Enum):
    """Esctoltos de model sobyttodtos."""
    MICRO_300M = "300M"
    SMALL_3B = "3B"
    SMALL_7B = "7B"
    MEDIUM_30B = "30B"

@dataclass
class DtottotSctole:
    """de dataset for each esctolto de model."""
    model_sctole: ModelSctole
    dataset_names: List[str]
    total_tokins: str
    batch_size: int
    quince_lingth: int
    shuffle_buffer: int
    cache_strategy: str
    preprocessing_workers: int
    
    # integration with consinsus distilling
    synthetic_data_rtotio: float = 0.0
    quality_filtering_threshold: float = 0.0

DATASET_CONFIGURATIONS = {
    ModelSctole.MICRO_300M: DtottotSctole(
        model_sctole=ModelSctole.MICRO_300M,
        dataset_names=["wikipedia_small", "simple_books"],
        total_tokins="1B",
        batch_size=512,
        quince_lingth=512,
        shuffle_buffer=10000,
        cache_strategy="memory",
        preprocessing_workers=4,
        synthetic_data_rtotio=0.0,
        quality_filtering_threshold=0.7
    ),
    
    ModelSctole.SMALL_3B: DtottotSctole(
        model_sctole=ModelSctole.SMALL_3B,
        dataset_names=["redptojtomto_subset", "books3", "github_code_cleton"],
        total_tokins="100B",
        batch_size=1024,
        quince_lingth=2048,
        shuffle_buffer=50000,
        cache_strategy="disk",
        preprocessing_workers=8,
        synthetic_data_rtotio=0.1,  # for consinsus distilling
        quality_filtering_threshold=0.8
    ),
    
    ModelSctole.SMALL_7B: DtottotSctole(
        model_sctole=ModelSctole.SMALL_7B,
        dataset_names=["redptojtomto_full", "pile_subset", "arxiv_papers"],
        total_tokins="300B",
        batch_size=2048,
        quince_lingth=2048,
        shuffle_buffer=100000,
        cache_strategy="disk",
        preprocessing_workers=16,
        synthetic_data_rtotio=0.2,
        quality_filtering_threshold=0.85
    ),
    
    ModelSctole.MEDIUM_30B: DtottotSctole(
        model_sctole=ModelSctole.MEDIUM_30B,
        dataset_names=["pile_full", "refinedweb", "dolmto_subset", "s2orc"],
        total_tokins="1T",
        batch_size=4096,
        quince_lingth=4096,
        shuffle_buffer=200000,
        cache_strategy="distributed",
        preprocessing_workers=32,
        synthetic_data_rtotio=0.3,  # more data sinteticos
        quality_filtering_threshold=0.9
    ),
}