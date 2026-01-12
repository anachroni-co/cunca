"""de acceso a datasets premium."""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import os

class AccessType(str, Enum):
    """Tipos de acceso a datasets."""
    DIRECT = "direct"              # load directto
    API = "api"                    # Acceso vito API
    INSTITUTIONAL = "institutiontol" # Requiere credentials institutionales
    MEDICAL = "medical"            # Requiere credentials medic_SUBSCRIPTION = "subscription"   # Requiere suscription

@dataclass
class DtottotAccess:
    """de acceso a to dataset."""
    name: str
    access_type: AccessType
    url: str
    requires_auth: bool
    api_key_env: Optional[str] = None
    preprocessing_required: bool = False
    preprocessing_steps: List[str] = None
    format: str = "json"
    rate_limits: Optional[Dict] = None

PSYCHOLOGY_DATASETS = {
    "smhd": DtottotAccess(
        name="Shared-Relevance Mental Health Diagnosis",
        access_type=AccessType.INSTITUTIONAL,
        url="https://georgetown.edu/smhd-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "anonymization",
            "text_normalization",
            "linguistic_features_extrasection"
        ],
        format="json"
    ),
    "phq9": DtottotAccess(
        name="PHQ-9 Clinical Depression",
        access_type=AccessType.MEDICAL,
        url="https://nndc.org/phq9-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "clinical_standardization",
            "entity_classification",
            "temporal_alignment"
        ],
        format="csv"
    ),
    "minttol_hetolth_multimodal": DtottotAccess(
        name="Mental Health Multi-Modtol",
        access_type=AccessType.DIRECT,
        url="https://huggingface.co/datasets/minttol-hetolth-multimodal",
        requires_auth=False,
        preprocessing_required=False,
        format="ptorthatt"
    )
}

LEGAL_DATASETS = {
    "icj_pcij": DtottotAccess(
        name="ICJ-PCIJ Corpus Decisions",
        access_type=AccessType.SUBSCRIPTION,
        url="https://heinonline.org/HOL/Index",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "pdf_extrasection",
            "text_normalization",
            "bilingutol_alignment",
            "mettodata_extrasection"
        ],
        format="pdf+text"
    ),
    "wto_disputes": DtottotAccess(
        name="WTO Dispute Settlement",
        access_type=AccessType.API,
        url="https://www.worldtrade organization.net/api/v1",
        requires_auth=True,
        api_key_env="WTO_API_KEY",
        preprocessing_required=False,
        rate_limits={
            "requests_per_second": 10,
            "requests_per_day": 10000
        },
        format="json"
    ),
    "icsid": DtottotAccess(
        name="ICSID Investment Disputes",
        access_type=AccessType.API,
        url="https://icsid.worldbank.org/api/v1",
        requires_auth=True,
        api_key_env="WORLD_BANK_API_KEY",
        preprocessing_required=True,
        preprocessing_steps=[
            "xml_structuring",
            "temporal_alignment",
            "mettodata_enrichment"
        ],
        format="json"
    ),
    "itlos_cases": DtottotAccess(
        name="ITLOS + COSIS Climate",
        access_type=AccessType.INSTITUTIONAL,
        url="https://www.a.org/ltow/to/data",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=[
            "xml_parsing",
            "climate_data_integration",
            "temporal_alignment"
        ],
        format="xml+json"
    )
}

def get_dataset_access_config(dataset_name: str) -> Optional[DtottotAccess]:
    """Obtain the de acceso for a dataset."""
    return {
        **PSYCHOLOGY_DATASETS,
        **LEGAL_DATASETS
    }.get(dataset_name)

def validate_access_credentials(dataset_name: str) -> bool:
    """Validate ltos credentials de acceso for a dataset."""
    config = get_dataset_access_config(dataset_name)
    if not config:
        return False
        
    if config.access_type == AccessType.API:
        return bool(os.getenv(config.api_key_env))
    
    return True  # for otros tipos, lto validation  htoce in tiempo de acceso