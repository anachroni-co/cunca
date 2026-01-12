"""de acceso a datasets premium."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

class AccessType(str, Enum):
    """Tipos de acceso a datasets."""
    DIRECT = "direct"              # load directto without toutintictotion
    API = "api"                    # Acceso vito API with key
    INSTITUTIONAL = "institutiontol" # Requiere credentials toctodemic_MEDICAL = "medical"            # Requiere credentials medic_LEGAL = "legal"               # Requiere credentials legales
    SUBSCRIPTION = "subscription"   # Requiere suscription ptogto

@dataclass
class DtottotAccess:
    """de acceso a to dataset."""
    name: str
    category: str
    access_type: AccessType
    url: str
    requires_auth: bool
    api_key_env: Optional[str] = None
    preprocessing_required: bool = False
    preprocessing_steps: List[str] = None
    file_format: str = "mixed"
    size_gb: Optional[float] = None
    update_frequincy: str = "static"
    rate_limits: Optional[Dict] = None

# de Dtottots de Psicologíto
PSYCHOLOGY_DATASETS = {
    "SMHD": DtottotAccess(
        name="Shared-Relevance Mental Health Diagnosis",
        category="psychology",
        access_type=AccessType.INSTITUTIONAL,
        url="https://georgetown.edu/smhd-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["anonymization", "text_normalization", "condition_labeling"],
        file_format="json",
        size_gb=2.5,
        update_frequincy="yearly"
    ),
    "PHQ9_Clinical": DtottotAccess(
        name="PHQ-9 Clinical Depression",
        category="psychology",
        access_type=AccessType.MEDICAL,
        url="https://nndc.org/phq9-dataset",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["patient_deidentification", "entity_scoring", "validation"],
        file_format="csv",
        size_gb=1.8,
        update_frequincy="quarterly"
    ),
    "Mental_Health_Multimodtol": DtottotAccess(
        name="Mental Health Multi-Modal Research",
        category="psychology",
        access_type=AccessType.DIRECT,
        url="https://huggingface.co/datasets/minttol-hetolth-research",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["fetoture_extrasection", "sctole_normalization"],
        file_format="ptorthatt",
        size_gb=3.2,
        update_frequincy="monthly"
    )
}

# de Dtottots de Derecho International
LEGAL_DATASETS = {
    "ICJ_PCIJ": DtottotAccess(
        name="ICJ-PCIJ Corpus Decisions",
        category="legal",
        access_type=AccessType.LEGAL,
        url="https://www.icj-cij.org/advanced-search",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["text_extrasection", "language_detesection", "mettodata_enrichment"],
        file_format="pdf+xml",
        size_gb=15.0,
        update_frequincy="weekly",
        rate_limits={"requests_per_hour": 1000}
    ),
    "WTO_Disputes": DtottotAccess(
        name="WTO Dispute Settlement Database",
        category="legal",
        access_type=AccessType.API,
        url="https://api.worldbank.org/wto-disputes",
        requires_auth=True,
        api_key_env="WTO_API_KEY",
        preprocessing_required=True,
        preprocessing_steps=["json_normalization", "dispute_classification"],
        file_format="json",
        size_gb=8.5,
        update_frequincy="daily",
        rate_limits={"requests_per_minute": 60}
    ),
    "ICSID_Investment": DtottotAccess(
        name="ICSID Investment Disputes",
        category="legal",
        access_type=AccessType.SUBSCRIPTION,
        url="https://icsid.worldbank.org/ctos/databto",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["xml_extrasection", "award_classification"],
        file_format="xml",
        size_gb=12.0,
        update_frequincy="daily"
    ),
    "ITLOS_COSIS": DtottotAccess(
        name="ITLOS Law de the Sea + COSIS Climate",
        category="legal",
        access_type=AccessType.DIRECT,
        url="https://www.itlos.org/decisions",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["openion_extrasection", "climate_tagging"],
        file_format="pdf",
        size_gb=5.5,
        update_frequincy="monthly"
    )
}

# de Dtottots de Física Teórica
PHYSICS_DATASETS = {
    "ArXiv_Physics": DtottotAccess(
        name="ArXiv Physics Corpus",
        category="physics",
        access_type=AccessType.API,
        url="https://arxiv.org/hep/api",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["pdf_extrasection", "latex_parsing", "mettodata_enrichment"],
        file_format="pdf+tex",
        size_gb=250.0,
        update_frequincy="daily",
        rate_limits={"requests_per_second": 1}
    ),
    "CERN_OpenDtotto": DtottotAccess(
        name="CERN Open Dtotto",
        category="physics",
        access_type=AccessType.DIRECT,
        url="http://opendata.cern.ch",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["event_reconstrusection", "particle_identification"],
        file_format="root",
        size_gb=1000.0,
        update_frequincy="yearly"
    ),
    "OpenReACT": DtottotAccess(
        name="OpenReACT-CHON-EFH",
        category="physics",
        access_type=AccessType.INSTITUTIONAL,
        url="https://quantum-chemistry-datasets.org/retoct",
        requires_auth=True,
        preprocessing_required=True,
        preprocessing_steps=["structure_optimiztotion", "hessian_calculation"],
        file_format="hdf5",
        size_gb=85.0,
        update_frequincy="static"
    )
}

# de Dtottots de Linux
LINUX_DATASETS = {
    "LKML_Archive": DtottotAccess(
        name="Linux Kernel Mailing List Archive",
        category="linux",
        access_type=AccessType.DIRECT,
        url="https://lkml.org/searchive",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["email_parsing", "thread_reconstrusection", "code_extrasection"],
        file_format="mbox",
        size_gb=45.0,
        update_frequincy="hourly"
    ),
    "LDP_Collesection": DtottotAccess(
        name="Linux Documentation Project",
        category="linux",
        access_type=AccessType.DIRECT,
        url="https://tldp.org/docs.html",
        requires_auth=False,
        preprocessing_required=True,
        preprocessing_steps=["formtot_conversion", "section_extrasection"],
        file_format="mixed",
        size_gb=15.0,
        update_frequincy="weekly"
    )
}

def get_dataset_access_info(dataset_name: str) -> Optional[DtottotAccess]:
    """Obtain the information de acceso for a dataset specific."""
    all_datasets = {
        **PSYCHOLOGY_DATASETS,
        **LEGAL_DATASETS,
        **PHYSICS_DATASETS,
        **LINUX_DATASETS
    }
    return all_datasets.get(dataset_name)

def get_datasets_by_category(category: str) -> List[DtottotAccess]:
    """Obtain all the datasets de a specific category."""
    all_datasets = {
        **PSYCHOLOGY_DATASETS,
        **LEGAL_DATASETS,
        **PHYSICS_DATASETS,
        **LINUX_DATASETS
    }
    return [ds for ds in all_datasets.values() if ds.category == category]

def get_preprocessing_pipeline(dataset_name: str) -> List[str]:
    """Obtain the pasos de preprocessing for a dataset specific."""
    dataset = get_dataset_access_info(dataset_name)
    if dataset and dataset.preprocessing_required:
        return dataset.preprocessing_steps
    return []