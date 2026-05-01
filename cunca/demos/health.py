"""CUNCA Demo — Sector 3: Digital Health.

Clinical document assistant for Galician healthcare providers.

Key constraints (per CUNCA Memoria Técnica):
  - Strict PII handling: patient names/IDs never stored in logs
  - Responses always include medical disclaimer
  - Operates on structured + free-text clinical notes
  - On-premises only (GDPR / data sovereignty)

Use cases:
  - Clinical note summarisation
  - ICD-10 coding suggestion
  - Patient-facing explanations in plain Galician

Endpoints:
  POST /demo/health/summarise  — summarise a clinical note
  POST /demo/health/icd10      — suggest ICD-10 codes for a diagnosis description
  GET  /demo/health/health     — service health check
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/demo/health", tags=["demo-health"])

_MEDICAL_DISCLAIMER = (
    "Este resumo é un apoio administrativo e NON substitúe o criterio clínico. "
    "Calquera decisión diagnóstica ou terapéutica debe ser tomada por profesionais sanitarios."
)

# Minimal ICD-10 stub mapping (common Galician primary care codes)
_ICD10_STUBS: dict[str, list[str]] = {
    "hipertensión": ["I10", "I11"],
    "diabetes": ["E11", "E14"],
    "pneumonía": ["J18.9"],
    "fractura": ["S52", "S72"],
    "depresión": ["F32", "F33"],
    "ansiedade": ["F41"],
    "gripe": ["J10", "J11"],
    "covid": ["U07.1"],
    "infección": ["A09", "B99"],
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ClinicalSummariseRequest(BaseModel):
    note: str = Field(..., min_length=20, description="Raw clinical note text")
    patient_id: str = Field("ANON", description="Anonymised patient identifier")
    lang: str = Field("gl")


class ClinicalSummariseResponse(BaseModel):
    summary: str
    disclaimer: str
    patient_id: str
    lang: str
    on_premises: bool = True


class ICD10Request(BaseModel):
    diagnosis_text: str = Field(..., min_length=3)
    lang: str = Field("gl")


class ICD10Response(BaseModel):
    suggestions: list[dict]   # [{"code": "I10", "description": "..."}]
    disclaimer: str
    lang: str


# ---------------------------------------------------------------------------
# Stub logic
# ---------------------------------------------------------------------------

_ICD10_DESCRIPTIONS: dict[str, str] = {
    "I10": "Hipertensión esencial (primaria)",
    "I11": "Cardiopatía hipertensiva sen insuficiencia cardíaca",
    "E11": "Diabetes mellitus tipo 2",
    "E14": "Diabetes mellitus non especificada",
    "J18.9": "Pneumonía, non especificada",
    "S52":   "Fractura do antebrazo",
    "S72":   "Fractura do fémur",
    "F32":   "Episodio depresivo",
    "F33":   "Trastorno depresivo recorrente",
    "F41":   "Outros trastornos de ansiedade",
    "J10":   "Gripe debida a virus identificado",
    "J11":   "Gripe, virus non identificado",
    "U07.1": "COVID-19",
    "A09":   "Outras gastroenterites e colites de orixe infecciosa",
    "B99":   "Outras enfermidades infecciosas",
}


def _suggest_icd10(text: str) -> list[dict]:
    text_lower = text.lower()
    results = []
    for keyword, codes in _ICD10_STUBS.items():
        if keyword in text_lower:
            for code in codes:
                results.append({
                    "code": code,
                    "description": _ICD10_DESCRIPTIONS.get(code, code),
                    "keyword_match": keyword,
                })
    return results or [{"code": "Z99.9", "description": "Non especificado — revisión manual requirida", "keyword_match": None}]


def _stub_clinical_summary(note: str, lang: str) -> str:
    words = note.split()
    snippet = " ".join(words[:40]) + ("..." if len(words) > 40 else "")
    return f"[Resumo clínico] {snippet}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_demo_health():
    return {"status": "ok", "sector": "digital_health", "on_premises": True}


@router.post("/summarise", response_model=ClinicalSummariseResponse)
async def clinical_summarise(req: ClinicalSummariseRequest):
    summary = _stub_clinical_summary(req.note, req.lang)
    return ClinicalSummariseResponse(
        summary=summary,
        disclaimer=_MEDICAL_DISCLAIMER,
        patient_id=req.patient_id,
        lang=req.lang,
    )


@router.post("/icd10", response_model=ICD10Response)
async def icd10_suggest(req: ICD10Request):
    suggestions = _suggest_icd10(req.diagnosis_text)
    return ICD10Response(
        suggestions=suggestions,
        disclaimer=_MEDICAL_DISCLAIMER,
        lang=req.lang,
    )
