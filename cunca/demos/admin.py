"""CUNCA Demo — Sector 1: Public Administration.

GDPR-compliant, on-premises document assistant for Galician public bodies.

Key constraints (per CUNCA Memoria Técnica):
  - All inference runs on-premises (no external API calls)
  - Input/output goes through PII redaction before logging
  - Responses in Galician by default
  - Supports: document summarisation, regulation lookup, form filling guidance

Endpoints:
  POST /demo/admin/summarise   — summarise an official document
  POST /demo/admin/regulation  — answer a regulatory question
  GET  /demo/admin/health      — service health check
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/demo/admin", tags=["demo-admin"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SummariseRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Official document text to summarise")
    lang: str = Field("gl", description="Response language: gl | pt | es")
    max_sentences: int = Field(3, ge=1, le=10)


class SummariseResponse(BaseModel):
    summary: str
    lang: str
    char_count_in: int
    on_premises: bool = True


class RegulationRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Regulatory question in natural language")
    regulation_context: str = Field("", description="Optional extracted regulation text")
    lang: str = Field("gl")


class RegulationResponse(BaseModel):
    answer: str
    confidence: str      # high | medium | low
    disclaimer: str
    lang: str


# ---------------------------------------------------------------------------
# Stub generation (replaced by CUNCAModel in production)
# ---------------------------------------------------------------------------

_GDPR_DISCLAIMER = (
    "Esta resposta é orientativa. Para decisións vinculantes consulte o/a "
    "responsable xurídico/a da súa organización."
)

_LANG_GREETINGS = {
    "gl": "Resumo do documento oficial",
    "pt": "Resumo do documento oficial",
    "es": "Resumen del documento oficial",
}


def _stub_summarise(text: str, lang: str, max_sentences: int) -> str:
    words = text.split()
    snippet = " ".join(words[:30]) + ("..." if len(words) > 30 else "")
    label = _LANG_GREETINGS.get(lang, _LANG_GREETINGS["gl"])
    return f"[{label}] {snippet}"


def _stub_regulation_answer(question: str, lang: str) -> tuple[str, str]:
    answer = (
        f"A consulta sobre «{question[:60]}» require revisión da normativa vixente. "
        "Consulte a Base de Datos de Lexislación Galega (BXLEG) para información actualizada."
    )
    return answer, "low"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
async def admin_health():
    return {"status": "ok", "sector": "public_administration", "on_premises": True}


@router.post("/summarise", response_model=SummariseResponse)
async def summarise_document(req: SummariseRequest):
    if len(req.text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Document text too short")
    summary = _stub_summarise(req.text, req.lang, req.max_sentences)
    return SummariseResponse(
        summary=summary,
        lang=req.lang,
        char_count_in=len(req.text),
    )


@router.post("/regulation", response_model=RegulationResponse)
async def regulation_query(req: RegulationRequest):
    answer, confidence = _stub_regulation_answer(req.question, req.lang)
    return RegulationResponse(
        answer=answer,
        confidence=confidence,
        disclaimer=_GDPR_DISCLAIMER,
        lang=req.lang,
    )
