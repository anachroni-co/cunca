"""CUNCA Demo — Sector 2: Industry 4.0 / Predictive Maintenance.

Galician-language AI assistant for industrial facilities.

Use cases (per CUNCA Memoria Técnica):
  - Interpret sensor anomaly alerts in natural language
  - Generate maintenance work orders in Galician
  - Answer technical FAQ from equipment manuals

Endpoints:
  POST /demo/industry/anomaly   — describe sensor anomaly in natural language
  POST /demo/industry/workorder — generate a maintenance work order
  GET  /demo/industry/health
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/demo/industry", tags=["demo-industry"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AnomalyRequest(BaseModel):
    sensor_id: str = Field(..., description="Sensor identifier, e.g. 'PUMP-A1'")
    metric: str = Field(..., description="Metric name, e.g. 'vibration_hz'")
    value: float = Field(..., description="Current reading")
    threshold: float = Field(..., description="Normal operating threshold")
    lang: str = Field("gl")


class AnomalyResponse(BaseModel):
    description: str
    severity: str       # low | medium | high | critical
    recommended_action: str
    lang: str


class WorkOrderRequest(BaseModel):
    equipment_id: str
    fault_description: str
    technician: str = ""
    lang: str = Field("gl")


class WorkOrderResponse(BaseModel):
    order_id: str
    equipment_id: str
    description: str
    priority: str
    lang: str


# ---------------------------------------------------------------------------
# Stub logic
# ---------------------------------------------------------------------------

def _severity(value: float, threshold: float) -> str:
    ratio = value / max(threshold, 1e-9)
    if ratio < 1.1:
        return "low"
    elif ratio < 1.5:
        return "medium"
    elif ratio < 2.0:
        return "high"
    return "critical"


_SEVERITY_ACTIONS_GL = {
    "low": "Continuar vixilancia. Revisión na próxima parada programada.",
    "medium": "Aumentar frecuencia de inspección. Planificar substitución preventiva.",
    "high": "Parar equipamento en 24 h. Contactar coa empresa de mantemento.",
    "critical": "PARADA INMEDIATA. Risco de fallo catastrófico.",
}

import hashlib as _hl


def _make_order_id(equipment_id: str) -> str:
    h = _hl.md5(equipment_id.encode()).hexdigest()[:6].upper()
    return f"WO-{h}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
async def industry_health():
    return {"status": "ok", "sector": "industry_4_0"}


@router.post("/anomaly", response_model=AnomalyResponse)
async def anomaly_description(req: AnomalyRequest):
    sev = _severity(req.value, req.threshold)
    desc = (
        f"O sensor {req.sensor_id} rexistrou un valor de {req.value:.2f} "
        f"para a métrica '{req.metric}' (limiar normal: {req.threshold:.2f}). "
        f"Severidade estimada: {sev.upper()}."
    )
    action = _SEVERITY_ACTIONS_GL.get(sev, "Revisar manualmente.")
    return AnomalyResponse(description=desc, severity=sev, recommended_action=action, lang=req.lang)


@router.post("/workorder", response_model=WorkOrderResponse)
async def create_work_order(req: WorkOrderRequest):
    order_id = _make_order_id(req.equipment_id)
    description = (
        f"Orde de traballo para o equipamento {req.equipment_id}. "
        f"Fallo reportado: {req.fault_description}. "
        + (f"Técnico asignado: {req.technician}." if req.technician else "Técnico por asignar.")
    )
    return WorkOrderResponse(
        order_id=order_id,
        equipment_id=req.equipment_id,
        description=description,
        priority="normal",
        lang=req.lang,
    )
