# GitHub Issues Pendientes (backlog técnico)

Fecha de actualización: 2026-02-07

## ISSUE-001 - `training`: eliminar mocks restantes de consenso TPU

**Labels:** `training`, `consensus`, `tpu`, `high-priority`

**Scope**
- `training/tpu/tpu_v6_consensus_optimizer.py`

**Problema**
- Persisten embeddings y métricas mock en ruta principal.

**Criterio de cierre**
- Embeddings reales en flujo principal.
- Métricas de rendimiento basadas en ejecución real.
- Prueba mínima de integración documentada.

---

## ISSUE-002 - `training`: consenso meta aún usa `mock_response`

**Labels:** `training`, `consensus`, `high-priority`

**Scope**
- `training/consensus/meta_consensus_system.py`
- `training/consensus/advance_meta_consensus_integration.py`

**Problema**
- Existen respuestas/métricas simuladas en lógica de consenso.

**Criterio de cierre**
- Sustituir respuestas simuladas por inferencia real de expertos.
- Quitar `mock_metrics` y usar métricas calculadas.
- Agregar test de no-regresión.

---

## ISSUE-003 - `services/automation`: rutas simuladas en ejecutor

**Labels:** `services`, `automation`, `n8n`, `high-priority`

**Scope**
- `services/automation/agent_executor.py`
- `services/automation/n8n_service.py`

**Problema**
- Hay paths de ejecución simulada en runtime.

**Criterio de cierre**
- Ejecución real para nodo estándar/fallback.
- Contratos de entrada/salida estables.
- Smoke test de flujo básico.

---

## ISSUE-004 - `inference`: motores híbridos/quantized con secciones simuladas

**Labels:** `inference`, `quantization`, `high-priority`

**Scope**
- `inference/hybrid_inference_engine.py`
- `inference/engines/advanced_quantized_engine.py`

**Problema**
- Carga de parámetros/generación aún depende de simulaciones.

**Criterio de cierre**
- Carga real de parámetros desde checkpoint/model hub.
- Eliminación de delays/sampling simulado en ruta principal.

---

## ISSUE-005 - `training/data_lineage`: separar demo mock de runtime real

**Labels:** `training`, `data-lineage`, `medium-priority`

**Scope**
- `training/data_lineage/demo_traceability_system.py`
- `training/data_lineage/inference_safe_parameter_controller.py`

**Problema**
- Mezcla de rutas demo con rutas potencialmente productivas.

**Criterio de cierre**
- Modo demo explícito y aislado.
- Runtime real sin dependencia de mocks.

---

## ISSUE-006 - saneamiento de documentación de TODOs por carpeta

**Labels:** `docs`, `maintenance`, `medium-priority`

**Scope**
- `training/TODOs.md`, `core/TODOs.md`, `services/TODOs.md`, etc.

**Problema**
- Muchos TODOs desactualizados respecto al estado real.

**Criterio de cierre**
- Marcar items resueltos.
- Conservar solo pendientes verificables por ruta/línea.

