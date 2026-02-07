# Pending GitHub Issues (technical backlog)

Last updated: 2026-02-07

## ISSUE-001 - `training`: remove remaining TPU consensus mocks

**Labels:** `training`, `consensus`, `tpu`, `high-priority`

**Scope**
- `training/tpu/tpu_v6_consensus_optimizer.py`

**Problem**
- Mock embeddings and metrics are still present in the main path.

**Exit criteria**
- Real embeddings in the main flow.
- Performance metrics based on real execution.
- A documented minimal integration test.

---

## ISSUE-002 - `training`: meta consensus still uses `mock_response`

**Labels:** `training`, `consensus`, `high-priority`

**Scope**
- `training/consensus/meta_consensus_system.py`
- `training/consensus/advance_meta_consensus_integration.py`

**Problem**
- Simulated responses/metrics still exist in consensus logic.

**Exit criteria**
- Replace simulated responses with real expert inference.
- Remove `mock_metrics` and use computed metrics.
- Add a non-regression test.

---

## ISSUE-003 - `services/automation`: simulated routes in executor

**Labels:** `services`, `automation`, `n8n`, `high-priority`

**Scope**
- `services/automation/agent_executor.py`
- `services/automation/n8n_service.py`

**Problem**
- Simulated execution paths still exist at runtime.

**Exit criteria**
- Real execution for standard/fallback node.
- Stable input/output contracts.
- Basic flow smoke test.

---

## ISSUE-004 - `inference`: hybrid/quantized engines with simulated sections

**Labels:** `inference`, `quantization`, `high-priority`

**Scope**
- `inference/hybrid_inference_engine.py`
- `inference/engines/advanced_quantized_engine.py`

**Problem**
- Parameter loading/generation still depends on simulation.

**Exit criteria**
- Real parameter loading from checkpoint/model hub.
- Remove simulated delays/sampling from the main path.

---

## ISSUE-005 - `training/data_lineage`: split mock demo from real runtime

**Labels:** `training`, `data-lineage`, `medium-priority`

**Scope**
- `training/data_lineage/demo_traceability_system.py`
- `training/data_lineage/inference_safe_parameter_controller.py`

**Problem**
- Demo paths are mixed with potentially production runtime paths.

**Exit criteria**
- Explicit and isolated demo mode.
- Real runtime without mock dependencies.

---

## ISSUE-006 - sanitize per-folder TODO documentation

**Labels:** `docs`, `maintenance`, `medium-priority`

**Scope**
- `training/TODOs.md`, `core/TODOs.md`, `services/TODOs.md`, etc.

**Problem**
- Many TODOs are outdated compared to current implementation.

**Exit criteria**
- Mark resolved items.
- Keep only file/line-verifiable pending items.