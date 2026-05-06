# CUNCA-Hybrid

Galician-focused hybrid Mamba-Transformer language model built on the Capibara Slim base.
Designed for low-resource Romance-language NLP with efficient fine-tuning via QLoRA
and production-ready sector demos for public administration, industry, and digital health.

---

## What this is

CUNCA (Corpus Unificado pola Normalización da Computación en Arda) is the Galician-language
specialisation of CapibaraGPT v3. It extends the JAX/Flax Capibara Slim base with:

- **Grouped Query Attention (GQA)** — fewer KV heads reduce memory pressure at inference
- **Sliding-Window Attention** — O(L·w) local context instead of full O(L²)
- **Configurable SSM ratio** — mix Mamba blocks and Transformer blocks per use case
- **QLoRA fine-tuning** — NF4-frozen base + trainable LoRA adapters; <2 % trainable params
- **CUNCA-Bench** — 12-task evaluation suite (comprehension, translation, reasoning, safety) in gl/pt/es
- **Sector demos** — public administration, Industry 4.0, digital health (FastAPI routers)
- **Think-Anywhere** reasoning — GRPO-trained inline `<thinkanywhere>` tokens at any position
- **L-MTP** — Leap Multi-Token Prediction (NeurIPS 2025) for 7 tokens/decode step

**Removed vs main:**
- `training/consensus/` — meta-consensus, federated consensus (experimental, mock-heavy)
- `training/federated_consensus/` — distributed consensus
- `training/data_lineage/` — blockchain audit, smart contracts
- `training/cython_kernels/` — uncompiled Cython stubs
- `agents/` — agent orchestration framework (disconnected from core)
- `sub_models/` — experimental expert submodels
- `COCOMO_II/` — cost estimation model

---

## Architecture

```
Input IDs
   ↓
Embedding  (vocab → hidden_size)
   ↓
Block 0  → CUNCATransformerBlock  (GQA + sliding-window + SwiGLU MLP)
Block 1  → CUNCAMambaBlock        (selective SSM: conv1d → ZOH → scan)
Block 2  → CUNCATransformerBlock
Block 3  → CUNCAMambaBlock
   ...   (ssm_ratio controls the alternating pattern)
   ↓
RMSNorm
   ↓
LM head  (weight-tied to embedding)
```

Training path:
```
User
 ↓
app/  (FastAPI — auth, rate limiting, streaming SSE)
 ↓
inference/hybrid_inference_engine.py  (backend selection)
 ├── TPU v5  →  JAX/Flax  →  training/tpu/
 ├── GPU     →  PyTorch + QLoRA (peft + bitsandbytes)
 └── CPU     →  models/pretrained_backbone.py  (NumPy fallback)
 ↓
core/think_anywhere/   (inline reasoning tokens — GRPO training)
core/special_tokens/   (verify / plan / search / fact_check / lang / debug)
 ↓
inference/quantization/  (INT8/INT4, KV-cache quantisation — Flax layers)
 ↓
rag/  (retrieval-augmented generation)
 ↓
safety/  (input/output filters)
 ↓
Response
```

### Size presets

| Preset | Hidden | Layers | Heads | KV heads | Window | ~Params |
|---|---|---|---|---|---|---|
| `1.3b` | 2048 | 24 | 16 | 4 | 512 | 1.3B |
| `3b`   | 2560 | 32 | 20 | 4 | 1024 | 3B |
| `7b`   | 4096 | 32 | 32 | 8 | 2048 | 7B |

---

## Directory structure

```
cunca/
  config.py          CUNCAConfig — GQA, window_size, ssm_ratio, presets
  architecture.py    CUNCAModel, CUNCAAttention, CUNCAMambaBlock
  qlora.py           NF4 quantisation, LoRALinear, apply_qlora(), qlora_stats()
  bench/
    tasks.py         12-task evaluation suite (gl / pt / es)
    evaluator.py     CUNCABench — model-agnostic runner + BenchReport
  energy/
    profiler.py      EnergyProfiler — nvidia-smi Joules/token measurement
  demos/
    admin.py         Public administration — GDPR/on-premises document assistant
    industry.py      Industry 4.0 — sensor anomaly + maintenance work orders
    health.py        Digital health — clinical note summary + ICD-10 coding

data/
  cunca_corpus.py               C4-style quality filter + MinHash-LSH dedup
  mix_configs/
    cunca_pretraining.yaml      gl 60 % / pt 40 % training mix
    base_pretraining.yaml       en 80 % / es 10 % / pt 10 % (Phase 1)
    galician_continual.yaml     gl 50 % / pt 25 % / es 25 % (Phase 2)

app/                API layer (FastAPI, auth, rate limiting, SSE streaming)
config/             TOML configuration files
core/
  think_anywhere/   Think-Anywhere inline reasoning (GRPO, streaming filter)
  special_tokens/   Structured meta-token framework (search, fact_check, …)
  backends/         Backend abstraction (TPU / GPU / CPU)
  moe/              Mixture-of-Experts routing
  cot/              Chain-of-thought helpers
inference/
  hybrid_inference_engine.py  Main inference orchestrator
  quantization/               INT8/INT4 Flax quantised layers
  engines/                    Quantised inference engine + KV cache
  int8_inference.py           NumPy INT8 fallback (no JAX required)
models/
  pretrained_backbone.py      TransformerNumpyBackbone + LlamaCppBackbone
  architecture.py             SlimModel (RMSNorm, RoPE, Mamba, Attention)
  lmtp.py                     L-MTP multi-token prediction heads
rag/                Vector store, ingestion, retriever
safety/             Input / output safety filters
training/
  tpu/              TPU v5 trainer (JAX/Flax)
  optimizations/    TPU v4/v5 config, XLA settings
  strategies/       Convexity-aware LR control
  data_capture/     Auto-capture training pairs from inference
  data_preprocessing/  Quality filter, deduplicator, TPU processor
  lmtp_trainer.py   L-MTP two-stage training recipe
  btx_training_system.py  Branch-Train-MiX expert training
  moe_hierarchical_router.py  Multi-tier MoE router
tests/              Unit / integration / security / benchmark tests
```

---

## Quick start

### TPU v5 (JAX/Flax)

```bash
pip install -e ".[tpu]"

# Phase 1 — multilingual base (en/es/pt)
python -m training.tpu.tpu_v6e_trainer \
    --config config/configs_toml/production/training.toml \
    --mix data/mix_configs/base_pretraining.yaml

# Phase 2 — Galician continual pretraining
python -m training.tpu.tpu_v6e_trainer \
    --config config/configs_toml/production/training.toml \
    --mix data/mix_configs/galician_continual.yaml \
    --resume checkpoints/phase1/latest
```

### GPU with QLoRA (PyTorch)

```bash
pip install -e ".[gpu]"
```

```python
from cunca.config import CUNCAConfig
from cunca.architecture import CUNCAModel
from cunca.qlora import apply_qlora, qlora_stats

model = CUNCAModel(CUNCAConfig.preset("1.3b"))
n = apply_qlora(model, rank=16, alpha=32, target_modules=["q_proj", "v_proj"])

stats = qlora_stats(model)
print(f"Trainable: {stats.trainable_ratio:.2%} of parameters ({n} LoRA layers)")
```

### CPU development (NumPy only)

```bash
pip install numpy gguf

python scripts/train_and_export_gguf.py \
    --steps 2000 --hidden 384 --n-layers 6 --n-heads 6 \
    --seq 128 --batch 4 --lr 1e-3 \
    --out models/cunca_dev.gguf
```

### API server

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

---

## Think-Anywhere

`core/think_anywhere/` implements the **Think-Anywhere** mechanism
([Jiang et al., 2026](https://arxiv.org/abs/2603.29957)) — the model inserts
`<thinkanywhere>` blocks at any token position, focusing compute where generation
is hardest.

| Class | Purpose |
|---|---|
| `ThinkAnywhereProcessor` | Format prompts, parse responses, strip thinking blocks |
| `ThinkAnywhereReward` | R = 0.1·R_struct + 0.9·R_correct, GRPO advantages |
| `ThinkAnywhereStreamFilter` | Real-time streaming filter — suppresses thinking tokens |

```python
from core.think_anywhere import ThinkAnywhereProcessor

proc = ThinkAnywhereProcessor()
prompt = proc.format_prompt("Traduce ao galego: 'The patient has hypertension.'")
result = proc.parse(model_response)
print(result.clean_text)   # output with thinking blocks stripped
```

Enable during inference:

```python
from inference.hybrid_inference_engine import InferenceConfig
config = InferenceConfig(think_anywhere_mode=True)
```

---

## Special-token framework

`core/special_tokens/` provides structured meta-tokens with semantic-aware
embedding initialisation and real-time streaming filters.

| Token | Stripped? | Purpose |
|---|---|---|
| `<verify>` | yes | Self-verification before continuing |
| `<plan>` | yes | Task decomposition |
| `<uncertain>` | no | Low-confidence marker for caller |
| `<search>` | yes | Local RAG trigger |
| `<web_search>` | yes | Live internet search |
| `<fact_check>` | no | Contradiction signal |
| `<lang:XX>` | yes | Inline language switch (e.g. `<lang:gl>`) |
| `<debug>` | yes | Error diagnosis |

---

## L-MTP (multi-token prediction)

`models/lmtp.py` implements **Leap Multi-Token Prediction** (NeurIPS 2025,
arXiv:2505.17505) — additional prediction heads at leaping positions improve
training signal and enable look-backward decoding with 7 tokens per step.

```python
from models.lmtp import LMTPConfig, wrap_with_lmtp

config = LMTPConfig(n_head=4, leap_k=2)   # → 7 tokens/decode step
model  = wrap_with_lmtp(cunca_model, config)
```

---

## CUNCA-Bench

```python
from cunca.bench import CUNCABench

def my_generate(prompt: str) -> str:
    ...  # call your model

bench = CUNCABench(generate_fn=my_generate)
report = bench.run()
print(report.summary())
```

12 tasks across 4 categories:

| Category | Tasks | Languages |
|---|---|---|
| Comprehension | T01–T03 | gl, pt, es |
| Translation | T04–T07 | gl↔pt, gl↔es |
| Reasoning / Math | T08–T10 | gl |
| Safety | T11–T12 | gl |

---

## Corpus pipeline

```python
from data.cunca_corpus import CUNCACorpusProcessor

proc = CUNCACorpusProcessor(
    output_dir="data/processed/gl",
    lang="gl",
    min_words=50,
    dedup=True,
)
stats = proc.process_file("data/raw/gl/corpus.txt")
print(f"Kept {stats.kept_ratio:.1%} ({stats.deduplicated} near-duplicates removed)")
```

C4-style filters: word count, stop-word ratio, URL density, repeated punctuation, lorem-ipsum.
Deduplication: MinHash (128 perms) + LSH (16 bands × 8 rows) — Jaccard threshold ≈ 0.50.

Pretraining data sources:
- **CorpusNos** — Galician native corpus
- **CulturaX-GL** — web-crawl Galician subset
- **Gigaverbo** — pt/gl parallel corpus

---

## Sector demos

Three FastAPI routers implementing real-world use cases from the CUNCA Memoria Técnica:

```python
from cunca.demos.admin    import router as admin_router
from cunca.demos.industry import router as industry_router
from cunca.demos.health   import router as health_router

app.include_router(admin_router)
app.include_router(industry_router)
app.include_router(health_router)
```

| Sector | Endpoint | Description |
|---|---|---|
| Public administration | `POST /demo/admin/summarise` | Official document summarisation (GDPR/on-premises) |
| Public administration | `POST /demo/admin/regulation` | Regulation QA in Galician |
| Industry 4.0 | `POST /demo/industry/anomaly` | Sensor anomaly description + severity |
| Industry 4.0 | `POST /demo/industry/workorder` | Maintenance work order generation |
| Digital health | `POST /demo/health/summarise` | Clinical note summarisation |
| Digital health | `POST /demo/health/icd10` | ICD-10 code suggestion |

---

## Energy profiling

```python
from cunca.energy.profiler import EnergyProfiler

profiler = EnergyProfiler()
with profiler.measure() as ctx:
    output = model.generate(input_ids)
result = ctx.result(n_tokens=output.shape[-1])
print(f"{result.joules_per_token:.4f} J/token")
```

Falls back to wall-clock-only mode when `nvidia-smi` is unavailable.

---

## Requirements

| Environment | Packages |
|---|---|
| TPU training | `jax[tpu]`, `flax`, `optax` |
| GPU + QLoRA | `torch`, `peft`, `bitsandbytes` |
| CPU / dev | `numpy`, `gguf`, `llama-cpp-python` (optional) |
| API server | `fastapi`, `uvicorn`, `pydantic` |

Python >= 3.9.

```bash
pip install -e ".[tpu]"    # TPU v5
pip install -e ".[gpu]"    # GPU + QLoRA
pip install -e ".[dev]"    # local dev + tests
pip install numpy gguf     # CPU-only, no extras
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/unit/test_think_anywhere.py tests/unit/test_special_tokens.py -v
pytest tests/ --cov=cunca --cov=core --cov-report=term-missing
```

---

## Pending work

See [`BACKLOG.md`](./BACKLOG.md) for concrete pending items with scope and exit criteria.

---

## License

Dual licensing (open + commercial). See `LICENSE`.

## Contact

- GitHub: `https://github.com/anachroni-co/cunca`
- Website: `https://www.anachroni.co`
- Email: `info@anachroni.co`
