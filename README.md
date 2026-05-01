# CUNCA-Hybrid

Galician-focused hybrid Mamba-Transformer language model built on the Capibara Slim base.
Implements grouped query attention (GQA), sliding-window attention, and a configurable
SSM/attention ratio — designed for low-resource Romance-language NLP with efficient
fine-tuning via QLoRA.

> **Branch layout**
> `main` — original Capibara project |
> `capibara-slim` — multilingual inference + training base (Weeks 1-9) |
> `cunca-hybrid` — this branch: CUNCA architecture extensions

---

## Architecture

CUNCA-Hybrid extends the Capibara Slim Transformer + Mamba base with:

| Feature | Description |
|---|---|
| Grouped Query Attention (GQA) | `num_kv_heads < num_heads` — reduces KV-cache memory at inference |
| Sliding-Window Attention | Local context of `window_size` tokens — O(L·w) instead of O(L²) |
| Configurable SSM ratio | `ssm_ratio` controls fraction of Mamba vs Transformer blocks |
| QLoRA fine-tuning | NF4-quantized frozen base + trainable LoRA adapters (A·B) |

```
Input IDs
   ↓
Embedding (vocab → hidden)
   ↓
Block 0  → CUNCATransformerBlock  (GQA + sliding-window + SwiGLU MLP)
Block 1  → CUNCAMambaBlock        (selective SSM: conv1d → ZOH → scan)
Block 2  → CUNCATransformerBlock
Block 3  → CUNCAMambaBlock
   ...   (ssm_ratio controls the pattern)
   ↓
RMSNorm
   ↓
LM head (weight-tied to embedding)
```

### Presets

| Preset | Hidden | Layers | Heads | KV heads | Window | Params |
|---|---|---|---|---|---|---|
| `1.3b` | 2048 | 24 | 16 | 4 | 512 | ~1.3B |
| `3b` | 2560 | 32 | 20 | 4 | 1024 | ~3B |
| `7b` | 4096 | 32 | 32 | 8 | 2048 | ~7B |

---

## Modules

```
cunca/
  config.py          CUNCAConfig — GQA, window_size, ssm_ratio, presets
  architecture.py    CUNCAModel, CUNCAAttention, CUNCAMambaBlock
  qlora.py           NF4 quantization, LoRALinear, apply_qlora()
  bench/
    tasks.py         12-task evaluation suite (comprehension, translation,
                     reasoning, safety) for gl / pt / es
    evaluator.py     CUNCABench — model-agnostic runner + BenchReport
  energy/
    profiler.py      EnergyProfiler — NVIDIA SMI Joules/token measurement
  demos/
    admin.py         Public administration — GDPR/on-premises document assistant
    industry.py      Industry 4.0 — sensor anomaly + maintenance work orders
    health.py        Digital health — clinical note summary + ICD-10 coding

data/
  cunca_corpus.py               C4-style quality filter + MinHash-LSH dedup
  mix_configs/
    cunca_pretraining.yaml      gl 60% / pt 40% training mix
    base_pretraining.yaml       en 80% / es 10% / pt 10%  (Phase 1)
    galician_continual.yaml     gl 50% / pt 25% / es 25%  (Phase 2)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements-slim.txt

# Run API server
uvicorn app.main:app --reload

# Run all Slim + CUNCA tests
python -m pytest tests/slim/ -v

# Run only CUNCA tests
python -m pytest tests/slim/test_cunca.py -v
```

---

## Training

### Two-phase Capibara Slim base (Weeks 1-9)

```bash
# Phase 1 — multilingual base (en/es/pt)
python -m training.train --preset 3b \
    --mix data/mix_configs/base_pretraining.yaml \
    --output checkpoints/phase1

# Phase 2 — Galician continual pretraining
python -m training.train --preset 3b \
    --mix data/mix_configs/galician_continual.yaml \
    --resume checkpoints/phase1/latest \
    --output checkpoints/phase2 \
    --lr 1e-4 --max-steps 30000
```

### CUNCA pretraining (gl/pt corpus)

```bash
# Download and preprocess CUNCA corpus
python -m data.download get corpusnos   --output data/raw
python -m data.download get culturax-gl --output data/raw
python -m data.download get gigaverbo   --output data/raw

# Filter + deduplicate
python -m data.cunca_corpus process data/raw/gl --output data/processed/gl --lang gl
python -m data.cunca_corpus process data/raw/pt --output data/processed/pt --lang pt

# Train CUNCA-1.3B
python -m training.train --preset 1.3b \
    --mix data/mix_configs/cunca_pretraining.yaml \
    --output checkpoints/cunca_base \
    --lr 3e-4 --max-steps 50000
```

### QLoRA fine-tuning

```python
from cunca.config import CUNCAConfig
from cunca.architecture import CUNCAModel
from cunca.qlora import apply_qlora, qlora_stats

model = CUNCAModel(CUNCAConfig.preset("1.3b"))
n = apply_qlora(model, rank=16, alpha=32, target_modules=["q_proj", "v_proj"])

stats = qlora_stats(model)
print(f"Trainable: {stats.trainable_ratio:.2%} of parameters ({n} LoRA layers)")
```

---

## Evaluation

```python
from cunca.bench import CUNCABench

def my_generate(prompt: str) -> str:
    ...  # call your model

bench = CUNCABench(generate_fn=my_generate)
report = bench.run()
print(report.summary())
```

CUNCA-Bench covers 12 tasks across 4 categories:

| Category | Tasks | Languages |
|---|---|---|
| Comprehension | 3 (T01-T03) | gl, pt, es |
| Translation | 4 (T04-T07) | gl↔pt, gl↔es |
| Reasoning / Math | 3 (T08-T10) | gl |
| Safety | 2 (T11-T12) | gl |

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

## Sector demos

Three FastAPI routers implementing real-world use cases per the CUNCA Memoria Técnica:

```bash
# Mount all three demos on a FastAPI app
from cunca.demos.admin    import router as admin_router
from cunca.demos.industry import router as industry_router
from cunca.demos.health   import router as health_router

app.include_router(admin_router)    # POST /demo/admin/summarise
app.include_router(industry_router) # POST /demo/industry/anomaly
app.include_router(health_router)   # POST /demo/health/icd10
```

| Sector | Endpoint | Description |
|---|---|---|
| Public administration | `/demo/admin/summarise` | Official document summarisation (GDPR/on-premises) |
| Public administration | `/demo/admin/regulation` | Regulation QA in Galician |
| Industry 4.0 | `/demo/industry/anomaly` | Sensor anomaly description + severity |
| Industry 4.0 | `/demo/industry/workorder` | Maintenance work order generation |
| Digital health | `/demo/health/summarise` | Clinical note summarisation |
| Digital health | `/demo/health/icd10` | ICD-10 code suggestion |

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
print(f"Kept {stats.kept_ratio:.1%} of documents ({stats.deduplicated} near-duplicates removed)")
```

Filters applied (C4-style):
- Minimum / maximum word count
- Stop-word ratio (language heuristic)
- URL density
- Repeated punctuation
- Lorem ipsum detection

Deduplication: MinHash (128 permutations) + LSH (16 bands × 8 rows) — Jaccard threshold ≈ 0.50.

---

## Requirements

- Python >= 3.9
- `pip install -r requirements-slim.txt`

Optional:
- PyTorch >= 2.0 — required for model training and inference
- `nvidia-smi` — required for energy profiling (CPU fallback available)

---

## License

Dual licensing (open + commercial). See `LICENSE`.

## Contact

- GitHub: `https://github.com/anachroni-co/capibaraGPT_v3`
- Website: `https://www.anachroni.co`
- Email: `info@anachroni.co`
