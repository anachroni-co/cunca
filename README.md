# CapibaraGPT v3

<div align="center">

<img src="https://img.shields.io/badge/CapibaraGPT-v3.0-orange?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQxIDAtOC0zLjU5LTgtOHMzLjU5LTggOC04IDggMy41OSA4IDgtMy41OSA4LTggOHoiLz48L3N2Zz4=" alt="CapibaraGPT"/>

**An Experimental Open-Source Foundation Model for Research and Education**

[![License](https://img.shields.io/badge/License-Dual%20(Open%20%2B%20Commercial)-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-392%20passed-success.svg?logo=pytest)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg?logo=ruff)](https://github.com/astral-sh/ruff)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![JAX](https://img.shields.io/badge/JAX-0.4+-A435F0.svg)](https://github.com/google/jax)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg?logo=numpy&logoColor=white)](https://numpy.org)

</div>

---

## Project Statistics

<div align="center">

| Metric | Value |
|--------|-------|
| **Total Python Files** | 519 |
| **Lines of Code** | ~130,000+ |
| **Test Cases** | 392 |
| **Modules** | 15+ |
| **Backends** | 3 (CPU, GPU, TPU) |

</div>

### Lines of Code by Module

```
core          ████████████████████████████████████████  30,097 (23%)
training      ██████████████████████████████████████████████████  35,915 (28%)
data          ████████████████  11,231 (9%)
services      ███████████████  10,993 (8%)
sub_models    ███████████████  10,497 (8%)
jax           ████████████  9,582 (7%)
utils         ██████████  8,200 (6%)
agents        ██████████  8,807 (7%)
inference     ███████  6,446 (5%)
config        ████  3,659 (3%)
benchmarks    ██  2,133 (2%)
pipeline      ██  2,201 (2%)
```

---

## Architecture Overview

### System Architecture

```mermaid
graph TB
    subgraph Input["📥 Input Layer"]
        TEXT[Text Input]
        IMG[Image Input]
        AUDIO[Audio Input]
    end

    subgraph Encoders["🔄 Encoders"]
        TOK[Tokenizer]
        VE[Vision Encoder]
        AE[Audio Encoder]
    end

    subgraph Core["⚙️ Core Model"]
        EMB[Embeddings]

        subgraph Layers["Transformer Layers"]
            ATT[Multi-Head Attention]
            MOE[Mixture of Experts]
            SSM[State Space Model]
            FFN[Feed Forward]
        end

        COT[Chain-of-Thought]
        NORM[Layer Norm]
    end

    subgraph Backends["🖥️ Hardware Backends"]
        CPU[CPU Backend<br/>NumPy]
        GPU[GPU Backend<br/>PyTorch/CUDA]
        TPU[TPU Backend<br/>JAX/Flax]
    end

    subgraph Output["📤 Output"]
        LM[Language Model Head]
        GEN[Generation]
    end

    TEXT --> TOK
    IMG --> VE
    AUDIO --> AE

    TOK --> EMB
    VE --> EMB
    AE --> EMB

    EMB --> ATT
    ATT --> MOE
    MOE --> SSM
    SSM --> FFN
    FFN --> COT
    COT --> NORM
    NORM --> LM
    LM --> GEN

    Layers -.-> CPU
    Layers -.-> GPU
    Layers -.-> TPU
```

### Module Structure

```mermaid
graph LR
    subgraph core["🧠 Core (30K lines)"]
        backends[backends]
        attention[attention]
        moe[moe]
        ssm[ssm]
        cot[cot]
        encoders[encoders]
        kernels[kernels]
        optimizers[optimizers]
    end

    subgraph training["🎯 Training (36K lines)"]
        consensus[consensus]
        strategies[strategies]
        safety[safety]
        preprocessing[preprocessing]
        federated[federated]
    end

    subgraph data["📊 Data (11K lines)"]
        datasets[datasets]
        loaders[loaders]
        processors[processors]
    end

    subgraph services["🔧 Services (11K lines)"]
        automation[automation]
        generation[generation]
        analysis[analysis]
    end

    core --> training
    data --> training
    training --> services
```

---

## Key Components

### Backend Architecture

```mermaid
flowchart TD
    subgraph API["Unified API"]
        GET[get_backend]
        OPS[Tensor Operations]
        ATT[Attention Ops]
    end

    subgraph Backends
        subgraph CPU["CPU Backend"]
            NP[NumPy]
            SP[SciPy]
        end

        subgraph GPU["GPU Backend"]
            PT[PyTorch]
            CUDA[CUDA Kernels]
            FA[Flash Attention]
        end

        subgraph TPU["TPU Backend"]
            JAX[JAX]
            FLAX[Flax]
            XLA[XLA Compiler]
        end
    end

    GET --> CPU
    GET --> GPU
    GET --> TPU

    OPS --> NP
    OPS --> PT
    OPS --> JAX

    ATT --> SP
    ATT --> FA
    ATT --> XLA
```

### Mixture of Experts (MoE)

```mermaid
flowchart LR
    INPUT[Input Tokens] --> ROUTER[Router Network]

    ROUTER --> |Top-K| E1[Expert 1<br/>Language]
    ROUTER --> |Top-K| E2[Expert 2<br/>Math]
    ROUTER --> |Top-K| E3[Expert 3<br/>Code]
    ROUTER --> |Top-K| E4[Expert 4<br/>Reasoning]
    ROUTER --> |Top-K| EN[Expert N<br/>Domain]

    E1 --> COMBINE[Weighted<br/>Combination]
    E2 --> COMBINE
    E3 --> COMBINE
    E4 --> COMBINE
    EN --> COMBINE

    COMBINE --> OUTPUT[Output]

    style ROUTER fill:#f9f,stroke:#333
    style COMBINE fill:#9f9,stroke:#333
```

### State Space Model (SSM/Mamba)

```mermaid
flowchart LR
    X[Input x] --> PROJ1[Input Projection]
    PROJ1 --> CONV[Conv1D]
    CONV --> SSM_BLOCK[SSM Block]

    subgraph SSM_BLOCK[Selective SSM]
        A[State Matrix A]
        B[Input Matrix B]
        C[Output Matrix C]
        D[Delta Δ]
    end

    SSM_BLOCK --> MULT[Gated Multiplication]
    X --> PROJ2[Gate Projection]
    PROJ2 --> SILU[SiLU]
    SILU --> MULT

    MULT --> OUT[Output Projection]
    OUT --> Y[Output y]
```

---

## Benchmark Results

### Performance Benchmarks (CPU Backend)

| Benchmark | Mean Time | Std Dev | Throughput |
|-----------|-----------|---------|------------|
| attention_small (128 seq) | 30.7 ms | ±0.7 ms | 32.5 ops/s |
| attention_medium (512 seq) | 174.8 ms | ±3.0 ms | 5.7 ops/s |
| matmul (512x512) | 438.0 ms | ±3.0 ms | 2.3 ops/s |
| gelu | 1416.2 ms | ±10.7 ms | 0.7 ops/s |
| softmax | 454.2 ms | ±1.8 ms | 2.2 ops/s |
| layer_norm | 475.3 ms | ±4.0 ms | 2.1 ops/s |

```
Benchmark Performance (lower is better)
═══════════════════════════════════════════════════════════════

attention_small   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  30.7ms
attention_medium  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░  174.8ms
matmul            █████████████████████████░░░░░░░░░░░  438.0ms
softmax           ██████████████████████████░░░░░░░░░░  454.2ms
layer_norm        ███████████████████████████░░░░░░░░░  475.3ms
gelu              █████████████████████████████████████  1416.2ms
```

### Memory Profile

```mermaid
pie title Memory Distribution by Component
    "Model Parameters" : 45
    "Activations" : 25
    "Optimizer States" : 20
    "Buffers" : 10
```

---

## Test Coverage

### Test Results Summary

```
Total Tests: 411
├── Passed:  392 (95.4%)
├── Skipped: 45  (10.9%)  [Hardware-specific]
└── Failed:  0   (0%)
```

### Coverage by Module

```
Module Coverage
═══════════════════════════════════════════════════════════════

core/backends     ████████████████████████████████████████  98%
core/attention    ██████████████████████████████████████░░  95%
core/moe          █████████████████████████████████████░░░  92%
utils             ████████████████████████████████████░░░░  90%
benchmarks        ████████████████████████████████████░░░░  90%
training          ██████████████████████████████░░░░░░░░░░  75%
config            █████████████████████████████████░░░░░░░  85%
```

### Test Categories

| Category | Tests | Status |
|----------|-------|--------|
| Unit Tests | 280 | ✅ All Pass |
| Benchmarks | 36 | ✅ All Pass |
| Security | 45 | ✅ All Pass |
| Integration | 31 | ✅ All Pass |

---

## Sub-Models & Experts

```mermaid
graph TB
    subgraph SubModels["🤖 Sub-Models"]
        SSM_TPU[SSM TPU<br/>State Space Model]
        BYTE_TPU[Byte TPU<br/>Byte-level Processing]
        CSA[CSA Expert<br/>Context-aware]
        ALEPH[Aleph Tilde<br/>Mathematical]
        DIALOG[Deep Dialog<br/>Conversation]
        REASON[Reasoning<br/>Enhancement]
    end

    subgraph Orchestrator["🎭 Ultra Orchestrator"]
        ROUTER[Model Router]
        ENSEMBLE[Ensemble Manager]
        WEIGHT[Weight Allocator]
    end

    subgraph Output["📤 Unified Output"]
        MERGE[Response Merger]
        RANK[Quality Ranker]
    end

    SSM_TPU --> ROUTER
    BYTE_TPU --> ROUTER
    CSA --> ROUTER
    ALEPH --> ROUTER
    DIALOG --> ROUTER
    REASON --> ROUTER

    ROUTER --> ENSEMBLE
    ENSEMBLE --> WEIGHT
    WEIGHT --> MERGE
    MERGE --> RANK
```

---

## Services Architecture

```mermaid
flowchart TB
    subgraph Services["🔧 Services Layer"]
        subgraph Meta["Meta Generation"]
            TTS[Text-to-Speech]
            TTG[Text-to-Gen]
        end

        subgraph Analysis["Analysis"]
            GENOMIC[Genomic Analysis]
            BIM[BIM Generation]
        end

        subgraph Automation["Automation"]
            PIPELINE[Pipeline Runner]
            SCHEDULER[Task Scheduler]
        end
    end

    subgraph Core["Core Model"]
        MODEL[CapibaraGPT Core]
    end

    MODEL --> Meta
    MODEL --> Analysis
    MODEL --> Automation
```

---

## Training Pipeline

```mermaid
flowchart LR
    subgraph Data["📊 Data Pipeline"]
        RAW[Raw Data]
        CLEAN[Cleaning]
        TOK[Tokenization]
        BATCH[Batching]
    end

    subgraph Training["🎯 Training Loop"]
        FWD[Forward Pass]
        LOSS[Loss Computation]
        BWD[Backward Pass]
        OPT[Optimizer Step]
    end

    subgraph Consensus["🤝 Consensus"]
        FED[Federated Learning]
        AGG[Gradient Aggregation]
        SYNC[Model Sync]
    end

    subgraph Safety["🛡️ Safety"]
        FILTER[Content Filter]
        ALIGN[Alignment Check]
        AUDIT[Audit Log]
    end

    RAW --> CLEAN --> TOK --> BATCH
    BATCH --> FWD --> LOSS --> BWD --> OPT
    OPT --> FED --> AGG --> SYNC
    SYNC --> FILTER --> ALIGN --> AUDIT
    AUDIT --> FWD
```

---

## Installation

### Requirements

- Python 3.9+
- NumPy >= 1.24.0

**Optional dependencies (for accelerated backends):**
- PyTorch >= 2.0.0 (GPU support)
- JAX >= 0.4.0 (TPU support)
- Flax >= 0.7.0

### Quick Install

```bash
# Clone the repository
git clone https://github.com/anachroni-co/capibaraGPT_v3.git
cd capibaraGPT_v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install base package
pip install -e .

# Install with GPU support
pip install -e ".[gpu]"

# Install with TPU support
pip install -e ".[tpu]"

# Install development dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### Basic Usage

```python
from core.backends import get_backend, BackendType

# Automatically select best available backend
backend = get_backend(BackendType.AUTO)
print(f"Using backend: {backend.name}")

# Create tensors
x = backend.randn((batch_size, seq_len, hidden_dim))

# Perform operations
y = backend.layer_norm(x, normalized_shape=(hidden_dim,))
z = backend.gelu(y)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

### Running Benchmarks

```bash
# Run automated benchmarks
python -m benchmarks run

# Generate HTML report
python -m benchmarks report --input results.json --output report.html

# CI/CD mode with regression detection
python -m benchmarks run --ci --baseline baseline.json --threshold 10
```

---

## Component Status

| Component | Status | Progress |
|-----------|--------|----------|
| CPU Backend | ✅ Stable | █████████████████████ 100% |
| GPU Backend | 🟡 Beta | ████████████████░░░░░ 80% |
| TPU Backend | 🟠 Alpha | ████████████░░░░░░░░░ 60% |
| MoE Routing | 🟡 Beta | ████████████████░░░░░ 80% |
| SSM/Mamba | 🟠 Alpha | ██████████░░░░░░░░░░░ 50% |
| CoT Reasoning | 🟠 Alpha | ████████░░░░░░░░░░░░░ 40% |
| Memory Profiler | ✅ Stable | █████████████████████ 100% |
| Benchmarks | ✅ Stable | █████████████████████ 100% |
| Documentation | 🟡 Beta | ██████████████░░░░░░░ 70% |

---

## Project Structure

```
capibaraGPT_v3/
├── 🧠 core/                 # Core model components (30K lines)
│   ├── backends/            # Hardware abstraction (CPU/GPU/TPU)
│   ├── attention/           # Multi-head and sparse attention
│   ├── moe/                 # Mixture of Experts
│   ├── ssm/                 # State Space Models
│   ├── cot/                 # Chain-of-Thought reasoning
│   ├── encoders/            # Multimodal encoders
│   ├── kernels/             # Optimized compute kernels
│   └── optimizers/          # Training optimizers
│
├── 🎯 training/             # Training system (36K lines)
│   ├── consensus/           # Distributed consensus
│   ├── strategies/          # Training strategies
│   ├── safety/              # Safety mechanisms
│   └── federated/           # Federated learning
│
├── 📊 data/                 # Data handling (11K lines)
│   ├── datasets/            # Dataset implementations
│   └── loaders/             # Data loaders
│
├── 🤖 sub_models/           # Specialized sub-models (10K lines)
│   ├── SSM_TPU/             # TPU-optimized SSM
│   ├── csa_expert/          # Context-aware expert
│   └── reasoning/           # Reasoning enhancement
│
├── 🔧 services/             # Application services (11K lines)
│   ├── meta_generation/     # Content generation
│   └── automation/          # Task automation
│
├── 🛠️ utils/                # Utilities (8K lines)
│   ├── memory_profiler/     # Memory profiling
│   └── monitoring/          # System monitoring
│
├── 📈 benchmarks/           # Benchmark system (2K lines)
│   ├── runner.py            # Benchmark execution
│   ├── comparator.py        # Result comparison
│   └── reporter.py          # Report generation
│
├── ⚙️ config/               # Configuration (4K lines)
│   ├── config.yaml          # Main configuration
│   └── config_loader.py     # Config utilities
│
├── 🧪 tests/                # Test suite (7K lines)
│   ├── unit/                # Unit tests
│   ├── benchmarks/          # Performance tests
│   └── security/            # Security tests
│
└── 📚 docs/                 # Documentation
    ├── conf.py              # Sphinx config
    └── api/                 # API reference
```

---

## Research References

### Transformer Architecture
- Vaswani et al. (2017). "Attention Is All You Need" [[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)]

### Mixture of Experts
- Fedus et al. (2021). "Switch Transformers" [[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)]
- Lepikhin et al. (2020). "GShard" [[arXiv:2006.16668](https://arxiv.org/abs/2006.16668)]

### State Space Models
- Gu et al. (2023). "Mamba" [[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)]

### Efficient Attention
- Dao et al. (2022). "FlashAttention" [[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)]

---

## License

<div align="center">

| Use Case | License |
|----------|---------|
| 🎓 Academic Research | ✅ Free |
| 📚 Education | ✅ Free |
| 🏛️ Non-profit | ✅ Free |
| 👤 Personal Projects | ✅ Free |
| 🏢 Commercial Use | 📧 Contact Us |

</div>

**Contact for commercial licensing:**
- **Email**: info@anachroni.co
- **Website**: https://www.anachroni.co

---

## Acknowledgments

### Compute Partners

<div align="center">
<table>
<tr>
<td align="center">
<a href="https://www.cesga.es">
<img src="https://www.cesga.es/wp-content/uploads/2024/10/logo_Blanco450x113.png" alt="CESGA" width="220"/>
</a>
<br/>
<b>CESGA</b> — Centro de Supercomputación de Galicia
<br/>
Providing access to the <b>Finisterrae III</b> supercomputer
</td>
<td align="center">
<a href="https://eurocc-spain.res.es/">
<img src="https://eurocc-spain.res.es/wp-content/uploads/2020/12/LOGOTIPO-EURO-CC-RES-SPAIN.png" alt="EuroCC Spain" width="220"/>
</a>
<br/>
<b>EuroCC Spain</b> — European HPC Competence Centre
<br/>
European programme supporting HPC access for research
</td>
</tr>
</table>
</div>

---

## Contact

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-anachroni--co/capibaraGPT__v3-181717?logo=github)](https://github.com/anachroni-co/capibaraGPT_v3)
[![Website](https://img.shields.io/badge/Website-anachroni.co-blue)](https://www.anachroni.co)
[![Email](https://img.shields.io/badge/Email-info@anachroni.co-red)](mailto:info@anachroni.co)

</div>

---

<div align="center">

**CapibaraGPT v3** — Open Foundation Model Research

*Free for science, education, and research. Commercial use requires license.*

```
Copyright (c) 2024-2026 Anacroni S.Coop.Gal. All rights reserved.
```

</div>
