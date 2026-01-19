# CapibaraGPT

**An Experimental Open-Source Foundation Model for Research and Education**

[![License](https://img.shields.io/badge/License-Dual%20(Open%20%2B%20Commercial)-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Experimental-yellow.svg)]()
[![Free for Research](https://img.shields.io/badge/Research%20%26%20Education-Free-green.svg)](LICENSE)

---

## Abstract

CapibaraGPT is an experimental open-source foundation model designed for research, education, and community-driven AI development. This project explores modern deep learning architectures including Mixture of Experts (MoE), State Space Models (SSM/Mamba), and multi-backend hardware optimization across CPU, GPU (CUDA), and TPU platforms.

The project aims to provide a transparent, well-documented codebase that serves as both a learning resource and a platform for experimenting with novel AI techniques. CapibaraGPT is not intended to compete with production-grade commercial models, but rather to democratize access to foundation model development practices.

## Project Goals

1. **Educational Resource**: Provide clear, documented implementations of modern AI architectures
2. **Research Platform**: Enable experimentation with novel techniques in a modular framework
3. **Hardware Democratization**: Support diverse compute backends (CPU, GPU, TPU) to lower barriers to entry
4. **Community Development**: Foster open collaboration on foundation model research
5. **Transparency**: Maintain fully open-source code with no hidden components

## Architecture Overview

CapibaraGPT implements a modular architecture that combines several research directions:

```
capibaraGPT_v3/
├── core/                    # Core model components
│   ├── backends/            # Hardware abstraction (CPU/GPU/TPU)
│   ├── attention/           # Multi-head and sparse attention variants
│   ├── moe/                 # Mixture of Experts routing and experts
│   ├── ssm/                 # State Space Models (Mamba-style)
│   ├── cot/                 # Chain-of-Thought reasoning modules
│   ├── encoders/            # Multimodal encoders (vision, audio)
│   ├── kernels/             # Optimized compute kernels
│   └── optimizers/          # Training optimizers
├── data/                    # Data loading and preprocessing
├── training/                # Training loops and utilities
├── inference/               # Inference pipelines
├── tests/                   # Unit and integration tests
└── configs/                 # Model and training configurations
```

### Key Components

#### Multi-Backend Support
The project implements a unified backend abstraction layer supporting:
- **CPU**: NumPy-based reference implementation
- **GPU**: PyTorch/CUDA with Flash Attention support
- **TPU**: JAX/Flax with XLA optimization

#### Mixture of Experts (MoE)
Implementation of sparse MoE layers with:
- Top-k expert routing with load balancing
- Auxiliary loss for balanced expert utilization
- Support for domain-specialized experts

#### State Space Models
Experimental SSM implementations inspired by Mamba architecture:
- Linear-time sequence modeling
- Selective state spaces
- Hardware-efficient scan operations

#### Chain-of-Thought Reasoning
Modules for structured reasoning:
- Step-by-step reasoning decomposition
- Self-reflection and verification
- Configurable reasoning depth

## Installation

### Requirements

- Python 3.9+
- NumPy >= 1.21.0

**Optional dependencies (for accelerated backends):**
- PyTorch >= 2.0.0 (GPU support)
- JAX >= 0.4.0 (TPU support)
- Flax >= 0.7.0

### Setup

```bash
# Clone the repository
git clone https://github.com/anacronic-io/capibaraGPT_v3.git
cd capibaraGPT_v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install base package
pip install -e .

# Install with GPU support
pip install -e ".[gpu]"

# Install with TPU support
pip install -e ".[tpu]"

# Install development dependencies
pip install -e ".[dev]"
```

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

## Current Status

CapibaraGPT is under active development. Current status of major components:

| Component | Status | Notes |
|-----------|--------|-------|
| CPU Backend | Stable | Full reference implementation |
| GPU Backend | Beta | Requires PyTorch, Flash Attention optional |
| TPU Backend | Alpha | Requires JAX, experimental |
| MoE Routing | Beta | Basic top-k routing implemented |
| SSM/Mamba | Alpha | Experimental implementation |
| CoT Reasoning | Alpha | Basic modules available |
| Multimodal | Planned | Vision encoder in progress |

## Limitations and Disclaimers

**This is an experimental research project.** Please be aware of the following:

1. **Not Production Ready**: CapibaraGPT is designed for research and education, not production deployment
2. **No Pre-trained Weights**: This repository contains architecture code only; no trained model weights are provided
3. **Performance**: Performance may vary significantly from commercial models
4. **Correctness**: While we strive for correctness, implementations may contain bugs or diverge from reference papers
5. **Resource Requirements**: Training foundation models requires significant computational resources

## Contributing

We welcome contributions from the community. Please see our contributing guidelines:

### How to Contribute

1. **Report Issues**: Use GitHub Issues for bugs and feature requests
2. **Submit PRs**: Fork the repository and submit pull requests
3. **Documentation**: Help improve documentation and examples
4. **Testing**: Add tests for new functionality

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all public functions
- Write docstrings for modules, classes, and functions
- Maintain test coverage above 80%

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linters
ruff check .
mypy core/

# Run tests
pytest tests/ -v
```

## Research References

This project draws inspiration from the following research:

### Transformer Architecture
- Vaswani et al. (2017). "Attention Is All You Need" [[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)]

### Mixture of Experts
- Fedus et al. (2021). "Switch Transformers: Scaling to Trillion Parameter Models" [[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)]
- Lepikhin et al. (2020). "GShard: Scaling Giant Models with Conditional Computation" [[arXiv:2006.16668](https://arxiv.org/abs/2006.16668)]

### State Space Models
- Gu et al. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" [[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)]
- Gu et al. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces" [[arXiv:2111.00396](https://arxiv.org/abs/2111.00396)]

### Chain-of-Thought Reasoning
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" [[arXiv:2201.11903](https://arxiv.org/abs/2201.11903)]

### Efficient Attention
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention" [[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)]

## License

This project uses a **Dual License** model. See [LICENSE](LICENSE) for full details.

### Open Source License (Free)

Free for:
- Scientific research
- Educational purposes
- Non-profit organizations
- Personal projects
- Open source contributions

**Conditions**: Attribution required, Share-Alike, Non-Commercial only.

### Commercial License

**Required for**: Commercial products, internal business use, SaaS/Cloud services, paid consulting, startups, and any for-profit use.

**Contact for commercial licensing:**
- **Email**: info@anachroni.co
- **Website**: https://www.anachroni.co

| Use Case | License |
|----------|---------|
| Academic research | Free |
| Education | Free |
| Non-profit | Free |
| Personal projects | Free |
| Commercial use | Contact us |

```
Copyright (c) 2024-2026 Anacroni S.Coop.Gal. All rights reserved.
```

## Acknowledgments

- The open-source AI research community
- Contributors and maintainers of JAX, PyTorch, and NumPy
- Research teams whose papers inspired this implementation

## Contact

- **Repository**: [github.com/anacronic-io/capibaraGPT_v3](https://github.com/anacronic-io/capibaraGPT_v3)
- **Organization**: [Anacroni S.Coop.Gal.](https://www.anachroni.co)
- **Email**: info@anachroni.co
- **Issues**: [GitHub Issues](https://github.com/anacronic-io/capibaraGPT_v3/issues)

---

<div align="center">

**CapibaraGPT** — Open Foundation Model Research

*Free for science, education, and research. Commercial use requires license.*

</div>
