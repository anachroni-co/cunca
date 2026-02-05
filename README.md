# CapibaraGPT v3

An experimental, modular foundation model system for research and education.

## What It Is
CapibaraGPT v3 is a multi-backend AI stack that supports CPU, GPU, and TPU
execution, with modular routing and extensible training and inference pipelines.

## Core Ideas
- Backend-agnostic compute (CPU/GPU/TPU)
- Modular model components (MoE, SSM, CoT, encoders)
- Routing layer for dynamic module selection
- Training and inference pipelines with optional safety and monitoring

## Project Structure (High Level)
- `core/` model components, routing, backends, kernels
- `training/` training strategies, optimizations, consensus
- `inference/` inference engines and runtime
- `data/` datasets, loaders, processors
- `services/` automation and analysis
- `benchmarks/` performance benchmarks
- `config/` configuration and presets

## Quick Start
```bash
# create environment
python -m venv venv

# activate (Windows)
venv\Scriptsctivate

# install base package
pip install -e .
```

```bash
# basic health check
python main.py --health

# demo mode
python main.py --demo
```

## Status
This repository is a research codebase. Some components are experimental or
hardware-specific. CPU backend is the default fallback if GPU/TPU is not
available.

## License
Dual license: free for research, education, non-profit, and personal projects.
Commercial use requires a separate license. See `LICENSE` for details.
