# CapibaraGPT Makefile
# Usage: make <target>

.PHONY: help install install-tpu install-gpu install-dev test test-unit test-integration lint format clean build docs

# Default target
help:
	@echo "CapibaraGPT Development Commands"
	@echo "================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install core dependencies"
	@echo "  make install-tpu      - Install TPU/JAX dependencies"
	@echo "  make install-gpu      - Install GPU/PyTorch dependencies (A-100)"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-all      - Install all dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-core        - Run core module tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make benchmark        - Run performance benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run linter (ruff)"
	@echo "  make format           - Format code (black + isort)"
	@echo "  make typecheck        - Run type checking (mypy)"
	@echo "  make check            - Run all checks (lint + typecheck)"
	@echo ""
	@echo "Training:"
	@echo "  make train-tpu        - Start TPU training"
	@echo "  make train-gpu        - Start GPU training (A-100)"
	@echo "  make train-synthetic  - Train with synthetic data"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  make build            - Build package"
	@echo "  make docs             - Build documentation"
	@echo "  make clean            - Clean build artifacts"

# Python interpreter
PYTHON := python3
PIP := pip3

# Directories
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs

# Installation targets
install:
	$(PIP) install -e .

install-tpu:
	$(PIP) install -r requirements-tpu.txt
	$(PIP) install -e .

install-gpu:
	$(PIP) install -r requirements-gpu.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all:
	$(PIP) install -r requirements-tpu.txt
	$(PIP) install -r requirements-gpu.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[all]"

# Testing targets
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

test-unit:
	$(PYTHON) -m pytest $(TEST_DIR)/unit -v -m "unit"

test-integration:
	$(PYTHON) -m pytest $(TEST_DIR)/integration -v -m "integration"

test-core:
	$(PYTHON) -m pytest $(TEST_DIR)/unit/test_core*.py -v

test-cov:
	$(PYTHON) -m pytest $(TEST_DIR) --cov=capibara --cov-report=html --cov-report=term-missing

test-tpu:
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "tpu"

test-gpu:
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "gpu"

benchmark:
	$(PYTHON) -m pytest $(TEST_DIR)/benchmarks -v --benchmark-enable

# Code quality targets
lint:
	$(PYTHON) -m ruff check $(SRC_DIR) --fix

format:
	$(PYTHON) -m black $(SRC_DIR)
	$(PYTHON) -m isort $(SRC_DIR)

typecheck:
	$(PYTHON) -m mypy $(SRC_DIR) --ignore-missing-imports

check: lint typecheck

# Training targets
train-tpu:
	$(PYTHON) -m capibara.training.cli train --backend=tpu --config=configs/production/tpu_v4.toml

train-gpu:
	$(PYTHON) -m capibara.training.cli train --backend=gpu --config=configs/production/gpu_a100.toml

train-synthetic:
	$(PYTHON) -m capibara.training.cli train --synthetic --config=configs/development/testing.toml

# Inference targets
infer-tpu:
	$(PYTHON) -m capibara.inference.cli --backend=tpu

infer-gpu:
	$(PYTHON) -m capibara.inference.cli --backend=gpu

# Build targets
build:
	$(PYTHON) -m build

docs:
	cd $(DOCS_DIR) && make html

# Clean targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

clean-all: clean
	rm -rf .venv/
	rm -rf node_modules/
