#!/bin/bash
# Script to create GitHub Issues from TODOs
# Run: bash scripts/create_todo_issues.sh
# Requires: gh cli authenticated (gh auth login)

set -e

REPO="anachroni-co/capibaraGPT_v3"

create_issue() {
  local title="$1"
  local body="$2"
  local labels="$3"
  echo "Creating issue: $title"
  gh issue create --repo "$REPO" --title "$title" --body "$body" --label "$labels"
}

# Create labels first (ignore errors if they exist)
gh label create "todo" --repo "$REPO" --color "fbca04" --description "Pending TODO from codebase" 2>/dev/null || true
gh label create "agents" --repo "$REPO" --color "0e8a16" 2>/dev/null || true
gh label create "core" --repo "$REPO" --color "1d76db" 2>/dev/null || true
gh label create "pipeline" --repo "$REPO" --color "5319e7" 2>/dev/null || true
gh label create "config" --repo "$REPO" --color "d93f0b" 2>/dev/null || true
gh label create "training" --repo "$REPO" --color "c5def5" 2>/dev/null || true
gh label create "datasets" --repo "$REPO" --color "bfdadc" 2>/dev/null || true
gh label create "services" --repo "$REPO" --color "f9d0c4" 2>/dev/null || true
gh label create "jax-tpu" --repo "$REPO" --color "e99695" 2>/dev/null || true

# Agents
create_issue \
  "Implement real LLM integration (OpenAI, Anthropic, local models)" \
  "## Context
\`agents/capibara_agent_factory.py:69\`

The agent factory currently has a placeholder for LLM integration. Need to implement actual connections to LLM providers (OpenAI, Anthropic, local models, etc.)." \
  "todo,agents"

create_issue \
  "Implement real Qdrant vector database connection" \
  "## Context
\`agents/capibara_agent_factory.py:92\`

Replace the placeholder with an actual Qdrant vector database connection for the agent factory." \
  "todo,agents"

# Core
create_issue \
  "Use capibara's struct implementation instead of flax" \
  "## Context
\`core/optimization.py:81\`

Replace the flax struct import with capibara's own struct implementation." \
  "todo,core"

# Pipeline
create_issue \
  "Implement stage-specific execution in pipeline" \
  "## Context
\`pipeline/run_pipeline.py:196\`

The pipeline runner needs stage-specific execution logic implemented." \
  "todo,pipeline"

# Config
create_issue \
  "Implement config validation logic" \
  "## Context
- \`config/__init__.py:61\` — Implement actual config validation logic
- \`config/__init__.py:66\` — Implement file-based config validation

Both validation paths need real implementations." \
  "todo,config"

# Training
create_issue \
  "Implement perplexity model and filtering in quality_filter" \
  "## Context
- \`training/data_preprocessing/quality_filter.py:359\` — Initialize perplexity model (GPT-2 or similar)
- \`training/data_preprocessing/quality_filter.py:381\` — Implement perplexity calculation and filtering" \
  "todo,training"

create_issue \
  "Implement gradient-based importance scoring and influence calculation" \
  "## Context
- \`training/data_lineage/dataset_parameter_controller.py:207\` — Gradient-based importance scoring
- \`training/data_lineage/dataset_parameter_controller.py:215\` — Actual influence calculation" \
  "todo,training"

# Datasets
create_issue \
  "Implement emotional audio dataset download and processing" \
  "## Context
- \`data/datasets/multimodal/emotional_audio_datasets.py:342\` — Download and processing
- \`data/datasets/multimodal/emotional_audio_datasets.py:361\` — Download with authentication" \
  "todo,datasets"

# Sub-models
create_issue \
  "Connect CSA expert to RAG/Knowledge Base for evidence retrieval" \
  "## Context
\`sub_models/csa_expert.py:332\`

The CSA expert currently returns empty evidence. Connect it to a RAG or Knowledge Base system." \
  "todo,core"

# Services
create_issue \
  "Add detailed descriptions for system_info and tts services" \
  "## Context
- \`utils/system_info.py:23\` — Add detailed description
- \`services/tts.py:14\` — Add detailed description" \
  "todo,services"

create_issue \
  "Implement text_to_gen service-specific logic" \
  "## Context
\`services/meta_generation/text_to_gen_service.py:482\`

Implement the actual service-specific logic for text-to-gen." \
  "todo,services"

# JAX/TPU
create_issue \
  "Implement TPU v4 scan kernel operations" \
  "## Context
\`jax/tpu_v4/scan_kernels.cc\` — 7 pending implementations:

- [ ] Llamada a kernel TPU específico (:130)
- [ ] Propagación de carries (:134)
- [ ] Aplicación final (:137)
- [ ] Operación vectorizada en TPU (:183)
- [ ] Scan de ventana en TPU (:230)
- [ ] Cumulative ops paralelas (:271)
- [ ] Cumulative ops secuenciales (:274)" \
  "todo,jax-tpu"

echo ""
echo "Done! All issues created."
