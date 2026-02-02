# TODOs - capibaraGPT v3

## Agents
- [ ] Implement real LLM integration (OpenAI, Anthropic, local models, etc.) — `agents/capibara_agent_factory.py:69`
- [ ] Implement real Qdrant vector database connection — `agents/capibara_agent_factory.py:92`

## Core
- [ ] Use capibara's struct implementation (flax) — `core/optimization.py:81`

## Pipeline
- [ ] Implement stage-specific execution — `pipeline/run_pipeline.py:196`

## Config
- [ ] Implement actual config validation logic — `config/__init__.py:61`
- [ ] Implement file-based config validation — `config/__init__.py:66`

## Training
- [ ] Initialize perplexity model (GPT-2 or similar for scoring) — `training/data_preprocessing/quality_filter.py:359`
- [ ] Implement perplexity calculation and filtering — `training/data_preprocessing/quality_filter.py:381`
- [ ] Implement gradient-based importance scoring — `training/data_lineage/dataset_parameter_controller.py:207`
- [ ] Implement actual influence calculation — `training/data_lineage/dataset_parameter_controller.py:215`

## Datasets
- [ ] Implement download and processing (emotional audio) — `data/datasets/multimodal/emotional_audio_datasets.py:342`
- [ ] Implement download with authentication (emotional audio) — `data/datasets/multimodal/emotional_audio_datasets.py:361`

## Sub-models
- [ ] Connect to RAG/Knowledge Base for evidence retrieval — `sub_models/csa_expert.py:332`

## Services
- [ ] Add detailed description (system_info) — `utils/system_info.py:23`
- [ ] Add detailed description (tts) — `services/tts.py:14`
- [ ] Implement service-specific logic (text_to_gen) — `services/meta_generation/text_to_gen_service.py:482`

## JAX / TPU v4 Kernels (`jax/tpu_v4/scan_kernels.cc`)
- [ ] Implementar llamada a kernel TPU específico — `:130`
- [ ] Implementar propagación de carries — `:134`
- [ ] Implementar aplicación final — `:137`
- [ ] Implementar operación vectorizada en TPU — `:183`
- [ ] Implementar scan de ventana en TPU — `:230`
- [ ] Implementar cumulative ops paralelas — `:271`
- [ ] Implementar cumulative ops secuenciales — `:274`
