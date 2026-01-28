# JSON-LD + TOON en capibaraGPT_v3

## Objetivo

Establecer JSON-LD como formato canónico para datos persistentes/estructurados y TOON como vista compacta para consumo por LLMs. Esto permite trazabilidad semántica (JSON-LD) y eficiencia de tokens (TOON) sin perder información.

## Características clave

- **JSON-LD como fuente de verdad**
  - Usa `@context`, `@id`, `@type` para entidades y relaciones.
  - Apto para auditoría, versionado y razonamiento por grafo.
- **TOON como vista para modelos**
  - Representación determinista, compacta y lossless.
  - Reduce tokens y mejora el parsing al exponer estructuras con indentación.
- **Separación de capas**
  - Persistencia y RAG interno mantienen JSON-LD.
  - Entrada a modelos usa TOON derivado.

## Flujo recomendado (alto nivel)

1. **Ingesta** → Normalización JSON-LD (`@id`, `@type`, `@context`).
2. **RAG interno** → Fusión, filtrado y selección sobre JSON-LD.
3. **Encoder** → JSON-LD → TOON antes del prompt.

## Implementación inicial

Se añadió un módulo utilitario para:
- Normalizar documentos/chunks a JSON-LD.
- Construir un contexto JSON-LD para RAG.
- Serializar JSON-LD a TOON siguiendo reglas deterministas.

Revisar:
- `utils/jsonld_toon.py`
- `core/pipelines/rag_pipeline.py` (opción `use_toon`)
