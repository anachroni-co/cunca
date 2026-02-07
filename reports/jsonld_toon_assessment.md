# Informe: viabilidad de JSON-LD/"json-dl" + TOON en capibaraGPT_v3

## Resumen ejecutivo

El proyecto ya cuenta con pipelines RAG y mecanismos de preparación de contexto basados en texto plano y metadatos genéricos. En particular, el RAG avanzado almacena documentos y chunks como diccionarios con claves simples (`text`, `metadata`, `chunks`, `timestamp`) y recupera por relevancia sin un esquema semántico explicit. Asimismo, el integrador RAG prepara prompts concatenando texto plano. Esto implica que **hoy no existe un modelo semántico formal ni un formato canónico** para datos persistentes o de contexto, por lo que introducir JSON-LD ("json-dl") como fuente de verdad y TOON como vista para LLMs sería un cambio de arquitectura relevante. 【F:core/pipelines/advanced_rag_pipeline.py†L130-L223】【F:core/pipelines/rag_pipeline.py†L17-L73】

**Conclusión rápida:**
- **Sí es viable** adoptar JSON-LD como formato canónico y TOON como vista para LLMs, pero **requiere capas nuevas** (normalización, encoder/decoder, contratos de datos y validación). La ganancia en robustez puede ser alta (trazabilidad, auditoría, consistencia), mientras que la mejora en velocidad dependerá de la implementación del encoder y del costo de transformación en tiempo de execution. Hoy no hay evidencia de benchmarking en el repositorio para cuantificar esa mejora, por lo que cualquier estimación de rendimiento debe validarse con pruebas específicas. 【F:core/pipelines/advanced_rag_pipeline.py†L130-L223】【F:core/pipelines/rag_pipeline.py†L17-L73】

---

## Current project state (RAG y preparación de contexto)

### RAG avanzado
- Los documentos se almacenan en un diccionario en memoria (`document_store`) con estructura plana y campos libres (`text`, `metadata`, `chunks`, `timestamp`). No existe un modelo semántico standard ni campos obligatorios equivalentes a `@id` o `@type`. 【F:core/pipelines/advanced_rag_pipeline.py†L130-L223】
- La recuperación opera sobre texto y metadatos (no estructurados semánticamente) y el contexto se construye por concatenación de texto. 【F:core/pipelines/advanced_rag_pipeline.py†L190-L223】

### RAG integrator (pipeline minimal)
- El `RAGIntegrator` espera `documents` con `text` y `score`, y genera un prompt que concatena texto plano. 【F:core/pipelines/rag_pipeline.py†L17-L73】
- No existe en el pipeline un modelo estructurado persistente para los documentos ni un punto de serialización formal (p. ej., encoder JSON-LD → TOON). 【F:core/pipelines/rag_pipeline.py†L17-L73】

---

## Encaje con JSON-LD ("json-dl") + TOON

### Encaje conceptual
Tu especificación propone:
- **JSON-LD** como fuente de verdad (persistencia, grafos, auditoría, versionado).
- **TOON** como vista compacta para LLMs, generada por un encoder determinista y lossless.

En el estado actual, el sistema **no tiene**:
- Un esquema explicit de entidades/relaciones.
- Un mecanismo de normalización semántica de la información recuperada.
- Una capa formal de serialización/normalización para entrada de LLMs.

Por tanto, **el encaje es arquitectónicamente coherente**, pero requiere construir una capa semántica encima del pipeline actual. 【F:core/pipelines/advanced_rag_pipeline.py†L130-L223】【F:core/pipelines/rag_pipeline.py†L17-L73】

---

## Impacto esperado

### Robustez y trazabilidad
**Alta mejora potencial**, porque:
- JSON-LD fuerza un contrato semántico para entidades y relaciones (IDs estables, tipos, contexto).
- Permite auditoría y razonamiento por grafo, útil para RAG explicable.
- Evita ambigüedad en los datos de entrada al modelo (cada entidad anclada a `@id`).

### Velocidad / tokens
**Ganancia potencial, no garantizada**, porque:
- TOON reduce tokens en el prompt (mejor densidad de información) → puede acelerar inferencia al reducir longitud del contexto.
- Pero introduce costo de transformación JSON-LD → TOON en tiempo de execution.
- El impacto real depende del **tamaño del contexto**, el **tiempo de serialización**, y **si el pipeline actual ya está optimizado**.

No hay benchmarks en el repositorio que midan tiempo de serialización o ahorro de tokens, por lo que esto debe validarse con pruebas específicas. 【F:core/pipelines/advanced_rag_pipeline.py†L130-L223】【F:core/pipelines/rag_pipeline.py†L17-L73】

---

## Cambios necesarios para adoptar JSON-LD + TOON

### 1) Contrato de datos y normalización
- Definir un esquema JSON-LD minimal para documentos, chunks, episodios y metrics (incluye `@id`, `@type`, `@context`).
- Mapear los datos actuales (`text`, `metadata`, `chunks`) a entidades explícitas.

### 2) Capa de serialización TOON
- Implementar un encoder JSON-LD → TOON siguiendo reglas deterministas y lossless.
- Enviar TOON solo en la etapa de entrada a modelos (no persistirlo).

### 3) RAG interno en JSON-LD
- Tras la recuperación, normalizar documentos a JSON-LD.
- Aplicar filtrado y fusión sobre JSON-LD antes de serializar a TOON.

### 4) Validación y trazabilidad
- Añadir validadores que aseguren que todo objeto persistente tenga `@id`, `@type` y `@context`.
- Agregar verificaciones de reversibilidad TOON → JSON-LD.

---

## Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Incremento de complejidad en el pipeline | Medio/Alto | Implementar por fases: primero JSON-LD en ingestión, luego encoder TOON, luego RAG completo en JSON-LD. |
| Costo de serialización TOON | Medio | Benchmark dedicado con tamaños de contexto reales; optimizar el encoder y cachear resultados. |
| Coste de migración de datos existentes | Medio | Introducir un “adapter layer” que traduzca los registros actuales a JSON-LD sin cambiar su origen. |

---

## Recomendación

**Recomendado con cautela** si el objetivo principal es **robustez, trazabilidad y consistencia semántica** del RAG. Esto encaja muy bien con el enfoque actual de pipelines y monitoreo en el proyecto. Sin embargo, **no se puede afirmar mejora de velocidad sin benchmarks**. Se sugiere:

1. Prototipo en un pipeline de RAG minimal: ingesta → normalización JSON-LD → encoder TOON → prompt.
   - **Ingestion**: mapear cada documento a JSON-LD con `@id`, `@type` y `@context` (incluyendo `text`, `metadata`, `timestamp`).
   - **Normalización**: garantizar un JSON-LD único por documento y por chunk (si aplica), con IDs estables y referencias entre entidades.
   - **Encoder TOON**: convertir el JSON-LD final siguiendo las reglas deterministas (bloques por indentación, arrays tabulares cuando aplique).
   - **Prompt**: sustituir el contexto concatenado por TOON serializado para el modelo.
2. Benchmark de tokens y latencia (antes vs. después) midiendo:
   - Tokens del contexto (JSON-LD vs. TOON).
   - Tiempo de serialización (JSON-LD → TOON).
   - Latencia total de inferencia.
3. Si el ahorro de tokens es significativo y la serialización no introduce overhead alto, avanzar a integración completa.

---

## Apéndice: relación con componentes actuales

- **RAG avanzado:** Base viable para incorporar JSON-LD en `document_store` y en la salida de `retrieve_documents`. 【F:core/pipelines/advanced_rag_pipeline.py†L130-L223】
- **RAG integrator:** Lugar natural para serializar a TOON antes de generar el prompt. 【F:core/pipelines/rag_pipeline.py†L17-L73】
