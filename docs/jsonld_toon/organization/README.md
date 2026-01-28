# Organización y ahorro esperado (JSON-LD + TOON)

## Organización de datos

- **Documento JSON-LD (canónico)**
  - Campos mínimos: `@context`, `@id`, `@type`.
  - Propiedades del documento (`text`, `metadata`, `timestamp`).
  - Chunks como entidades separadas con relación `parent`.
- **Contexto RAG JSON-LD**
  - Contiene `query`, `context` y colecciones de `documents` y `chunks`.
  - IDs estables para trazabilidad entre etapas.
- **Vista TOON para LLMs**
  - Serialización determinista con bloques y arrays tabulares.
  - `@id` y `@type` siempre visibles.

## Ahorro y métricas sugeridas

Para validar ahorro y robustez:

- **Tokens del contexto**
  - Comparar JSON-LD vs. TOON para el mismo contenido.
- **Costo de serialización**
  - Medir tiempo de `JSON-LD → TOON` por tamaño de contexto.
- **Latencia total**
  - Medir tiempo end-to-end (recuperación + serialización + inferencia).
- **Robustez de parsing**
  - Verificar reconstrucción lossless TOON → JSON-LD.

## Recomendaciones de uso

- Persistencia: almacenar siempre JSON-LD.
- RAG interno: operar y fusionar JSON-LD.
- LLM: enviar solo TOON.
