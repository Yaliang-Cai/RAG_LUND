# Entity Surface Normalization And Strict Relation Endpoint Switches (2026-04-17)

## Scope
- Runtime: `LocalRagService` (`Neo4j + Qdrant` path).
- Ingest path: entity/relation extraction and relation merge/rebuild.
- Goal: all new behavior is switch-gated and orthogonal. Turning all new switches off falls back to previous behavior.

## New Switches

### 1) `enable_entity_surface_normalization`
- Location:
  - `raganything.constants.DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION`
  - `lightrag.constants.DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION`
- Default: `False`
- Effect when `True`:
  - Applies only during extraction parsing.
  - Order: sanitize/filter first, then surface normalization.
  - Only all-lowercase names are normalized.
  - Existing uppercase/mixed-case names are preserved.
  - Extraction prompt injects normalization-specific casing rule and extra examples.
    When disabled, these prompt additions are not injected.

### 2) `entity_uppercase_allowlist`
- Location:
  - `raganything.constants.DEFAULT_ENTITY_UPPERCASE_ALLOWLIST`
  - `lightrag.constants.DEFAULT_ENTITY_UPPERCASE_ALLOWLIST`
- Default: predefined acronym list (`LLM`, `RAG`, `API`, `BERT`, `6G`, etc.)
- Effect:
  - Used only when `enable_entity_surface_normalization=True`.
  - For lowercase-only names, tokens in allowlist are uppercased before title-casing.
  - Example: `llm application` -> `LLM Application`.

### 3) `strict_relation_endpoint_entity_match`
- Location:
  - `raganything.constants.DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
  - `lightrag.constants.DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
- Default: `False`
- Effect when `True`:
  - Relation write is skipped if either endpoint does not exist as a graph node.
  - Relation rebuild is also skipped under the same condition.
  - Prevents fallback creation of `UNKNOWN` endpoint nodes.

## LocalRagService Environment Variables
- `RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION`
- `RAGANYTHING_ENTITY_UPPERCASE_ALLOWLIST`
  - Accepts CSV (`LLM,RAG,API,6G`) or JSON list (`["LLM","RAG","API","6G"]`).
- `RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`

## LightRAG Environment Variables
- `ENABLE_ENTITY_SURFACE_NORMALIZATION`
- `ENTITY_UPPERCASE_ALLOWLIST` (JSON list preferred)
- `STRICT_RELATION_ENDPOINT_ENTITY_MATCH`

## Orthogonality / Rollback Matrix
- `enable_entity_surface_normalization=False`
  - No case normalization is applied in extraction parsing.
- `strict_relation_endpoint_entity_match=False`
  - Keeps previous fallback behavior (missing relation endpoints may be created as `UNKNOWN`).
- Both off:
  - New behavior is effectively disabled, matching previous runtime behavior.
