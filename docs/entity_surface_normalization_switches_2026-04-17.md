# Entity Surface Normalization And Strict Relation Endpoint Switches (2026-04-17)

## Scope
- Runtime: `LocalRagService` (`Neo4j + Qdrant` path).
- Ingest/query path: entity/relation extraction, relation merge/rebuild, query keyword post-processing.
- Goal: all new behavior is switch-gated and orthogonal. Turning all new switches off falls back to previous behavior.

## New Switches

### 1) `enable_entity_surface_normalization`
- Location:
  - `raganything.constants.DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION`
  - `lightrag.constants.DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION`
- Default: `True`
- Effect when `True`:
  - Applies only during extraction parsing.
  - Order: sanitize/filter first, then surface normalization.
  - Applies word-by-word canonical casing:
    - allowlist/acronym tokens -> uppercase
    - words with meaningful internal capitals (e.g., `OpenAI`, `iPhone`) -> preserved
    - remaining case-insensitive words -> title-style casing
  - Example: `Machine learning` -> `Machine Learning`.
  - Extraction prompt injects normalization-specific casing rule and extra examples.
    When disabled, these prompt additions are not injected.

### 2) `enable_keyword_case_normalization`
- Location:
  - `raganything.constants.DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION`
  - `lightrag.constants.DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION`
- Default: `True`
- Effect when `True`:
  - Query `high_level_keywords`: lowercase by default, while preserving meaningful mixed/uppercase proper nouns and acronyms.
  - Query `low_level_keywords`: normalized with the same entity-surface normalization function.
  - Relation keyword merge: case-insensitive dedupe + normalized output to avoid duplicates like `Retrieval` / `retrieval`.
  - `high_level_keywords` and relation `keywords` now share the same normalization function, so query/relation matching uses the same casing semantics.

### 3) `entity_uppercase_allowlist`
- Location:
  - `raganything.constants.DEFAULT_ENTITY_UPPERCASE_ALLOWLIST`
  - `lightrag.constants.DEFAULT_ENTITY_UPPERCASE_ALLOWLIST`
- Default: predefined acronym list (`LLM`, `RAG`, `API`, `BERT`, `6G`, etc.)
- Effect:
  - Used only when `enable_entity_surface_normalization=True`.
  - For lowercase-only names, tokens in allowlist are uppercased before title-casing.
  - Example: `llm application` -> `LLM Application`.

### 4) `strict_relation_endpoint_entity_match`
- Location:
  - `raganything.constants.DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
  - `lightrag.constants.DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
- Default: `True`
- Effect when `True`:
  - Relation write is skipped if either endpoint does not exist as a graph node.
  - Relation rebuild is also skipped under the same condition.
  - Prevents fallback creation of `UNKNOWN` endpoint nodes.

## LocalRagService Environment Variables
- `RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION`
- `RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION`
- `RAGANYTHING_ENTITY_UPPERCASE_ALLOWLIST`
  - Accepts CSV (`LLM,RAG,API,6G`) or JSON list (`["LLM","RAG","API","6G"]`).
- `RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`

## LightRAG Environment Variables
- `ENABLE_ENTITY_SURFACE_NORMALIZATION`
- `ENABLE_KEYWORD_CASE_NORMALIZATION`
- `ENTITY_UPPERCASE_ALLOWLIST` (JSON list preferred)
- `STRICT_RELATION_ENDPOINT_ENTITY_MATCH`

## Orthogonality / Rollback Matrix
- `enable_entity_surface_normalization=False`
  - No case normalization is applied in extraction parsing.
- `enable_keyword_case_normalization=False`
  - Query/relation keyword casing stays as extracted without normalization.
- `strict_relation_endpoint_entity_match=False`
  - Keeps previous fallback behavior (missing relation endpoints may be created as `UNKNOWN`).
- All new switches off:
  - New behavior is effectively disabled, matching previous runtime behavior.
