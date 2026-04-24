# Retrieval Router Design

**Date:** 2026-04-24  
**Status:** Approved  
**Scope:** `rag-anything/raganything/retrieval/`

---

## 1. Problem Statement

The system currently exposes multiple retrieval modes (`naive`, `hybrid`, `mix`, `ppr`, `qdrant_hybrid`) as separate `mode=` values. Users must choose manually, and ablation experiments require explicit per-group configuration. There is no unified path that:

1. Automatically selects the best retrieval strategy for a given query
2. Supports multi-path parallel recall with RRF fusion
3. Provides fine-grained ablation control via named configuration profiles

This design introduces a `RetrievalRouter` that satisfies all three requirements under a new `mode="auto"` entry point, while leaving all existing modes untouched.

---

## 2. Goals

- Add `mode="auto"` to `aquery()` that routes queries to the optimal retrieval profile
- Support explicit `profile=` override to bypass LLM classification (for ablation/evaluation)
- Parallel multi-path recall with RRF → reranker → threshold fusion
- Full observability via `latency_per_path` and per-stage chunk counts in the trace
- Zero breaking changes to existing modes

---

## 3. Architecture

### 3.1 New Module Structure

```
rag-anything/raganything/retrieval/
├── __init__.py
├── profiles.py        # RetrievalProfile dataclass + built-in profile registry
├── classifier.py      # LLM query classifier → profile name
├── router.py          # RetrievalRouter main class
└── paths.py           # Per-path async execution functions (thin wrappers)
```

### 3.2 Call Chain

```
aquery(query, mode="auto", ...)
    └── RetrievalRouter.route(query, param)
            ├── classifier.classify(query)          → profile_name
            ├── profiles.get(profile_name)           → Profile
            ├── paths.run_parallel(profile.paths)    → {path: [chunks]}
            ├── rrf_merge(chunks_by_path, weights)   → merged_chunks
            ├── reranker.rerank(merged_chunks)        → scored_chunks
            └── threshold_filter(scored_chunks)       → final_chunks
```

### 3.3 Integration Boundary

- `aquery()` adds a single `mode="auto"` branch that delegates to `RetrievalRouter`; all other modes unchanged
- `RetrievalRouter` accepts the existing `QueryParam` + `LightRAG` instance; holds no storage references of its own
- RRF merge and reranker reuse LightRAG's existing `_rrf_merge` and rerank infrastructure
- When `profile=` is passed explicitly, the LLM classifier call is skipped entirely

### 3.4 VLM Dereference Integration

`RetrievalRouter.route()` returns `list[Chunk]` — it does **not** render a prompt string. The VLM image-path dereference pipeline (`_process_image_paths_for_vlm`) operates on a rendered prompt string and must therefore run after prompt assembly. The `mode="auto"` call path for VLM-enhanced queries is:

```
aquery_vlm_enhanced(query, mode="auto", ...)
    └── RetrievalRouter.route(query, param)      → list[Chunk]
    └── lightrag.build_prompt(query, chunks)     → raw_prompt string
    └── _process_image_paths_for_vlm(raw_prompt) → enhanced_prompt, images_base64
    └── _build_vlm_messages_with_images(...)     → messages
    └── _call_vlm_with_multimodal_content(...)   → answer
```

This keeps the router's responsibility strictly retrieval + fusion. Prompt rendering and image dereference remain in `aquery_vlm_enhanced()`, unchanged from the non-auto path. The only new step is calling `lightrag.build_prompt()` with the router's chunk output instead of invoking `lightrag.aquery_llm(only_need_prompt=True)`.

---

## 4. Profile System

### 4.1 Data Class

```python
@dataclass
class RetrievalProfile:
    name: str
    description: str                          # used in classifier prompt
    paths: list[str]                          # retrieval path names to activate
    rrf_weights: dict[str, float]             # path → RRF weight (default 1.0)
    rrf_k: int = 60
    enable_rerank: bool = True
    min_rerank_score: float = 0.3
    rerank_candidate_cap: int = 60            # top-N fed to reranker after RRF
    path_overrides: dict[str, dict] = field(default_factory=dict)
```

### 4.2 Built-in Profile Registry

| Profile | Paths | Key path_overrides | Query Type |
|---|---|---|---|
| `precise` | `qdrant_sparse` | `qdrant_retrieval_mode="bm25"` | Error codes, CVE IDs, rare proper nouns |
| `local` | `hybrid`, `naive` | — | Specific entity or clear single-hop fact; hybrid already subsumes VDB retrieval |
| `multihop` | `ppr`, `hybrid` | — | Cross-entity chain reasoning |
| `descriptive` | `mix`, `qdrant_hybrid` | `kg_chunk_selection_source="untruncated"`, `answer_context_mode="kg_prompt"` | Open-ended description, broad coverage |
| `full` | `naive`, `hybrid`, `mix`, `ppr`, `qdrant_hybrid`, `qdrant_sparse` | — | Fallback / evaluation / low-confidence |

**Rationale for merging `factual` + `graph_hybrid` → `local`:** Most single-hop facts are graph triples (entity–attribute–value); `hybrid` mode already calls the VDB path internally, so `factual` (naive-only) is a strict subset. The classifier boundary between the two was unstable in zero-shot settings. `local` unifies them under a clearer semantic: "direct query targeting a specific entity or fact."

**Future iteration:** `descriptive` profile will gain a `summary_node_boost` parameter that up-weights Summary-type graph nodes during PPR seed construction, improving coverage for document-level descriptive queries. Not in scope for this release.

### 4.3 Ablation Integration

Existing experiment groups (e.g., `ppr_hybrid_per_keyword`) can be registered as named profiles and selected via `profile=` with no LLM classification cost. Evaluation scripts require no changes beyond substituting `mode="auto", profile="<name>"`, or continuing to use legacy `mode=` parameters (fully backward-compatible).

`profile` is passed as a `**kwargs` key to `aquery()` and forwarded to `RetrievalRouter`. It is **not** added to `QueryParam` to avoid polluting the existing LightRAG interface. `RetrievalRouter` pops it before constructing any downstream `QueryParam` copies.

---

## 5. LLM Classifier

### 5.1 Classification Flow

```
query
  → classify_prompt (profiles with descriptions + few-shot examples)
  → LLM output: {"reasoning": "...", "profile": "multihop", "confidence": 0.85}
  → confidence < 0.6 OR unknown profile → fallback to "full"
  → return profile_name
```

### 5.2 Prompt Template (English)

```
You are a retrieval routing classifier. Given a user query, select the most
appropriate retrieval profile from the list below.

Available profiles and typical examples:

- precise: Exact character-level match (error codes, IDs, rare proper nouns)
  Examples: "What is the impact scope of CVE-2026-001?"
            "Status of order ID ORD-20260424-8821"

- local: Direct query targeting a specific entity or clear single-hop fact
  Examples: "How many parameters does BERT have?"
            "What are the architectural differences between BERT and GPT?"
            "When should you use RAG vs fine-tuning?"

- multihop: Chain reasoning across multiple entities or documents
  Examples: "What other papers have been published by the authors cited in HippoRAG2?"
            "Which components of LightRAG were influenced by HippoRAG2?"

- descriptive: Open-ended question requiring broad, complete context
  Examples: "Describe the overall architecture of LightRAG."
            "Provide a survey of PPR algorithms used in RAG systems."

- full: Fallback when query type is unclear or ambiguous

First briefly state your reasoning in one sentence, then output JSON.
Output format: {"reasoning": "...", "profile": "<name>", "confidence": <0.0-1.0>}

Query: {query}
```

### 5.3 Structured Output Guarantee

```python
async def classify(query: str) -> str:
    try:
        raw = await llm_func(prompt, response_format={"type": "json_object"})
        result = json.loads(raw)
        profile = result.get("profile", "full")
        confidence = float(result.get("confidence", 0.0))
        if confidence < 0.6 or profile not in PROFILE_REGISTRY:
            return "full"
        return profile
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Classifier output parse failed, fallback to 'full'")
        return "full"
```

- Uses `response_format={"type": "json_object"}` when the backend supports it (OpenAI / vLLM compatible)
- `try-except` catches all parse failures regardless of model support
- All failure paths return `"full"`; classifier never raises

### 5.4 Caching

Classifier results are cached via LightRAG's existing `llm_response_cache` keyed on query text. Identical queries skip the LLM call entirely.

---

## 6. Retrieval Path Execution Layer

### 6.1 Path Interface

```python
async def run_path(
    name: str,
    query: str,
    param: QueryParam,
    lightrag: LightRAG,
    overrides: dict,
) -> list[Chunk]:
    ...
```

All paths return a standardized `list[Chunk]`. `overrides` are merged into a copy of `param`; the original is never mutated.

### 6.2 Path Mapping

| Path Name | Implementation | Notes |
|---|---|---|
| `naive` | `lightrag.aquery(mode="naive")` | Qdrant dense VDB |
| `hybrid` | `lightrag.aquery(mode="hybrid")` | KG entity/relation + dense VDB |
| `mix` | `lightrag.aquery(mode="mix")` | hybrid + naive combined |
| `ppr` | `lightrag.aquery(mode="ppr")` | Global PPR + recognition memory |
| `qdrant_hybrid` | `lightrag.aquery(mode="hybrid")` + `qdrant_retrieval_mode="hybrid"` | Dense + BM25 dual-vector |
| `qdrant_sparse` | `lightrag.aquery(mode="naive")` + `qdrant_retrieval_mode="bm25"` | Pure BM25 sparse retrieval |

### 6.3 Parallel Execution

```python
results = await asyncio.gather(
    *[run_path(name, query, param, lightrag, profile.path_overrides.get(name, {}))
      for name in profile.paths],
    return_exceptions=True,
)
# Failed paths are logged as warnings and excluded from fusion; other paths proceed normally
```

### 6.4 Fusion Pipeline

```
chunks from all paths, each as an independent ranked list
    → _rrf_merge(ranked_lists_by_path, weights=profile.rrf_weights, k=profile.rrf_k)
         # RRF scores each chunk: Score(d) = Σ_p  weight_p × 1/(k + rank(d,p))
         # chunks appearing in multiple paths accumulate higher scores (rank consensus)
         # output is a single deduplicated list sorted by RRF score
    → reranker.rerank(rrf_output[:rerank_candidate_cap])
    → filter(score >= profile.min_rerank_score)
    → final[:chunk_top_k]
```

**Critical ordering constraint:** Deduplication must NOT happen before `_rrf_merge`. Each path's ranked list must be passed intact so that a chunk appearing in multiple paths accumulates its cross-path rank signals. Pre-deduplication would discard these signals and reduce RRF to a simple rank-boost, losing its statistical robustness. `_rrf_merge` produces a deduplicated, score-sorted output as a natural consequence of the scoring formula.

`rerank_candidate_cap` is defined per-Profile (default 60). A failed path contributes an empty list to RRF, which naturally degrades without breaking the pipeline.

---

## 7. Error Handling

| Scenario | Handling |
|---|---|
| Classifier timeout / parse failure | Return `"full"` profile, log warning |
| Single path raises exception | Exclude from fusion via `return_exceptions=True`, log warning |
| All paths fail | Raise `RetrievalError`, propagated to `aquery()` caller |
| Unknown profile name | `ConfigError` raised at startup during registry validation |
| Reranker OOM | Reuse existing `rerank_batch_backoff` mechanism |

---

## 8. Observability

The `routing` key is added to the existing `return_trace=True` trace structure:

```json
"routing": {
  "profile": "multihop",
  "confidence": 0.87,
  "reasoning": "Query requires cross-entity chain reasoning across multiple documents",
  "paths_activated": ["ppr", "hybrid"],
  "paths_failed": [],
  "chunks_per_path": {"ppr": 18, "hybrid": 24},
  "chunks_after_rrf": 31,
  "chunks_after_rerank": 12,
  "chunks_after_threshold": 9,
  "latency_per_path": {
    "classifier": 0.35,
    "ppr": 1.25,
    "hybrid": 0.08
  }
}
```

`latency_per_path` includes the classifier as a named key, giving a complete latency breakdown in a single dict.

---

## 9. Testing Strategy

- **Unit tests** (`classifier.py`): Mock LLM to cover all parse failure paths — missing fields, non-JSON output, `confidence < 0.6`, unknown profile name
- **Integration tests** (`router.py`): Use existing ablation workspaces with explicit `profile=` (no classifier), verify chunk overlap with single-mode baseline ≥ 80%
- **Registry validation test**: At startup, assert all built-in profile path names exist in the known path registry

---

## 10. Out of Scope

- `summary_node_boost` for `descriptive` profile (future iteration)
- Changes to any existing `mode=` behavior
- New Qdrant collection schema (assumes `DEFAULT_QDRANT_ENABLE_SPARSE_BM25=True` collections already exist for `qdrant_sparse` path)
- RRF v2 sparse/HyDE paths (tracked separately in `docs/TODO_rrf_v2_upgrade.md`)
