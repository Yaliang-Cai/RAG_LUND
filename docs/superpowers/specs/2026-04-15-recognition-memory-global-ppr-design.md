# Recognition Memory for Global PPR — Design Spec

**Date:** 2026-04-15  
**Status:** Approved  
**Scope:** `lightrag/lightrag/operate.py`, `lightrag/lightrag/base.py`

---

## 1. Background

The current `mode="ppr"` (GlobalPPREngine) builds entity seeds from two sources:

| Source | Mechanism |
|--------|-----------|
| Entity VDB | Cosine similarity to query keywords (`vdb_score`) |
| Relation VDB | Fact similarity → `src_id` / `tgt_id` extracted with `distance` score |

Both sources are merged by `max()`, hub-penalized, then fed into `engine.run_ppr()`.

**Problem:** Seeds are selected by pure vector similarity. There is no semantic judgment of whether a retrieved entity is actually relevant to answering the query. High-similarity-but-irrelevant entities pollute the PPR personalisation vector, degrading multi-hop retrieval quality.

**HippoRAG2 solution:** Recognition Memory — a 3-step hybrid filter on retrieved triplets before PPR:

1. **Numpy / argsort** — vector retrieval selects top-K candidate facts  
2. **LLM (DSPyFilter)** — model judges which facts are truly query-relevant  
3. **Difflib** — LLM text output remapped back to system indices

---

## 2. Goals

- Add recognition memory to `mode="ppr"` (global PPR only)
- Unified LLM recognition over both entity VDB results and relation VDB triplets
- Preserve the existing passage-node (DPR chunk) seed path — recognition memory is entity-seed-only
- Zero new VDB queries — reuse results already retrieved by hybrid retrieval
- Graceful fallback: if LLM fails or returns empty, continue with original seeds unchanged
- Default seed count `recognition_top_k=10` matching HippoRAG2 `link_top_k` default

---

## 3. Non-Goals

- No changes to `mode="ppr_local"` (local BFS path)
- No changes to `_ppr_rank_chunks_global()` signature or internals
- No new external LLM configuration — uses `global_config["llm_model_func"]`
- No reranking of PPR output chunks (separate concern, tracked in TODO_ppr_improvements.md)

---

## 4. Data Flow

```
_ppr_rank_chunks(use_global=True)
│
├── [EXISTING] entity VDB → raw_vdb_scores {entity_id: score}
│
├── [EXISTING] relation VDB → rel_results [{src_id, description, tgt_id, distance}]
│
├── [NEW] if use_global → _recognition_memory_filter()
│         │
│         ├── normalize separately:
│         │     norm_vdb  = min_max_normalize(raw_vdb_scores)
│         │     norm_fact = min_max_normalize(fact_scores from rel_results)
│         │
│         ├── build unified candidate pool:
│         │     standalone entities  → "EntityName"
│         │     triplet entities     → "Src | description | Tgt"
│         │
│         ├── LLM prompt (exact-string constraint)
│         │     → recognized entity name strings
│         │
│         ├── difflib.get_close_matches(cutoff=0.85)
│         │     → recognized entity_ids
│         │
│         ├── merge: max(norm_vdb[eid], norm_fact[eid]) for recognized eids
│         │
│         └── fallback: if empty → return original entity_seed_weights
│
├── [EXISTING] hub penalty on entity_seed_weights
│
└── [EXISTING] _ppr_rank_chunks_global()
      ├── chunk VDB (DPR) → chunk_seed_weights × passage_node_weight
      └── engine.run_ppr(entity_seed_weights, chunk_seed_weights)
```

---

## 5. New Function: `_recognition_memory_filter()`

**Location:** `operate.py`, defined above `_ppr_rank_chunks()`

```python
async def _recognition_memory_filter(
    query: str,
    node_datas: list[dict],          # entity VDB results (entity_id, vdb_score)
    rel_results: list[dict],         # relation VDB results (src_id, tgt_id, description, distance)
    llm_model_func: Callable,
    recognition_top_k: int = 10,     # max triplets to show LLM; HippoRAG2 default
) -> dict[str, float]:               # entity_id → normalised merged weight
```

### 5.1 Step-by-step

**Step 1 — Separate normalisation**

```python
def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo if hi > lo else 1.0
    return {k: (v - lo) / span for k, v in scores.items()}

norm_vdb  = _min_max_norm({nd["entity_id"]: nd["vdb_score"] for nd in node_datas if nd.get("entity_id")})
norm_fact = _min_max_norm({...})   # see Step 2
```

**Step 2 — Candidate pool construction**

Take top `recognition_top_k` relation results (already sorted by distance desc from VDB).

```
Standalone entity section:
  "{entity_id}"           (one per line, from node_datas)

Triplet section:
  "{src_id} | {description} | {tgt_id}"   (one per line, from rel_results[:recognition_top_k])
```

Fact score mapping: `{src_id: distance, tgt_id: distance}` for each triplet.

**Step 3 — LLM prompt**

```
You are an entity relevance judge.

Query: {query}

Below are retrieved entities and facts. Select ONLY those directly relevant to answering the query.
You MUST copy entity names EXACTLY as they appear in the list. Do not rephrase, abbreviate, or invent new entities.

Standalone entities:
{entity list, one per line}

Retrieved facts:
{triplet list, one per line}

Return the relevant entity names only, one per line. If none are relevant, return an empty response.
```

**Step 4 — Parse + Difflib mapping**

```python
import difflib

all_candidate_ids = list(norm_vdb.keys()) + [src/tgt from rel_results[:recognition_top_k]]
all_candidate_ids_dedup = list(dict.fromkeys(all_candidate_ids))

recognized_ids = set()
for line in llm_output.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    matches = difflib.get_close_matches(line, all_candidate_ids_dedup, n=1, cutoff=0.85)
    if matches:
        recognized_ids.add(matches[0])
```

**Step 5 — Merge weights**

```python
result = {}
for eid in recognized_ids:
    w = max(norm_vdb.get(eid, 0.0), norm_fact.get(eid, 0.0))
    if w > 0.0:
        result[eid] = w
return result   # empty dict triggers fallback in caller
```

---

## 6. Changes to `_ppr_rank_chunks()`

### 6.1 Restructure seed-building block (use_global path)

**Current code (~line 5592–5616):**
```python
# From entity VDB scores
for nd in node_datas:
    entity_seed_weights[eid] = max(...)

# From relation VDB scores
rel_results = await relationships_vdb.query(...)
for rel in rel_results:
    entity_seed_weights[eid] = max(...)
```

**New code:**
```python
# Always retrieve relation VDB results (used by both paths)
rel_results = []
try:
    rel_results = await relationships_vdb.query(
        query, top_k=query_param.top_k, query_embedding=query_embedding
    )
except Exception as e:
    logger.warning(f"PPR: relation VDB query failed: {e}")

if use_global:
    # --- Recognition Memory path ---
    llm_func = global_config.get("llm_model_func")
    if llm_func and (node_datas or rel_results):
        recognized = await _recognition_memory_filter(
            query=query,
            node_datas=node_datas,
            rel_results=rel_results,
            llm_model_func=llm_func,
            recognition_top_k=query_param.recognition_top_k,
        )
        if recognized:
            entity_seed_weights = recognized
        else:
            # Fallback: build seeds the old way
            _build_seeds_from_raw(node_datas, rel_results, entity_seed_weights)
    else:
        _build_seeds_from_raw(node_datas, rel_results, entity_seed_weights)
else:
    # Local PPR path: existing direct extraction (unchanged)
    _build_seeds_from_raw(node_datas, rel_results, entity_seed_weights)
```

Where `_build_seeds_from_raw` is an inline helper (or just the existing 8-line loop inlined) that does the current max-merge without recognition.

### 6.2 `_ppr_rank_chunks_global()` call site

Add `global_config` forwarding — the function already uses `text_chunks_db.global_config` internally, so no signature change needed. The call site in `_ppr_rank_chunks` is unchanged.

---

## 7. QueryParam Addition

**File:** `lightrag/lightrag/base.py`

```python
recognition_top_k: int = 10
"""HippoRAG2 Recognition Memory: max triplet candidates shown to LLM for entity seed filtering.
Candidate pool = rel_results[:recognition_top_k] from relation VDB.
Only active when mode="ppr". Default 10 matches HippoRAG2 link_top_k.
Set to 0 to disable recognition memory (falls back to direct score merge)."""
```

---

## 8. Interaction with Existing Mechanisms

| Mechanism | Interaction |
|-----------|-------------|
| Hub penalty | Applied **after** recognition, to the filtered `entity_seed_weights`. Still useful as structural safety net (complements semantic recognition). |
| Passage node weights (DPR chunks) | Completely unaffected — handled inside `_ppr_rank_chunks_global` independently. |
| `query_param.top_k` | Controls relation VDB retrieval size (candidate pool for recognition). |
| `recognition_top_k=0` | Disables recognition memory; falls back to original direct score merge. |
| LLM failure / empty output | Silent fallback to original `entity_seed_weights` computed via `_build_seeds_from_raw`. |

---

## 9. Engineering Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM hallucinates entity names | Prompt constraint: "copy exactly as listed"; difflib cutoff=0.85 rejects near-misses |
| Ambiguous difflib mapping (e.g. "Apple" → "Apple Inc" vs "Apple Fruit") | cutoff=0.85 is strict; ties broken by first match; failure is safe (entity excluded) |
| `vdb_score` and `distance` on different scales | Separate min-max normalisation before merge |
| LLM latency added to every PPR query | Cached via `llm_model_func` call-level caching; `recognition_top_k=0` opt-out |
| Passage seeds lost | Not affected — DPR chunk path in `_ppr_rank_chunks_global` is independent |

---

## 10. Files Changed

| File | Change |
|------|--------|
| `lightrag/lightrag/operate.py` | New `_recognition_memory_filter()` (~60 lines); refactor seed-building block in `_ppr_rank_chunks()` (~20 lines) |
| `lightrag/lightrag/base.py` | `QueryParam.recognition_top_k: int = 10` |

---

## 11. Out of Scope (Future)

- DSPy-style few-shot examples in the recognition prompt
- Caching recognition results at query-hash level
- Adaptive `recognition_top_k` based on graph size
