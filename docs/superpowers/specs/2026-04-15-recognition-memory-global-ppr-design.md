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

**Step 1 — Candidate pool sizing**

```python
# Relation triplets: cap at recognition_top_k (controls LLM context size)
top_rels = rel_results[:recognition_top_k]

# Entity VDB candidates: cap at recognition_top_k * 2 to keep prompt manageable
# (entity count is governed by query_param.top_k; recognition_top_k only limits
#  how many are sent to the LLM)
top_nodes = node_datas[:recognition_top_k * 2]
```

> **Note on `recognition_top_k` scope:** This parameter controls only the number of
> relation triplets presented to the LLM. The entity VDB candidate count is governed
> by `query_param.top_k`; `recognition_top_k * 2` is an internal prompt-size cap,
> not a retrieval limit.

**Step 2 — fact_scores construction (explicit max-merge)**

When the same entity appears as src/tgt in multiple triplets, take `max` across all occurrences:

```python
fact_scores: dict[str, float] = {}
for rel in top_rels:
    for eid in (rel.get("src_id"), rel.get("tgt_id")):
        if eid:
            fact_scores[eid] = max(fact_scores.get(eid, 0.0), rel.get("distance", 0.0))
```

**Step 3 — Separate normalisation**

```python
def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    """Normalise to [0, 1]. If all scores are equal and > 0, return uniform 1.0
    to preserve the seed signal rather than collapsing everything to 0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        # All scores identical: uniform weight 1.0 if signal exists, else 0.0
        uniform = 1.0 if hi > 0.0 else 0.0
        return {k: uniform for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}

norm_vdb  = _min_max_norm({
    nd["entity_id"]: nd.get("vdb_score", 0.0)
    for nd in top_nodes if nd.get("entity_id")
})
norm_fact = _min_max_norm(fact_scores)
```

**Step 4 — Candidate pool construction**

`entity_id` in this system is the entity name string (e.g. `"苹果|ORGANIZATION"` when
entity disambiguation is active). Pass entity_ids as-is — the LLM sees the raw id and
is instructed to copy it exactly, including any `|TYPE` suffix.

```
Standalone entities:
  "{entity_id}"    (one per line, from top_nodes — at most recognition_top_k * 2 entries)

Retrieved facts:
  "{src_id} | {description} | {tgt_id}"   (one per line, from top_rels)
```

Build the difflib candidate list from the same entity_ids used in the prompt:

```python
import difflib

entity_vdb_ids = [nd["entity_id"] for nd in top_nodes if nd.get("entity_id")]
triplet_eids   = list(dict.fromkeys(
    eid for rel in top_rels
    for eid in (rel.get("src_id"), rel.get("tgt_id")) if eid
))
all_candidate_ids = list(dict.fromkeys(entity_vdb_ids + triplet_eids))
```

**Step 5 — LLM prompt**

```
You are an entity relevance judge.

Query: {query}

Below are retrieved entities and facts. Select ONLY those directly relevant to answering the query.
You MUST copy each entity identifier EXACTLY as it appears in the list below, including any
"|TYPE" suffixes and special characters. Do not rephrase, abbreviate, or invent new identifiers.

Standalone entities:
{entity_id, one per line}

Retrieved facts:
{src_id | description | tgt_id, one per line}

Return the relevant entity identifiers only, one per line. If none are relevant, return an empty response.
```

**Step 6 — Parse + Difflib mapping**

```python
recognized_ids = set()
for line in llm_output.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    matches = difflib.get_close_matches(line, all_candidate_ids, n=1, cutoff=0.85)
    if matches:
        recognized_ids.add(matches[0])
    # No match → silently skip (never hallucinate an entity into seeds)
```

**Step 7 — Merge weights**

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

if use_global and query_param.recognition_top_k > 0:
    # --- Recognition Memory path ---
    llm_func = global_config.get("llm_model_func")
    if llm_func and (node_datas or rel_results):
        try:
            recognized = await _recognition_memory_filter(
                query=query,
                node_datas=node_datas,
                rel_results=rel_results,
                llm_model_func=llm_func,
                recognition_top_k=query_param.recognition_top_k,
            )
        except Exception as e:
            logger.warning(f"PPR(global): recognition memory failed, falling back: {e}")
            recognized = {}
        if recognized:
            entity_seed_weights = recognized
        else:
            # Empty result (LLM returned nothing useful) — fall back to direct merge
            logger.warning("PPR(global): recognition memory returned empty; using direct seed merge")
            _build_seeds_from_raw(node_datas, rel_results, entity_seed_weights)
    else:
        _build_seeds_from_raw(node_datas, rel_results, entity_seed_weights)
else:
    # Local PPR path OR recognition_top_k=0 (disabled): existing direct extraction
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
"""HippoRAG2 Recognition Memory: controls how many relation triplets are shown to the LLM.
- Relation triplets sent to LLM: rel_results[:recognition_top_k]
- Entity VDB candidates sent to LLM: node_datas[:recognition_top_k * 2]
  (entity VDB retrieval size is still governed by query_param.top_k)
- Only active when mode="ppr". Default 10 matches HippoRAG2 link_top_k.
- Set to 0 to disable recognition memory (falls back to direct score merge)."""
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
| `lightrag/lightrag/operate.py` | New `_recognition_memory_filter()` (~70 lines); refactor seed-building block in `_ppr_rank_chunks()` (~25 lines) |
| `lightrag/lightrag/base.py` | `QueryParam.recognition_top_k: int = 10` |
| `tests/test_recognition_memory.py` | Mock-LLM unit tests: difflib mapping, fallback path, hi==lo normalisation edge case, recognition_top_k=0 opt-out |

---

## 11. Out of Scope (Future)

- DSPy-style few-shot examples in the recognition prompt
- Caching recognition results at query-hash level
- Adaptive `recognition_top_k` based on graph size
