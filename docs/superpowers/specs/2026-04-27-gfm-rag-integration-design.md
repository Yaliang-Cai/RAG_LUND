# GFM-RAG Integration Design

**Date:** 2026-04-27  
**Scope:** Add GFM-RAG as a new retrieval path in RAGAnything's RetrievalRouter, and as a standalone query mode.

---

## 1. Goals

- `aquery(mode="gfm")` — standalone multi-hop retrieval via GFM graph neural network
- `"gfm"` path available in `RetrievalRouter` profiles for RRF fusion with `ppr`, `hybrid`, etc.
- New `"gfm_multihop"` profile in `PROFILE_REGISTRY`; GFM path commented out by default, enabling manual A/B testing against PPR
- Zero-translation chunk ID alignment: GFM document node `name` == LightRAG `chunk_id`

**Out of scope (Phase 2):** Automatic incremental graph sync on document insert.

---

## 2. Architecture Overview

```
Neo4j (entities + relations)          LightRAG KV Store (chunk_id → content)
        │                                            │
        └──────── scripts/export_lightrag_to_gfm.py ┘
                               │
               ./data/<graph_name>/processed/stage1/
                  nodes.csv / edges.csv / relations.csv
               ./data/<graph_name>/raw/
                  documents.json
                               │
                  GFMRetriever (lazy singleton)
                  raganything/retrieval/gfm_retriever.py
                               │
               retrieval/paths.py  ── special-case "gfm" branch
                               │
                    RetrievalRouter (mode="auto")
                               │
                  RRF merge → rerank → final chunks
```

---

## 3. Graph Export Format

**Target directory:** `./data/<graph_name>/processed/stage1/`  
GFMRetriever auto-detects this path from `data_dir` + `data_name`.

### nodes.csv

| Column | Value |
|---|---|
| `name` | `chunk_id` for document nodes; `"entity_" + entity_name` for entity nodes |
| `type` | `"document"` or `"entity"` |
| `attributes` | JSON string, e.g. `{"content": "...chunk text..."}` for documents |

```
name,type,attributes
chunk_abc123,document,"{""content"": ""France is a country in Western Europe...""}"
entity_France,entity,"{""description"": ""France is a country...""}"
```

### edges.csv

| Column | Value |
|---|---|
| `source` | node `name` |
| `relation` | relation type string |
| `target` | node `name` |
| `attributes` | JSON string (empty `{}` is valid) |

```
source,relation,target,attributes
entity_France,mentioned_in,chunk_abc123,"{}"
entity_France,capital_of,entity_Paris,"{}"
```

### relations.csv

| Column | Value |
|---|---|
| `name` | relation type string |
| `attributes` | JSON string |

```
name,attributes
mentioned_in,"{}"
capital_of,"{}"
```

---

## 4. Export Script

**File:** `scripts/export_lightrag_to_gfm.py`

**Inputs:**
- LightRAG working directory (reads `text_chunks` KV store for chunk_id → content)
- Neo4j connection (reads entity nodes and relations)
- `--data-dir` (default `./data`)
- `--graph-name` (e.g. the LightRAG workspace name)

**Steps:**
1. Enumerate all chunks from LightRAG KV store → emit as document nodes
2. Enumerate Neo4j entity nodes → emit as entity nodes
3. For each entity, query Neo4j for which chunk IDs it appears in → emit `mentioned_in` edges
4. Enumerate Neo4j relations → emit entity→entity edges and relation types
5. Write the three CSV files with correct headers to `./data/<graph_name>/processed/stage1/`
6. Write `./data/<graph_name>/raw/documents.json` — flat mapping `{"chunk_id": "chunk content", ...}` for all chunks; prevents errors from GFM-RAG downstream APIs that probe the raw directory

**Key invariant:** document node `name` in `nodes.csv` must exactly match the `chunk_id` key in LightRAG's KV store. No ID translation layer is needed downstream.

---

## 5. GFMRetriever Wrapper

**File:** `raganything/retrieval/gfm_retriever.py`

### Initialization (lazy singleton)

```python
GFMRetrieverWrapper.get_instance(
    data_dir: str,       # e.g. "./data"
    data_name: str,      # e.g. "My_Graph_Name"
    model_path: str,     # e.g. "rmanluo/G-reasoner-34M"
) -> GFMRetrieverWrapper
```

Calls `GFMRetriever.from_index(data_dir, data_name, model_path)` once; subsequent calls return the cached instance.

### Retrieve

```python
async def retrieve(
    query: str,
    top_k: int,
    text_chunks_kv,      # LightRAG text_chunks KV store
) -> list[dict]
```

1. Call `self._retriever.retrieve(query, top_k)` → `{"document": [{"id": chunk_id, "score": float}, ...]}`
2. For each result, fetch `content = await text_chunks_kv.get_by_id(chunk_id)`
3. Return `[{"chunk_id": chunk_id, "content": content, "score": score}, ...]`

Zero translation: the `id` returned by GFM-RAG is the `chunk_id` used directly as KV key.

---

## 6. paths.py Changes

**`KNOWN_PATHS`:** add `"gfm"`.

**`_PATH_CONFIG`:** `"gfm"` is NOT added here — it does not call `lightrag.aquery_data`.

**`run_path()`:** add special-case before the existing config lookup:

```python
if name == "gfm":
    # GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH read from env via constants
    wrapper = GFMRetrieverWrapper.get_instance(GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH)
    t0 = time.monotonic()
    chunks = await wrapper.retrieve(query, getattr(param, "chunk_top_k", 10), lightrag.text_chunks)
    latency = time.monotonic() - t0
    return chunks, latency
```

All other paths continue to use `lightrag.aquery_data` unchanged.

---

## 7. profiles.py Changes

Add `"gfm"` to `KNOWN_PATHS` frozenset.

Add new profile to `PROFILE_REGISTRY`:

```python
RetrievalProfile(
    name="gfm_multihop",
    description="Multi-hop reasoning via GFM graph + PPR walk (toggle either path below)",
    paths=[
        # "gfm",    # uncomment to enable GFM graph retrieval
        "ppr",
        "hybrid",
    ],
    rrf_weights={
        # "gfm": 1.0,   # uncomment together with path above
        "ppr":    0.8,
        "hybrid": 0.6,
    },
),
```

GFM commented out by default. To A/B test: uncomment both lines and comment out PPR's two lines.

---

## 8. query.py — Standalone mode="gfm"

In `aquery()`, add a branch after `mode="auto"` handling:

```python
if mode == "gfm":
    return_trace_gfm = bool(kwargs.pop("return_trace", False))
    wrapper = GFMRetrieverWrapper.get_instance(GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH)
    chunks = await wrapper.retrieve(query, kwargs.get("chunk_top_k", DEFAULT_CHUNK_TOP_K), self.lightrag.text_chunks)
    answer = await self._generate_answer_from_chunks(
        query, chunks, system_prompt=system_prompt,
        response_type=kwargs.get("response_type", "Multiple Paragraphs"),
    )
    if return_trace_gfm:
        return {"answer": answer, "trace": {"mode": "gfm", "chunks_retrieved": len(chunks)}}
    return answer
```

---

## 9. Configuration

New environment variables (added to `constants.py` and `env.example`):

| Variable | Default | Description |
|---|---|---|
| `GFM_DATA_DIR` | `./data` | Root data directory for GFM-RAG index |
| `GFM_DATA_NAME` | `""` | Graph name (subdirectory under `GFM_DATA_DIR`) |
| `GFM_MODEL_PATH` | `rmanluo/G-reasoner-34M` | HuggingFace model path or local path |

If `GFM_DATA_NAME` is empty, `GFMRetrieverWrapper.get_instance()` raises `RuntimeError` with a clear message directing the user to run the export script first.

---

## 10. Phase 2 — Semi-Automatic Sync (not implemented in Phase 1)

**Interface pre-reserved in `processor.py`:**

- After each successful document insert, increment counter in `gfm_sync_state.json`
- When counter reaches threshold (`GFM_SYNC_THRESHOLD`, default 100), schedule a background `asyncio` task that runs the export script
- Counter resets after successful export

**New file:** `raganything/retrieval/gfm_sync.py` — encapsulates background export task logic.

---

## 11. File Inventory

### New files

| File | Purpose |
|---|---|
| `scripts/export_lightrag_to_gfm.py` | One-time offline export: LightRAG → GFM-RAG CSVs |
| `raganything/retrieval/gfm_retriever.py` | Lazy singleton wrapper around `gfmrag.GFMRetriever` |

### Modified files

| File | Change |
|---|---|
| `raganything/retrieval/paths.py` | Add `"gfm"` to `KNOWN_PATHS`; add special-case branch in `run_path()` |
| `raganything/retrieval/profiles.py` | Add `"gfm"` to `KNOWN_PATHS`; add `"gfm_multihop"` profile |
| `raganything/query.py` | Add `mode="gfm"` branch in `aquery()` |
| `raganything/constants.py` | Add `GFM_DATA_DIR`, `GFM_DATA_NAME`, `GFM_MODEL_PATH` defaults |
| `env.example` | Document new GFM env vars |

### Phase 2 additions (not in Phase 1)

| File | Purpose |
|---|---|
| `raganything/retrieval/gfm_sync.py` | Background export task |
| `processor.py` | Insert counter + sync trigger |
