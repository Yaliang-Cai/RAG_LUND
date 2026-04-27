# Adaptive Retrieval Routing: Auto Query Mode

**Date:** 2026-04-27  
**Branch:** `feat/retrieval-router`  
**Status:** Implemented, 37 tests passing

---

## 1. Overview

The `mode="auto"` query entry point introduces **profile-based adaptive retrieval routing** on top of the existing retrieval components (LightRAG naive/hybrid/mix, custom PPR, Qdrant sparse BM25, reranker). Instead of requiring the caller to select a fixed retrieval strategy, the system classifies the query at runtime, selects the optimal retrieval profile, executes multiple retrieval paths in parallel, and fuses the results via weighted Reciprocal Rank Fusion (RRF) followed by neural reranking.

The design targets two orthogonal use cases:

- **Production inference**: LLM classifier automatically selects the profile that best matches the query type. No manual mode selection needed.
- **Ablation / evaluation**: Caller passes `profile=<name>` to bypass the classifier and force a specific profile. Fully reproducible, no LLM call overhead.

---

## 2. Architecture

```
query
  │
  ▼
┌─────────────────────────────┐
│    QueryClassifier (LLM)    │  ← skipped if profile= override
│  "precise" / "local" /      │
│  "multihop" / "descriptive" │
│  / "full"                   │
└────────────┬────────────────┘
             │ profile name
             ▼
┌─────────────────────────────┐
│      RetrievalProfile       │
│  paths, weights, overrides  │
│  rrf_k, rerank settings     │
│  max_concurrent_paths       │
└────────────┬────────────────┘
             │
    ┌────────┴──────────┐
    │  parallel paths   │  (asyncio.gather + optional Semaphore)
    │                   │
  ppr  hybrid  naive  qdrant_sparse          (full profile)
  ppr  hybrid                                (multihop profile)
  hybrid  naive                              (local profile)
  mix  qdrant_hybrid                         (descriptive profile)
  qdrant_sparse                              (precise profile)
    │       │     │    │        │              │
    └───────┴─────┴────┴────────┴──────────────┘
             │ ranked chunk lists (intact, no pre-dedup)
             ▼
┌─────────────────────────────┐
│   Weighted RRF Fusion       │
│  Score(d) = Σ w_p/(k+rank) │
│  Natural dedup via scores   │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Neural Reranker (BGE)      │
│  capped at candidate_cap    │
│  filtered by min_score      │
└────────────┬────────────────┘
             ▼
          top-k chunks
             │
             ▼
┌─────────────────────────────┐
│   Answer Generation (LLM)   │
│   or VLM dereference        │
└─────────────────────────────┘
```

---

## 3. Retrieval Profiles

Each profile encodes a retrieval strategy as a named, reusable configuration. Profiles are the unit of ablation: swapping profiles changes retrieval behavior without touching query logic.

### 3.1 Built-in Profiles

| Profile | Paths | Rationale |
|---------|-------|-----------|
| `precise` | `qdrant_sparse` (BM25 only) | Exact lexical match for error codes, IDs, rare proper nouns. Avoids semantic drift from vector similarity. |
| `local` | `hybrid` + `naive` | Single-hop factual queries. Hybrid covers entity-centric KG paths; naive adds dense vector recall for short answer facts. |
| `multihop` | `ppr` + `hybrid` | Multi-entity chain reasoning. PPR propagates across the KG to surface indirect connections; hybrid anchors on direct mentions. |
| `descriptive` | `mix` + `qdrant_hybrid` | Open-ended, broad-context questions. `mix` activates untruncated KG context (`kg_chunk_selection_source=untruncated`); Qdrant hybrid adds dense+sparse complementary recall. |
| `full` | `ppr` + `hybrid` + `naive` + `qdrant_sparse` | Fallback for ambiguous queries. All 4 paths run fully in parallel (no semaphore). Weights are calibrated to counter dense-overlap inflation: `ppr=1.2, hybrid=1.0, qdrant_sparse=0.9, naive=0.7`. |

### 3.2 Profile Fields

| Field | Default | Description |
|-------|---------|-------------|
| `paths` | — | Ordered list of retrieval path names to activate |
| `rrf_weights` | — | Per-path weight in the RRF formula (must cover all paths) |
| `rrf_k` | 60 | RRF smoothing constant |
| `enable_rerank` | True | Whether to apply neural reranker after fusion |
| `min_rerank_score` | 0.3 | Minimum BGE score to retain a chunk |
| `rerank_candidate_cap` | 60 | Max chunks passed to reranker |
| `max_concurrent_paths` | None | Semaphore limit; None = no limit |
| `path_overrides` | {} | Per-path `QueryParam` overrides (e.g. `kg_chunk_selection_source`) |

---

## 4. Query Classifier

The classifier is an LLM call with a structured English few-shot prompt. It outputs a JSON object:

```json
{"reasoning": "...", "profile": "local", "confidence": 0.88}
```

**Fallback rules** (applied in order):
1. If `confidence < 0.6` → fall back to `"full"`
2. If `profile` is not a registered profile name → fall back to `"full"`
3. If any exception occurs (parse error, LLM timeout) → fall back to `"full"`

The classifier is entirely bypassed when `profile=` is passed explicitly by the caller. In that case, the classifier latency is recorded as 0.0 s in the routing trace.

---

## 5. Weighted RRF Fusion

Given ranked chunk lists from _P_ active paths, the fusion score for document _d_ is:

$$\text{Score}(d) = \sum_{p \in P} w_p \cdot \frac{1}{k + \text{rank}_p(d)}$$

where $w_p$ is the profile-defined weight for path _p_, $k = 60$ is the smoothing constant (default), and $\text{rank}_p(d)$ is the 0-based rank of _d_ in path _p_'s result list (chunks not returned by path _p_ contribute 0).

**Key design decision:** each path's ranked list enters the fusion intact — **no pre-deduplication**. A chunk appearing in multiple paths naturally accumulates higher score because each occurrence adds a positive $w_p/(k + \text{rank})$ term. Deduplication is a byproduct of score accumulation: each unique chunk ID appears once in the output, carrying the sum of its cross-path contributions. This preserves cross-path rank consensus signals that would be lost by pre-deduplication.

---

## 6. Routing Trace

Every `mode="auto"` call returns a routing trace (accessible via `return_trace=True`):

```python
{
  "profile": "local",
  "confidence": 0.88,
  "reasoning": "...",
  "paths_activated": ["hybrid", "naive"],
  "paths_failed": [],
  "chunks_per_path": {"hybrid": 14, "naive": 11},
  "chunks_after_rrf": 18,
  "chunks_after_rerank": 12,
  "chunks_after_threshold": 5,
  "latency_per_path": {
    "classifier": 0.34,
    "hybrid": 0.87,
    "naive": 0.45
  }
}
```

The trace is the primary instrument for diagnosing retrieval behavior and comparing profiles in ablation experiments.

---

## 7. Usage

### 7.1 Python API

```python
# Auto routing (LLM classifies query type)
answer = await rag.aquery("Describe the overall architecture of LightRAG.", mode="auto")

# Explicit profile (bypasses classifier — for ablation)
answer = await rag.aquery("CVE-2026-001 impact scope?", mode="auto", profile="precise")

# With routing trace
result = await rag.aquery("...", mode="auto", return_trace=True)
answer = result["answer"]
routing = result["trace"]["routing"]  # full trace dict

# VLM-enhanced (router provides chunks, then image dereference runs normally)
answer = await rag.aquery_vlm_enhanced("Describe this figure.", mode="auto")
```

### 7.2 Service Layer (`LocalRagService`)

`mode="auto"` flows transparently through `service.query()` and `service.query_with_trace()`:

```python
# Via service
response = await service.query_with_trace(
    workspace_id="My_Graph",
    query="What is the main contribution of HippoRAG2?",
    mode="auto",
    profile="multihop",   # optional
)
routing_trace = response["trace"]["routing"]
```

Note: for `mode="auto"`, `response["trace"]` has structure `{"routing": {...}}`, whereas other modes return `{"data": {"chunks": [...], "entities": [...], ...}}`.

### 7.3 CLI (`scripts/query_ppr.py`)

```bash
# LLM 自动路由（分类器决定 profile）
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode auto --trace

# 强制指定 profile=precise（绕过分类器）
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode auto --profile precise --trace

# ppr 模式对比基线
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode ppr --trace
```

Available `--profile` values: `precise`, `local`, `multihop`, `descriptive`, `full`.  
`--profile` is only effective when `--mode auto` is set.

---

## 8. Methodology (Paper Section)

### 8.1 Motivation

Existing RAG systems apply a single, fixed retrieval strategy to all queries regardless of query type. A factual lookup ("How many parameters does BERT have?") and a multi-document synthesis question ("What architectural choices from HippoRAG2 influenced LightRAG?") have fundamentally different retrieval requirements. The former benefits from high-precision lexical or entity-centric recall; the latter requires propagation across multi-hop relational chains in the knowledge graph. Applying a uniform hybrid retrieval strategy to both introduces unnecessary noise in the first case and insufficient graph traversal depth in the second.

### 8.2 Adaptive Retrieval Routing

We introduce a **profile-based adaptive retrieval routing** mechanism that selects the retrieval strategy at query time. The system maintains a registry of named retrieval profiles, each specifying: (i) which retrieval paths to activate, (ii) their relative fusion weights, and (iii) per-path parameter overrides. A lightweight LLM classifier maps an incoming query to a profile name by reasoning over query-type signals (specificity, entity count, scope of synthesis required). The classifier uses a structured few-shot prompt and outputs a confidence score; when confidence falls below a threshold (0.6), or when the predicted profile is unrecognized, the system falls back to the full multi-path profile.

For evaluation and ablation settings, the classifier can be bypassed by explicitly specifying a profile name, ensuring reproducibility without LLM classification overhead.

### 8.3 Parallel Multi-Path Recall

Once a profile is selected, all configured retrieval paths are executed concurrently via asynchronous parallel dispatch. The system supports four retrieval paths in the full profile: Personalized PageRank over the knowledge graph (PPR), entity-centric KG retrieval with dense entity seeds (hybrid), dense vector chunk retrieval (naive), and BM25 sparse chunk retrieval (qdrant_sparse). `mix` and `qdrant_hybrid` are available for targeted profiles (`descriptive`) but excluded from `full` to avoid redundant dense overlap. All four `full` paths run in fully parallel dispatch with no concurrency semaphore.

Individual path failures are isolated: a failed path is recorded in the routing trace but does not abort the overall retrieval. The system raises an error only if all paths fail simultaneously.

### 8.4 Weighted Reciprocal Rank Fusion

Retrieved chunk lists from all active paths are fused using a weighted extension of Reciprocal Rank Fusion (RRF). For a candidate document $d$, the fusion score is defined as:

$$\text{Score}(d) = \sum_{p \in P} w_p \cdot \frac{1}{k + \text{rank}_p(d)}$$

where $P$ is the set of active paths, $w_p$ is the profile-assigned weight of path $p$, $\text{rank}_p(d)$ is the zero-based rank of $d$ in path $p$'s result list, and $k$ is a smoothing constant (default: 60). Documents not returned by path $p$ contribute zero for that path. Crucially, each path's ranked list enters the fusion intact without prior deduplication, preserving cross-path rank consensus: a document appearing at rank 1 in two independent paths accumulates a higher fused score than a document appearing at rank 1 in only one path. Deduplication is a natural byproduct of the score accumulation rather than a preprocessing step.

Path weights in the `full` profile are calibrated to counter the dense-overlap inflation effect: `hybrid` (KG traversal, dense entity seeds) and `naive` (direct dense chunk retrieval) share the same underlying dense retrieval signal at the chunk level. Without weight adjustment, a chunk retrieved by both paths accumulates a doubled RRF contribution, causing dense results to dominate and diluting unique signals from PPR and BM25. The calibrated weights (`ppr=1.2, hybrid=1.0, qdrant_sparse=0.9, naive=0.7`) dampen the naive path's contribution relative to the non-overlapping paths while preserving its role as a recall safety net.

### 8.5 Neural Reranking and Threshold Filtering

The top-$C$ candidates from RRF (where $C$ is a profile-defined cap, default: 60) are passed to a CrossEncoder reranker (BGE-reranker-v2-m3). The reranker scores each candidate against the original query. Candidates with a reranker score below a minimum threshold (default: 0.3) are discarded. The remaining candidates are truncated to the final `chunk_top_k` window before answer generation.

This two-stage pipeline (RRF→rerank) decouples recall (handled by multi-path retrieval and fusion) from precision (handled by neural reranking), allowing each stage to be tuned independently.

### 8.6 Routing Trace for Interpretability

Every query in auto mode produces a structured routing trace recording: the selected profile and classifier confidence, which paths were activated and how many chunks each returned, the chunk counts at each pipeline stage (after RRF, after reranking, after threshold), and per-path latency including classifier latency. This trace serves as the primary diagnostic instrument for understanding retrieval behavior and as the ground truth for per-query profile attribution in ablation experiments.

---

## 9. Implementation Notes

- **`raganything/retrieval/profiles.py`** — `RetrievalProfile` dataclass and `PROFILE_REGISTRY`
- **`raganything/retrieval/classifier.py`** — `QueryClassifier` with few-shot English prompt
- **`raganything/retrieval/paths.py`** — `run_path()` per-path wrapper; handles dynamic `QueryParam` fields via `setattr`
- **`raganything/retrieval/router.py`** — `RetrievalRouter`, `_weighted_rrf_merge`, `RetrievalError`
- **`raganything/query.py`** — `mode="auto"` branches in `aquery()` and `aquery_vlm_enhanced()`

All modules have full unit and integration test coverage (37 tests, `tests/retrieval/`).
