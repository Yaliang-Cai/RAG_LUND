# Agentic RAG Redesign (V4)

**Date:** 2026-05-05
**Workspace:** `.worktrees/agentic-rag/` (in-place evolution)
**Status:** Approved design, ready for implementation plan

---

## 1. Problem Statement

The current `AdaptiveAgentGraph` (in `.worktrees/agentic-rag/`) has structural gaps:

- **Two-tier classification redundancy.** `ComplexityClassifier` (simple/medium/complex) and `RetrievalRouter` (5 profiles) both use LLM calls; their outputs largely overlap.
- **No pre-generation grader.** Retrieved chunks go straight to the generator. If retrieval is poor, the LLM still generates from irrelevant context.
- **No hallucination check.** Generator output is returned to the user without verification against retrieved evidence.
- **Up to 130 chunks reach the reranker** in multi-hop / parallel paths; reranker OOM triggers a 32→16→8 batch-size backoff that wastes 10s+ per query.
- **`full` profile chosen too eagerly.** Used as low-confidence fallback in `QueryClassifier`, which dilutes its semantics.
- **No router caching.** Identical or near-identical queries re-pay LLM classification cost.
- **Decompose runs upfront.** `complexity == "complex"` triggers query decomposition before any retrieval is attempted, wasting LLM calls when single-shot multi-hop retrieval would have sufficed.

This redesign produces a single-tier router, adds grader and hallucination-check nodes, introduces failure-driven escalation (rewrite → decompose), and locks in concrete reliability fixes.

## 2. Goals & Non-Goals

**Goals:**
- Eliminate hallucination by gating generator output on a grounding check.
- Single-LLM-call routing with deterministic fallback.
- Bounded cost: hard cycle limits prevent runaway loops.
- Fix concrete production issues (rerank OOM, 130-chunk pipelines).
- Backward-compatible CLI (`query_ppr.py --mode agentic --phoenix` keeps working).

**Non-goals:**
- Changing rag-anything's standard prompt templates.
- Citation formatting (`[DC1]` markers) in agentic mode — defer.
- Multi-process router cache (Redis) — single-process LRU is sufficient.
- Replacing `RetrievalRouter` (the multi-path executor); only the LangGraph orchestration layer changes.

## 3. Architecture

### 3.1 LangGraph Topology

```
START
  ↓
router (1 LLM, with cache)
  ↓ [profile ∈ {precise, semantic, local, multihop}]
retriever (calls RetrievalRouter.route)
  ↓
grader (1 LLM, batch over all chunks, shares prefix with generator)
  ├─ sufficient → generator
  └─ insufficient:
       cycle 0 → rewriter → retriever (cycle++)
       cycle 1 → decomposer → parallel_retriever (force profile=full, cycle++)
       cycle ≥ 2 → END_INSUFFICIENT (skip generator)

generator (1 LLM, shares prefix with grader)
  ↓
hallucination_check (1 LLM, shares prefix)
  ├─ grounded → END (confidence=high)
  ├─ ungrounded & check_cycle < 2 → targeted_retriever → generator (check_cycle++)
  └─ ungrounded & check_cycle ≥ 2 → END_INSUFFICIENT (drop answer)

targeted_retriever
  - Uses ungrounded_claims as new query
  - Same profile as current state
  - Appends new chunks to state, dedupe by chunk_id
```

**Two independent counters:**
- `retrieve_cycle: 0..3` — controls retrieve→grade loop
- `check_cycle: 0..2` — controls generate→check loop

**Two terminal failure states:**
- `END_INSUFFICIENT` — return `{"answer": None, "confidence": "none"}`. Generator never called or its output rejected.
- `END_LOW_CONFIDENCE` — only fires if hallucination_check itself crashes (LLM error / JSON parse failure). Returns generator's last answer with `confidence="low"`.

### 3.2 State Schema

```python
class AgentState(TypedDict):
    query: str                    # original query (for trace)
    current_query: str            # rewriter-modified query
    profile: str                  # active profile
    chunks: list[dict]            # accumulated across targeted retries
    answer: str
    grounded: bool
    ungrounded_claims: list[str]
    retrieve_cycle: int
    check_cycle: int
    routing_trace: dict
```

Removed from existing schema: `complexity`, `sub_questions` (now node-local), `eval_score`, `eval_gap`.

### 3.3 File Layout

```
.worktrees/agentic-rag/rag-anything/raganything/retrieval/
├── router.py              # UNCHANGED (RetrievalRouter, multi-path executor)
├── profiles.py            # MODIFIED (defaults: cap=30, min_rrf_score=0.01)
├── paths.py               # UNCHANGED
├── classifier.py          # MODIFIED (4 candidates, fallback=semantic, avoid param)
├── complexity.py          # DELETED
├── evaluator.py           # RENAMED → hallucination_checker.py, fully rewritten
├── agent_graph.py         # REWRITTEN (_build_graph, new state, new nodes)
├── grader.py              # NEW
├── rewriter.py            # NEW
└── router_cache.py        # NEW (LRU 2048)
```

## 4. Component Specifications

### 4.1 `router` Node + `RouterClassifier`

**`classifier.py` changes:**
- Candidate set reduced from 5 to 4: `{precise, semantic, local, multihop}`. `full` is removed from router's choice set; it is now used only by cycle-3 escalation.
- Low-confidence fallback changed from `full` → `semantic`.
- New parameter `avoid: list[str] = None` — when retrying via cache-failure path, the prompt instructs the classifier to exclude profiles in this list.
- Prompt updated: remove `full` description; add note that ambiguous queries should fall through to `semantic`.

**`router_cache.py` (new):**
- `functools.lru_cache`-style wrapper, `maxsize=2048`.
- Key: `sha256(prompt_template_hash + normalize(query))[:16]`.
- Value: `{"profile": str, "outcome": "success" | "failed" | "unknown", "fail_count": int}`.
- Lifecycle:
  - Router writes `outcome="unknown"` after first classification.
  - On `END` with `grounded=True`, mark cache entry `success`.
  - On `END_INSUFFICIENT` or `END_LOW_CONFIDENCE`, increment `fail_count`; mark `failed` when `fail_count >= 2`; evict at `fail_count >= 3`.
  - On cache hit with `outcome="failed"`, router calls classifier with `avoid=[cached_profile]`.

### 4.2 `retriever` Node

Calls `RetrievalRouter.route(current_query, param, profile_name=state["profile"])`. No changes to `router.py`. Existing `routing_trace` is merged into `state["routing_trace"]`.

### 4.3 `grader` Node + `Grader`

**`grader.py` (new):**
```python
class Grader:
    PROMPT_SUFFIX = """\
Question: {query}

Are the chunks above sufficient to accurately answer this question?
Output JSON: {"sufficient": true|false, "reason": "<one short sentence>"}
"""

    async def grade(self, query: str, chunks: list[dict]) -> dict:
        # Build SHARED_PREFIX (chunks full text) + suffix
        # Single LLM call, JSON output
        # On JSON parse failure: return {"sufficient": True, "reason": "fallback"}
```

**Prompt prefix structure (shared with generator and hallucination_check):**
```
You are a RAG quality controller.

Context (10 chunks):
[1] Source: {file_path}
{chunk_1_full_1200_tokens}

[2] Source: {file_path}
{chunk_2_full_1200_tokens}

... [10] ...

---
{node-specific suffix}
```

The chunks block is identical across all three node prompts → vLLM APC reuses the prefill.

### 4.4 `rewriter` Node + `Rewriter`

**`rewriter.py` (new):**
```python
PROMPT = """\
The following query did not retrieve sufficient evidence.

Query: {original_query}
Retrieval feedback: {grader_reason}

Rewrite the query to improve retrieval. Strategies:
- Replace ambiguous terms with synonyms
- Add explicit domain context
- Decompose compound noun phrases

Output the rewritten query only. No explanation, no quotation marks.
"""
```

Single LLM call. On exception, returns the original query unchanged (allows the next cycle to escalate to decomposer).

### 4.5 `decomposer` + `parallel_retriever` Nodes

- `decomposer`: reuses existing `_decompose_query` from `agent_graph.py`. Generates 2–4 sub-questions.
- `parallel_retriever`: reuses existing `_node_parallel_retrieve`, with two changes:
  - **Forces `profile_name="full"`** when called as cycle-3 escalation.
  - After concatenating sub-question chunks, runs RRF merge + caps at 30 (prevents 130-chunk cascades into grader).

### 4.6 `generator` Node

Reuses existing `_generate_answer`, prompt restructured to share prefix with grader/check:

```python
GENERATOR_SUFFIX = """\
Question: {query}

Answer the question based ONLY on the context above.
If the context lacks the information needed to answer accurately,
say so explicitly rather than speculating.

Provide a comprehensive response.
"""
```

The honesty hint at the end ensures that "I cannot answer based on the context" responses are produced naturally and pass hallucination_check (no claims to verify).

**Generator prompt remains independent from rag-anything / LightRAG standard prompts** (out of scope to unify).

### 4.7 `hallucination_check` Node + `HallucinationChecker`

**`hallucination_checker.py` (renamed from `evaluator.py`, fully rewritten):**

```python
class HallucinationChecker:
    PROMPT_SUFFIX = """\
Answer:
{answer}

Question being answered: {query}

For every factual claim in the Answer, verify it against the Context above.
A claim is "grounded" only if the Context contains explicit support for it.
Statements like "I cannot determine X from the context" make no factual claims
and are considered grounded.

Output JSON:
{
  "grounded": true | false,
  "ungrounded_claims": ["<claim 1>", "<claim 2>", ...]
}
"""

    async def verify(self, query: str, answer: str, chunks: list[dict]) -> dict:
        # Single LLM call
        # On exception or JSON parse failure: return {"grounded": True, "ungrounded_claims": []}
        # (Defaults to grounded=True so service-level errors don't block valid answers;
        #  caller marks confidence="low" when this fallback fires.)
```

Binary judgment only — no ratio, no scoring. Any ungrounded claim → trigger retry or END.

### 4.8 `targeted_retriever` Node

```python
async def targeted_retriever(state):
    new_query = " ".join(state["ungrounded_claims"])
    new_chunks, trace = await RetrievalRouter.route(
        new_query, param, profile_name=state["profile"]
    )
    combined = dedupe_by_chunk_id(state["chunks"] + new_chunks)
    combined = combined[:30]  # cap to prevent unbounded growth
    return {
        "chunks": combined,
        "check_cycle": state["check_cycle"] + 1,
    }
```

Does not re-router, does not re-grade. Goes straight back to generator.

## 5. Configuration

### 5.1 New constants in `raganything/constants.py`

```python
# Query mode (string list updated, no new constant needed)
DEFAULT_QUERY_MODE = "hybrid"  # ... | "ppr" | "gfm" | "agentic"

# Agentic RAG (V4)
DEFAULT_AGENTIC_MAX_RETRIEVE_CYCLES = 3
DEFAULT_AGENTIC_MAX_CHECK_CYCLES = 2

DEFAULT_AGENTIC_ROUTER_CACHE_SIZE = 2048
DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE = "semantic"

DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS = 4
DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY = 3

DEFAULT_AGENTIC_GRADER_FALLBACK_SUFFICIENT = True
```

### 5.2 Modified existing constants

```python
# Lock down rerank batch size — eliminates 32→16→8 OOM backoff (saves ~10s/query)
DEFAULT_RERANK_BATCH_SIZE = 8           # was 32
DEFAULT_RERANK_ENABLE_OOM_BACKOFF = False  # was True
```

### 5.3 `profiles.py` changes

```python
@dataclass
class RetrievalProfile:
    # ...
    rerank_candidate_cap: int = 30          # was 60 (prevents 130-chunk cascades)
    min_rrf_score: float = 0.01             # NEW (filter long-tail RRF results)
```

`RetrievalRouter._weighted_rrf_merge` adds a post-filter:
```python
merged = [c for c in merged if c["rrf_score"] >= profile.min_rrf_score]
```

### 5.4 vLLM startup (documentation only, not code)

`start_vllm.sh` annotation:
```bash
--enable-prefix-caching   # REQUIRED for grader/generator/check prefix sharing
--max-num-seqs 6
```

### 5.5 API integration

`raganything/query.py` (`QueryMixin.aquery`): add dispatch branch for `mode == "agentic"`:

```python
async def aquery(self, query, mode="hybrid", **kwargs):
    if mode == "agentic":
        from .retrieval.agent_graph import AdaptiveAgentGraph
        graph = AdaptiveAgentGraph(self.lightrag, ...)
        return await graph.run(query, return_trace=kwargs.get("return_trace", False))
    # ... existing dispatch for other modes
```

`scripts/query_ppr.py` already has `--mode agentic --phoenix` wiring (lines 116, 193, 221). No CLI changes needed.

## 6. Return Value

### 6.1 `return_trace=False` (default, backward-compatible)
Returns `str` (the answer) or `None` (if `END_INSUFFICIENT`).

### 6.2 `return_trace=True`

```python
{
    "answer": str | None,                # None when END_INSUFFICIENT
    "confidence": "high" | "low" | "none",
    "grounded": bool,                    # meaningful only when confidence != "none"
    "ungrounded_claims": list[str],      # empty when grounded=True
    "trace": {
        "profile": str,                  # final profile used
        "router_cache_hit": bool,
        "retrieve_cycles_used": int,
        "check_cycles_used": int,
        "rewrite_history": list[str],    # successive query rewrites
        "sub_questions": list[str] | None,
        "chunks_per_path": dict[str, int],
        "llm_call_count": int,
        "latency_total_ms": int,
    }
}
```

**Confidence semantics:**
- `high`: hallucination_check returned grounded=True
- `low`: hallucination_check itself crashed; answer is unverified
- `none`: explicit failure (END_INSUFFICIENT — generator never called or its output rejected)

## 7. Failure Handling

| Failure point | Type | Behavior |
|---|---|---|
| Router LLM exception | infrastructure | Use `semantic` profile, continue |
| Retriever: all paths fail | data | Raise `RetrievalError`, END_INSUFFICIENT |
| Retriever: some paths fail | data | Existing logic: drop failed paths, continue with rest |
| Grader JSON parse fail | infrastructure | `sufficient=True` (avoids infinite retrieve loop) |
| Rewriter exception | infrastructure | Return original query unchanged |
| Decomposer exception | infrastructure | Use original query as sole "sub-question" |
| Generator exception | infrastructure | Retry once; on second failure → END_INSUFFICIENT |
| Hallucination check crashes | infrastructure | `grounded=True`, return answer, mark `confidence="low"`, trace `check_status="error"` |
| Hallucination check returns ungrounded + check_cycle < 2 | business | `targeted_retriever` → re-generate |
| Hallucination check returns ungrounded + check_cycle ≥ 2 | business | END_INSUFFICIENT, drop answer (do not return partial fabrication) |
| `retrieve_cycle ≥ 3` reached | business | END_INSUFFICIENT |

**Core principle:** infrastructure failures never deadlock the loop — every fallback either continues or terminates, never retries the same call indefinitely.

## 8. Cost Bounds

| Path | LLM calls | Estimated latency |
|---|---|---|
| Happy (cycle 0, grounded) | 4 (router + grade + generate + check) | 5–7s |
| Cycle-1 rewrite recovers | 6 | 8–10s |
| Cycle-2 decompose recovers | 7+ | 14–18s |
| All retrieve cycles fail | 4–5 (no generator) | 8–10s |
| 1 hallucination retry | +2 | +4–6s |
| Worst case | ~9 | ~18–22s |

Soft monitoring: log warning if `llm_call_count > 12` per query.

## 9. Testing

### 9.1 Unit tests (`tests/retrieval/`)
- `test_router_classifier.py` — 4 candidates, low-confidence → semantic, `avoid` parameter respected
- `test_router_cache.py` — LRU eviction, prompt-hash key, success/failed state transitions
- `test_grader.py` — JSON parse fallback, prompt contains all chunks
- `test_rewriter.py` — feedback embedded, exception → returns original
- `test_hallucination_checker.py` — grounded/ungrounded judgment, claim extraction, parse-fail fallback
- `test_agent_graph.py` (mock LLM) — happy path, cycle 1 rewrite, cycle 2 decompose, cycle 3 END_INSUFFICIENT (generator NOT called), check retry, all counters correct

### 9.2 Integration tests (`evaluate_local/agentic/`)
On a 20–50 query subset of DocBench, compare `mode=hybrid` vs `mode=agentic`:
- Accuracy delta
- Latency distribution (p50, p95)
- LLM call count distribution
- END_INSUFFICIENT / END_LOW_CONFIDENCE rates

### 9.3 Phoenix observability
LangGraph node-level OpenTelemetry spans (auto-emitted) with project name `agentic-rag`:
- `router.classify` (cache hit/miss)
- `retriever.route` (paths, latencies)
- `grader.grade` (sufficient + reason)
- `rewriter.rewrite`
- `decomposer.decompose`
- `parallel_retriever.gather`
- `generator.generate`
- `hallucination_check.verify`

## 10. Documentation Updates

- `docs/auto_query_mode_2026-04-27.md` — add "V4: Agentic mode" section
- `rag-anything/CLAUDE.md` — extend "查询层" section with agentic mode
- `scripts/query_ppr.py` docstring — refresh agentic example block

## 11. Out of Scope

- Multi-process router cache (Redis / SQLite). Single-process LRU is sufficient for current deployment.
- Citation formatting (`[DC1]` markers) in agentic answers.
- Replacing generator prompt with LightRAG standard prompt template.
- GFM-RAG path integration into agentic flow (existing `gfm_multihop` profile remains untouched).
- Streaming responses through the LangGraph state machine.
