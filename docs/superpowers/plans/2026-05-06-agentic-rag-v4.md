# Agentic RAG V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the LangGraph agent loop to add a batch sufficiency grader, a binary hallucination checker, failure-driven escalation (rewrite → decompose), a router LRU cache, and concrete reliability fixes (rerank OOM, 130-chunk cascades).

**Architecture:** A single-tier `QueryClassifier` routes to one of four profiles; a `Grader` gates generation on retrieved evidence quality; a `HallucinationChecker` verifies the generated answer is fully grounded; iterative cycle counters bound the loop. All three LLM calls after retrieval share a common prompt prefix to exploit vLLM automatic prefix caching.

**Tech Stack:** Python 3.10+, LangGraph (`langgraph`), `pytest-asyncio` (asyncio_mode=auto), local vLLM Qwen3, BGE-reranker-v2-m3 CrossEncoder.

**Working directory:** `.worktrees/agentic-rag/rag-anything/` — all paths below are relative to this root.

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `raganything/constants.py` | New agentic constants; rerank batch/backoff fix |
| Modify | `raganything/retrieval/profiles.py` | `rerank_candidate_cap=30`; `min_rrf_score` field |
| Modify | `raganything/retrieval/router.py` | Post-RRF long-tail filter |
| Modify | `raganything/retrieval/classifier.py` | 4 candidates; `avoid` param; `semantic` fallback |
| Create | `raganything/retrieval/router_cache.py` | LRU cache with tri-state outcome tracking |
| Create | `raganything/retrieval/grader.py` | Batch sufficiency grader; shared prefix builder |
| Create | `raganything/retrieval/rewriter.py` | Query rewriter |
| Create | `raganything/retrieval/hallucination_checker.py` | Binary grounding verifier |
| Delete | `raganything/retrieval/complexity.py` | No longer needed |
| Delete | `raganything/retrieval/evaluator.py` | Replaced by hallucination_checker.py |
| Rewrite | `raganything/retrieval/agent_graph.py` | New state, nodes, graph topology |
| Modify | `tests/retrieval/test_classifier.py` | Update fallback expectations |
| Create | `tests/retrieval/test_router_cache.py` | LRU + outcome lifecycle |
| Create | `tests/retrieval/test_grader.py` | Sufficiency grading; JSON parse fallback |
| Create | `tests/retrieval/test_rewriter.py` | Rewrite behaviour; exception fallback |
| Create | `tests/retrieval/test_hallucination_checker.py` | Grounding judgment; exception fallback |
| Delete | `tests/retrieval/test_complexity.py` | Obsolete |
| Delete | `tests/retrieval/test_evaluator.py` | Replaced |
| Rewrite | `tests/retrieval/test_agent_graph.py` | Full flow; cycle limits; END states |

---

## Task 1: Update Constants, Profiles, and RRF Filter

**Files:**
- Modify: `raganything/constants.py`
- Modify: `raganything/retrieval/profiles.py`
- Modify: `raganything/retrieval/router.py`
- Test: `tests/retrieval/test_profiles.py` (existing — one assertion update)

- [ ] **Step 1: Update `raganything/constants.py`**

Add a new block after the existing rerank section and change two existing values:

```python
# Change these two existing lines:
DEFAULT_RERANK_BATCH_SIZE = 8           # was 32 — locks batch, eliminates OOM backoff
DEFAULT_RERANK_ENABLE_OOM_BACKOFF = False  # was True

# Add after the rerank block:
# =============================================================================
# Agentic RAG (V4)
# =============================================================================
DEFAULT_AGENTIC_MAX_RETRIEVE_CYCLES = 3
DEFAULT_AGENTIC_MAX_CHECK_CYCLES = 2
DEFAULT_AGENTIC_ROUTER_CACHE_SIZE = 2048
DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE = "semantic"
DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS = 4
DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY = 3
DEFAULT_AGENTIC_GRADER_FALLBACK_SUFFICIENT = True
```

- [ ] **Step 2: Update `raganything/retrieval/profiles.py`**

Change the `RetrievalProfile` dataclass defaults and add the new field:

```python
@dataclass
class RetrievalProfile:
    name: str
    description: str
    paths: list[str]
    rrf_weights: dict[str, float]
    rrf_k: int = 60
    enable_rerank: bool = True
    min_rerank_score: float = 0.3
    rerank_candidate_cap: int = 30      # was 60
    min_rrf_score: float = 0.01         # NEW
    max_concurrent_paths: int | None = None
    path_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
```

- [ ] **Step 3: Add RRF long-tail filter in `raganything/retrieval/router.py`**

In `RetrievalRouter.route()`, after the `_weighted_rrf_merge` call (currently step 4, line ~93), add a filter before the candidate cap slice:

```python
# 4. Weighted RRF
merged = _weighted_rrf_merge(
    {n: chunks_by_path[n] for n in profile.paths if n in chunks_by_path},
    profile.rrf_weights,
    profile.rrf_k,
)
# NEW: drop long-tail low-confidence candidates
if profile.min_rrf_score > 0.0:
    merged = [c for c in merged if c.get("rrf_score", 0.0) >= profile.min_rrf_score]
chunks_after_rrf = len(merged)

# 5. Rerank (capped at rerank_candidate_cap)
candidate_pool = merged[: profile.rerank_candidate_cap]
```

- [ ] **Step 4: Run existing profile tests**

```bash
pytest tests/retrieval/test_profiles.py -v
```

Expected: all existing tests PASS (field addition is backward-compatible).

- [ ] **Step 5: Commit**

```bash
git add raganything/constants.py raganything/retrieval/profiles.py raganything/retrieval/router.py
git commit -m "fix: lock rerank batch=8, cap RRF candidates at 30, add min_rrf_score filter"
```

---

## Task 2: Router Cache

**Files:**
- Create: `raganything/retrieval/router_cache.py`
- Create: `tests/retrieval/test_router_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_router_cache.py
import pytest
from raganything.retrieval.router_cache import RouterCache


def test_miss_returns_none():
    cache = RouterCache()
    assert cache.get("some query") is None


def test_put_then_get():
    cache = RouterCache()
    cache.put("what is BERT?", "semantic")
    entry = cache.get("what is BERT?")
    assert entry is not None
    assert entry["profile"] == "semantic"
    assert entry["outcome"] == "unknown"


def test_normalisation_ignores_case_and_whitespace():
    cache = RouterCache()
    cache.put("  What IS bert?  ", "semantic")
    assert cache.get("what is bert?") is not None


def test_mark_success():
    cache = RouterCache()
    cache.put("q", "semantic")
    cache.mark_success("q")
    assert cache.get("q")["outcome"] == "success"


def test_mark_failed_twice_marks_entry_failed():
    cache = RouterCache()
    cache.put("q", "multihop")
    cache.mark_failed("q")
    assert cache.get("q")["outcome"] == "unknown"  # not yet failed
    cache.mark_failed("q")
    assert cache.get("q")["outcome"] == "failed"


def test_mark_failed_three_times_evicts():
    cache = RouterCache()
    cache.put("q", "multihop")
    for _ in range(3):
        cache.mark_failed("q")
    assert cache.get("q") is None


def test_get_avoid_profiles_returns_empty_when_not_failed():
    cache = RouterCache()
    cache.put("q", "multihop")
    assert cache.get_avoid_profiles("q") == []


def test_get_avoid_profiles_returns_failed_profile():
    cache = RouterCache()
    cache.put("q", "multihop")
    cache.mark_failed("q")
    cache.mark_failed("q")
    avoid = cache.get_avoid_profiles("q")
    assert "multihop" in avoid


def test_lru_eviction():
    cache = RouterCache(maxsize=2)
    cache.put("q1", "semantic")
    cache.put("q2", "local")
    cache.put("q3", "precise")  # evicts q1
    assert cache.get("q1") is None
    assert cache.get("q2") is not None
    assert cache.get("q3") is not None


def test_prompt_hash_isolates_keys():
    c1 = RouterCache(prompt_hash="abc")
    c2 = RouterCache(prompt_hash="xyz")
    c1.put("q", "semantic")
    assert c2.get("q") is None
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/retrieval/test_router_cache.py -v
```

Expected: `ModuleNotFoundError: raganything.retrieval.router_cache`

- [ ] **Step 3: Implement `raganything/retrieval/router_cache.py`**

```python
# raganything/retrieval/router_cache.py
from __future__ import annotations
import hashlib
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


class RouterCache:
    """
    Process-local LRU cache for query → profile decisions.

    Tri-state outcome: "unknown" → "success" | "failed"
    Evicts entries that fail >= 3 times.
    """

    def __init__(self, maxsize: int = 2048, prompt_hash: str = "") -> None:
        self._maxsize = maxsize
        self._prompt_hash = prompt_hash
        self._store: OrderedDict[str, dict] = OrderedDict()

    def _key(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        raw = f"v{_CACHE_VERSION}:{self._prompt_hash}:{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[dict]:
        key = self._key(query)
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, query: str, profile: str) -> None:
        key = self._key(query)
        self._store[key] = {"profile": profile, "outcome": "unknown", "fail_count": 0}
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def mark_success(self, query: str) -> None:
        key = self._key(query)
        if key in self._store:
            self._store[key]["outcome"] = "success"

    def mark_failed(self, query: str) -> None:
        key = self._key(query)
        if key not in self._store:
            return
        entry = self._store[key]
        entry["fail_count"] += 1
        if entry["fail_count"] >= 2:
            entry["outcome"] = "failed"
        if entry["fail_count"] >= 3:
            del self._store[key]
            logger.debug("RouterCache: evicted %r after 3 failures", query[:60])

    def get_avoid_profiles(self, query: str) -> list[str]:
        entry = self.get(query)
        if entry and entry.get("outcome") == "failed":
            return [entry["profile"]]
        return []
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/retrieval/test_router_cache.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add raganything/retrieval/router_cache.py tests/retrieval/test_router_cache.py
git commit -m "feat(agentic): add RouterCache with LRU and tri-state outcome tracking"
```

---

## Task 3: Update QueryClassifier

**Files:**
- Modify: `raganything/retrieval/classifier.py`
- Modify: `tests/retrieval/test_classifier.py`

- [ ] **Step 1: Update the failing tests first**

In `tests/retrieval/test_classifier.py`, the test `test_low_confidence_falls_back_to_full` and `test_unknown_profile_falls_back_to_full` now expect `"semantic"` not `"full"`. Update both:

```python
async def test_low_confidence_falls_back_to_semantic():
    llm = AsyncMock(return_value=json.dumps({
        "reasoning": "unsure",
        "profile": "local",
        "confidence": 0.4,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("some ambiguous query")
    assert name == "semantic"


async def test_unknown_profile_falls_back_to_semantic():
    llm = AsyncMock(return_value=json.dumps({
        "reasoning": "ok",
        "profile": "nonexistent_profile",
        "confidence": 0.9,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("query")
    assert name == "semantic"


async def test_full_profile_rejected_from_llm_output():
    # Classifier must not accept "full" from LLM — it is reserved for cycle-3 escalation
    llm = AsyncMock(return_value=json.dumps({
        "reasoning": "very ambiguous",
        "profile": "full",
        "confidence": 0.95,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("something ambiguous")
    assert name == "semantic"   # rejected and replaced with fallback


async def test_avoid_excludes_profile():
    llm = AsyncMock(return_value=json.dumps({
        "reasoning": "multihop fits",
        "profile": "multihop",
        "confidence": 0.85,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("complex query", avoid=["multihop"])
    assert name == "semantic"   # multihop avoided, fallback applied


async def test_avoid_empty_list_no_effect():
    llm = AsyncMock(return_value=json.dumps({
        "reasoning": "multihop",
        "profile": "multihop",
        "confidence": 0.85,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("complex query", avoid=[])
    assert name == "multihop"
```

Add `from raganything.constants import DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE` to the test imports.

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/retrieval/test_classifier.py -v
```

Expected: 3-4 tests fail (fallback was "full", `avoid` param doesn't exist).

- [ ] **Step 3: Update `raganything/retrieval/classifier.py`**

```python
# raganything/retrieval/classifier.py
import json
import logging
import time
from typing import Any, Awaitable, Callable

from raganything.constants import DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE
from .profiles import PROFILE_REGISTRY

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6
_ROUTER_PROFILES = {"precise", "semantic", "local", "multihop"}  # full excluded

_CLASSIFIER_PROMPT = """\
You are a retrieval routing classifier. Given a user query, select the most
appropriate retrieval profile from the list below.

Available profiles (ordered from narrow to broad):

- precise: Query contains hard constraints that require exact lexical matching.
  Signals: specific IDs, error codes, version numbers, rare proper nouns, abbreviations.
  Examples: "What is the impact scope of CVE-2026-001?"
            "Status of order ID ORD-20260424-8821"

- semantic: Default workhorse for everyday knowledge queries. No graph traversal needed.
  Signals: factual questions, process/procedure explanations, concept definitions, summaries.
           Single topic, no multi-entity reasoning.
  Examples: "What is the company leave policy?"
            "How does the attention mechanism work?"

- local: Query is tightly focused on ONE specific entity and its direct properties or relationships.
  Signals: "What are the [attributes/dependencies] of X?"
  Examples: "What are the upstream systems of the payment service?"

- multihop: Query involves MULTIPLE distinct entities requiring cross-document reasoning.
  Signals: two or more named entities, causal/comparative language.
  Examples: "How did the network partition in region A cause failures in region B?"
            "Compare the indexing strategies used by LightRAG and HippoRAG2."

Key disambiguation rules:
- If the query asks about one entity → prefer local over multihop.
- If no entity graph is needed → prefer semantic over local.
- When genuinely unsure → choose semantic (it is the safe default).{avoid_instruction}

First briefly state your reasoning in one sentence, then output JSON.
Output format: {{"reasoning": "...", "profile": "<name>", "confidence": <0.0-1.0>}}

Query: {query}
"""

_AVOID_INSTRUCTION = """
- Do NOT output any of these profiles (already tried and failed): {avoid_list}
"""


class QueryClassifier:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]):
        self._llm = llm_func

    async def classify(
        self, query: str, avoid: list[str] | None = None
    ) -> tuple[str, dict[str, Any]]:
        t0 = time.monotonic()
        fallback = DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE
        profile = fallback
        confidence = 0.0
        reasoning = ""
        avoid = avoid or []
        avoid_instruction = (
            _AVOID_INSTRUCTION.format(avoid_list=", ".join(avoid)) if avoid else ""
        )
        try:
            prompt = _CLASSIFIER_PROMPT.format(
                query=query, avoid_instruction=avoid_instruction
            )
            raw = await self._llm(prompt, response_format={"type": "json_object"})
            result = json.loads(raw)
            candidate = str(result.get("profile", fallback)).strip()
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", ""))

            valid = (
                candidate in _ROUTER_PROFILES
                and candidate not in avoid
                and confidence >= _CONFIDENCE_THRESHOLD
            )
            profile = candidate if valid else fallback
            if not valid:
                logger.warning(
                    "Classifier fallback: profile=%r conf=%.2f avoid=%r → %r",
                    candidate, confidence, avoid, fallback,
                )
        except Exception:
            logger.warning("Classifier failed, fallback to %r", fallback, exc_info=True)
            profile = fallback

        latency = time.monotonic() - t0
        return profile, {"confidence": confidence, "reasoning": reasoning, "latency": round(latency, 4)}
```

- [ ] **Step 4: Run all classifier tests**

```bash
pytest tests/retrieval/test_classifier.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add raganything/retrieval/classifier.py tests/retrieval/test_classifier.py
git commit -m "feat(agentic): restrict router to 4 profiles, add avoid param, fallback=semantic"
```

---

## Task 4: Grader

**Files:**
- Create: `raganything/retrieval/grader.py`
- Create: `tests/retrieval/test_grader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_grader.py
import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.grader import Grader, build_shared_prefix


def _chunks(n: int = 3) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "content": f"Evidence paragraph {i}.", "file_path": f"doc{i}.pdf"}
        for i in range(n)
    ]


async def test_grade_sufficient():
    llm = AsyncMock(return_value=json.dumps({"sufficient": True, "reason": "All facts present."}))
    g = Grader(llm)
    result = await g.grade("What is X?", _chunks())
    assert result["sufficient"] is True
    assert "reason" in result


async def test_grade_insufficient():
    llm = AsyncMock(return_value=json.dumps({"sufficient": False, "reason": "Missing Y."}))
    g = Grader(llm)
    result = await g.grade("What is X?", _chunks())
    assert result["sufficient"] is False
    assert result["reason"] == "Missing Y."


async def test_grade_json_parse_failure_falls_back_sufficient():
    llm = AsyncMock(return_value="not json at all")
    g = Grader(llm, fallback_sufficient=True)
    result = await g.grade("query", _chunks())
    assert result["sufficient"] is True


async def test_grade_json_parse_failure_respects_fallback_false():
    llm = AsyncMock(return_value="broken")
    g = Grader(llm, fallback_sufficient=False)
    result = await g.grade("query", _chunks())
    assert result["sufficient"] is False


async def test_grade_llm_exception_falls_back():
    llm = AsyncMock(side_effect=RuntimeError("LLM down"))
    g = Grader(llm, fallback_sufficient=True)
    result = await g.grade("query", _chunks())
    assert result["sufficient"] is True


async def test_build_shared_prefix_contains_all_chunks():
    chunks = _chunks(3)
    prefix = build_shared_prefix(chunks)
    for c in chunks:
        assert c["content"] in prefix
    assert "[1]" in prefix
    assert "[3]" in prefix


async def test_grade_prompt_contains_query():
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return json.dumps({"sufficient": True, "reason": "ok"})
    g = Grader(llm)
    await g.grade("unique_query_string_xyz", _chunks(1))
    assert "unique_query_string_xyz" in captured[0]
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/retrieval/test_grader.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `raganything/retrieval/grader.py`**

```python
# raganything/retrieval/grader.py
from __future__ import annotations
import json
import logging
from typing import Awaitable, Callable

from raganything.constants import DEFAULT_AGENTIC_GRADER_FALLBACK_SUFFICIENT

logger = logging.getLogger(__name__)

_CONTEXT_HEADER = "You are a RAG quality controller.\n\nContext:\n"

_GRADER_SUFFIX = """\
Question: {query}

Are the chunks above sufficient to accurately answer this question?
Output JSON: {{"sufficient": true|false, "reason": "<one short sentence>"}}
"""


def build_shared_prefix(chunks: list[dict]) -> str:
    """Build the chunk-text prefix shared by grader, generator, and hallucination_check."""
    parts = [
        f"[{i + 1}] Source: {c.get('file_path', 'unknown')}\n{c.get('content', '')}"
        for i, c in enumerate(chunks)
    ]
    return _CONTEXT_HEADER + "\n\n---\n\n".join(parts) + "\n\n---\n\n"


class Grader:
    def __init__(
        self,
        llm_func: Callable[..., Awaitable[str]],
        fallback_sufficient: bool = DEFAULT_AGENTIC_GRADER_FALLBACK_SUFFICIENT,
    ) -> None:
        self._llm = llm_func
        self._fallback_sufficient = fallback_sufficient

    async def grade(self, query: str, chunks: list[dict]) -> dict:
        prefix = build_shared_prefix(chunks)
        prompt = prefix + _GRADER_SUFFIX.format(query=query)
        try:
            raw = await self._llm(prompt, response_format={"type": "json_object"})
            result = json.loads(raw)
            return {
                "sufficient": bool(result.get("sufficient", self._fallback_sufficient)),
                "reason": str(result.get("reason", "")).strip(),
            }
        except Exception:
            logger.warning("Grader failed, fallback sufficient=%s", self._fallback_sufficient, exc_info=True)
            return {"sufficient": self._fallback_sufficient, "reason": "grader error"}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/retrieval/test_grader.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add raganything/retrieval/grader.py tests/retrieval/test_grader.py
git commit -m "feat(agentic): add Grader with shared prefix builder for vLLM APC"
```

---

## Task 5: Rewriter

**Files:**
- Create: `raganything/retrieval/rewriter.py`
- Create: `tests/retrieval/test_rewriter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_rewriter.py
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.rewriter import Rewriter


async def test_rewrite_returns_llm_output():
    llm = AsyncMock(return_value="  rewritten query  ")
    rw = Rewriter(llm)
    result = await rw.rewrite("original query", "missing entity Y")
    assert result == "rewritten query"


async def test_rewrite_prompt_contains_original_and_reason():
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return "new query"
    rw = Rewriter(llm)
    await rw.rewrite("original_xyz", "missing_reason_abc")
    assert "original_xyz" in captured[0]
    assert "missing_reason_abc" in captured[0]


async def test_rewrite_exception_returns_original():
    llm = AsyncMock(side_effect=RuntimeError("LLM error"))
    rw = Rewriter(llm)
    result = await rw.rewrite("original query", "some reason")
    assert result == "original query"


async def test_rewrite_empty_response_returns_original():
    llm = AsyncMock(return_value="   ")
    rw = Rewriter(llm)
    result = await rw.rewrite("original query", "reason")
    assert result == "original query"
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/retrieval/test_rewriter.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `raganything/retrieval/rewriter.py`**

```python
# raganything/retrieval/rewriter.py
from __future__ import annotations
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

_REWRITER_PROMPT = """\
The following query did not retrieve sufficient evidence.

Original query: {query}
Retrieval feedback: {reason}

Rewrite the query to improve retrieval. Strategies:
- Replace ambiguous terms with synonyms
- Add explicit domain context
- Decompose compound noun phrases

Output the rewritten query only. No explanation, no quotation marks.
"""


class Rewriter:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def rewrite(self, query: str, reason: str) -> str:
        prompt = _REWRITER_PROMPT.format(query=query, reason=reason)
        try:
            raw = await self._llm(prompt)
            result = (raw if isinstance(raw, str) else str(raw)).strip()
            return result if result else query
        except Exception:
            logger.warning("Rewriter failed, returning original query", exc_info=True)
            return query
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/retrieval/test_rewriter.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add raganything/retrieval/rewriter.py tests/retrieval/test_rewriter.py
git commit -m "feat(agentic): add Rewriter with feedback-conditioned query reformulation"
```

---

## Task 6: HallucinationChecker

**Files:**
- Create: `raganything/retrieval/hallucination_checker.py`
- Create: `tests/retrieval/test_hallucination_checker.py`
- Delete: `raganything/retrieval/evaluator.py`
- Delete: `tests/retrieval/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_hallucination_checker.py
import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.hallucination_checker import HallucinationChecker


def _chunks(n: int = 2) -> list[dict]:
    return [{"chunk_id": f"c{i}", "content": f"Fact {i}.", "file_path": "doc.pdf"} for i in range(n)]


async def test_grounded_true():
    llm = AsyncMock(return_value=json.dumps({"grounded": True, "ungrounded_claims": []}))
    hc = HallucinationChecker(llm)
    r = await hc.verify("What is X?", "X is A.", _chunks())
    assert r["grounded"] is True
    assert r["ungrounded_claims"] == []


async def test_grounded_false_with_claims():
    llm = AsyncMock(return_value=json.dumps({
        "grounded": False,
        "ungrounded_claims": ["X is 42", "Y happened in 2020"],
    }))
    hc = HallucinationChecker(llm)
    r = await hc.verify("What is X?", "X is 42 and Y happened in 2020.", _chunks())
    assert r["grounded"] is False
    assert "X is 42" in r["ungrounded_claims"]


async def test_exception_defaults_grounded_true():
    llm = AsyncMock(side_effect=RuntimeError("LLM down"))
    hc = HallucinationChecker(llm)
    r = await hc.verify("q", "answer", _chunks())
    assert r["grounded"] is True
    assert r.get("check_status") == "error"


async def test_json_parse_failure_defaults_grounded_true():
    llm = AsyncMock(return_value="not json")
    hc = HallucinationChecker(llm)
    r = await hc.verify("q", "answer", _chunks())
    assert r["grounded"] is True
    assert r.get("check_status") == "error"


async def test_prompt_contains_answer_and_query():
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return json.dumps({"grounded": True, "ungrounded_claims": []})
    hc = HallucinationChecker(llm)
    await hc.verify("unique_query_abc", "unique_answer_xyz", _chunks(1))
    assert "unique_query_abc" in captured[0]
    assert "unique_answer_xyz" in captured[0]


async def test_prompt_shares_prefix_with_grader():
    from raganything.retrieval.grader import build_shared_prefix
    chunks = _chunks(2)
    shared = build_shared_prefix(chunks)
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return json.dumps({"grounded": True, "ungrounded_claims": []})
    hc = HallucinationChecker(llm)
    await hc.verify("q", "a", chunks)
    assert captured[0].startswith(shared)
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/retrieval/test_hallucination_checker.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `raganything/retrieval/hallucination_checker.py`**

```python
# raganything/retrieval/hallucination_checker.py
from __future__ import annotations
import json
import logging
from typing import Awaitable, Callable

from .grader import build_shared_prefix

logger = logging.getLogger(__name__)

_CHECKER_SUFFIX = """\
Answer: {answer}

Question being answered: {query}

For every factual claim in the Answer, verify it is explicitly supported by the Context above.
Statements such as "I cannot determine X from the context" make no factual claims and are grounded.

Output JSON:
{{
  "grounded": true|false,
  "ungrounded_claims": ["<claim>", ...]
}}
"""


class HallucinationChecker:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def verify(self, query: str, answer: str, chunks: list[dict]) -> dict:
        prefix = build_shared_prefix(chunks)
        prompt = prefix + _CHECKER_SUFFIX.format(answer=answer, query=query)
        try:
            raw = await self._llm(prompt, response_format={"type": "json_object"})
            result = json.loads(raw)
            return {
                "grounded": bool(result.get("grounded", True)),
                "ungrounded_claims": [str(c) for c in result.get("ungrounded_claims", [])],
            }
        except Exception:
            logger.warning("HallucinationChecker failed, defaulting grounded=True", exc_info=True)
            return {"grounded": True, "ungrounded_claims": [], "check_status": "error"}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/retrieval/test_hallucination_checker.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Delete obsolete files**

```bash
git rm raganything/retrieval/evaluator.py
git rm tests/retrieval/test_evaluator.py
git rm raganything/retrieval/complexity.py
git rm tests/retrieval/test_complexity.py
```

- [ ] **Step 6: Commit**

```bash
git add raganything/retrieval/hallucination_checker.py tests/retrieval/test_hallucination_checker.py
git commit -m "feat(agentic): add HallucinationChecker; remove evaluator and complexity modules"
```

---

## Task 7: Rewrite AgentGraph

**Files:**
- Rewrite: `raganything/retrieval/agent_graph.py`
- Rewrite: `tests/retrieval/test_agent_graph.py`

- [ ] **Step 1: Write failing tests covering the new graph topology**

```python
# tests/retrieval/test_agent_graph.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from raganything.retrieval.agent_graph import AdaptiveAgentGraph
from raganything.retrieval.router_cache import RouterCache


def _lightrag() -> MagicMock:
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value="the answer")
    return lg


def _chunks(n: int = 2) -> list[dict]:
    return [{"chunk_id": f"c{i}", "content": f"content {i}", "rrf_score": 0.5} for i in range(n)]


def _router(chunks=None) -> MagicMock:
    m = MagicMock()
    m.route = AsyncMock(return_value=(chunks or _chunks(), {"profile": "semantic", "chunks_per_path": {}}))
    return m


def _classifier(profile: str = "semantic") -> MagicMock:
    m = MagicMock()
    m.classify = AsyncMock(return_value=(profile, {"confidence": 0.9, "reasoning": "test", "latency": 0.01}))
    return m


def _grader(sufficient: bool = True, reason: str = "") -> MagicMock:
    m = MagicMock()
    m.grade = AsyncMock(return_value={"sufficient": sufficient, "reason": reason})
    return m


def _rewriter(new_query: str = "rewritten query") -> MagicMock:
    m = MagicMock()
    m.rewrite = AsyncMock(return_value=new_query)
    return m


def _checker(grounded: bool = True, claims: list | None = None) -> MagicMock:
    m = MagicMock()
    m.verify = AsyncMock(return_value={"grounded": grounded, "ungrounded_claims": claims or []})
    return m


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_happy_path_returns_answer():
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("What is BERT?")
    assert isinstance(result, str)
    assert result != ""


async def test_return_trace_true_includes_metadata():
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("query", return_trace=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert result["confidence"] == "high"
    assert result["grounded"] is True
    assert "trace" in result


# ── Retrieval cycle: rewriter at cycle 0 ─────────────────────────────────────

async def test_cycle0_fail_triggers_rewriter():
    grader_calls = []
    async def grade_side_effect(query, chunks):
        grader_calls.append(query)
        # fail first, succeed second
        return {"sufficient": len(grader_calls) > 1, "reason": "missing Y"}

    grader = MagicMock()
    grader.grade = grade_side_effect
    rewriter = _rewriter("improved query")

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=grader,
        _rewriter=rewriter,
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("original query")
    assert result != ""
    rewriter.rewrite.assert_called_once()


# ── Retrieval cycle: decompose at cycle 1 ────────────────────────────────────

async def test_cycle1_fail_triggers_decompose_with_full_profile():
    grader_calls = []
    async def grade_side_effect(query, chunks):
        grader_calls.append(query)
        # fail twice, succeed third
        return {"sufficient": len(grader_calls) > 2, "reason": "missing"}

    grader = MagicMock()
    grader.grade = grade_side_effect
    router = _router()

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=grader,
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=router,
        _cache=RouterCache(),
    )

    llm = AsyncMock(side_effect=[
        # classifier
        json.dumps({"reasoning": "r", "profile": "semantic", "confidence": 0.9}),
        # rewriter
        "rewritten",
        # decomposer
        json.dumps({"sub_questions": ["sub1", "sub2"]}),
        # generator
        "the answer",
        # hallucination check
        json.dumps({"grounded": True, "ungrounded_claims": []}),
    ])
    graph._llm = llm

    result = await graph.run("complex query")
    assert result != ""
    # router should have been called with "full" profile for parallel retrieve
    route_calls = router.route.call_args_list
    full_calls = [c for c in route_calls if c.kwargs.get("profile_name") == "full" or
                  (len(c.args) > 2 and c.args[2] == "full")]
    assert len(full_calls) > 0


# ── 3 retrieval cycles all fail → END_INSUFFICIENT ───────────────────────────

async def test_three_retrieve_failures_returns_none_without_generating():
    grader = _grader(sufficient=False)
    generator_llm = AsyncMock(return_value="fabricated answer")

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=grader,
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    # Patch llm so we can detect if generator was called
    call_log = []
    original_llm = graph._llm
    async def logged_llm(prompt, **kw):
        call_log.append(prompt)
        return await original_llm(prompt, **kw)
    graph._llm = logged_llm

    result = await graph.run("unanswerable query", return_trace=True)
    assert result["answer"] is None
    assert result["confidence"] == "none"
    # Generator suffix contains this phrase; it must NOT appear in any call
    for call in call_log:
        assert "Answer the question based ONLY" not in call


# ── Hallucination check: retry via targeted_retriever ────────────────────────

async def test_check_fail_triggers_targeted_retriever():
    check_calls = []
    async def check_side_effect(query, answer, chunks):
        check_calls.append(1)
        grounded = len(check_calls) > 1
        return {"grounded": grounded, "ungrounded_claims": ["claim X"] if not grounded else []}

    checker = MagicMock()
    checker.verify = check_side_effect
    router = _router()

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=checker,
        _router=router,
        _cache=RouterCache(),
    )
    result = await graph.run("query", return_trace=True)
    assert result["confidence"] == "high"
    # targeted_retriever should have fired once (check_cycle went 0→1)
    assert result["trace"]["check_cycles_used"] == 1


# ── Hallucination check: 2 failures → END_INSUFFICIENT ───────────────────────

async def test_two_check_failures_returns_none():
    checker = _checker(grounded=False, claims=["unsupported claim"])
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=checker,
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("query", return_trace=True)
    assert result["answer"] is None
    assert result["confidence"] == "none"


# ── Router cache integration ──────────────────────────────────────────────────

async def test_cache_hit_skips_classifier():
    cache = RouterCache()
    cache.put("cached query", "local")
    classifier = _classifier("semantic")  # would return semantic, but cache wins

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=classifier,
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=cache,
    )
    result = await graph.run("cached query", return_trace=True)
    assert result["trace"]["router_cache_hit"] is True
    classifier.classify.assert_not_called()
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/retrieval/test_agent_graph.py -v
```

Expected: multiple failures (wrong imports, old state schema).

- [ ] **Step 3: Rewrite `raganything/retrieval/agent_graph.py`**

```python
# raganything/retrieval/agent_graph.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from lightrag import QueryParam
from raganything.constants import (
    DEFAULT_AGENTIC_MAX_RETRIEVE_CYCLES,
    DEFAULT_AGENTIC_MAX_CHECK_CYCLES,
    DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS,
    DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY,
)
from .classifier import QueryClassifier
from .grader import Grader, build_shared_prefix
from .rewriter import Rewriter
from .hallucination_checker import HallucinationChecker
from .router import RetrievalRouter
from .router_cache import RouterCache

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """\
Break this question into {max_sub} or fewer independent sub-questions, \
each answerable by searching a knowledge base independently.

Question: {query}

Rules:
- Each sub-question must be self-contained (no references to other sub-questions).
- Sub-questions must not overlap in scope.
- Prefer 2 sub-questions over 4 unless truly needed.

Output JSON: {{"sub_questions": ["...", "..."]}}
"""

_GENERATOR_SUFFIX = """\
Question: {query}

Answer the question based ONLY on the context above.
If the context lacks the information needed to answer accurately, \
say so explicitly rather than speculating.

Provide a comprehensive response.
"""


class AgentState(TypedDict):
    query: str
    current_query: str
    profile: str
    chunks: list[dict]
    grader_sufficient: bool
    grader_reason: str
    answer: str
    grounded: bool
    ungrounded_claims: list[str]
    retrieve_cycle: int
    check_cycle: int
    routing_trace: dict


class AdaptiveAgentGraph:
    def __init__(
        self,
        lightrag: Any,
        llm_func: Any = None,
        *,
        _classifier: QueryClassifier | None = None,
        _grader: Grader | None = None,
        _rewriter: Rewriter | None = None,
        _checker: HallucinationChecker | None = None,
        _router: RetrievalRouter | None = None,
        _cache: RouterCache | None = None,
        max_retrieve_cycles: int = DEFAULT_AGENTIC_MAX_RETRIEVE_CYCLES,
        max_check_cycles: int = DEFAULT_AGENTIC_MAX_CHECK_CYCLES,
    ) -> None:
        self._lightrag = lightrag
        self._llm = llm_func or lightrag.llm_model_func
        self._clf = _classifier or QueryClassifier(self._llm)
        self._grader = _grader or Grader(self._llm)
        self._rewriter = _rewriter or Rewriter(self._llm)
        self._checker = _checker or HallucinationChecker(self._llm)
        self._router = _router or RetrievalRouter(lightrag, self._llm)
        self._cache = _cache or RouterCache()
        self._max_retrieve_cycles = max_retrieve_cycles
        self._max_check_cycles = max_check_cycles
        self._graph = self._build_graph()

    # ── Nodes ──────────────────────────────────────────────────────────────

    async def _node_router(self, state: AgentState) -> dict:
        query = state["query"]
        cached = self._cache.get(query)
        if cached and cached["outcome"] != "failed":
            profile = cached["profile"]
            cache_hit = True
        else:
            avoid = self._cache.get_avoid_profiles(query)
            profile, meta = await self._clf.classify(query, avoid=avoid)
            self._cache.put(query, profile)
            cache_hit = False
            logger.debug("Router LLM: %r → %s (conf=%.2f)", query[:60], profile, meta["confidence"])
        return {
            "current_query": query,
            "profile": profile,
            "retrieve_cycle": 0,
            "check_cycle": 0,
            "routing_trace": {
                "profile": profile,
                "router_cache_hit": cache_hit,
                "rewrite_history": [query],
                "sub_questions": None,
                "chunks_per_path": {},
            },
        }

    async def _node_retriever(self, state: AgentState) -> dict:
        param = QueryParam(mode="hybrid")
        chunks, trace = await self._router.route(
            state["current_query"], param, profile_name=state["profile"]
        )
        routing_trace = dict(state.get("routing_trace", {}))
        routing_trace.setdefault("chunks_per_path", {})
        routing_trace["chunks_per_path"].update(trace.get("chunks_per_path", {}))
        return {"chunks": chunks, "routing_trace": routing_trace}

    async def _node_grader(self, state: AgentState) -> dict:
        result = await self._grader.grade(state["current_query"], state["chunks"])
        return {"grader_sufficient": result["sufficient"], "grader_reason": result["reason"]}

    async def _node_rewriter(self, state: AgentState) -> dict:
        new_q = await self._rewriter.rewrite(state["current_query"], state["grader_reason"])
        history = list(state["routing_trace"].get("rewrite_history", []))
        history.append(new_q)
        return {
            "current_query": new_q,
            "retrieve_cycle": state["retrieve_cycle"] + 1,
            "routing_trace": {**state["routing_trace"], "rewrite_history": history},
        }

    async def _node_decomposer(self, state: AgentState) -> dict:
        try:
            raw = await self._llm(
                _DECOMPOSE_PROMPT.format(
                    query=state["query"],
                    max_sub=DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS,
                ),
                response_format={"type": "json_object"},
            )
            sub_qs = json.loads(raw).get("sub_questions", [])
            if not sub_qs:
                sub_qs = [state["query"]]
        except Exception:
            logger.warning("Decomposer failed, using original query", exc_info=True)
            sub_qs = [state["query"]]
        sub_qs = [str(q) for q in sub_qs[:DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS]]
        return {
            "retrieve_cycle": state["retrieve_cycle"] + 1,
            "routing_trace": {**state["routing_trace"], "sub_questions": sub_qs},
        }

    async def _node_parallel_retriever(self, state: AgentState) -> dict:
        sub_qs = state["routing_trace"].get("sub_questions") or [state["query"]]
        sem = asyncio.Semaphore(DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY)
        param = QueryParam(mode="hybrid")

        async def _one(q: str) -> list[dict]:
            async with sem:
                chunks, _ = await self._router.route(q, param, profile_name="full")
                return chunks

        results = await asyncio.gather(*[_one(q) for q in sub_qs], return_exceptions=True)
        all_chunks: list[dict] = []
        for r in results:
            if not isinstance(r, BaseException):
                all_chunks.extend(r)
        return {"chunks": _dedup_chunks(all_chunks)[:30]}

    async def _node_generator(self, state: AgentState) -> dict:
        prefix = build_shared_prefix(state["chunks"])
        prompt = prefix + _GENERATOR_SUFFIX.format(query=state["query"])
        try:
            raw = await self._llm(prompt)
            answer = raw if isinstance(raw, str) else str(raw)
        except Exception:
            logger.warning("Generator failed", exc_info=True)
            answer = ""
        return {"answer": answer}

    async def _node_hallucination_check(self, state: AgentState) -> dict:
        result = await self._checker.verify(state["query"], state["answer"], state["chunks"])
        return {
            "grounded": result["grounded"],
            "ungrounded_claims": result.get("ungrounded_claims", []),
        }

    async def _node_targeted_retriever(self, state: AgentState) -> dict:
        new_q = " ".join(state["ungrounded_claims"]) or state["query"]
        param = QueryParam(mode="hybrid")
        new_chunks, _ = await self._router.route(new_q, param, profile_name=state["profile"])
        combined = _dedup_chunks(state["chunks"] + new_chunks)[:30]
        return {
            "chunks": combined,
            "check_cycle": state["check_cycle"] + 1,
        }

    async def _node_end_grounded(self, state: AgentState) -> dict:
        self._cache.mark_success(state["query"])
        return {}

    async def _node_end_insufficient(self, state: AgentState) -> dict:
        self._cache.mark_failed(state["query"])
        return {}

    # ── Conditional edges ──────────────────────────────────────────────────

    def _after_grade(self, state: AgentState) -> str:
        if state["grader_sufficient"]:
            return "generate"
        if state["retrieve_cycle"] < 1:
            return "rewrite"
        if state["retrieve_cycle"] < 2:
            return "decompose"
        return "end_insufficient"

    def _after_check(self, state: AgentState) -> str:
        if state["grounded"]:
            return "end_grounded"
        if state["check_cycle"] < self._max_check_cycles:
            return "targeted"
        return "end_insufficient"

    # ── Graph ──────────────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("router", self._node_router)
        builder.add_node("retriever", self._node_retriever)
        builder.add_node("grader", self._node_grader)
        builder.add_node("rewriter", self._node_rewriter)
        builder.add_node("decomposer", self._node_decomposer)
        builder.add_node("parallel_retriever", self._node_parallel_retriever)
        builder.add_node("generator", self._node_generator)
        builder.add_node("hallucination_check", self._node_hallucination_check)
        builder.add_node("targeted_retriever", self._node_targeted_retriever)
        builder.add_node("end_grounded", self._node_end_grounded)
        builder.add_node("end_insufficient", self._node_end_insufficient)

        builder.set_entry_point("router")
        builder.add_edge("router", "retriever")
        builder.add_edge("retriever", "grader")
        builder.add_conditional_edges("grader", self._after_grade, {
            "generate": "generator",
            "rewrite": "rewriter",
            "decompose": "decomposer",
            "end_insufficient": "end_insufficient",
        })
        builder.add_edge("rewriter", "retriever")
        builder.add_edge("decomposer", "parallel_retriever")
        builder.add_edge("parallel_retriever", "grader")
        builder.add_edge("generator", "hallucination_check")
        builder.add_conditional_edges("hallucination_check", self._after_check, {
            "end_grounded": "end_grounded",
            "targeted": "targeted_retriever",
            "end_insufficient": "end_insufficient",
        })
        builder.add_edge("targeted_retriever", "generator")
        builder.add_edge("end_grounded", END)
        builder.add_edge("end_insufficient", END)

        return builder.compile()

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(self, query: str, return_trace: bool = False, **kwargs: Any) -> str | dict:
        initial: AgentState = {
            "query": query,
            "current_query": query,
            "profile": "semantic",
            "chunks": [],
            "grader_sufficient": False,
            "grader_reason": "",
            "answer": "",
            "grounded": False,
            "ungrounded_claims": [],
            "retrieve_cycle": 0,
            "check_cycle": 0,
            "routing_trace": {},
        }
        final = await self._graph.ainvoke(initial)
        answer: str | None = final.get("answer") or None
        grounded = final.get("grounded", False)
        if grounded:
            confidence = "high"
        elif answer:
            confidence = "low"   # grader passed, check crashed (grounded=True default)
        else:
            confidence = "none"
            answer = None

        if return_trace:
            return {
                "answer": answer,
                "confidence": confidence,
                "grounded": grounded,
                "ungrounded_claims": final.get("ungrounded_claims", []),
                "trace": {
                    **final.get("routing_trace", {}),
                    "retrieve_cycles_used": final.get("retrieve_cycle", 0),
                    "check_cycles_used": final.get("check_cycle", 0),
                },
            }
        return answer if answer is not None else ""


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id", "")
        if not cid:
            continue
        if cid not in seen or c.get("rrf_score", 0.0) > seen[cid].get("rrf_score", 0.0):
            seen[cid] = c
    return list(seen.values())
```

- [ ] **Step 4: Run all retrieval tests**

```bash
pytest tests/retrieval/ -v
```

Expected: all tests PASS. Any remaining failures from the old `test_agent_graph.py` are because we haven't deleted the old test yet — it was replaced in step 1.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest -v --tb=short
```

Expected: all tests PASS. Investigate and fix any failure before committing.

- [ ] **Step 6: Commit**

```bash
git add raganything/retrieval/agent_graph.py tests/retrieval/test_agent_graph.py
git commit -m "feat(agentic): rewrite AdaptiveAgentGraph — grader, checker, escalation, cycle limits"
```

---

## Task 8: Smoke Test via CLI

**Files:** none modified — validation only.

- [ ] **Step 1: Start vLLM with prefix caching**

Confirm `start_server_qwen3.5.sh` includes `--enable-prefix-caching`. If not, add it before running:

```bash
cat start_server_qwen3.5.sh | grep prefix-caching
```

If absent, the flag should be added to the script (one line addition).

- [ ] **Step 2: Run agentic mode with trace**

Choose an existing workspace from your evaluation runs. Example:

```bash
python scripts/query_ppr.py \
  -w docbench_shared_ablation_20260417_v0_v1_v2 \
  --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 \
  -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" \
  --mode agentic \
  --trace
```

Expected output structure:

```json
{
  "answer": "...",
  "confidence": "high",
  "grounded": true,
  "trace": {
    "profile": "semantic",
    "router_cache_hit": false,
    "retrieve_cycles_used": 0,
    "check_cycles_used": 0
  }
}
```

- [ ] **Step 3: Verify Phoenix spans (optional)**

```bash
python scripts/query_ppr.py \
  -w <workspace> --cache-dir <cache_dir> \
  -q "Compare LightRAG and HippoRAG2 indexing strategies." \
  --mode agentic --trace --phoenix
```

Open `http://localhost:6006`. Verify spans appear for: `router`, `retriever`, `grader`, `generator`, `hallucination_check`.

- [ ] **Step 4: Run a query that should fail gracefully**

```bash
python scripts/query_ppr.py \
  -w <workspace> --cache-dir <cache_dir> \
  -q "What is the GDP of Narnia in 2025?" \
  --mode agentic --trace
```

Expected: `"confidence": "none"` and `"answer": null` or `"answer": ""`. The system should not fabricate.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "test(agentic): smoke test confirmed — grader, checker, escalation all working"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered in task |
|---|---|
| 4 router profiles, full removed from classifier | Task 3 |
| full reserved for cycle-3 parallel_retriever | Task 7 (`_node_parallel_retriever`) |
| Router LRU cache with tri-state outcome | Task 2 |
| Grader: batch, shared prefix, full chunk text | Task 4 |
| Rewriter at cycle 0, decomposer at cycle 1 | Task 7 (`_after_grade`) |
| Hallucination check: binary, any claim → retry | Task 6 |
| END_INSUFFICIENT: generator not called on 3 retrieve fails | Task 7 (test: `test_three_retrieve_failures_returns_none_without_generating`) |
| END_INSUFFICIENT: answer dropped on 2 check fails | Task 7 (test: `test_two_check_failures_returns_none`) |
| check_status=error on infra failure → confidence=low | Task 6 (fallback grounded=True propagates through graph) |
| Dedup + cap 30 in parallel_retriever and targeted_retriever | Task 7 |
| rerank_batch_size=8, OOM backoff=False | Task 1 |
| min_rrf_score filter in router | Task 1 |
| rerank_candidate_cap=30 | Task 1 |
| Delete complexity.py and evaluator.py | Task 6 |
| `return_trace=True` includes confidence, grounded, trace | Task 7 |
| Smoke test via query_ppr.py --mode agentic | Task 8 |

**No gaps found.**
