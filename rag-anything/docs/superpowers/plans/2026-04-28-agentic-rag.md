# Adaptive Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `mode="agentic"` to RAGAnything that routes queries to simple/medium/complex execution tracks with self-reflection, iterative retrieval, and Arize Phoenix observability — without modifying any existing code.

**Architecture:** A LangGraph `StateGraph` sits behind a new `elif mode == "agentic"` branch in `query.py`. It classifies query complexity, then runs one of three tracks: simple (retrieve→generate), medium (+ evaluate→retry once), or complex (decompose→parallel retrieve→evaluate→retry up to twice). All existing retrieval primitives (`RetrievalRouter`, `run_path`) are reused as workers.

**Tech Stack:** `langgraph>=0.2`, `arize-phoenix[otel]>=4.0`, `opentelemetry-sdk`, `pytest`, `pytest-asyncio`, `unittest.mock`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Add `agentic` optional-dependency group |
| `raganything/observability.py` | Create | Phoenix OTEL init with graceful degradation |
| `raganything/retrieval/complexity.py` | Create | LLM-based 3-way complexity classifier |
| `raganything/retrieval/evaluator.py` | Create | LLM-based answer quality scorer (0-1) |
| `raganything/retrieval/agent_graph.py` | Create | LangGraph graph: state, nodes, edges |
| `raganything/query.py` | Modify | Add `mode="agentic"` elif branch only |
| `tests/test_observability.py` | Create | Graceful-degradation smoke test |
| `tests/retrieval/test_complexity.py` | Create | Classifier unit tests |
| `tests/retrieval/test_evaluator.py` | Create | Evaluator unit tests |
| `tests/retrieval/test_agent_graph.py` | Create | Graph track tests (simple/medium/complex) |
| `tests/test_agentic_integration.py` | Create | End-to-end smoke test via `aquery()` |

---

## Task 1: Add agentic dependencies

**Files:**
- Modify: `rag-anything/pyproject.toml`

- [ ] **Step 1: Add optional-dependency group**

In `pyproject.toml`, add after the existing `[project.optional-dependencies]` entries:

```toml
agentic = [
    "langgraph>=0.2",
    "arize-phoenix[otel]>=4.0",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
]
```

- [ ] **Step 2: Verify the file parses**

```bash
cd rag-anything && python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag-anything/pyproject.toml
git commit -m "chore(agentic): add agentic optional-dependency group"
```

---

## Task 2: Observability module

**Files:**
- Create: `rag-anything/raganything/observability.py`
- Create: `rag-anything/tests/test_observability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_observability.py
from raganything.observability import setup_phoenix


def test_setup_phoenix_is_idempotent_and_does_not_raise_without_phoenix(monkeypatch):
    """setup_phoenix must not raise even when arize-phoenix is not installed."""
    import sys
    # Simulate phoenix not installed
    monkeypatch.setitem(sys.modules, "phoenix", None)
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)
    # Should not raise
    setup_phoenix()
    setup_phoenix()  # second call: idempotent


def test_setup_phoenix_sets_initialized_flag():
    import raganything.observability as obs
    obs._phoenix_initialized = False
    setup_phoenix.__wrapped__ if hasattr(setup_phoenix, "__wrapped__") else None
    # After first call the flag must be set
    setup_phoenix()
    assert obs._phoenix_initialized is True
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd rag-anything && pytest tests/test_observability.py -v
```

Expected: `ImportError` — `raganything.observability` does not exist yet.

- [ ] **Step 3: Implement**

```python
# raganything/observability.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
_phoenix_initialized: bool = False


def setup_phoenix(project_name: str = "rag-agentic", port: int = 6006) -> None:
    """Initialize Arize Phoenix local OTEL tracing.

    Safe to call multiple times; initializes only once.
    If arize-phoenix is not installed, logs a warning and returns silently.
    """
    global _phoenix_initialized
    if _phoenix_initialized:
        return
    try:
        from phoenix.otel import register  # type: ignore[import]
        register(project_name=project_name, auto_instrument=True)
        _phoenix_initialized = True
        logger.info("Arize Phoenix initialized at http://localhost:%d", port)
    except (ImportError, TypeError):
        logger.warning(
            "arize-phoenix not installed; observability disabled. "
            "Install with: pip install 'arize-phoenix[otel]>=4.0'"
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd rag-anything && pytest tests/test_observability.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/observability.py rag-anything/tests/test_observability.py
git commit -m "feat(agentic): add Phoenix observability module with graceful degradation"
```

---

## Task 3: Complexity classifier

**Files:**
- Create: `rag-anything/raganything/retrieval/complexity.py`
- Create: `rag-anything/tests/retrieval/test_complexity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_complexity.py
import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.complexity import ComplexityClassifier

def _llm(response: str) -> AsyncMock:
    return AsyncMock(return_value=response)


async def test_returns_simple_on_high_confidence():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "one hop", "complexity": "simple", "confidence": 0.9})))
    level, meta = await clf.classify("What does BERT stand for?")
    assert level == "simple"
    assert meta["confidence"] == 0.9


async def test_returns_complex_on_high_confidence():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "multi-entity", "complexity": "complex", "confidence": 0.85})))
    level, _ = await clf.classify("Compare indexing in LightRAG vs HippoRAG.")
    assert level == "complex"


async def test_low_confidence_downgrades_to_medium():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "unsure", "complexity": "complex", "confidence": 0.4})))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_unknown_complexity_falls_back_to_medium():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "ok", "complexity": "impossible_value", "confidence": 0.9})))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_non_json_falls_back_to_medium():
    clf = ComplexityClassifier(_llm("Not JSON at all"))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_llm_exception_falls_back_to_medium():
    clf = ComplexityClassifier(AsyncMock(side_effect=RuntimeError("LLM down")))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_metadata_contains_latency():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "ok", "complexity": "simple", "confidence": 0.8})))
    _, meta = await clf.classify("test")
    assert "latency" in meta
    assert meta["latency"] >= 0.0
```

- [ ] **Step 2: Run to confirm all fail**

```bash
cd rag-anything && pytest tests/retrieval/test_complexity.py -v
```

Expected: `ImportError` — module does not exist yet.

- [ ] **Step 3: Implement**

```python
# raganything/retrieval/complexity.py
from __future__ import annotations
import json
import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6
_FALLBACK = "medium"

_PROMPT = """\
You are a query complexity classifier for a RAG system.
Classify the query into one complexity level:

- simple: Single-hop factual question; one retrieval pass is sufficient.
  Examples: "What does BERT stand for?", "What is the capital of France?"

- medium: Moderate depth; one entity or topic, may need one follow-up retrieval.
  Examples: "What are all the config options for the Redis module?",
            "Explain the attention mechanism in detail."

- complex: Multi-entity; requires decomposition, causal chains, or cross-document
  reasoning across multiple distinct entities.
  Examples: "How did the network partition in region A cause failures in region B?",
            "Compare the indexing strategies used by LightRAG and HippoRAG."

Rules:
- When unsure between simple and medium → choose medium.
- When unsure between medium and complex → choose medium.
- Only choose complex when multiple distinct entities clearly need cross-document reasoning.

Output JSON: {{"reasoning": "...", "complexity": "<simple|medium|complex>", "confidence": <0.0-1.0>}}

Query: {query}
"""


class ComplexityClassifier:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def classify(self, query: str) -> tuple[str, dict[str, Any]]:
        """Return (complexity, metadata). complexity is 'simple', 'medium', or 'complex'."""
        t0 = time.monotonic()
        complexity = _FALLBACK
        confidence = 0.0
        reasoning = ""
        try:
            raw = await self._llm(_PROMPT.format(query=query), response_format={"type": "json_object"})
            result = json.loads(raw)
            complexity = str(result.get("complexity", _FALLBACK)).strip()
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", ""))
            if complexity not in {"simple", "medium", "complex"}:
                complexity = _FALLBACK
            elif confidence < _CONFIDENCE_THRESHOLD:
                logger.warning("Low confidence %.2f for %r → %r", confidence, complexity, _FALLBACK)
                complexity = _FALLBACK
        except Exception:
            logger.warning("ComplexityClassifier failed, fallback to %r", _FALLBACK, exc_info=True)
            complexity = _FALLBACK
        return complexity, {"confidence": confidence, "reasoning": reasoning, "latency": round(time.monotonic() - t0, 4)}
```

- [ ] **Step 4: Run tests to confirm all pass**

```bash
cd rag-anything && pytest tests/retrieval/test_complexity.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/retrieval/complexity.py rag-anything/tests/retrieval/test_complexity.py
git commit -m "feat(agentic): add ComplexityClassifier (simple/medium/complex)"
```

---

## Task 4: Evaluator node

**Files:**
- Create: `rag-anything/raganything/retrieval/evaluator.py`
- Create: `rag-anything/tests/retrieval/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_evaluator.py
import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.evaluator import AnswerEvaluator


def _llm(response: str) -> AsyncMock:
    return AsyncMock(return_value=response)


async def test_high_score_with_no_gap():
    ev = AnswerEvaluator(_llm(json.dumps({"score": 0.92, "gap": ""})))
    result = await ev.evaluate("What is BERT?", "BERT is a transformer model.")
    assert result["score"] == 0.92
    assert result["gap"] == ""


async def test_low_score_with_gap():
    ev = AnswerEvaluator(_llm(json.dumps({"score": 0.45, "gap": "Missing info about training data"})))
    result = await ev.evaluate("How was BERT trained?", "BERT uses transformers.")
    assert result["score"] == 0.45
    assert "training" in result["gap"].lower()


async def test_score_clamped_to_0_1():
    ev = AnswerEvaluator(_llm(json.dumps({"score": 1.5, "gap": ""})))
    result = await ev.evaluate("q", "a")
    assert result["score"] == 1.0

    ev2 = AnswerEvaluator(_llm(json.dumps({"score": -0.3, "gap": "something"})))
    result2 = await ev2.evaluate("q", "a")
    assert result2["score"] == 0.0


async def test_non_json_returns_default_score():
    ev = AnswerEvaluator(_llm("Sorry, cannot evaluate."))
    result = await ev.evaluate("q", "a")
    assert result["score"] == 0.5
    assert isinstance(result["gap"], str)


async def test_llm_exception_returns_default():
    ev = AnswerEvaluator(AsyncMock(side_effect=RuntimeError("LLM down")))
    result = await ev.evaluate("q", "a")
    assert result["score"] == 0.5
```

- [ ] **Step 2: Run to confirm all fail**

```bash
cd rag-anything && pytest tests/retrieval/test_evaluator.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement**

```python
# raganything/retrieval/evaluator.py
from __future__ import annotations
import json
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_PROMPT = """\
Evaluate whether the generated answer adequately addresses the original question.

Original question: {query}

Generated answer:
{answer}

Score 0.0–1.0:
  0.9–1.0  Complete, accurate, well-supported
  0.7–0.9  Mostly complete, minor gaps
  0.5–0.7  Partial, notable gaps
  0.0–0.5  Incomplete or off-topic

If score < 0.7, describe in one sentence what specific information is missing.

Output JSON: {{"score": <float>, "gap": "<missing info or empty string>"}}
"""


class AnswerEvaluator:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def evaluate(self, query: str, answer: str) -> dict[str, Any]:
        """Return dict with keys: score (float 0–1), gap (str)."""
        try:
            raw = await self._llm(_PROMPT.format(query=query, answer=answer), response_format={"type": "json_object"})
            result = json.loads(raw)
            score = float(result.get("score", 0.5))
            gap = str(result.get("gap", "")).strip()
            return {"score": max(0.0, min(1.0, score)), "gap": gap}
        except Exception:
            logger.warning("AnswerEvaluator failed, returning default score", exc_info=True)
            return {"score": 0.5, "gap": "Evaluation failed; please verify the answer."}
```

- [ ] **Step 4: Run tests to confirm all pass**

```bash
cd rag-anything && pytest tests/retrieval/test_evaluator.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/retrieval/evaluator.py rag-anything/tests/retrieval/test_evaluator.py
git commit -m "feat(agentic): add AnswerEvaluator (0-1 score with gap description)"
```

---

## Task 5: Agent graph — full implementation

**Files:**
- Create: `rag-anything/raganything/retrieval/agent_graph.py`
- Create: `rag-anything/tests/retrieval/test_agent_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_agent_graph.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from raganything.retrieval.agent_graph import AdaptiveAgentGraph


def _lightrag(answer: str = "the answer") -> MagicMock:
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value=answer)
    return lg


def _chunks(n: int = 2) -> list[dict]:
    return [{"chunk_id": f"c{i}", "content": f"content {i}", "rrf_score": 0.5 + i * 0.1} for i in range(n)]


def _clf(complexity: str = "simple") -> MagicMock:
    m = MagicMock()
    m.classify = AsyncMock(return_value=(complexity, {"confidence": 0.9, "reasoning": "test", "latency": 0.01}))
    return m


def _router(chunks=None) -> MagicMock:
    m = MagicMock()
    m.route = AsyncMock(return_value=(chunks or _chunks(), {"profile": "semantic"}))
    return m


def _evaluator(score: float = 0.9, gap: str = "") -> MagicMock:
    m = MagicMock()
    m.evaluate = AsyncMock(return_value={"score": score, "gap": gap})
    return m


# ── Simple path ─────────────────────────────────────────────────────────────

async def test_simple_path_returns_answer():
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("simple"), _router=_router())
    result = await graph.run("What is BERT?")
    assert result == "the answer"


async def test_simple_path_does_not_call_evaluator():
    ev = _evaluator()
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("simple"), _router=_router(), _evaluator=ev)
    await graph.run("simple question")
    ev.evaluate.assert_not_called()


async def test_simple_path_calls_router_once():
    router = _router()
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("simple"), _router=router)
    await graph.run("simple question")
    assert router.route.call_count == 1


# ── Medium path ──────────────────────────────────────────────────────────────

async def test_medium_path_no_retry_when_eval_passes():
    router = _router()
    graph = AdaptiveAgentGraph(
        _lightrag(), _complexity_clf=_clf("medium"), _router=router, _evaluator=_evaluator(0.85)
    )
    result = await graph.run("Explain attention.")
    assert isinstance(result, str)
    assert router.route.call_count == 1


async def test_medium_path_retries_once_on_low_score():
    router = _router()
    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.4, "gap": "missing details about multi-head"},
        {"score": 0.82, "gap": ""},
    ])
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("medium"), _router=router, _evaluator=ev)
    result = await graph.run("What is multi-head attention?")
    assert isinstance(result, str)
    assert router.route.call_count == 2  # initial + 1 targeted


async def test_medium_path_stops_at_max_iter_even_if_eval_fails():
    router = _router()
    ev = _evaluator(0.2, "always bad")
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("medium"), _router=router, _evaluator=ev)
    await graph.run("hard medium question")
    assert router.route.call_count == 2  # initial + max 1 retry for medium


# ── Complex path ─────────────────────────────────────────────────────────────

async def test_complex_path_decomposes_and_parallel_retrieves():
    router = _router()
    # LLM: first call is decompose (returns JSON), subsequent calls are generate
    llm = AsyncMock(side_effect=[
        '{"sub_questions": ["sub1", "sub2"]}',
        "synthesized answer",
    ])
    lg = MagicMock()
    lg.llm_model_func = llm
    graph = AdaptiveAgentGraph(
        lg,
        _complexity_clf=_clf("complex"),
        _router=router,
        _evaluator=_evaluator(0.9),
    )
    result = await graph.run("Compare LightRAG and HippoRAG indexing.")
    assert isinstance(result, str)
    assert router.route.call_count == 2  # one per sub-question


async def test_complex_path_retries_once_targeted_not_full_decompose():
    route_calls = []

    async def fake_route(q, param, **kw):
        route_calls.append(q)
        return _chunks(), {}

    router = MagicMock()
    router.route = fake_route

    llm = AsyncMock(side_effect=[
        '{"sub_questions": ["q1", "q2"]}',  # decompose
        "partial answer",                    # generate round 1
        "better answer",                     # generate round 2
    ])
    lg = MagicMock()
    lg.llm_model_func = llm

    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.4, "gap": "missing region B details"},
        {"score": 0.85, "gap": ""},
    ])

    graph = AdaptiveAgentGraph(lg, _complexity_clf=_clf("complex"), _router=router, _evaluator=ev)
    await graph.run("Complex multi-hop question.")
    # 2 sub-questions + 1 targeted retry = 3 route calls total
    assert len(route_calls) == 3
    # targeted query should contain the eval_gap keyword
    assert "region B" in route_calls[2]


async def test_return_trace_includes_complexity_and_score():
    graph = AdaptiveAgentGraph(
        _lightrag(), _complexity_clf=_clf("simple"), _router=_router()
    )
    result = await graph.run("test", return_trace=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert result["trace"]["complexity"] == "simple"


# ── Chunk deduplication ───────────────────────────────────────────────────────

async def test_chunks_deduplicated_keeping_highest_rrf():
    """Second retrieval returns same chunk_id with lower score; dedup keeps first."""
    chunk_high = {"chunk_id": "c1", "content": "high quality", "rrf_score": 0.9}
    chunk_low = {"chunk_id": "c1", "content": "high quality", "rrf_score": 0.3}

    route_calls = [0]
    async def fake_route(q, param, **kw):
        route_calls[0] += 1
        if route_calls[0] == 1:
            return [chunk_high], {}
        return [chunk_low], {}

    router = MagicMock()
    router.route = fake_route

    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.4, "gap": "needs more"},
        {"score": 0.9, "gap": ""},
    ])

    lg = MagicMock()
    captured_prompts = []
    async def capture_llm(prompt, **kw):
        captured_prompts.append(prompt)
        return "answer"
    lg.llm_model_func = capture_llm

    graph = AdaptiveAgentGraph(lg, _complexity_clf=_clf("medium"), _router=router, _evaluator=ev)
    await graph.run("dedup test")
    # The generate prompt in round 2 should list each chunk_id only once
    last_prompt = captured_prompts[-1]
    assert last_prompt.count("c1") <= 2  # at most "[1] Source:" and content mention
```

- [ ] **Step 2: Run to confirm all fail**

```bash
cd rag-anything && pytest tests/retrieval/test_agent_graph.py -v
```

Expected: `ImportError` — `agent_graph` does not exist yet.

- [ ] **Step 3: Implement `agent_graph.py`**

```python
# raganything/retrieval/agent_graph.py
from __future__ import annotations

import asyncio
import json
import logging
import operator
from typing import Annotated, Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from lightrag import QueryParam
from .complexity import ComplexityClassifier
from .evaluator import AnswerEvaluator
from .router import RetrievalRouter

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SUBQUESTIONS = 3
MAX_ITER_BY_COMPLEXITY: dict[str, int] = {"simple": 0, "medium": 1, "complex": 2}
EVAL_PASS_THRESHOLD = 0.7

_DECOMPOSE_PROMPT = """\
Break this complex question into 2-4 independent sub-questions that can each be \
answered by searching a knowledge base independently.

Complex question: {query}

Rules:
- Each sub-question must be self-contained (no references to other sub-questions).
- Sub-questions must not overlap in scope.
- Prefer 2 sub-questions over 4 unless truly needed.

Output JSON: {{"sub_questions": ["...", "..."]}}
"""


class AgentState(TypedDict):
    query: str
    complexity: str
    sub_questions: list[str]
    retrieved_chunks: Annotated[list[dict], operator.add]
    answer: str
    eval_score: float
    eval_gap: str
    current_search_query: str
    iteration: int
    routing_trace: dict


class AdaptiveAgentGraph:
    def __init__(
        self,
        lightrag: Any,
        llm_func: Any = None,
        *,
        _complexity_clf: ComplexityClassifier | None = None,
        _evaluator: AnswerEvaluator | None = None,
        _router: RetrievalRouter | None = None,
    ) -> None:
        self._lightrag = lightrag
        self._llm = llm_func or lightrag.llm_model_func
        self._complexity_clf = _complexity_clf or ComplexityClassifier(self._llm)
        self._evaluator = _evaluator or AnswerEvaluator(self._llm)
        self._router = _router or RetrievalRouter(lightrag, self._llm)
        self._graph = self._build_graph()

    # ── Node implementations ────────────────────────────────────────────────

    async def _node_classify(self, state: AgentState) -> dict:
        complexity, meta = await self._complexity_clf.classify(state["query"])
        return {
            "complexity": complexity,
            "current_search_query": state["query"],
            "routing_trace": {"complexity": meta},
            "iteration": 0,
        }

    async def _node_retrieve(self, state: AgentState) -> dict:
        param = QueryParam(mode="hybrid")
        chunks, trace = await self._router.route(state["current_search_query"], param)
        return {
            "retrieved_chunks": chunks,
            "routing_trace": {**state.get("routing_trace", {}), "retrieve": trace},
        }

    async def _node_decompose(self, state: AgentState) -> dict:
        sub_questions = await self._decompose_query(state["query"])
        return {"sub_questions": sub_questions}

    async def _node_parallel_retrieve(self, state: AgentState) -> dict:
        sem = asyncio.Semaphore(MAX_CONCURRENT_SUBQUESTIONS)
        param = QueryParam(mode="hybrid")

        async def _one(sub_q: str) -> list[dict]:
            async with sem:
                chunks, _ = await self._router.route(sub_q, param)
                return chunks

        results = await asyncio.gather(*[_one(q) for q in state["sub_questions"]])
        all_chunks = [c for batch in results for c in batch]
        return {"retrieved_chunks": all_chunks}

    async def _node_generate(self, state: AgentState) -> dict:
        deduped = _dedup_chunks(state["retrieved_chunks"])
        answer = await self._generate_answer(state["query"], deduped)
        return {"answer": answer}

    async def _node_evaluate(self, state: AgentState) -> dict:
        result = await self._evaluator.evaluate(state["query"], state["answer"])
        return {"eval_score": result["score"], "eval_gap": result["gap"]}

    async def _node_targeted_retrieve(self, state: AgentState) -> dict:
        new_query = f"{state['query']} — supplementary retrieval: {state['eval_gap']}"
        param = QueryParam(mode="hybrid")
        chunks, trace = await self._router.route(new_query, param)
        iter_key = f"targeted_retrieve_{state['iteration']}"
        return {
            "retrieved_chunks": chunks,
            "current_search_query": new_query,
            "iteration": state["iteration"] + 1,
            "routing_trace": {**state.get("routing_trace", {}), iter_key: trace},
        }

    # ── Conditional edge ────────────────────────────────────────────────────

    def _should_retry(self, state: AgentState) -> str:
        max_iter = MAX_ITER_BY_COMPLEXITY.get(state["complexity"], 1)
        if state["eval_score"] >= EVAL_PASS_THRESHOLD or state["iteration"] >= max_iter:
            return "end"
        return "retry"

    # ── Graph construction ──────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("classify", self._node_classify)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("decompose", self._node_decompose)
        builder.add_node("parallel_retrieve", self._node_parallel_retrieve)
        builder.add_node("generate", self._node_generate)
        builder.add_node("evaluate", self._node_evaluate)
        builder.add_node("targeted_retrieve", self._node_targeted_retrieve)

        builder.set_entry_point("classify")

        builder.add_conditional_edges(
            "classify",
            lambda s: "decompose" if s["complexity"] == "complex" else "retrieve",
            {"decompose": "decompose", "retrieve": "retrieve"},
        )
        builder.add_edge("retrieve", "generate")
        builder.add_edge("decompose", "parallel_retrieve")
        builder.add_edge("parallel_retrieve", "generate")
        builder.add_conditional_edges(
            "generate",
            lambda s: "end" if s["complexity"] == "simple" else "evaluate",
            {"end": END, "evaluate": "evaluate"},
        )
        builder.add_conditional_edges(
            "evaluate",
            self._should_retry,
            {"end": END, "retry": "targeted_retrieve"},
        )
        builder.add_edge("targeted_retrieve", "generate")

        return builder.compile()

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(self, query: str, return_trace: bool = False, **kwargs: Any) -> str | dict:
        initial: AgentState = {
            "query": query,
            "complexity": "medium",
            "sub_questions": [],
            "retrieved_chunks": [],
            "answer": "",
            "eval_score": 0.0,
            "eval_gap": "",
            "current_search_query": query,
            "iteration": 0,
            "routing_trace": {},
        }
        final = await self._graph.ainvoke(initial)
        if return_trace:
            return {
                "answer": final["answer"],
                "trace": {
                    "complexity": final["complexity"],
                    "eval_score": final["eval_score"],
                    "iteration": final["iteration"],
                    "routing": final["routing_trace"],
                },
            }
        return final["answer"]

    # ── Helpers ─────────────────────────────────────────────────────────────

    async def _generate_answer(self, query: str, chunks: list[dict]) -> str:
        if not chunks:
            context = "No relevant information found."
        else:
            parts = [
                f"[{i + 1}] Source: {c.get('file_path', 'unknown')}\n{c.get('content', '')}"
                for i, c in enumerate(chunks)
            ]
            context = "\n\n---\n\n".join(parts)
        prompt = (
            f"Answer the following question based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a comprehensive response."
        )
        answer = await self._llm(prompt)
        return answer if isinstance(answer, str) else str(answer)

    async def _decompose_query(self, query: str) -> list[str]:
        try:
            raw = await self._llm(
                _DECOMPOSE_PROMPT.format(query=query),
                response_format={"type": "json_object"},
            )
            result = json.loads(raw)
            sub_qs = result.get("sub_questions", [])
            if not sub_qs or not isinstance(sub_qs, list):
                return [query]
            return [str(q) for q in sub_qs[:4]]
        except Exception:
            logger.warning("Decompose failed, using original query as single sub-question", exc_info=True)
            return [query]


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate by chunk_id, keeping the entry with the highest rrf_score."""
    seen: dict[str, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id", "")
        if not cid:
            continue
        if cid not in seen or c.get("rrf_score", 0.0) > seen[cid].get("rrf_score", 0.0):
            seen[cid] = c
    return list(seen.values())
```

- [ ] **Step 4: Run all agent graph tests**

```bash
cd rag-anything && pytest tests/retrieval/test_agent_graph.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/retrieval/agent_graph.py rag-anything/tests/retrieval/test_agent_graph.py
git commit -m "feat(agentic): implement AdaptiveAgentGraph with simple/medium/complex tracks"
```

---

## Task 6: query.py integration

**Files:**
- Modify: `rag-anything/raganything/query.py:266-294` (after the `auto` block, before `gfm`)
- Create: `rag-anything/tests/test_agentic_query_mode.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_agentic_query_mode.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_rag():
    """Minimal RAGAnything-like object with QueryMixin."""
    from raganything.raganything import RAGAnything
    rag = MagicMock(spec=RAGAnything)
    rag.lightrag = MagicMock()
    rag.lightrag.llm_model_func = AsyncMock(return_value="mocked answer")
    # Bind the real aquery method
    from raganything.query import QueryMixin
    rag.aquery = QueryMixin.aquery.__get__(rag)
    rag.logger = MagicMock()
    return rag


async def test_agentic_mode_returns_string():
    rag = _make_rag()
    with patch("raganything.query.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value="agentic answer")
        result = await rag.aquery("test query", mode="agentic")
    assert result == "agentic answer"
    MockGraph.assert_called_once_with(rag.lightrag)


async def test_agentic_mode_passes_return_trace():
    rag = _make_rag()
    with patch("raganything.query.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value={"answer": "x", "trace": {}})
        result = await rag.aquery("test query", mode="agentic", return_trace=True)
    assert result == {"answer": "x", "trace": {}}
    instance.run.assert_called_once_with("test query", return_trace=True)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd rag-anything && pytest tests/test_agentic_query_mode.py -v
```

Expected: `FAILED` — `agentic` mode raises `ValueError("Unknown mode agentic")` or `AttributeError`.

- [ ] **Step 3: Add the `mode="agentic"` branch to `query.py`**

Locate the line in `raganything/query.py` at approximately line 294 (right after the `# ── end mode="auto"` comment, before `# ── mode="gfm"`):

```python
        # ── end mode="auto" ───────────────────────────────────────────────

        # ── mode="agentic": adaptive LangGraph agent ─────────────────────
        if mode == "agentic":
            from raganything.retrieval.agent_graph import AdaptiveAgentGraph
            return_trace_agentic = bool(kwargs.pop("return_trace", False))
            graph = AdaptiveAgentGraph(self.lightrag)
            return await graph.run(query, return_trace=return_trace_agentic)
        # ── end mode="agentic" ────────────────────────────────────────────

        # ── mode="gfm": GFM graph neural retrieval ────────────────────────
```

- [ ] **Step 4: Run the integration tests**

```bash
cd rag-anything && pytest tests/test_agentic_query_mode.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Confirm existing tests are unaffected**

```bash
cd rag-anything && pytest tests/ -v --ignore=tests/test_agentic_integration.py -x
```

Expected: all existing tests pass; no regressions.

- [ ] **Step 6: Commit**

```bash
git add rag-anything/raganything/query.py rag-anything/tests/test_agentic_query_mode.py
git commit -m "feat(agentic): wire mode='agentic' into query.py"
```

---

## Task 7: Integration smoke test

**Files:**
- Create: `rag-anything/tests/test_agentic_integration.py`

- [ ] **Step 1: Write integration smoke test**

This test spins up the full `AdaptiveAgentGraph` with a real (mocked-at-LLM-boundary) LightRAG. It verifies the complete graph executes end-to-end without exceptions for all three tracks.

```python
# tests/test_agentic_integration.py
"""
End-to-end smoke tests for AdaptiveAgentGraph.
LLM and retrieval are mocked at their boundaries; graph logic is real.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from raganything.retrieval.agent_graph import AdaptiveAgentGraph
from raganything.retrieval.complexity import ComplexityClassifier
from raganything.retrieval.evaluator import AnswerEvaluator


def _chunks(n=3):
    return [{"chunk_id": f"c{i}", "content": f"relevant content {i}", "rrf_score": 0.6 + i * 0.05} for i in range(n)]


def _lightrag():
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value="integrated answer")
    return lg


def _router(chunks=None):
    m = MagicMock()
    m.route = AsyncMock(return_value=(chunks or _chunks(), {"profile": "semantic", "confidence": 0.88}))
    return m


async def test_full_simple_track():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("simple", {"confidence": 0.95, "reasoning": "one hop", "latency": 0.01}))
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=_router())
    result = await graph.run("What does BERT stand for?")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_full_medium_track_with_retry():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("medium", {"confidence": 0.8, "reasoning": "moderate", "latency": 0.02}))
    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.55, "gap": "missing information about layer normalization"},
        {"score": 0.88, "gap": ""},
    ])
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=_router(), _evaluator=ev)
    result = await graph.run("Explain BERT's architecture in detail.")
    assert isinstance(result, str)
    assert ev.evaluate.call_count == 2  # failed once, passed second time


async def test_full_complex_track():
    llm_responses = [
        json.dumps({"sub_questions": ["What is BERT?", "What is GPT?"]}),
        "synthesized comparison answer",
    ]
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(side_effect=llm_responses)

    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("complex", {"confidence": 0.87, "reasoning": "multi-entity", "latency": 0.03}))
    ev = MagicMock()
    ev.evaluate = AsyncMock(return_value={"score": 0.92, "gap": ""})

    graph = AdaptiveAgentGraph(lg, _complexity_clf=clf, _router=_router(), _evaluator=ev)
    result = await graph.run("Compare BERT and GPT architectures.")
    assert isinstance(result, str)


async def test_return_trace_structure():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("simple", {"confidence": 0.9, "reasoning": "ok", "latency": 0.01}))
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=_router())
    result = await graph.run("test", return_trace=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "trace" in result
    assert result["trace"]["complexity"] == "simple"
    assert "eval_score" in result["trace"]
    assert "iteration" in result["trace"]
    assert "routing" in result["trace"]


async def test_empty_retrieval_does_not_crash():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("simple", {"confidence": 0.9, "reasoning": "ok", "latency": 0.01}))
    router = MagicMock()
    router.route = AsyncMock(return_value=([], {}))  # empty retrieval
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=router)
    result = await graph.run("question with no matching docs")
    assert isinstance(result, str)  # must not raise
```

- [ ] **Step 2: Run integration tests**

```bash
cd rag-anything && pytest tests/test_agentic_integration.py -v
```

Expected: `5 passed`

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
cd rag-anything && pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/tests/test_agentic_integration.py
git commit -m "test(agentic): add integration smoke tests for all three graph tracks"
```

---

## Self-Review Checklist

### Spec coverage

| Spec section | Covered by task |
|---|---|
| mode="agentic" neue branch | Task 6 |
| Simple/medium/complex tracks | Task 5 |
| AgentState with Reducer | Task 5 |
| ComplexityClassifier | Task 3 |
| AnswerEvaluator | Task 4 |
| targeted_retrieve (not re-decompose) | Task 5 — `_node_targeted_retrieve` |
| current_search_query field | Task 5 — AgentState |
| retrieved_chunks Reducer (append) | Task 5 — `Annotated[list[dict], operator.add]` |
| Dedup chunks by rrf_score | Task 5 — `_dedup_chunks()` |
| Outer semaphore for parallel sub-questions | Task 5 — `asyncio.Semaphore(MAX_CONCURRENT_SUBQUESTIONS)` |
| asyncio context propagation note | Implemented via `asyncio.gather` (no `create_task`) |
| MAX_ITER per complexity | Task 5 — `MAX_ITER_BY_COMPLEXITY` dict read in `_should_retry` |
| Phoenix observability | Task 2 |
| Dependencies | Task 1 |
| Zero modification to existing code | All tasks use new files; query.py gets 4-line elif only |

### Type consistency check

- `AgentState.retrieved_chunks` → `Annotated[list[dict], operator.add]` — used as `list[dict]` in all nodes ✅
- `ComplexityClassifier.classify` → `tuple[str, dict]` — consumed in `_node_classify` ✅
- `AnswerEvaluator.evaluate` → `dict[str, Any]` with keys `score`, `gap` — consumed in `_node_evaluate` ✅
- `RetrievalRouter.route` → `tuple[list[dict], dict]` — consumed in all retrieve nodes ✅
- `AdaptiveAgentGraph.run` → `str | dict` — tested in Task 7 ✅
