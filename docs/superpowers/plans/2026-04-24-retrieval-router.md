# Retrieval Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `mode="auto"` to `aquery()` that routes queries to the optimal retrieval profile using an LLM classifier, parallel multi-path recall, weighted RRF fusion, reranking, and threshold filtering.

**Architecture:** An independent `RetrievalRouter` class in `raganything/retrieval/` handles query classification, parallel path execution, and chunk fusion. The `aquery()` and `aquery_vlm_enhanced()` methods in `query.py` gain a `mode="auto"` branch that delegates to this router and pops a `profile=` kwarg to allow explicit profile selection (bypassing the LLM classifier). All other modes remain untouched.

**Tech Stack:** Python 3.10+, `asyncio`, `pytest-asyncio`, `unittest.mock`. Depends on `lightrag.operate._rrf_merge` (indirectly, via our own weighted variant), `lightrag.utils.apply_rerank_if_enabled`, `lightrag.QueryParam`, `lightrag.LightRAG.aquery_data`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `raganything/retrieval/__init__.py` | Package exports |
| Create | `raganything/retrieval/profiles.py` | `RetrievalProfile` dataclass + `PROFILE_REGISTRY` |
| Create | `raganything/retrieval/classifier.py` | `QueryClassifier` — LLM call → profile name |
| Create | `raganything/retrieval/paths.py` | `run_path()` — one retrieval path via `aquery_data` |
| Create | `raganything/retrieval/router.py` | `RetrievalRouter`, `_weighted_rrf_merge`, `RetrievalError` |
| Modify | `raganything/query.py` | `aquery()` + `aquery_vlm_enhanced()` `mode="auto"` branches |
| Create | `tests/__init__.py` | pytest package marker |
| Create | `tests/retrieval/__init__.py` | pytest package marker |
| Create | `tests/retrieval/test_profiles.py` | Profile dataclass + registry tests |
| Create | `tests/retrieval/test_classifier.py` | Classifier parse/fallback tests |
| Create | `tests/retrieval/test_paths.py` | Per-path retrieval tests |
| Create | `tests/retrieval/test_router.py` | Router fusion + trace tests |
| Create | `tests/retrieval/test_query_integration.py` | `mode="auto"` wire-up tests |
| Create | `tests/conftest.py` | `asyncio_mode = "auto"` pytest config |

---

## Task 1: RetrievalProfile dataclass + registry

**Files:**
- Create: `raganything/retrieval/__init__.py`
- Create: `raganything/retrieval/profiles.py`
- Create: `tests/__init__.py`
- Create: `tests/retrieval/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/retrieval/test_profiles.py`

- [ ] **Step 1.1: Create test file**

```python
# tests/retrieval/test_profiles.py
from raganything.retrieval.profiles import (
    RetrievalProfile,
    PROFILE_REGISTRY,
    KNOWN_PATHS,
)


def test_all_builtin_profiles_present():
    assert set(PROFILE_REGISTRY.keys()) == {"precise", "local", "multihop", "descriptive", "full"}


def test_profile_paths_are_known():
    for profile in PROFILE_REGISTRY.values():
        for path in profile.paths:
            assert path in KNOWN_PATHS, f"Profile '{profile.name}' uses unknown path '{path}'"


def test_profile_rrf_weights_cover_all_paths():
    for profile in PROFILE_REGISTRY.values():
        for path in profile.paths:
            assert path in profile.rrf_weights, (
                f"Profile '{profile.name}' path '{path}' missing from rrf_weights"
            )


def test_full_profile_has_semaphore():
    assert PROFILE_REGISTRY["full"].max_concurrent_paths == 3


def test_simple_profiles_have_no_semaphore():
    for name in ("precise", "local", "multihop", "descriptive"):
        assert PROFILE_REGISTRY[name].max_concurrent_paths is None


def test_descriptive_path_overrides():
    profile = PROFILE_REGISTRY["descriptive"]
    assert "mix" in profile.path_overrides
    assert profile.path_overrides["mix"]["kg_chunk_selection_source"] == "untruncated"


def test_profile_defaults():
    p = RetrievalProfile(
        name="test",
        description="test",
        paths=["naive"],
        rrf_weights={"naive": 1.0},
    )
    assert p.rrf_k == 60
    assert p.enable_rerank is True
    assert p.min_rerank_score == 0.3
    assert p.rerank_candidate_cap == 60
    assert p.max_concurrent_paths is None
    assert p.path_overrides == {}
```

- [ ] **Step 1.2: Run test to verify it fails**

```
cd rag-anything
pytest tests/retrieval/test_profiles.py -v
```

Expected: `ModuleNotFoundError: No module named 'raganything.retrieval'`

- [ ] **Step 1.3: Create package init files**

```python
# tests/__init__.py
# (empty)
```

```python
# tests/retrieval/__init__.py
# (empty)
```

```ini
# tests/conftest.py
import pytest

# Enable asyncio mode globally so all async test functions work without decoration
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 1.4: Create `raganything/retrieval/__init__.py`**

```python
# raganything/retrieval/__init__.py
from .profiles import RetrievalProfile, PROFILE_REGISTRY, KNOWN_PATHS
from .classifier import QueryClassifier
from .paths import run_path
from .router import RetrievalRouter, RetrievalError

__all__ = [
    "RetrievalProfile",
    "PROFILE_REGISTRY",
    "KNOWN_PATHS",
    "QueryClassifier",
    "run_path",
    "RetrievalRouter",
    "RetrievalError",
]
```

- [ ] **Step 1.5: Create `raganything/retrieval/profiles.py`**

```python
# raganything/retrieval/profiles.py
from dataclasses import dataclass, field


@dataclass
class RetrievalProfile:
    name: str
    description: str
    paths: list[str]
    rrf_weights: dict[str, float]
    rrf_k: int = 60
    enable_rerank: bool = True
    min_rerank_score: float = 0.3
    rerank_candidate_cap: int = 60
    max_concurrent_paths: int | None = None
    path_overrides: dict[str, dict] = field(default_factory=dict)


KNOWN_PATHS: frozenset[str] = frozenset(
    ["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse"]
)

PROFILE_REGISTRY: dict[str, RetrievalProfile] = {
    p.name: p
    for p in [
        RetrievalProfile(
            name="precise",
            description="Exact character-level match queries (error codes, IDs, rare proper nouns)",
            paths=["qdrant_sparse"],
            rrf_weights={"qdrant_sparse": 1.0},
        ),
        RetrievalProfile(
            name="local",
            description="Direct query targeting a specific entity or clear single-hop fact",
            paths=["hybrid", "naive"],
            rrf_weights={"hybrid": 1.0, "naive": 1.0},
        ),
        RetrievalProfile(
            name="multihop",
            description="Chain reasoning across multiple entities or documents",
            paths=["ppr", "hybrid"],
            rrf_weights={"ppr": 1.0, "hybrid": 0.8},
        ),
        RetrievalProfile(
            name="descriptive",
            description="Open-ended question requiring broad, complete context",
            paths=["mix", "qdrant_hybrid"],
            rrf_weights={"mix": 1.0, "qdrant_hybrid": 0.8},
            path_overrides={
                "mix": {
                    "kg_chunk_selection_source": "untruncated",
                    "answer_context_mode": "kg_prompt",
                },
                "qdrant_hybrid": {
                    "kg_chunk_selection_source": "untruncated",
                    "answer_context_mode": "kg_prompt",
                },
            },
        ),
        RetrievalProfile(
            name="full",
            description="Fallback when query type is unclear or ambiguous",
            paths=["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse"],
            rrf_weights={
                p: 1.0
                for p in ["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse"]
            },
            max_concurrent_paths=3,
        ),
    ]
}
```

- [ ] **Step 1.6: Run test to verify it passes**

```
pytest tests/retrieval/test_profiles.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 1.7: Commit**

```bash
git add raganything/retrieval/__init__.py raganything/retrieval/profiles.py \
        tests/__init__.py tests/retrieval/__init__.py tests/conftest.py \
        tests/retrieval/test_profiles.py pyproject.toml
git commit -m "feat: add RetrievalProfile dataclass and built-in profile registry"
```

---

## Task 2: QueryClassifier

**Files:**
- Create: `raganything/retrieval/classifier.py`
- Test: `tests/retrieval/test_classifier.py`

- [ ] **Step 2.1: Write failing tests**

```python
# tests/retrieval/test_classifier.py
import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.classifier import QueryClassifier


async def _make_llm(response_str: str) -> AsyncMock:
    mock = AsyncMock(return_value=response_str)
    return mock


async def test_valid_classification():
    llm = await _make_llm(json.dumps({
        "reasoning": "clear factual query",
        "profile": "local",
        "confidence": 0.9,
    }))
    clf = QueryClassifier(llm)
    name, meta = await clf.classify("How many parameters does BERT have?")
    assert name == "local"
    assert meta["confidence"] == 0.9
    assert "reasoning" in meta
    assert meta["latency"] >= 0.0


async def test_low_confidence_falls_back_to_full():
    llm = await _make_llm(json.dumps({
        "reasoning": "unsure",
        "profile": "local",
        "confidence": 0.4,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("some ambiguous query")
    assert name == "full"


async def test_unknown_profile_falls_back_to_full():
    llm = await _make_llm(json.dumps({
        "reasoning": "ok",
        "profile": "nonexistent_profile",
        "confidence": 0.95,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "full"


async def test_non_json_output_falls_back_to_full():
    llm = await _make_llm("Sorry, I cannot classify this query.")
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "full"


async def test_missing_profile_key_falls_back_to_full():
    llm = await _make_llm(json.dumps({"reasoning": "ok", "confidence": 0.9}))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    # missing "profile" key → default "full" (json.get returns "full")
    # but confidence=0.9 >= threshold, and "full" IS in registry → returns "full"
    assert name == "full"


async def test_llm_exception_falls_back_to_full():
    llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "full"
```

- [ ] **Step 2.2: Run test to verify it fails**

```
pytest tests/retrieval/test_classifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'raganything.retrieval.classifier'`

- [ ] **Step 2.3: Create `raganything/retrieval/classifier.py`**

```python
# raganything/retrieval/classifier.py
import json
import logging
import time
from typing import Any, Awaitable, Callable

from .profiles import PROFILE_REGISTRY

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6

_CLASSIFIER_PROMPT = """\
You are a retrieval routing classifier. Given a user query, select the most
appropriate retrieval profile from the list below.

Available profiles and typical examples:

- precise: Exact character-level match queries (error codes, IDs, rare proper nouns)
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
Output format: {{"reasoning": "...", "profile": "<name>", "confidence": <0.0-1.0>}}

Query: {query}
"""


class QueryClassifier:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]):
        self._llm = llm_func

    async def classify(self, query: str) -> tuple[str, dict[str, Any]]:
        """Classify query into a profile name.

        Returns:
            (profile_name, metadata) where metadata contains confidence,
            reasoning, and latency_seconds.
        """
        t0 = time.monotonic()
        profile = "full"
        confidence = 0.0
        reasoning = ""
        try:
            prompt = _CLASSIFIER_PROMPT.format(query=query)
            raw = await self._llm(
                prompt,
                response_format={"type": "json_object"},
            )
            result = json.loads(raw)
            profile = str(result.get("profile", "full")).strip()
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", ""))
            if confidence < _CONFIDENCE_THRESHOLD or profile not in PROFILE_REGISTRY:
                logger.warning(
                    "Classifier fallback: profile=%r confidence=%.2f → 'full'",
                    profile,
                    confidence,
                )
                profile = "full"
        except Exception:
            logger.warning("Classifier output parse failed, fallback to 'full'", exc_info=True)
            profile = "full"
        latency = time.monotonic() - t0
        return profile, {
            "confidence": confidence,
            "reasoning": reasoning,
            "latency": round(latency, 4),
        }
```

- [ ] **Step 2.4: Run test to verify it passes**

```
pytest tests/retrieval/test_classifier.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 2.5: Commit**

```bash
git add raganything/retrieval/classifier.py tests/retrieval/test_classifier.py
git commit -m "feat: add QueryClassifier with few-shot prompt and safe fallback"
```

---

## Task 3: Per-path retrieval (paths.py)

**Files:**
- Create: `raganything/retrieval/paths.py`
- Test: `tests/retrieval/test_paths.py`

- [ ] **Step 3.1: Write failing tests**

```python
# tests/retrieval/test_paths.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from lightrag import QueryParam
from raganything.retrieval.paths import run_path, _PATH_CONFIG


def _make_lightrag(chunks: list[dict]) -> MagicMock:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(return_value={
        "status": "success",
        "data": {"chunks": chunks, "entities": [], "relations": []},
    })
    return lightrag


async def test_naive_path_uses_naive_mode():
    lg = _make_lightrag([{"chunk_id": "c1", "content": "hello", "file_path": "a.pdf"}])
    param = QueryParam(mode="hybrid")  # initial mode should be overridden
    chunks, latency = await run_path("naive", "test query", param, lg, overrides={})
    call_param = lg.aquery_data.call_args[0][1]
    assert call_param.mode == "naive"
    assert len(chunks) == 1
    assert latency >= 0.0


async def test_qdrant_sparse_sets_bm25_mode():
    lg = _make_lightrag([])
    param = QueryParam(mode="naive")
    await run_path("qdrant_sparse", "test", param, lg, overrides={})
    call_param = lg.aquery_data.call_args[0][1]
    assert call_param.mode == "naive"
    assert call_param.qdrant_retrieval_mode == "bm25"


async def test_qdrant_hybrid_sets_hybrid_qdrant_mode():
    lg = _make_lightrag([])
    param = QueryParam(mode="naive")
    await run_path("qdrant_hybrid", "test", param, lg, overrides={})
    call_param = lg.aquery_data.call_args[0][1]
    assert call_param.mode == "hybrid"
    assert call_param.qdrant_retrieval_mode == "hybrid"


async def test_overrides_applied():
    lg = _make_lightrag([])
    param = QueryParam(mode="naive")
    await run_path("mix", "test", param, lg, overrides={"kg_chunk_selection_source": "untruncated"})
    call_param = lg.aquery_data.call_args[0][1]
    assert call_param.kg_chunk_selection_source == "untruncated"


async def test_original_param_not_mutated():
    lg = _make_lightrag([])
    param = QueryParam(mode="hybrid", top_k=5)
    await run_path("naive", "test", param, lg, overrides={})
    assert param.mode == "hybrid"
    assert param.top_k == 5


async def test_failed_query_returns_empty_chunks():
    lg = MagicMock()
    lg.aquery_data = AsyncMock(return_value={"status": "failure", "data": {}})
    param = QueryParam(mode="naive")
    chunks, _ = await run_path("naive", "test", param, lg, overrides={})
    assert chunks == []


async def test_unknown_path_raises():
    lg = MagicMock()
    param = QueryParam(mode="naive")
    with pytest.raises(ValueError, match="Unknown path"):
        await run_path("nonexistent", "test", param, lg, overrides={})


def test_all_known_paths_configured():
    from raganything.retrieval.profiles import KNOWN_PATHS
    assert set(_PATH_CONFIG.keys()) == KNOWN_PATHS
```

- [ ] **Step 3.2: Run test to verify it fails**

```
pytest tests/retrieval/test_paths.py -v
```

Expected: `ModuleNotFoundError: No module named 'raganything.retrieval.paths'`

- [ ] **Step 3.3: Create `raganything/retrieval/paths.py`**

```python
# raganything/retrieval/paths.py
import logging
import time
from dataclasses import replace

logger = logging.getLogger(__name__)

# path_name → (lightrag_mode, qdrant_retrieval_mode_override_or_None)
_PATH_CONFIG: dict[str, tuple[str, str | None]] = {
    "naive":         ("naive",   None),
    "hybrid":        ("hybrid",  None),
    "mix":           ("mix",     None),
    "ppr":           ("ppr",     None),
    "qdrant_hybrid": ("hybrid",  "hybrid"),
    "qdrant_sparse": ("naive",   "bm25"),
}


async def run_path(
    name: str,
    query: str,
    param,           # QueryParam — never mutated
    lightrag,        # LightRAG instance
    overrides: dict,
) -> tuple[list[dict], float]:
    """Execute one retrieval path.

    Returns:
        (chunks, latency_seconds). chunks is an empty list on failure.
    """
    if name not in _PATH_CONFIG:
        raise ValueError(f"Unknown path: {name!r}")

    mode, qdrant_mode = _PATH_CONFIG[name]

    # Build an isolated param copy; never mutate the caller's param
    path_param = replace(param, mode=mode)
    if qdrant_mode is not None:
        path_param = replace(path_param, qdrant_retrieval_mode=qdrant_mode)
    for k, v in overrides.items():
        path_param = replace(path_param, **{k: v})

    t0 = time.monotonic()
    result = await lightrag.aquery_data(query, path_param)
    latency = time.monotonic() - t0

    chunks: list[dict] = []
    if result and result.get("status") == "success":
        chunks = result.get("data", {}).get("chunks", [])

    logger.debug("Path '%s': %d chunks in %.3fs", name, len(chunks), latency)
    return chunks, latency
```

- [ ] **Step 3.4: Run test to verify it passes**

```
pytest tests/retrieval/test_paths.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 3.5: Commit**

```bash
git add raganything/retrieval/paths.py tests/retrieval/test_paths.py
git commit -m "feat: add run_path per-path retrieval wrapper"
```

---

## Task 4: RetrievalRouter + weighted RRF fusion

**Files:**
- Create: `raganything/retrieval/router.py`
- Test: `tests/retrieval/test_router.py`

- [ ] **Step 4.1: Write failing tests**

```python
# tests/retrieval/test_router.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lightrag import QueryParam
from raganything.retrieval.router import RetrievalRouter, RetrievalError, _weighted_rrf_merge
from raganything.retrieval.profiles import PROFILE_REGISTRY


# ── _weighted_rrf_merge unit tests ──────────────────────────────────────────

def test_weighted_rrf_deduplicates_by_chunk_id():
    path_a = [{"chunk_id": "c1", "content": "a"}, {"chunk_id": "c2", "content": "b"}]
    path_b = [{"chunk_id": "c1", "content": "a"}, {"chunk_id": "c3", "content": "c"}]
    result = _weighted_rrf_merge({"a": path_a, "b": path_b}, {"a": 1.0, "b": 1.0}, k=60)
    ids = [c["chunk_id"] for c in result]
    assert len(ids) == len(set(ids)), "Duplicates not removed"
    assert set(ids) == {"c1", "c2", "c3"}


def test_weighted_rrf_chunk_in_two_paths_ranks_higher():
    path_a = [{"chunk_id": "shared", "content": "x"}, {"chunk_id": "only_a", "content": "y"}]
    path_b = [{"chunk_id": "shared", "content": "x"}, {"chunk_id": "only_b", "content": "z"}]
    result = _weighted_rrf_merge({"a": path_a, "b": path_b}, {"a": 1.0, "b": 1.0}, k=60)
    top_id = result[0]["chunk_id"]
    assert top_id == "shared", "Chunk appearing in both paths should rank first"


def test_weighted_rrf_respects_weights():
    path_a = [{"chunk_id": "c1", "content": "a"}]
    path_b = [{"chunk_id": "c2", "content": "b"}]
    result = _weighted_rrf_merge({"a": path_a, "b": path_b}, {"a": 2.0, "b": 1.0}, k=60)
    assert result[0]["chunk_id"] == "c1"  # higher weight → higher score


def test_weighted_rrf_attaches_rrf_score():
    chunks = [{"chunk_id": "c1", "content": "hello"}]
    result = _weighted_rrf_merge({"a": chunks}, {"a": 1.0}, k=60)
    assert "rrf_score" in result[0]
    assert result[0]["rrf_score"] > 0.0


# ── RetrievalRouter integration tests ───────────────────────────────────────

def _make_router(profile_chunks: dict[str, list[dict]]):
    """Build a router whose paths return pre-defined chunks."""
    lightrag = MagicMock()
    lightrag.llm_model_func = AsyncMock()

    async def fake_aquery_data(query, param):
        name_map = {
            ("naive",  "dense"):  "naive",
            ("naive",  "bm25"):   "qdrant_sparse",
            ("hybrid", "dense"):  "hybrid",
            ("hybrid", "hybrid"): "qdrant_hybrid",
            ("mix",    "dense"):  "mix",
            ("ppr",    "dense"):  "ppr",
        }
        key = (param.mode, param.qdrant_retrieval_mode)
        path_name = name_map.get(key, param.mode)
        return {
            "status": "success",
            "data": {"chunks": profile_chunks.get(path_name, []), "entities": [], "relations": []},
        }

    lightrag.aquery_data = fake_aquery_data

    # Stub apply_rerank_if_enabled to return input unchanged (no reranker in unit tests)
    return lightrag


async def test_router_returns_chunks_and_trace():
    chunks = [{"chunk_id": f"c{i}", "content": f"text {i}", "file_path": "f.pdf"} for i in range(5)]
    lightrag = _make_router({"hybrid": chunks[:3], "naive": chunks[3:]})

    with patch("raganything.retrieval.router.apply_rerank_if_enabled", new=AsyncMock(side_effect=lambda **kw: kw["retrieved_docs"])):
        router = RetrievalRouter(lightrag, llm_func=AsyncMock())
        param = QueryParam(mode="hybrid", chunk_top_k=10)
        final_chunks, trace = await router.route("test query", param, profile_name="local")

    assert isinstance(final_chunks, list)
    assert isinstance(trace, dict)
    assert trace["profile"] == "local"
    assert "latency_per_path" in trace
    assert "classifier" in trace["latency_per_path"]


async def test_router_trace_has_all_fields():
    lightrag = _make_router({"hybrid": [{"chunk_id": "c1", "content": "hi", "file_path": "f"}]})
    with patch("raganything.retrieval.router.apply_rerank_if_enabled", new=AsyncMock(side_effect=lambda **kw: kw["retrieved_docs"])):
        router = RetrievalRouter(lightrag, llm_func=AsyncMock())
        param = QueryParam(mode="hybrid", chunk_top_k=10)
        _, trace = await router.route("q", param, profile_name="local")

    required = {
        "profile", "confidence", "reasoning", "paths_activated",
        "paths_failed", "chunks_per_path", "chunks_after_rrf",
        "chunks_after_rerank", "chunks_after_threshold", "latency_per_path",
    }
    assert required.issubset(trace.keys()), f"Missing trace keys: {required - trace.keys()}"


async def test_router_all_paths_fail_raises():
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(side_effect=RuntimeError("DB down"))
    with patch("raganything.retrieval.router.apply_rerank_if_enabled", new=AsyncMock()):
        router = RetrievalRouter(lightrag, llm_func=AsyncMock())
        param = QueryParam(mode="hybrid", chunk_top_k=10)
        with pytest.raises(RetrievalError):
            await router.route("q", param, profile_name="local")


async def test_router_explicit_profile_skips_classifier():
    classifier_called = []
    lightrag = _make_router({"hybrid": [{"chunk_id": "c1", "content": "x", "file_path": "f"}]})
    with patch("raganything.retrieval.router.apply_rerank_if_enabled", new=AsyncMock(side_effect=lambda **kw: kw["retrieved_docs"])):
        router = RetrievalRouter(lightrag, llm_func=AsyncMock())
        # Patch classify to detect if it's called
        router._classifier.classify = AsyncMock(side_effect=lambda q: classifier_called.append(q))
        param = QueryParam(mode="hybrid", chunk_top_k=10)
        await router.route("q", param, profile_name="local")

    assert classifier_called == [], "Classifier should not be called when profile_name is explicit"
```

- [ ] **Step 4.2: Run test to verify it fails**

```
pytest tests/retrieval/test_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'raganything.retrieval.router'`

- [ ] **Step 4.3: Create `raganything/retrieval/router.py`**

```python
# raganything/retrieval/router.py
import asyncio
import logging
from dataclasses import asdict

from lightrag.utils import apply_rerank_if_enabled

from .classifier import QueryClassifier
from .paths import run_path
from .profiles import PROFILE_REGISTRY, RetrievalProfile

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    pass


class RetrievalRouter:
    def __init__(self, lightrag, llm_func=None):
        self._lightrag = lightrag
        self._classifier = QueryClassifier(llm_func or lightrag.llm_model_func)

    async def route(
        self,
        query: str,
        param,                          # QueryParam — not mutated
        profile_name: str | None = None,
    ) -> tuple[list[dict], dict]:
        """Run routing → parallel retrieval → RRF → rerank → threshold.

        Returns:
            (final_chunks, routing_trace)
        """
        # 1. Select profile
        if profile_name is not None:
            profile = PROFILE_REGISTRY.get(profile_name)
            if profile is None:
                raise ValueError(f"Unknown profile: {profile_name!r}")
            classifier_meta = {
                "confidence": 1.0,
                "reasoning": "explicit override",
                "latency": 0.0,
            }
        else:
            profile_name, classifier_meta = await self._classifier.classify(query)
            profile = PROFILE_REGISTRY[profile_name]

        # 2. Parallel path execution with optional semaphore
        sem = (
            asyncio.Semaphore(profile.max_concurrent_paths)
            if profile.max_concurrent_paths
            else None
        )

        async def _guarded(name: str):
            overrides = profile.path_overrides.get(name, {})
            if sem:
                async with sem:
                    return name, await run_path(name, query, param, self._lightrag, overrides)
            return name, await run_path(name, query, param, self._lightrag, overrides)

        results = await asyncio.gather(
            *[_guarded(n) for n in profile.paths],
            return_exceptions=True,
        )

        # 3. Collect results; skip failed paths
        chunks_by_path: dict[str, list[dict]] = {}
        latency_by_path: dict[str, float] = {}
        failed_paths: list[str] = []

        for item in results:
            if isinstance(item, BaseException):
                logger.warning("Retrieval path exception: %s", item)
                continue
            name, (chunks, latency) = item
            chunks_by_path[name] = chunks
            latency_by_path[name] = round(latency, 3)

        for name in profile.paths:
            if name not in chunks_by_path:
                failed_paths.append(name)

        if not chunks_by_path:
            raise RetrievalError("All retrieval paths failed")

        # 4. Weighted RRF — ranked lists enter intact (no pre-dedup)
        merged = _weighted_rrf_merge(
            {n: chunks_by_path[n] for n in profile.paths if n in chunks_by_path},
            profile.rrf_weights,
            profile.rrf_k,
        )
        chunks_after_rrf = len(merged)

        # 5. Rerank (capped at rerank_candidate_cap)
        candidate_pool = merged[: profile.rerank_candidate_cap]
        global_config = asdict(self._lightrag)

        if profile.enable_rerank:
            reranked = await apply_rerank_if_enabled(
                query=query,
                retrieved_docs=candidate_pool,
                global_config=global_config,
                enable_rerank=True,
                top_n=None,
                item_label="chunks",
            )
        else:
            reranked = candidate_pool
        chunks_after_rerank = len(reranked)

        # 6. Threshold filter
        if profile.enable_rerank:
            filtered = [
                c for c in reranked
                if c.get("rerank_score", 1.0) >= profile.min_rerank_score
            ]
        else:
            filtered = reranked

        # 7. Final top-k
        chunk_top_k = getattr(param, "chunk_top_k", 10)
        final_chunks = filtered[:chunk_top_k]

        routing_trace = {
            "profile": profile_name,
            "confidence": classifier_meta["confidence"],
            "reasoning": classifier_meta["reasoning"],
            "paths_activated": list(chunks_by_path.keys()),
            "paths_failed": failed_paths,
            "chunks_per_path": {n: len(c) for n, c in chunks_by_path.items()},
            "chunks_after_rrf": chunks_after_rrf,
            "chunks_after_rerank": chunks_after_rerank,
            "chunks_after_threshold": len(final_chunks),
            "latency_per_path": {
                "classifier": round(classifier_meta["latency"], 3),
                **latency_by_path,
            },
        }

        return final_chunks, routing_trace


def _weighted_rrf_merge(
    chunks_by_path: dict[str, list[dict]],
    weights: dict[str, float],
    k: int,
) -> list[dict]:
    """Weighted RRF: Score(d) = Σ_p weight_p * 1/(k + rank(d,p)).

    Each path's ranked list is passed intact so that a chunk appearing
    in multiple paths accumulates cross-path rank signals.
    Output is deduplicated and sorted by descending score.
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for path_name, ranked in chunks_by_path.items():
        w = weights.get(path_name, 1.0)
        for rank, chunk in enumerate(ranked):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (k + rank + 1)
            if chunk_id not in meta:
                meta[chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    result = []
    for cid in sorted_ids:
        chunk = dict(meta[cid])
        chunk["rrf_score"] = round(scores[cid], 6)
        result.append(chunk)
    return result
```

- [ ] **Step 4.4: Run test to verify it passes**

```
pytest tests/retrieval/test_router.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 4.5: Run full test suite to check no regressions**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4.6: Commit**

```bash
git add raganything/retrieval/router.py tests/retrieval/test_router.py
git commit -m "feat: add RetrievalRouter with weighted RRF fusion and rerank pipeline"
```

---

## Task 5: Wire mode="auto" into aquery() (non-VLM path)

**Files:**
- Modify: `raganything/query.py` (lines ~214–334)
- Test: `tests/retrieval/test_query_integration.py`

- [ ] **Step 5.1: Write failing test**

```python
# tests/retrieval/test_query_integration.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lightrag import QueryParam
from raganything.query import QueryMixin


def _make_mixin(chunks: list[dict]) -> QueryMixin:
    """Build a minimal QueryMixin stand-in."""
    mixin = MagicMock(spec=QueryMixin)
    mixin.lightrag = MagicMock()
    mixin.lightrag.llm_model_func = AsyncMock(return_value="answer text")
    mixin.logger = MagicMock()
    mixin.callback_manager = None

    async def fake_ensure_initialized():
        return {"success": True}

    mixin._ensure_lightrag_initialized = fake_ensure_initialized
    return mixin


async def test_aquery_auto_mode_calls_router():
    mixin = _make_mixin([])
    router_mock = MagicMock()
    router_mock.route = AsyncMock(return_value=(
        [{"chunk_id": "c1", "content": "answer chunk", "file_path": "f.pdf"}],
        {"profile": "local", "confidence": 0.9, "reasoning": "r",
         "paths_activated": ["hybrid"], "paths_failed": [],
         "chunks_per_path": {"hybrid": 1}, "chunks_after_rrf": 1,
         "chunks_after_rerank": 1, "chunks_after_threshold": 1,
         "latency_per_path": {"classifier": 0.1, "hybrid": 0.3}},
    ))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        result = await QueryMixin.aquery(mixin, "test query", mode="auto")

    router_mock.route.assert_called_once()
    assert isinstance(result, str)


async def test_aquery_auto_mode_passes_profile_kwarg():
    mixin = _make_mixin([])
    router_mock = MagicMock()
    router_mock.route = AsyncMock(return_value=([], {
        "profile": "precise", "confidence": 1.0, "reasoning": "",
        "paths_activated": [], "paths_failed": [],
        "chunks_per_path": {}, "chunks_after_rrf": 0,
        "chunks_after_rerank": 0, "chunks_after_threshold": 0,
        "latency_per_path": {"classifier": 0.0},
    }))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        await QueryMixin.aquery(mixin, "CVE-2026-001", mode="auto", profile="precise")

    _, call_kwargs = router_mock.route.call_args
    assert call_kwargs.get("profile_name") == "precise"


async def test_aquery_non_auto_mode_unchanged():
    """Non-auto modes must not touch the router at all."""
    mixin = _make_mixin([])
    mixin.lightrag.aquery = AsyncMock(return_value="legacy answer")

    with patch("raganything.query.RetrievalRouter") as router_cls:
        result = await QueryMixin.aquery(mixin, "test", mode="hybrid")

    router_cls.assert_not_called()
```

- [ ] **Step 5.2: Run test to verify it fails**

```
pytest tests/retrieval/test_query_integration.py -v
```

Expected: `AssertionError` or `TypeError` — router is not wired yet

- [ ] **Step 5.3: Modify `raganything/query.py` — add imports and `mode="auto"` branch**

At the top of `query.py`, add:
```python
from raganything.retrieval import RetrievalRouter
```

Inside `QueryMixin.aquery()`, after the `kwargs.setdefault(...)` defaults block and before the `vlm_enhanced` / `return_trace` pops, add:

```python
        # ── mode="auto": delegate to RetrievalRouter ──────────────────────
        if mode == "auto":
            profile_name: str | None = kwargs.pop("profile", None)
            return_trace_auto = bool(kwargs.pop("return_trace", False))
            query_param_auto = QueryParam(mode="hybrid", **kwargs)  # mode ignored by router
            router = RetrievalRouter(self.lightrag)
            final_chunks, routing_trace = await router.route(
                query,
                query_param_auto,
                profile_name=profile_name,
            )
            answer = await self._generate_answer_from_chunks(
                query,
                final_chunks,
                system_prompt=system_prompt,
                response_type=query_param_auto.response_type,
            )
            if return_trace_auto:
                return {"answer": answer, "trace": {"routing": routing_trace}}
            return answer
        # ── end mode="auto" ───────────────────────────────────────────────
```

Also add the `_generate_answer_from_chunks` helper method to `QueryMixin`:

```python
    async def _generate_answer_from_chunks(
        self,
        query: str,
        chunks: list[dict],
        *,
        system_prompt: str | None,
        response_type: str = "Multiple Paragraphs",
    ) -> str:
        """Build a simple RAG prompt from final chunks and call LLM."""
        if not chunks:
            context = "No relevant information found."
        else:
            parts = []
            for i, chunk in enumerate(chunks, 1):
                fp = chunk.get("file_path", "unknown")
                content = chunk.get("content", "")
                parts.append(f"[{i}] Source: {fp}\n{content}")
            context = "\n\n---\n\n".join(parts)

        prompt = (
            f"Answer the following question based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a {response_type} response."
        )
        llm_model_func = getattr(self.lightrag, "llm_model_func", None)
        if llm_model_func is None:
            raise ValueError("LightRAG llm_model_func is not available")
        answer = await llm_model_func(prompt, system_prompt=system_prompt)
        return answer if isinstance(answer, str) else str(answer)
```

- [ ] **Step 5.4: Run test to verify it passes**

```
pytest tests/retrieval/test_query_integration.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5.5: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5.6: Commit**

```bash
git add raganything/query.py tests/retrieval/test_query_integration.py
git commit -m "feat: wire mode=auto into aquery() with RetrievalRouter delegation"
```

---

## Task 6: VLM integration for mode="auto"

**Files:**
- Modify: `raganything/query.py` — `aquery_vlm_enhanced()`
- Test: append to `tests/retrieval/test_query_integration.py`

- [ ] **Step 6.1: Write failing test**

Add to `tests/retrieval/test_query_integration.py`:

```python
async def test_aquery_vlm_enhanced_auto_mode_uses_router():
    """VLM enhanced + mode=auto: router provides chunks, then image dereference runs."""
    mixin = _make_mixin([])
    mixin.vision_model_func = AsyncMock(return_value="vlm answer")

    test_chunks = [
        {"chunk_id": "c1", "content": "Image Path: /data/img.jpg\nsome text", "file_path": "f.pdf"}
    ]
    router_mock = MagicMock()
    router_mock.route = AsyncMock(return_value=(test_chunks, {
        "profile": "descriptive", "confidence": 0.8, "reasoning": "r",
        "paths_activated": ["mix"], "paths_failed": [],
        "chunks_per_path": {"mix": 1}, "chunks_after_rrf": 1,
        "chunks_after_rerank": 1, "chunks_after_threshold": 1,
        "latency_per_path": {"classifier": 0.2, "mix": 0.5},
    }))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        with patch.object(mixin, "_process_image_paths_for_vlm",
                          new=AsyncMock(return_value=("processed prompt", []))):
            with patch.object(mixin, "_generate_text_answer_from_retrieval_prompt",
                              new=AsyncMock(return_value="text answer")):
                result = await QueryMixin.aquery_vlm_enhanced(
                    mixin, "describe image", mode="auto"
                )

    router_mock.route.assert_called_once()
    assert isinstance(result, str)
```

- [ ] **Step 6.2: Run test to verify it fails**

```
pytest tests/retrieval/test_query_integration.py::test_aquery_vlm_enhanced_auto_mode_uses_router -v
```

Expected: FAIL — VLM enhanced doesn't handle `mode="auto"` yet

- [ ] **Step 6.3: Modify `aquery_vlm_enhanced()` in `query.py`**

At the start of `aquery_vlm_enhanced()`, after the VLM availability check and `_ensure_lightrag_initialized`, add:

```python
        # ── mode="auto": get chunks from router, then run standard VLM dereference ──
        if mode == "auto":
            profile_name: str | None = kwargs.pop("profile", None)
            kwargs.setdefault("multimodal_top_k", DEFAULT_MULTIMODAL_TOP_K)
            kwargs.setdefault("rerank_score_scope", "all")
            kwargs.setdefault("qdrant_retrieval_mode", DEFAULT_QDRANT_RETRIEVAL_MODE)
            query_param_auto = QueryParam(mode="hybrid", **kwargs)
            image_cap = query_param_auto.multimodal_top_k or DEFAULT_MULTIMODAL_TOP_K

            router = RetrievalRouter(self.lightrag)
            final_chunks, routing_trace = await router.route(
                query,
                query_param_auto,
                profile_name=profile_name,
            )

            # Build context string from chunks; _process_image_paths_for_vlm
            # scans it for "Image Path:" lines → base64 dereference
            context_str = "\n\n---\n\n".join(
                c.get("content", "") for c in final_chunks
            )

            enhanced_prompt, images_base64 = await self._process_image_paths_for_vlm(
                context_str, max_images=image_cap
            )

            if not images_base64:
                answer = await self._generate_text_answer_from_retrieval_prompt(
                    enhanced_prompt,
                    query,
                    system_prompt=system_prompt,
                    conversation_history=query_param_auto.conversation_history,
                    history_summary=query_param_auto.history_summary,
                )
                if return_trace:
                    return {"answer": answer, "trace": {"routing": routing_trace}}
                return answer

            messages = self._build_vlm_messages_with_images(
                enhanced_prompt,
                query,
                system_prompt,
                images_base64=images_base64,
                conversation_history=query_param_auto.conversation_history,
                history_summary=query_param_auto.history_summary,
            )
            result = await self._call_vlm_with_multimodal_content(messages)
            if return_trace:
                return {"answer": result, "trace": {"routing": routing_trace}}
            return result
        # ── end mode="auto" ───────────────────────────────────────────────
```

- [ ] **Step 6.4: Run test to verify it passes**

```
pytest tests/retrieval/test_query_integration.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 6.5: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6.6: Commit**

```bash
git add raganything/query.py tests/retrieval/test_query_integration.py
git commit -m "feat: wire mode=auto into aquery_vlm_enhanced with router chunk dereference"
```

---

## Task 7: return_trace routing field end-to-end

**Files:**
- Test: append to `tests/retrieval/test_query_integration.py`

- [ ] **Step 7.1: Write failing test**

Add to `tests/retrieval/test_query_integration.py`:

```python
async def test_aquery_auto_return_trace_includes_routing():
    mixin = _make_mixin([])
    router_mock = MagicMock()
    routing_trace = {
        "profile": "local",
        "confidence": 0.9,
        "reasoning": "factual query",
        "paths_activated": ["hybrid", "naive"],
        "paths_failed": [],
        "chunks_per_path": {"hybrid": 3, "naive": 2},
        "chunks_after_rrf": 4,
        "chunks_after_rerank": 3,
        "chunks_after_threshold": 3,
        "latency_per_path": {"classifier": 0.12, "hybrid": 0.45, "naive": 0.08},
    }
    router_mock.route = AsyncMock(return_value=([], routing_trace))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        result = await QueryMixin.aquery(
            mixin, "test query", mode="auto", return_trace=True
        )

    assert isinstance(result, dict)
    assert "answer" in result
    assert "trace" in result
    assert result["trace"]["routing"]["profile"] == "local"
    assert "latency_per_path" in result["trace"]["routing"]
    assert result["trace"]["routing"]["latency_per_path"]["classifier"] == 0.12
```

- [ ] **Step 7.2: Run test to verify it fails**

```
pytest tests/retrieval/test_query_integration.py::test_aquery_auto_return_trace_includes_routing -v
```

Expected: FAIL — trace structure not yet verified

- [ ] **Step 7.3: Run test after Task 5 code is in place**

The `mode="auto"` branch already returns `{"answer": ..., "trace": {"routing": routing_trace}}` when `return_trace=True`. This test should PASS without additional code changes.

```
pytest tests/retrieval/test_query_integration.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 7.4: Run full test suite one final time**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 7.5: Final commit**

```bash
git add tests/retrieval/test_query_integration.py
git commit -m "test: verify routing trace latency_per_path in return_trace output"
```

---

## Self-Review Against Spec

| Spec Section | Covered by Task |
|---|---|
| 3.1 Module structure (`retrieval/` package) | Task 1 |
| 3.2 Call chain (classifier → profile → paths → RRF → rerank → threshold) | Tasks 2–4 |
| 3.3 `mode="auto"` entry point, no mutation of existing modes | Task 5 |
| 3.4 VLM dereference: router returns chunks, caller runs `_process_image_paths_for_vlm` | Task 6 |
| 4.1 `RetrievalProfile` dataclass with all fields including `max_concurrent_paths` | Task 1 |
| 4.2 Five built-in profiles with correct overrides | Task 1 |
| 4.3 `profile=` kwarg bypasses classifier | Tasks 5–6 |
| 5.2 English few-shot prompt | Task 2 |
| 5.3 JSON parse fallback → `"full"` | Task 2 |
| 5.4 Classifier caching | ⚠ Not implemented — noted below |
| 6.3 Semaphore for `full` profile | Task 4 |
| 6.4 RRF ordering: ranked lists enter intact, no pre-dedup | Task 4 |
| 7. Error handling (single-path failure, all-paths failure) | Task 4 |
| 8. Routing trace with `latency_per_path` | Tasks 4 + 7 |
| 9. Unit tests for classifier fallbacks, integration tests with explicit profile | Tasks 2 + 4 |

**⚠ Gap: Classifier caching (Spec §5.4)** — The spec says classifier results should be cached via `llm_response_cache`. This is left as a follow-up; the classifier is functionally correct without caching and adding cache integration would require wiring the LightRAG KV store into `QueryClassifier`, which is a non-trivial change best done after end-to-end validation.
