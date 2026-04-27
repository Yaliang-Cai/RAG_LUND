# GFM-RAG Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate GFM-RAG as a new `"gfm"` retrieval path in RetrievalRouter and as a standalone `mode="gfm"` in `aquery()`, backed by a one-time offline export of the LightRAG Neo4j graph to GFM-RAG CSV format.

**Architecture:** GFMRetrieverWrapper is a lazy singleton that wraps `gfmrag.GFMRetriever`; it is called by a special-case branch in `run_path()` (instead of the usual `lightrag.aquery_data`) and by a new `mode="gfm"` branch in `aquery()`. Chunk IDs are aligned at export time so GFM document node names equal LightRAG KV store keys — no translation layer needed.

**Tech Stack:** Python 3.10+, `gfmrag` (PyPI), `neo4j` Python driver, `pytest-asyncio` (`asyncio_mode="auto"`), existing `raganything` + `lightrag` packages.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `rag-anything/raganything/retrieval/gfm_retriever.py` | Lazy singleton wrapper around `gfmrag.GFMRetriever` |
| Create | `rag-anything/scripts/export_lightrag_to_gfm.py` | CLI: export LightRAG KV + Neo4j → GFM-RAG CSV files |
| Create | `rag-anything/tests/__init__.py` | Make tests a package |
| Create | `rag-anything/tests/retrieval/__init__.py` | Sub-package |
| Create | `rag-anything/tests/retrieval/test_gfm_retriever.py` | Unit tests for GFMRetrieverWrapper |
| Create | `rag-anything/tests/retrieval/test_paths_gfm.py` | Unit tests for "gfm" path in run_path() |
| Create | `rag-anything/tests/retrieval/test_profiles_gfm.py` | Unit tests for gfm_multihop profile |
| Create | `rag-anything/tests/test_query_gfm_mode.py` | Unit tests for aquery(mode="gfm") |
| Create | `rag-anything/tests/scripts/__init__.py` | Sub-package |
| Create | `rag-anything/tests/scripts/test_export_lightrag_to_gfm.py` | Unit tests for export helper functions |
| Modify | `rag-anything/raganything/constants.py` | Add GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH |
| Modify | `rag-anything/raganything/retrieval/profiles.py` | Add "gfm" to KNOWN_PATHS; add gfm_multihop profile |
| Modify | `rag-anything/raganything/retrieval/paths.py` | Add "gfm" to KNOWN_PATHS; add gfm special-case in run_path() |
| Modify | `rag-anything/raganything/query.py` | Add mode="gfm" branch in aquery() |
| Modify | `rag-anything/env.example` | Document GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH |

---

## Task 1: Add GFM constants to constants.py and env.example

**Files:**
- Modify: `rag-anything/raganything/constants.py`
- Modify: `rag-anything/env.example` (create if absent)

- [ ] **Step 1: Add three constants at the end of constants.py**

Open `rag-anything/raganything/constants.py` and append after the last section:

```python
# =============================================================================
# GFM-RAG retrieval
# =============================================================================
GFM_DATA_DIR = "./data"          # root data dir for GFM-RAG index
GFM_DATA_NAME = ""               # graph name; empty string disables GFM path
GFM_MODEL_PATH = "rmanluo/G-reasoner-34M"  # HuggingFace model id or local path
```

- [ ] **Step 2: Document in env.example**

If `rag-anything/env.example` does not exist, create it. Append (or add) these lines:

```bash
# GFM-RAG retrieval path
# Run scripts/export_lightrag_to_gfm.py first, then set these.
GFM_DATA_DIR=./data
GFM_DATA_NAME=
GFM_MODEL_PATH=rmanluo/G-reasoner-34M
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/raganything/constants.py rag-anything/env.example
git commit -m "feat(gfm): add GFM_DATA_DIR/NAME/MODEL_PATH constants"
```

---

## Task 2: Create GFMRetrieverWrapper

**Files:**
- Create: `rag-anything/raganything/retrieval/gfm_retriever.py`
- Create: `rag-anything/tests/__init__.py`
- Create: `rag-anything/tests/retrieval/__init__.py`
- Create: `rag-anything/tests/retrieval/test_gfm_retriever.py`

- [ ] **Step 1: Create test package init files**

```bash
touch rag-anything/tests/__init__.py
touch rag-anything/tests/retrieval/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `rag-anything/tests/retrieval/test_gfm_retriever.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys


@pytest.fixture(autouse=True)
def reset_singleton():
    import raganything.retrieval.gfm_retriever as mod
    mod._instance = None
    yield
    mod._instance = None


class TestGFMRetrieverWrapper:
    async def test_raises_without_data_name(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        with pytest.raises(RuntimeError, match="GFM_DATA_NAME"):
            GFMRetrieverWrapper.get_instance("./data", "", "model")

    async def test_raises_when_gfmrag_not_installed(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        with patch.dict(sys.modules, {"gfmrag": None}):
            with pytest.raises(ImportError, match="gfmrag"):
                GFMRetrieverWrapper.get_instance("./data", "graph", "model")

    async def test_get_instance_returns_singleton(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = MagicMock()
        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            a = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            b = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
        assert a is b
        mock_mod.GFMRetriever.from_index.assert_called_once()

    async def test_retrieve_maps_chunk_ids_to_content(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = {
            "document": [
                {"id": "chunk_abc", "score": 0.9},
                {"id": "chunk_def", "score": 0.7},
            ]
        }
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = mock_inner

        mock_kv = AsyncMock()
        mock_kv.get_by_id.side_effect = lambda cid: {
            "chunk_abc": {"content": "France is a country."},
            "chunk_def": {"content": "Paris is the capital."},
        }.get(cid)

        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            wrapper = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            result = await wrapper.retrieve("Who is president?", top_k=5, text_chunks_kv=mock_kv)

        assert len(result) == 2
        assert result[0] == {"chunk_id": "chunk_abc", "content": "France is a country.", "score": 0.9}
        assert result[1] == {"chunk_id": "chunk_def", "content": "Paris is the capital.", "score": 0.7}

    async def test_retrieve_skips_chunk_ids_missing_from_kv(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = {
            "document": [{"id": "chunk_gone", "score": 0.8}]
        }
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = mock_inner

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = None

        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            wrapper = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            result = await wrapper.retrieve("query", top_k=5, text_chunks_kv=mock_kv)

        assert result == []

    async def test_retrieve_handles_non_dict_chunk_data(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = {
            "document": [{"id": "chunk_str", "score": 0.5}]
        }
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = mock_inner

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = "raw string content"

        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            wrapper = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            result = await wrapper.retrieve("query", top_k=5, text_chunks_kv=mock_kv)

        assert len(result) == 1
        assert result[0]["content"] == "raw string content"
```

- [ ] **Step 3: Run tests — expect ImportError (module not yet created)**

```bash
cd rag-anything
pytest tests/retrieval/test_gfm_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'raganything.retrieval.gfm_retriever'`

- [ ] **Step 4: Create the implementation**

Create `rag-anything/raganything/retrieval/gfm_retriever.py`:

```python
"""Lazy singleton wrapper around gfmrag.GFMRetriever."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_instance: Optional["GFMRetrieverWrapper"] = None


class GFMRetrieverWrapper:
    def __init__(self, retriever) -> None:
        self._retriever = retriever

    @classmethod
    def get_instance(
        cls,
        data_dir: str,
        data_name: str,
        model_path: str,
    ) -> "GFMRetrieverWrapper":
        global _instance
        if _instance is not None:
            return _instance

        if not data_name:
            raise RuntimeError(
                "GFM_DATA_NAME is not configured. "
                "Run scripts/export_lightrag_to_gfm.py first, "
                "then set GFM_DATA_NAME in your .env file."
            )

        try:
            from gfmrag import GFMRetriever
        except ImportError as exc:
            raise ImportError(
                "gfmrag is not installed. Install it with: pip install gfmrag"
            ) from exc

        logger.info(
            "Initializing GFMRetriever (data_dir=%s, data_name=%s)", data_dir, data_name
        )
        retriever = GFMRetriever.from_index(data_dir, data_name, model_path)
        _instance = cls(retriever)
        return _instance

    async def retrieve(
        self,
        query: str,
        top_k: int,
        text_chunks_kv,
    ) -> list[dict]:
        """Retrieve chunks via GFM graph reasoning, fetch content from LightRAG KV."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._retriever.retrieve, query, top_k
        )
        documents = result.get("document", [])

        chunks: list[dict] = []
        for doc in documents:
            chunk_id = doc.get("id", "")
            score = float(doc.get("score", 0.0))
            if not chunk_id:
                continue
            chunk_data = await text_chunks_kv.get_by_id(chunk_id)
            if chunk_data is None:
                logger.warning("GFM returned chunk_id not found in KV store: %s", chunk_id)
                continue
            content = (
                chunk_data.get("content", "")
                if isinstance(chunk_data, dict)
                else str(chunk_data)
            )
            chunks.append({"chunk_id": chunk_id, "content": content, "score": score})

        return chunks
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
cd rag-anything
pytest tests/retrieval/test_gfm_retriever.py -v
```

Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add rag-anything/raganything/retrieval/gfm_retriever.py \
        rag-anything/tests/__init__.py \
        rag-anything/tests/retrieval/__init__.py \
        rag-anything/tests/retrieval/test_gfm_retriever.py
git commit -m "feat(gfm): add GFMRetrieverWrapper lazy singleton"
```

---

## Task 3: Add "gfm" to KNOWN_PATHS and wire into profiles.py

**Files:**
- Modify: `rag-anything/raganything/retrieval/profiles.py`
- Create: `rag-anything/tests/retrieval/test_profiles_gfm.py`

- [ ] **Step 1: Write failing tests**

Create `rag-anything/tests/retrieval/test_profiles_gfm.py`:

```python
class TestGFMMultihopProfile:
    def test_gfm_in_known_paths(self):
        from raganything.retrieval.profiles import KNOWN_PATHS
        assert "gfm" in KNOWN_PATHS

    def test_gfm_multihop_profile_exists(self):
        from raganything.retrieval.profiles import PROFILE_REGISTRY
        assert "gfm_multihop" in PROFILE_REGISTRY

    def test_gfm_multihop_default_excludes_gfm(self):
        from raganything.retrieval.profiles import PROFILE_REGISTRY
        profile = PROFILE_REGISTRY["gfm_multihop"]
        # GFM is commented out by default
        assert "gfm" not in profile.paths
        assert "ppr" in profile.paths
        assert "hybrid" in profile.paths

    def test_gfm_multihop_rrf_weights_match_paths(self):
        from raganything.retrieval.profiles import PROFILE_REGISTRY
        profile = PROFILE_REGISTRY["gfm_multihop"]
        assert set(profile.paths) == set(profile.rrf_weights.keys())
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd rag-anything
pytest tests/retrieval/test_profiles_gfm.py -v
```

Expected: `FAILED test_gfm_in_known_paths` and `FAILED test_gfm_multihop_profile_exists`

- [ ] **Step 3: Edit profiles.py**

In `rag-anything/raganything/retrieval/profiles.py`, make two changes:

**Change 1** — extend the `KNOWN_PATHS` frozenset (line 4–6):

```python
KNOWN_PATHS: frozenset[str] = frozenset(
    ["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse", "gfm"]
)
```

**Change 2** — add the new profile at the end of the list inside `PROFILE_REGISTRY` (after the `"full"` profile, before the closing `]`):

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

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd rag-anything
pytest tests/retrieval/test_profiles_gfm.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/retrieval/profiles.py \
        rag-anything/tests/retrieval/test_profiles_gfm.py
git commit -m "feat(gfm): add gfm to KNOWN_PATHS and gfm_multihop profile"
```

---

## Task 4: Wire "gfm" special-case into paths.py

**Files:**
- Modify: `rag-anything/raganything/retrieval/paths.py`
- Create: `rag-anything/tests/retrieval/test_paths_gfm.py`

- [ ] **Step 1: Write failing tests**

Create `rag-anything/tests/retrieval/test_paths_gfm.py`:

```python
import dataclasses
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@dataclasses.dataclass
class FakeParam:
    mode: str = "hybrid"
    chunk_top_k: int = 10


class TestGFMPath:
    def test_gfm_in_known_paths(self):
        from raganything.retrieval.paths import KNOWN_PATHS
        assert "gfm" in KNOWN_PATHS

    async def test_run_path_gfm_calls_wrapper_retrieve(self):
        from raganything.retrieval.paths import run_path

        mock_chunks = [{"chunk_id": "c1", "content": "hello", "score": 0.9}]
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = mock_chunks

        mock_lightrag = MagicMock()
        mock_lightrag.text_chunks = AsyncMock()

        with patch("raganything.retrieval.paths.GFMRetrieverWrapper") as MockCls:
            MockCls.get_instance.return_value = mock_wrapper
            with patch("raganything.retrieval.paths.GFM_DATA_DIR", "./data"), \
                 patch("raganything.retrieval.paths.GFM_DATA_NAME", "test_graph"), \
                 patch("raganything.retrieval.paths.GFM_MODEL_PATH", "model"):
                chunks, latency = await run_path(
                    "gfm", "Who is the president?", FakeParam(), mock_lightrag, {}
                )

        assert chunks == mock_chunks
        assert latency >= 0.0
        mock_wrapper.retrieve.assert_called_once_with(
            "Who is the president?", 10, mock_lightrag.text_chunks
        )

    async def test_run_path_gfm_does_not_call_aquery_data(self):
        from raganything.retrieval.paths import run_path

        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = []
        mock_lightrag = MagicMock()
        mock_lightrag.text_chunks = AsyncMock()

        with patch("raganything.retrieval.paths.GFMRetrieverWrapper") as MockCls:
            MockCls.get_instance.return_value = mock_wrapper
            with patch("raganything.retrieval.paths.GFM_DATA_DIR", "./data"), \
                 patch("raganything.retrieval.paths.GFM_DATA_NAME", "graph"), \
                 patch("raganything.retrieval.paths.GFM_MODEL_PATH", "model"):
                await run_path("gfm", "query", FakeParam(), mock_lightrag, {})

        mock_lightrag.aquery_data.assert_not_called()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd rag-anything
pytest tests/retrieval/test_paths_gfm.py -v
```

Expected: `FAILED test_gfm_in_known_paths`, `ERROR test_run_path_gfm_calls_wrapper_retrieve`

- [ ] **Step 3: Edit paths.py**

In `rag-anything/raganything/retrieval/paths.py`, make two changes:

**Change 1** — extend `KNOWN_PATHS` (currently on line ~3):

```python
KNOWN_PATHS: frozenset[str] = frozenset(
    ["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse", "gfm"]
)
```

**Change 2** — at the top of `run_path()`, add the GFM special-case before the existing `if name not in _PATH_CONFIG` check. Also add the two imports at the top of the file:

Add to the top-level imports at the top of `paths.py`:
```python
from raganything.constants import GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH
from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
```

Then inside `run_path()`, insert before the existing `if name not in _PATH_CONFIG:` line:

```python
    if name == "gfm":
        wrapper = GFMRetrieverWrapper.get_instance(GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH)
        t0 = time.monotonic()
        chunks = await wrapper.retrieve(
            query, getattr(param, "chunk_top_k", 10), lightrag.text_chunks
        )
        latency = time.monotonic() - t0
        return chunks, latency
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd rag-anything
pytest tests/retrieval/test_paths_gfm.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
cd rag-anything
pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add rag-anything/raganything/retrieval/paths.py \
        rag-anything/tests/retrieval/test_paths_gfm.py
git commit -m "feat(gfm): wire gfm special-case into run_path()"
```

---

## Task 5: Add mode="gfm" branch to aquery()

**Files:**
- Modify: `rag-anything/raganything/query.py`
- Create: `rag-anything/tests/test_query_gfm_mode.py`

- [ ] **Step 1: Write failing tests**

Create `rag-anything/tests/test_query_gfm_mode.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_query_mixin_obj():
    """Build a minimal QueryMixin instance with required attributes mocked."""
    from raganything.query import QueryMixin
    obj = QueryMixin.__new__(QueryMixin)
    obj.lightrag = MagicMock()
    obj.lightrag.text_chunks = AsyncMock()
    obj._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})
    obj._generate_answer_from_chunks = AsyncMock(return_value="France is in Europe.")
    obj.logger = MagicMock()
    obj.callback_manager = None
    obj.vision_model_func = None
    return obj


class TestQueryGFMMode:
    async def test_aquery_gfm_returns_answer_string(self):
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = [
            {"chunk_id": "c1", "content": "France is in Europe.", "score": 0.9}
        ]

        with patch("raganything.query.GFMRetrieverWrapper") as MockCls, \
             patch("raganything.query.GFM_DATA_DIR", "./data"), \
             patch("raganything.query.GFM_DATA_NAME", "graph"), \
             patch("raganything.query.GFM_MODEL_PATH", "model"):
            MockCls.get_instance.return_value = mock_wrapper
            obj = _make_query_mixin_obj()
            result = await obj.aquery("Where is France?", mode="gfm")

        assert result == "France is in Europe."
        obj._generate_answer_from_chunks.assert_called_once()

    async def test_aquery_gfm_return_trace_true(self):
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = [
            {"chunk_id": "c1", "content": "x", "score": 0.5},
            {"chunk_id": "c2", "content": "y", "score": 0.4},
        ]

        with patch("raganything.query.GFMRetrieverWrapper") as MockCls, \
             patch("raganything.query.GFM_DATA_DIR", "./data"), \
             patch("raganything.query.GFM_DATA_NAME", "graph"), \
             patch("raganything.query.GFM_MODEL_PATH", "model"):
            MockCls.get_instance.return_value = mock_wrapper
            obj = _make_query_mixin_obj()
            result = await obj.aquery("query", mode="gfm", return_trace=True)

        assert isinstance(result, dict)
        assert result["answer"] == "France is in Europe."
        assert result["trace"]["mode"] == "gfm"
        assert result["trace"]["chunks_retrieved"] == 2

    async def test_aquery_gfm_passes_chunk_top_k(self):
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = []

        with patch("raganything.query.GFMRetrieverWrapper") as MockCls, \
             patch("raganything.query.GFM_DATA_DIR", "./data"), \
             patch("raganything.query.GFM_DATA_NAME", "graph"), \
             patch("raganything.query.GFM_MODEL_PATH", "model"):
            MockCls.get_instance.return_value = mock_wrapper
            obj = _make_query_mixin_obj()
            await obj.aquery("query", mode="gfm", chunk_top_k=7)

        mock_wrapper.retrieve.assert_called_once()
        call_args = mock_wrapper.retrieve.call_args
        assert call_args[0][1] == 7  # top_k positional arg
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd rag-anything
pytest tests/test_query_gfm_mode.py -v
```

Expected: tests fail because the `mode="gfm"` branch does not yet exist.

- [ ] **Step 3: Edit query.py**

In `rag-anything/raganything/query.py`, make two changes:

**Change 1** — add imports near the top of the file, after the existing imports from `raganything.constants`:

```python
from raganything.constants import (
    DEFAULT_MULTIMODAL_TOP_K,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_QDRANT_RETRIEVAL_MODE,
    SUPPORTED_IMAGE_EXTENSIONS,
    GFM_DATA_DIR,
    GFM_DATA_NAME,
    GFM_MODEL_PATH,
)
from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
```

(Replace the existing `from raganything.constants import (...)` block with the expanded version.)

**Change 2** — in `aquery()`, insert the `mode="gfm"` branch immediately after the `mode="auto"` block ends (after the `# ── end mode="auto" ───` comment, before the `vlm_enhanced = kwargs.pop(...)` line):

```python
        # ── mode="gfm": GFM graph neural retrieval ────────────────────────
        if mode == "gfm":
            return_trace_gfm = bool(kwargs.pop("return_trace", False))
            top_k = kwargs.get("chunk_top_k", DEFAULT_CHUNK_TOP_K)
            wrapper = GFMRetrieverWrapper.get_instance(GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH)
            chunks = await wrapper.retrieve(query, top_k, self.lightrag.text_chunks)
            answer = await self._generate_answer_from_chunks(
                query,
                chunks,
                system_prompt=system_prompt,
                response_type=kwargs.get("response_type", "Multiple Paragraphs"),
            )
            answer = answer if isinstance(answer, str) else str(answer)
            if return_trace_gfm:
                return {"answer": answer, "trace": {"mode": "gfm", "chunks_retrieved": len(chunks)}}
            return answer
        # ── end mode="gfm" ────────────────────────────────────────────────
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd rag-anything
pytest tests/test_query_gfm_mode.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run full test suite**

```bash
cd rag-anything
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rag-anything/raganything/query.py \
        rag-anything/tests/test_query_gfm_mode.py
git commit -m "feat(gfm): add mode=gfm branch in aquery()"
```

---

## Task 6: Write the export script

**Files:**
- Create: `rag-anything/scripts/export_lightrag_to_gfm.py`
- Create: `rag-anything/tests/scripts/__init__.py`
- Create: `rag-anything/tests/scripts/test_export_lightrag_to_gfm.py`

- [ ] **Step 1: Write unit tests for export helper functions**

Create `rag-anything/tests/scripts/__init__.py` (empty).

Create `rag-anything/tests/scripts/test_export_lightrag_to_gfm.py`:

```python
import csv
import json
import io
from pathlib import Path
import pytest


# ── helpers: import the functions under test ──────────────────────────────────
def _import():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "export_lightrag_to_gfm",
        Path(__file__).parents[2] / "scripts" / "export_lightrag_to_gfm.py",
    )
    mod = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _import()


class TestWriteNodesCsv:
    def test_document_nodes_use_chunk_id_as_name(self, mod, tmp_path):
        chunks = {"chunk_abc": "France is a country."}
        entities = []
        mod.write_nodes_csv(tmp_path, chunks, entities)
        rows = list(csv.DictReader((tmp_path / "nodes.csv").open()))
        assert rows[0]["name"] == "chunk_abc"
        assert rows[0]["type"] == "document"
        attrs = json.loads(rows[0]["attributes"])
        assert attrs["content"] == "France is a country."

    def test_entity_nodes_prefixed_with_entity_(self, mod, tmp_path):
        chunks = {}
        entities = [{"name": "France", "description": "A country", "entity_id": "e1", "source_ids": ""}]
        mod.write_nodes_csv(tmp_path, chunks, entities)
        rows = list(csv.DictReader((tmp_path / "nodes.csv").open()))
        assert rows[0]["name"] == "entity_France"
        assert rows[0]["type"] == "entity"

    def test_header_is_name_type_attributes(self, mod, tmp_path):
        mod.write_nodes_csv(tmp_path, {}, [])
        with open(tmp_path / "nodes.csv") as f:
            header = f.readline().strip()
        assert header == "name,type,attributes"


class TestWriteEdgesCsv:
    def test_mentioned_in_edges_for_entity_source_ids(self, mod, tmp_path):
        chunks = {"chunk_abc": "content"}
        entities = [
            {"name": "France", "entity_id": "e1", "description": "", "source_ids": "chunk_abc"}
        ]
        mod.write_edges_csv(tmp_path, entities, [], chunks)
        rows = list(csv.DictReader((tmp_path / "edges.csv").open()))
        assert any(
            r["source"] == "entity_France"
            and r["relation"] == "mentioned_in"
            and r["target"] == "chunk_abc"
            for r in rows
        )

    def test_entity_entity_edges_from_relations(self, mod, tmp_path):
        chunks = {}
        entities = [
            {"name": "France", "entity_id": "e1", "description": "", "source_ids": ""},
            {"name": "Paris", "entity_id": "e2", "description": "", "source_ids": ""},
        ]
        relations = [{"src": "e1", "relation": "capital_of", "tgt": "e2"}]
        mod.write_edges_csv(tmp_path, entities, relations, chunks)
        rows = list(csv.DictReader((tmp_path / "edges.csv").open()))
        assert any(
            r["source"] == "entity_France"
            and r["relation"] == "capital_of"
            and r["target"] == "entity_Paris"
            for r in rows
        )

    def test_header_is_source_relation_target_attributes(self, mod, tmp_path):
        mod.write_edges_csv(tmp_path, [], [], {})
        with open(tmp_path / "edges.csv") as f:
            header = f.readline().strip()
        assert header == "source,relation,target,attributes"


class TestWriteRelationsCsv:
    def test_includes_mentioned_in_always(self, mod, tmp_path):
        mod.write_relations_csv(tmp_path, [])
        rows = list(csv.DictReader((tmp_path / "relations.csv").open()))
        names = [r["name"] for r in rows]
        assert "mentioned_in" in names

    def test_includes_relation_types_from_input(self, mod, tmp_path):
        relations = [{"src": "a", "relation": "capital_of", "tgt": "b"}]
        mod.write_relations_csv(tmp_path, relations)
        rows = list(csv.DictReader((tmp_path / "relations.csv").open()))
        names = [r["name"] for r in rows]
        assert "capital_of" in names

    def test_header_is_name_attributes(self, mod, tmp_path):
        mod.write_relations_csv(tmp_path, [])
        with open(tmp_path / "relations.csv") as f:
            header = f.readline().strip()
        assert header == "name,attributes"


class TestWriteDocumentsJson:
    def test_writes_chunk_id_to_content_mapping(self, mod, tmp_path):
        chunks = {"chunk_abc": "France is a country.", "chunk_def": "Paris is the capital."}
        mod.write_documents_json(tmp_path, chunks)
        data = json.loads((tmp_path / "documents.json").read_text())
        assert data == chunks
```

- [ ] **Step 2: Run tests — expect ImportError (script not yet created)**

```bash
cd rag-anything
pytest tests/scripts/test_export_lightrag_to_gfm.py -v
```

Expected: fails loading the module (file does not exist yet).

- [ ] **Step 3: Create the export script**

Create `rag-anything/scripts/export_lightrag_to_gfm.py`:

```python
#!/usr/bin/env python3
"""
Export LightRAG graph and chunks to GFM-RAG CSV format (Path B: pre-built index).

Usage:
    python scripts/export_lightrag_to_gfm.py \\
        --working-dir ./rag_storage/My_Graph \\
        --data-dir ./data \\
        --graph-name My_Graph \\
        --workspace My_Graph          # Neo4j workspace label (defaults to graph-name)

Output layout:
    <data-dir>/<graph-name>/processed/stage1/nodes.csv
    <data-dir>/<graph-name>/processed/stage1/edges.csv
    <data-dir>/<graph-name>/processed/stage1/relations.csv
    <data-dir>/<graph-name>/raw/documents.json

After running, set in .env:
    GFM_DATA_DIR=<data-dir>
    GFM_DATA_NAME=<graph-name>
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_chunks(working_dir: str) -> dict[str, str]:
    """Read all chunks from LightRAG JSON KV store.

    Returns {chunk_id: content_string}.
    """
    kv_path = Path(working_dir) / "kv_store_text_chunks.json"
    if not kv_path.exists():
        raise FileNotFoundError(
            f"LightRAG chunk KV store not found: {kv_path}\n"
            "Make sure --working-dir points to a LightRAG workspace directory."
        )
    with open(kv_path, encoding="utf-8") as f:
        raw = json.load(f)
    chunks: dict[str, str] = {}
    for chunk_id, chunk_data in raw.items():
        if isinstance(chunk_data, dict):
            content = chunk_data.get("content", "")
        else:
            content = str(chunk_data)
        chunks[chunk_id] = content
    logger.info("Loaded %d chunks from KV store", len(chunks))
    return chunks


def load_neo4j_graph(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    workspace: str,
) -> tuple[list[dict], list[dict]]:
    """Read entities and relations from Neo4j for the given workspace.

    Returns (entities, relations).

    Each entity dict: {entity_id, name, description, source_ids}
      source_ids is a <SEP>-separated string of chunk IDs stored by LightRAG.

    Each relation dict: {src, relation, tgt}
      src/tgt are entity_id values.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError(
            "neo4j Python driver not installed. Run: pip install neo4j"
        ) from exc

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    entities: list[dict] = []
    relations: list[dict] = []

    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {workspace: $ws}) "
                "RETURN e.entity_id AS entity_id, e.entity_name AS name, "
                "       e.description AS description, e.source_id AS source_ids",
                ws=workspace,
            )
            for record in result:
                entities.append(
                    {
                        "entity_id": record["entity_id"] or "",
                        "name": record["name"] or "",
                        "description": record["description"] or "",
                        "source_ids": record["source_ids"] or "",
                    }
                )
            logger.info("Loaded %d entities from Neo4j (workspace=%s)", len(entities), workspace)

            result = session.run(
                "MATCH (src:Entity {workspace: $ws})-[r:RELATES_TO]->(tgt:Entity {workspace: $ws}) "
                "RETURN src.entity_id AS src_id, r.relation AS relation, tgt.entity_id AS tgt_id",
                ws=workspace,
            )
            for record in result:
                relations.append(
                    {
                        "src": record["src_id"] or "",
                        "relation": record["relation"] or "",
                        "tgt": record["tgt_id"] or "",
                    }
                )
            logger.info("Loaded %d relations from Neo4j", len(relations))
    finally:
        driver.close()

    return entities, relations


def write_nodes_csv(out_dir: Path, chunks: dict[str, str], entities: list[dict]) -> None:
    """Write nodes.csv — header: name,type,attributes"""
    out_path = out_dir / "nodes.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "type", "attributes"])
        for chunk_id, content in chunks.items():
            attrs = json.dumps({"content": content}, ensure_ascii=False)
            writer.writerow([chunk_id, "document", attrs])
        for entity in entities:
            attrs = json.dumps({"description": entity["description"]}, ensure_ascii=False)
            writer.writerow([f"entity_{entity['name']}", "entity", attrs])
    logger.info(
        "Written %d document nodes + %d entity nodes → %s",
        len(chunks), len(entities), out_path,
    )


def write_edges_csv(
    out_dir: Path,
    entities: list[dict],
    relations: list[dict],
    chunks: dict[str, str],
) -> None:
    """Write edges.csv — header: source,relation,target,attributes"""
    out_path = out_dir / "edges.csv"
    entity_by_id: dict[str, str] = {e["entity_id"]: e["name"] for e in entities}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "relation", "target", "attributes"])

        # Entity → chunk (mentioned_in) edges
        for entity in entities:
            source_ids_str = entity.get("source_ids", "") or ""
            for chunk_id in source_ids_str.split("<SEP>"):
                chunk_id = chunk_id.strip()
                if chunk_id and chunk_id in chunks:
                    writer.writerow(
                        [f"entity_{entity['name']}", "mentioned_in", chunk_id, "{}"]
                    )

        # Entity → entity edges (from Neo4j relations)
        for rel in relations:
            src_name = entity_by_id.get(rel["src"], rel["src"])
            tgt_name = entity_by_id.get(rel["tgt"], rel["tgt"])
            if src_name and tgt_name and rel["relation"]:
                writer.writerow(
                    [f"entity_{src_name}", rel["relation"], f"entity_{tgt_name}", "{}"]
                )

    logger.info("Written edges → %s", out_path)


def write_relations_csv(out_dir: Path, relations: list[dict]) -> None:
    """Write relations.csv — header: name,attributes"""
    out_path = out_dir / "relations.csv"
    unique_relations = sorted({rel["relation"] for rel in relations if rel["relation"]} | {"mentioned_in"})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "attributes"])
        for rel_name in unique_relations:
            writer.writerow([rel_name, "{}"])
    logger.info("Written %d relation types → %s", len(unique_relations), out_path)


def write_documents_json(raw_dir: Path, chunks: dict[str, str]) -> None:
    """Write raw/documents.json — {chunk_id: chunk_content}"""
    out_path = raw_dir / "documents.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Written %d documents → %s", len(chunks), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LightRAG KV store + Neo4j graph → GFM-RAG CSV index (Path B)"
    )
    parser.add_argument(
        "--working-dir", required=True,
        help="LightRAG workspace directory (contains kv_store_text_chunks.json)",
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="GFM-RAG root data directory (default: ./data)",
    )
    parser.add_argument(
        "--graph-name", required=True,
        help="Graph name — becomes GFM_DATA_NAME and the subdirectory name",
    )
    parser.add_argument(
        "--workspace", default=None,
        help="Neo4j workspace label (default: same as --graph-name)",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    args = parser.parse_args()

    workspace = args.workspace or args.graph_name

    stage1_dir = Path(args.data_dir) / args.graph_name / "processed" / "stage1"
    raw_dir = Path(args.data_dir) / args.graph_name / "raw"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.working_dir)
    entities, relations = load_neo4j_graph(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password, workspace
    )

    write_nodes_csv(stage1_dir, chunks, entities)
    write_edges_csv(stage1_dir, entities, relations, chunks)
    write_relations_csv(stage1_dir, relations)
    write_documents_json(raw_dir, chunks)

    logger.info("Export complete.")
    logger.info("Set in .env:  GFM_DATA_DIR=%s  GFM_DATA_NAME=%s", args.data_dir, args.graph_name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests — expect all pass**

```bash
cd rag-anything
pytest tests/scripts/test_export_lightrag_to_gfm.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Run full test suite**

```bash
cd rag-anything
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rag-anything/scripts/export_lightrag_to_gfm.py \
        rag-anything/tests/scripts/__init__.py \
        rag-anything/tests/scripts/test_export_lightrag_to_gfm.py
git commit -m "feat(gfm): add export_lightrag_to_gfm.py offline export script"
```

---

## Manual Verification Checklist (after all tasks)

Run after completing all tasks and before declaring Phase 1 done:

- [ ] Export script runs end-to-end against real data:
  ```bash
  cd rag-anything
  python scripts/export_lightrag_to_gfm.py \
      --working-dir ./rag_storage/My_Graph \
      --graph-name My_Graph
  ```
  Verify that `./data/My_Graph/processed/stage1/` contains `nodes.csv`, `edges.csv`, `relations.csv` and `./data/My_Graph/raw/documents.json`.

- [ ] Set env vars and test standalone GFM mode (requires `pip install gfmrag`):
  ```bash
  GFM_DATA_DIR=./data GFM_DATA_NAME=My_Graph python - <<'EOF'
  import asyncio
  from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
  from raganything.constants import GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH
  wrapper = GFMRetrieverWrapper.get_instance(GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH)
  print("GFMRetriever initialized OK")
  EOF
  ```

- [ ] To enable GFM in `gfm_multihop` profile for A/B testing against PPR, open `rag-anything/raganything/retrieval/profiles.py` and uncomment the `"gfm"` lines (and optionally comment out `"ppr"` lines) in the `gfm_multihop` profile.
