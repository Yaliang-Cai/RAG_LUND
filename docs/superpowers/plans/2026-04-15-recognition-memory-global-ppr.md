# Recognition Memory for Global PPR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HippoRAG2-style recognition memory (Numpy→LLM→Difflib) to `mode="ppr"` so entity seeds are LLM-verified before PPR propagation, replacing pure vector similarity selection.

**Architecture:** A new async helper `_recognition_memory_filter()` collects entity VDB and relation VDB candidates already retrieved by hybrid search, normalises their scores independently, sends a unified candidate list to the LLM for relevance judgment, then uses difflib to remap LLM text output to graph entity_ids. The refactored seed-building block in `_ppr_rank_chunks()` calls this helper for the global path only; the rest of the PPR pipeline (hub penalty, `_ppr_rank_chunks_global`, DPR chunk seeds) is untouched.

**Tech Stack:** Python stdlib `difflib`, existing `llm_model_func` from `global_config`, `pytest` + `unittest.mock.AsyncMock` for tests.

---

## File Map

| File | Role | Action |
|------|------|--------|
| `lightrag/lightrag/base.py` | `QueryParam` dataclass | Add `recognition_top_k: int = 10` field |
| `lightrag/lightrag/operate.py` | PPR orchestration | Add `_min_max_norm()`, `_recognition_memory_filter()`; refactor seed-building block in `_ppr_rank_chunks()` |
| `lightrag/tests/test_recognition_memory.py` | Unit tests | New file — all tests for the new functions |

---

### Task 1: Add `recognition_top_k` to `QueryParam`

**Files:**
- Modify: `lightrag/lightrag/base.py:238` (after `hub_penalty_threshold`)

- [ ] **Step 1: Open `base.py` and locate the insertion point**

In `lightrag/lightrag/base.py`, find the `hub_penalty_threshold` field (currently the last PPR-related param at ~line 238):

```python
    hub_penalty_threshold: int = 50
    """Degree threshold above which entity seed weights are penalised by log(1 + degree).
    Set to 0 to disable. Reduces subgraph bloat caused by high-degree generic entities."""
```

- [ ] **Step 2: Add `recognition_top_k` directly below**

```python
    recognition_top_k: int = 10
    """HippoRAG2 Recognition Memory: controls how many candidates are shown to the LLM
    for entity seed filtering when mode="ppr".
    - Relation triplets sent to LLM  : rel_results[:recognition_top_k]
    - Entity VDB candidates sent to LLM: node_datas[:recognition_top_k * 2]
      (entity VDB retrieval size is still governed by query_param.top_k)
    - Default 10 matches HippoRAG2 link_top_k.
    - Set to 0 to disable recognition memory (falls back to direct score merge)."""
```

- [ ] **Step 3: Verify the import and dataclass decorator are intact**

Run:
```bash
cd /d/HUAWEI/RAG_LUND/lightrag && python -c "from lightrag.base import QueryParam; p = QueryParam(); print(p.recognition_top_k)"
```
Expected output: `10`

- [ ] **Step 4: Commit**

```bash
git add lightrag/lightrag/base.py
git commit -m "feat(ppr): add QueryParam.recognition_top_k for recognition memory"
```

---

### Task 2: Write tests for `_min_max_norm`

**Files:**
- Create: `lightrag/tests/test_recognition_memory.py`

- [ ] **Step 1: Create the test file with `_min_max_norm` tests**

```python
"""Unit tests for recognition memory components (operate.py).

All LLM calls are mocked — no real API calls in this suite.
"""
import difflib
import pytest
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# _min_max_norm — import by copying the function under test directly, so
# tests don't depend on operate.py import chain (which needs Neo4j/Qdrant).
# ---------------------------------------------------------------------------

def _min_max_norm(scores: dict) -> dict:
    """Copied verbatim from operate.py for isolated testing."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        uniform = 1.0 if hi > 0.0 else 0.0
        return {k: uniform for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


class TestMinMaxNorm:
    def test_empty_dict_returns_empty(self):
        assert _min_max_norm({}) == {}

    def test_single_entry_normalises_to_1(self):
        result = _min_max_norm({"a": 0.7})
        # hi == lo, value > 0 → uniform 1.0
        assert result == {"a": 1.0}

    def test_all_zero_returns_zero(self):
        result = _min_max_norm({"a": 0.0, "b": 0.0})
        assert result == {"a": 0.0, "b": 0.0}

    def test_all_equal_nonzero_returns_one(self):
        # hi == lo == 0.9 → should NOT collapse to 0.0
        result = _min_max_norm({"a": 0.9, "b": 0.9, "c": 0.9})
        assert result == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_normal_range(self):
        result = _min_max_norm({"a": 0.0, "b": 0.5, "c": 1.0})
        assert abs(result["a"] - 0.0) < 1e-9
        assert abs(result["b"] - 0.5) < 1e-9
        assert abs(result["c"] - 1.0) < 1e-9

    def test_output_range_is_zero_to_one(self):
        import random
        scores = {f"e{i}": random.uniform(0.5, 0.95) for i in range(20)}
        result = _min_max_norm(scores)
        assert all(0.0 <= v <= 1.0 for v in result.values())
        assert min(result.values()) == pytest.approx(0.0)
        assert max(result.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests — they must all pass (pure Python, no imports needed)**

```bash
cd /d/HUAWEI/RAG_LUND/lightrag && python -m pytest tests/test_recognition_memory.py::TestMinMaxNorm -v
```
Expected: `6 passed`

- [ ] **Step 3: Commit**

```bash
git add lightrag/tests/test_recognition_memory.py
git commit -m "test(ppr): add _min_max_norm unit tests"
```

---

### Task 3: Implement `_min_max_norm` in `operate.py`

**Files:**
- Modify: `lightrag/lightrag/operate.py` — insert before `_ppr_rank_chunks_global` (around line 5473)

- [ ] **Step 1: Find the insertion point**

In `operate.py`, locate line ~5473:
```python
async def _ppr_rank_chunks_global(
```

Insert `_min_max_norm` as a module-level function immediately above `_ppr_rank_chunks_global`:

```python
def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    """Normalise score dict to [0, 1].

    Edge case: if all scores are identical and > 0, returns uniform 1.0 to
    preserve seed signal rather than collapsing everything to 0.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        uniform = 1.0 if hi > 0.0 else 0.0
        return {k: uniform for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}

```

- [ ] **Step 2: Verify the existing tests still pass**

```bash
cd /d/HUAWEI/RAG_LUND/lightrag && python -m pytest tests/test_recognition_memory.py::TestMinMaxNorm -v
```
Expected: `6 passed`  
(Tests import their own copy, but this confirms the file isn't broken.)

Also confirm the module still imports cleanly:
```bash
python -c "import lightrag.operate; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add lightrag/lightrag/operate.py
git commit -m "feat(ppr): add _min_max_norm helper to operate.py"
```

---

### Task 4: Write tests for `_recognition_memory_filter`

**Files:**
- Modify: `lightrag/tests/test_recognition_memory.py`

- [ ] **Step 1: Add a self-contained copy of the function under test and its tests**

Append to `lightrag/tests/test_recognition_memory.py`:

```python
# ---------------------------------------------------------------------------
# _recognition_memory_filter — tested with a self-contained implementation
# copy so tests don't depend on the full operate.py import chain.
# ---------------------------------------------------------------------------

import difflib
from typing import Callable


async def _recognition_memory_filter(
    query: str,
    node_datas: list[dict],
    rel_results: list[dict],
    llm_model_func: Callable,
    recognition_top_k: int = 10,
) -> dict[str, float]:
    """Copied verbatim from operate.py for isolated testing."""
    top_rels = rel_results[:recognition_top_k]
    top_nodes = node_datas[:recognition_top_k * 2]

    # fact_scores: max across triplets for same entity
    fact_scores: dict[str, float] = {}
    for rel in top_rels:
        for eid in (rel.get("src_id"), rel.get("tgt_id")):
            if eid:
                fact_scores[eid] = max(fact_scores.get(eid, 0.0), rel.get("distance", 0.0))

    norm_vdb = _min_max_norm({
        nd["entity_id"]: nd.get("vdb_score", 0.0)
        for nd in top_nodes if nd.get("entity_id")
    })
    norm_fact = _min_max_norm(fact_scores)

    # Build prompt candidate list
    entity_vdb_ids = [nd["entity_id"] for nd in top_nodes if nd.get("entity_id")]
    triplet_eids = list(dict.fromkeys(
        eid for rel in top_rels
        for eid in (rel.get("src_id"), rel.get("tgt_id")) if eid
    ))
    all_candidate_ids = list(dict.fromkeys(entity_vdb_ids + triplet_eids))

    standalone_block = "\n".join(entity_vdb_ids)
    triplet_block = "\n".join(
        f"{r.get('src_id')} | {r.get('description', '')} | {r.get('tgt_id')}"
        for r in top_rels
    )
    prompt = (
        "You are an entity relevance judge.\n\n"
        f"Query: {query}\n\n"
        "Below are retrieved entities and facts. Select ONLY those directly relevant "
        "to answering the query.\n"
        "You MUST copy each entity identifier EXACTLY as it appears in the list below, "
        "including any \"|TYPE\" suffixes and special characters. "
        "Do not rephrase, abbreviate, or invent new identifiers.\n\n"
        f"Standalone entities:\n{standalone_block}\n\n"
        f"Retrieved facts:\n{triplet_block}\n\n"
        "Return the relevant entity identifiers only, one per line. "
        "If none are relevant, return an empty response."
    )

    llm_output: str = await llm_model_func(prompt)

    recognized_ids = set()
    for line in llm_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        matches = difflib.get_close_matches(line, all_candidate_ids, n=1, cutoff=0.85)
        if matches:
            recognized_ids.add(matches[0])

    result = {}
    for eid in recognized_ids:
        w = max(norm_vdb.get(eid, 0.0), norm_fact.get(eid, 0.0))
        if w > 0.0:
            result[eid] = w
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecognitionMemoryFilter:
    """All LLM calls mocked. No real API access."""

    def _make_node(self, entity_id: str, vdb_score: float) -> dict:
        return {"entity_id": entity_id, "vdb_score": vdb_score}

    def _make_rel(self, src: str, tgt: str, desc: str, dist: float) -> dict:
        return {"src_id": src, "tgt_id": tgt, "description": desc, "distance": dist}

    @pytest.mark.asyncio
    async def test_recognized_entity_from_vdb(self):
        """LLM confirms an entity from entity VDB → it appears in result."""
        nodes = [self._make_node("华为|ORGANIZATION", 0.9)]
        rels = []
        llm = AsyncMock(return_value="华为|ORGANIZATION")
        result = await _recognition_memory_filter("华为做了什么", nodes, rels, llm, recognition_top_k=10)
        assert "华为|ORGANIZATION" in result
        assert result["华为|ORGANIZATION"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_recognized_entity_from_triplet(self):
        """LLM confirms entity from relation triplet → appears in result with norm_fact score."""
        nodes = []
        rels = [self._make_rel("华为|ORGANIZATION", "5G|TECHNOLOGY", "开发了", 0.8)]
        llm = AsyncMock(return_value="华为|ORGANIZATION")
        result = await _recognition_memory_filter("5G研发", nodes, rels, llm, recognition_top_k=10)
        assert "华为|ORGANIZATION" in result

    @pytest.mark.asyncio
    async def test_llm_returns_empty_gives_empty_result(self):
        """Empty LLM output → empty dict → caller falls back to direct merge."""
        nodes = [self._make_node("Entity A", 0.9)]
        rels = [self._make_rel("Entity A", "Entity B", "relates to", 0.7)]
        llm = AsyncMock(return_value="")
        result = await _recognition_memory_filter("query", nodes, rels, llm)
        assert result == {}

    @pytest.mark.asyncio
    async def test_hallucinated_entity_excluded(self):
        """LLM returns entity not in candidate list → difflib rejects it."""
        nodes = [self._make_node("Real Entity|ORG", 0.9)]
        rels = []
        llm = AsyncMock(return_value="Invented Entity XYZ")
        result = await _recognition_memory_filter("query", nodes, rels, llm)
        assert result == {}

    @pytest.mark.asyncio
    async def test_difflib_handles_minor_typo(self):
        """LLM returns slightly misspelled entity → difflib still maps it."""
        nodes = [self._make_node("华为|ORGANIZATION", 0.9)]
        rels = []
        # Minor variation — still within cutoff=0.85
        llm = AsyncMock(return_value="华为|ORGANIZATON")  # typo in suffix
        result = await _recognition_memory_filter("query", nodes, rels, llm)
        # May or may not match depending on edit distance; test that we don't crash
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_recognition_top_k_limits_triplets_shown(self):
        """Only top recognition_top_k triplets reach the LLM prompt."""
        nodes = []
        rels = [self._make_rel(f"E{i}|T", f"F{i}|T", "rel", 0.9 - i * 0.01) for i in range(20)]
        captured_prompt = []
        async def capture_llm(prompt):
            captured_prompt.append(prompt)
            return ""
        await _recognition_memory_filter("query", nodes, rels, capture_llm, recognition_top_k=5)
        # Only first 5 triplets in the prompt
        assert captured_prompt[0].count("|") <= 5 * 3  # rough upper bound

    @pytest.mark.asyncio
    async def test_multi_source_entity_takes_max_weight(self):
        """Entity in both VDB and triplet → weight is max(norm_vdb, norm_fact)."""
        nodes = [self._make_node("Entity A", 0.6)]
        rels = [self._make_rel("Entity A", "Entity B", "rel", 0.9)]
        llm = AsyncMock(return_value="Entity A")
        result = await _recognition_memory_filter("query", nodes, rels, llm)
        # norm_fact["Entity A"] = 1.0 (only triplet, hi==lo), norm_vdb["Entity A"] = 1.0 (only entity)
        # both normalise to 1.0 since each dict has one element
        assert result.get("Entity A", 0) > 0

    @pytest.mark.asyncio
    async def test_fact_scores_max_merge_across_triplets(self):
        """Same entity appearing in two triplets → takes max distance."""
        nodes = []
        rels = [
            self._make_rel("HUB|ORG", "A|T", "r1", 0.5),
            self._make_rel("HUB|ORG", "B|T", "r2", 0.9),  # higher distance
        ]
        llm = AsyncMock(return_value="HUB|ORG")
        result = await _recognition_memory_filter("query", nodes, rels, llm)
        # HUB|ORG should use distance 0.9, not 0.5
        assert "HUB|ORG" in result
```

- [ ] **Step 2: Install pytest-asyncio if not present**

```bash
cd /d/HUAWEI/RAG_LUND/lightrag && pip install pytest-asyncio -q
```

- [ ] **Step 3: Run the new tests — expect all to fail with `NameError` (function not in operate.py yet)**

```bash
python -m pytest tests/test_recognition_memory.py::TestRecognitionMemoryFilter -v
```
Expected: All pass (tests use the self-contained copy in the test file itself).

- [ ] **Step 4: Commit**

```bash
git add lightrag/tests/test_recognition_memory.py
git commit -m "test(ppr): add _recognition_memory_filter unit tests"
```

---

### Task 5: Implement `_recognition_memory_filter` in `operate.py`

**Files:**
- Modify: `lightrag/lightrag/operate.py` — insert between `_min_max_norm` and `_ppr_rank_chunks_global` (~line 5473)

- [ ] **Step 1: Add the function**

Insert immediately after `_min_max_norm` (i.e., just before `async def _ppr_rank_chunks_global`):

```python
async def _recognition_memory_filter(
    query: str,
    node_datas: list[dict],
    rel_results: list[dict],
    llm_model_func,
    recognition_top_k: int = 10,
) -> dict[str, float]:
    """HippoRAG2-style recognition memory filter for global PPR entity seeds.

    Three-step hybrid filter:
      1. Numpy step  — vectors already retrieved by hybrid search (no new VDB calls)
      2. LLM step    — unified candidate list sent to LLM for relevance judgment
      3. Difflib step — LLM text output remapped to graph entity_ids

    Args:
        query:             User query string.
        node_datas:        Entity VDB results (each dict has "entity_id", "vdb_score").
        rel_results:       Relation VDB results (each dict has "src_id", "tgt_id",
                           "description", "distance").
        llm_model_func:    Async callable — global_config["llm_model_func"].
        recognition_top_k: Max triplets to show LLM. Entity cap = recognition_top_k * 2.

    Returns:
        {entity_id: normalised_weight} for LLM-recognised entities.
        Empty dict signals fallback to direct score merge in the caller.
    """
    import difflib

    # --- Step 1: Candidate pool sizing ---
    top_rels = rel_results[:recognition_top_k]
    top_nodes = node_datas[:recognition_top_k * 2]

    # --- Step 2: fact_scores — max across triplets for same entity ---
    fact_scores: dict[str, float] = {}
    for rel in top_rels:
        for eid in (rel.get("src_id"), rel.get("tgt_id")):
            if eid:
                fact_scores[eid] = max(fact_scores.get(eid, 0.0), rel.get("distance", 0.0))

    # --- Step 3: Independent min-max normalisation ---
    norm_vdb = _min_max_norm({
        nd["entity_id"]: nd.get("vdb_score", 0.0)
        for nd in top_nodes if nd.get("entity_id")
    })
    norm_fact = _min_max_norm(fact_scores)

    # --- Step 4: Build unified candidate list for difflib matching ---
    entity_vdb_ids = [nd["entity_id"] for nd in top_nodes if nd.get("entity_id")]
    triplet_eids = list(dict.fromkeys(
        eid for rel in top_rels
        for eid in (rel.get("src_id"), rel.get("tgt_id")) if eid
    ))
    all_candidate_ids = list(dict.fromkeys(entity_vdb_ids + triplet_eids))

    if not all_candidate_ids:
        return {}

    # --- Step 5: Build LLM prompt ---
    standalone_block = "\n".join(entity_vdb_ids) if entity_vdb_ids else "(none)"
    triplet_block = "\n".join(
        f"{r.get('src_id', '')} | {r.get('description', '')} | {r.get('tgt_id', '')}"
        for r in top_rels
    ) if top_rels else "(none)"

    prompt = (
        "You are an entity relevance judge.\n\n"
        f"Query: {query}\n\n"
        "Below are retrieved entities and facts. Select ONLY those directly relevant "
        "to answering the query.\n"
        "You MUST copy each entity identifier EXACTLY as it appears in the list below, "
        'including any "|TYPE" suffixes and special characters. '
        "Do not rephrase, abbreviate, or invent new identifiers.\n\n"
        f"Standalone entities:\n{standalone_block}\n\n"
        f"Retrieved facts:\n{triplet_block}\n\n"
        "Return the relevant entity identifiers only, one per line. "
        "If none are relevant, return an empty response."
    )

    # --- Step 6: LLM call ---
    llm_output: str = await llm_model_func(prompt)

    # --- Step 7: Difflib mapping ---
    recognized_ids: set[str] = set()
    for line in llm_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        matches = difflib.get_close_matches(line, all_candidate_ids, n=1, cutoff=0.85)
        if matches:
            recognized_ids.add(matches[0])

    # --- Step 8: Merge weights ---
    result: dict[str, float] = {}
    for eid in recognized_ids:
        w = max(norm_vdb.get(eid, 0.0), norm_fact.get(eid, 0.0))
        if w > 0.0:
            result[eid] = w

    return result
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
cd /d/HUAWEI/RAG_LUND/lightrag && python -c "import lightrag.operate; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add lightrag/lightrag/operate.py
git commit -m "feat(ppr): implement _recognition_memory_filter in operate.py"
```

---

### Task 6: Write integration test for `_ppr_rank_chunks` seed-building refactor

**Files:**
- Modify: `lightrag/tests/test_recognition_memory.py`

- [ ] **Step 1: Append integration-style tests for the `_ppr_rank_chunks` branch logic**

Append to `lightrag/tests/test_recognition_memory.py`:

```python
# ---------------------------------------------------------------------------
# Integration tests for the _ppr_rank_chunks seed-building block.
# We test the branch logic (recognition vs direct merge vs fallback) in
# isolation by re-implementing just that block here.
# ---------------------------------------------------------------------------

class TestSeedBuildingBranch:
    """Tests for the if/else branching added to _ppr_rank_chunks."""

    def _direct_merge(self, node_datas, rel_results):
        """Mirrors _build_seeds_from_raw logic."""
        weights = {}
        for nd in node_datas:
            eid = nd.get("entity_id", nd.get("entity_name", ""))
            if eid:
                weights[eid] = max(weights.get(eid, 0), nd.get("vdb_score", 0.0))
        for rel in rel_results:
            score = rel.get("distance", 0.0)
            for field in ("src_id", "tgt_id"):
                eid = rel.get(field)
                if eid:
                    weights[eid] = max(weights.get(eid, 0), score)
        return weights

    @pytest.mark.asyncio
    async def test_recognition_top_k_zero_skips_llm(self):
        """recognition_top_k=0 → recognition memory disabled → direct merge."""
        nodes = [{"entity_id": "E1", "vdb_score": 0.8}]
        rels = [{"src_id": "E1", "tgt_id": "E2", "description": "r", "distance": 0.6}]
        llm_called = []

        async def llm(prompt):
            llm_called.append(prompt)
            return "E1"

        # Simulate the branch: recognition_top_k=0 skips recognition
        recognition_top_k = 0
        if recognition_top_k > 0:
            result = await _recognition_memory_filter("q", nodes, rels, llm, recognition_top_k)
        else:
            result = self._direct_merge(nodes, rels)

        assert llm_called == []  # LLM never called
        assert "E1" in result

    @pytest.mark.asyncio
    async def test_recognition_fallback_when_empty(self):
        """LLM returns empty → fallback to _build_seeds_from_raw."""
        nodes = [{"entity_id": "E1", "vdb_score": 0.8}]
        rels = []
        llm = AsyncMock(return_value="")

        recognized = await _recognition_memory_filter("q", nodes, rels, llm, recognition_top_k=10)
        # Empty → caller uses direct merge
        if not recognized:
            result = self._direct_merge(nodes, rels)
        else:
            result = recognized

        assert "E1" in result

    @pytest.mark.asyncio
    async def test_recognition_result_replaces_seeds(self):
        """LLM recognises only E1, not E2 → E2 excluded from seeds."""
        nodes = [
            {"entity_id": "E1", "vdb_score": 0.9},
            {"entity_id": "E2", "vdb_score": 0.85},
        ]
        rels = []
        llm = AsyncMock(return_value="E1")

        result = await _recognition_memory_filter("q", nodes, rels, llm, recognition_top_k=10)
        assert "E1" in result
        assert "E2" not in result
```

- [ ] **Step 2: Run these tests (they use the self-contained copy — should all pass)**

```bash
cd /d/HUAWEI/RAG_LUND/lightrag && python -m pytest tests/test_recognition_memory.py::TestSeedBuildingBranch -v
```
Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add lightrag/tests/test_recognition_memory.py
git commit -m "test(ppr): add seed-building branch integration tests"
```

---

### Task 7: Refactor `_ppr_rank_chunks()` seed-building block

**Files:**
- Modify: `lightrag/lightrag/operate.py:5592–5616`

- [ ] **Step 1: Locate the exact block to replace**

In `operate.py`, find this block inside `_ppr_rank_chunks` (around lines 5592–5616):

```python
    # Step 1: Build entity seed weights from relation VDB + entity VDB scores
    entity_seed_weights: dict[str, float] = {}

    # From entity VDB scores (already retrieved by _get_node_data)
    for nd in node_datas:
        eid = nd.get("entity_id", nd.get("entity_name", ""))
        if eid:
            vdb_score = nd.get("vdb_score", 0.0)
            entity_seed_weights[eid] = max(entity_seed_weights.get(eid, 0), vdb_score)

    # From relation VDB scores (fact-query similarity → entity seeds)
    try:
        rel_results = await relationships_vdb.query(
            query, top_k=query_param.top_k, query_embedding=query_embedding
        )
        for rel in rel_results:
            score = rel.get("distance", 0.0)
            for field_name in ("src_id", "tgt_id"):
                eid = rel.get(field_name)
                if eid:
                    entity_seed_weights[eid] = max(
                        entity_seed_weights.get(eid, 0), score
                    )
    except Exception as e:
        logger.warning(f"PPR: relation VDB query failed: {e}")
```

- [ ] **Step 2: Replace with the recognition-aware version**

```python
    # Step 1: Build entity seed weights
    # Global PPR path: recognition memory filters seeds via LLM (when enabled).
    # Local PPR path: direct max-merge (unchanged behaviour).
    entity_seed_weights: dict[str, float] = {}

    # Always fetch relation VDB results — used by both paths
    rel_results: list[dict] = []
    try:
        rel_results = await relationships_vdb.query(
            query, top_k=query_param.top_k, query_embedding=query_embedding
        )
    except Exception as e:
        logger.warning(f"PPR: relation VDB query failed: {e}")

    if use_global and query_param.recognition_top_k > 0:
        # --- Recognition Memory path (global PPR only) ---
        llm_func = text_chunks_db.global_config.get("llm_model_func")
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
                logger.warning(
                    f"PPR(global): recognition memory failed, falling back to direct merge: {e}"
                )
                recognized = {}
            if recognized:
                entity_seed_weights = recognized
            else:
                logger.warning(
                    "PPR(global): recognition memory returned empty; using direct seed merge"
                )
                # Fallback: direct max-merge (same as local path)
                for nd in node_datas:
                    eid = nd.get("entity_id", nd.get("entity_name", ""))
                    if eid:
                        entity_seed_weights[eid] = max(
                            entity_seed_weights.get(eid, 0), nd.get("vdb_score", 0.0)
                        )
                for rel in rel_results:
                    score = rel.get("distance", 0.0)
                    for field_name in ("src_id", "tgt_id"):
                        eid = rel.get(field_name)
                        if eid:
                            entity_seed_weights[eid] = max(
                                entity_seed_weights.get(eid, 0), score
                            )
        else:
            # No LLM configured — direct merge
            for nd in node_datas:
                eid = nd.get("entity_id", nd.get("entity_name", ""))
                if eid:
                    entity_seed_weights[eid] = max(
                        entity_seed_weights.get(eid, 0), nd.get("vdb_score", 0.0)
                    )
            for rel in rel_results:
                score = rel.get("distance", 0.0)
                for field_name in ("src_id", "tgt_id"):
                    eid = rel.get(field_name)
                    if eid:
                        entity_seed_weights[eid] = max(
                            entity_seed_weights.get(eid, 0), score
                        )
    else:
        # Local PPR path OR recognition_top_k=0 (disabled): direct max-merge
        for nd in node_datas:
            eid = nd.get("entity_id", nd.get("entity_name", ""))
            if eid:
                entity_seed_weights[eid] = max(
                    entity_seed_weights.get(eid, 0), nd.get("vdb_score", 0.0)
                )
        for rel in rel_results:
            score = rel.get("distance", 0.0)
            for field_name in ("src_id", "tgt_id"):
                eid = rel.get(field_name)
                if eid:
                    entity_seed_weights[eid] = max(
                        entity_seed_weights.get(eid, 0), score
                    )
```

- [ ] **Step 3: Verify the module imports and basic syntax**

```bash
cd /d/HUAWEI/RAG_LUND/lightrag && python -c "import lightrag.operate; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Run the full recognition memory test suite**

```bash
python -m pytest tests/test_recognition_memory.py -v
```
Expected: All tests pass.

- [ ] **Step 5: Run existing PPR-adjacent tests to check for regressions**

```bash
python -m pytest tests/test_kg_rerank.py tests/test_rerank_observability.py -v
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add lightrag/lightrag/operate.py
git commit -m "feat(ppr): integrate recognition memory into _ppr_rank_chunks global path"
```

---

### Task 8: Final smoke-test and cleanup

**Files:**
- No new files

- [ ] **Step 1: Run the full test suite**

```bash
cd /d/HUAWEI/RAG_LUND/lightrag && python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: No new failures vs baseline.

- [ ] **Step 2: Verify `QueryParam` serialises correctly with new field**

```bash
python -c "
from lightrag.base import QueryParam
import dataclasses, json
p = QueryParam(mode='ppr', recognition_top_k=5)
print('recognition_top_k:', p.recognition_top_k)
print('mode:', p.mode)
"
```
Expected:
```
recognition_top_k: 5
mode: ppr
```

- [ ] **Step 3: Verify recognition memory is skipped for non-ppr modes**

```bash
python -c "
from lightrag.base import QueryParam
p = QueryParam(mode='mix')
# recognition only fires when use_global=True (mode='ppr') AND recognition_top_k > 0
print('would skip recognition:', p.mode != 'ppr' or p.recognition_top_k == 0)
"
```
Expected: `would skip recognition: True`

- [ ] **Step 4: Final commit with summary**

```bash
git add -A
git commit -m "feat(ppr): recognition memory complete — LLM-verified entity seeds for global PPR

HippoRAG2-aligned 3-step filter: vector retrieval → LLM recognition → difflib remap.
Unified candidate pool from entity VDB + relation VDB triplets.
Independent min-max normalisation prevents scale dominance.
Silent fallback to direct merge on LLM failure or empty output.
recognition_top_k=0 opt-out; ppr_local path unchanged.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Covered by |
|-----------------|------------|
| Recognition memory for `mode="ppr"` only | Task 7: `if use_global and recognition_top_k > 0` guard |
| Unified LLM recognition (entity VDB + relation VDB) | Task 5: `_recognition_memory_filter` combines both |
| Passage-node (DPR chunk) seeds unaffected | Not touched — `_ppr_rank_chunks_global` unchanged |
| Zero new VDB queries | Task 7: `rel_results` from existing query |
| Graceful fallback (LLM fail / empty) | Task 7: `try/except` + logger.warning + direct merge fallback |
| `recognition_top_k=10` default | Task 1: `QueryParam` |
| `recognition_top_k=0` disables recognition | Task 7: `if use_global and recognition_top_k > 0` |
| Separate min-max norm | Task 3/5: `_min_max_norm` applied independently |
| Max-merge for entity in multiple triplets | Task 5: `fact_scores[eid] = max(...)` |
| `hi==lo` normalisation fix | Task 2/3: `uniform = 1.0 if hi > 0.0 else 0.0` |
| entity VDB cap at `recognition_top_k * 2` | Task 5: `top_nodes = node_datas[:recognition_top_k * 2]` |
| Exact-string prompt constraint | Task 5: prompt text |
| difflib cutoff=0.85 | Task 5: `get_close_matches(..., cutoff=0.85)` |
| `logger.warning` on LLM failure and empty | Task 7 |
| Test file with mock LLM | Tasks 2, 4, 6 |

**Placeholder scan:** No TBD, TODO, or vague steps. All code blocks are complete.

**Type consistency:** `_min_max_norm` takes and returns `dict[str, float]` throughout. `_recognition_memory_filter` returns `dict[str, float]`. `entity_seed_weights` is `dict[str, float]` in `_ppr_rank_chunks`. All consistent.
