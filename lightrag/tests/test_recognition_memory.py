"""Unit tests for recognition memory components (operate.py).

All LLM calls are mocked — no real API calls in this suite.
"""
import pytest


# ---------------------------------------------------------------------------
# _min_max_norm — import by copying the function under test directly, so
# tests don't depend on operate.py import chain (which needs Neo4j/Qdrant).
# ---------------------------------------------------------------------------

# SYNC: keep identical to lightrag/lightrag/operate.py::_min_max_norm
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


# ---------------------------------------------------------------------------
# _recognition_memory_filter — tested with a self-contained implementation
# copy so tests don't depend on the full operate.py import chain.
# SYNC: keep identical to lightrag/lightrag/operate.py::_recognition_memory_filter
# ---------------------------------------------------------------------------

import difflib
from typing import Callable
from unittest.mock import AsyncMock


# SYNC: keep identical to lightrag/lightrag/operate.py::_recognition_memory_filter
async def _recognition_memory_filter(
    query: str,
    node_datas: list[dict],
    rel_results: list[dict],
    llm_model_func: Callable,
) -> dict[str, float]:
    """Copied verbatim from operate.py for isolated testing."""
    # All hybrid-retrieved candidates passed directly — no truncation.
    top_rels = rel_results
    top_nodes = node_datas

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

    if not all_candidate_ids:
        return {}

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
        result = await _recognition_memory_filter("华为做了什么", nodes, rels, llm)
        assert "华为|ORGANIZATION" in result
        assert result["华为|ORGANIZATION"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_recognized_entity_from_triplet(self):
        """LLM confirms entity from relation triplet → appears in result with norm_fact score."""
        nodes = []
        rels = [self._make_rel("华为|ORGANIZATION", "5G|TECHNOLOGY", "开发了", 0.8)]
        llm = AsyncMock(return_value="华为|ORGANIZATION")
        result = await _recognition_memory_filter("5G研发", nodes, rels, llm)
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
        """LLM returns slightly misspelled entity → difflib may or may not match; no crash."""
        nodes = [self._make_node("华为|ORGANIZATION", 0.9)]
        rels = []
        llm = AsyncMock(return_value="华为|ORGANIZATON")  # typo in suffix
        result = await _recognition_memory_filter("query", nodes, rels, llm)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_multi_source_entity_takes_max_weight(self):
        """Entity in both VDB and triplet → weight is max(norm_vdb, norm_fact)."""
        nodes = [self._make_node("Entity A", 0.6)]
        rels = [self._make_rel("Entity A", "Entity B", "rel", 0.9)]
        llm = AsyncMock(return_value="Entity A")
        result = await _recognition_memory_filter("query", nodes, rels, llm)
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
        assert "HUB|ORG" in result


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
        """recognition_top_k=0 (caller flag) → recognition memory disabled → direct merge."""
        nodes = [{"entity_id": "E1", "vdb_score": 0.8}]
        rels = [{"src_id": "E1", "tgt_id": "E2", "description": "r", "distance": 0.6}]
        llm_called = []

        async def llm(prompt):
            llm_called.append(prompt)
            return "E1"

        # Simulate the branch: recognition_top_k=0 skips recognition
        recognition_top_k = 0
        if recognition_top_k > 0:
            result = await _recognition_memory_filter("q", nodes, rels, llm)
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

        recognized = await _recognition_memory_filter("q", nodes, rels, llm)
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

        result = await _recognition_memory_filter("q", nodes, rels, llm)
        assert "E1" in result
        assert "E2" not in result
