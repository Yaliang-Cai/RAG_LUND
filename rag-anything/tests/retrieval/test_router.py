import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lightrag import QueryParam
from raganything.retrieval.router import (
    RetrievalRouter,
    RetrievalError,
    _select_with_path_floors,
    _weighted_rrf_merge,
)
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


def test_weighted_rrf_attaches_source_paths():
    chunks = [{"chunk_id": "shared", "content": "hello"}]
    result = _weighted_rrf_merge(
        {"ppr": chunks, "qdrant_chunks_hybrid": chunks},
        {"ppr": 2.0, "qdrant_chunks_hybrid": 0.7},
        k=60,
    )
    assert result[0]["rrf_source_paths"] == ["ppr", "qdrant_chunks_hybrid"]
    assert result[0]["rrf_sources"]["ppr"]["rank"] == 1


def test_path_floor_reserves_low_rank_ppr_candidates():
    chunks = [
        {"chunk_id": f"h{i}", "rrf_source_paths": ["qdrant_chunks_hybrid"]}
        for i in range(5)
    ] + [
        {"chunk_id": f"p{i}", "rrf_source_paths": ["ppr"]}
        for i in range(3)
    ]
    final, trace = _select_with_path_floors(
        chunks,
        chunk_top_k=5,
        path_floors={"ppr": 3},
    )
    ids = {chunk["chunk_id"] for chunk in final}
    assert {"p0", "p1", "p2"}.issubset(ids)
    assert len(final) == 5
    assert trace["ppr"] == {"requested": 3, "available": 3, "selected": 3}


def test_path_floor_uses_available_ppr_without_duplicates():
    chunks = [
        {"chunk_id": "h0", "rrf_source_paths": ["qdrant_chunks_hybrid"]},
        {"chunk_id": "p0", "rrf_source_paths": ["ppr"]},
        {"chunk_id": "p0", "rrf_source_paths": ["ppr"]},
    ]
    final, trace = _select_with_path_floors(
        chunks,
        chunk_top_k=5,
        path_floors={"ppr": 3},
    )
    assert [chunk["chunk_id"] for chunk in final] == ["h0", "p0"]
    assert trace["ppr"] == {"requested": 3, "available": 1, "selected": 1}


# ── RetrievalRouter integration tests ───────────────────────────────────────

def _make_router_lightrag(profile_chunks: dict[str, list[dict]]):
    """Build a lightrag mock whose aquery_data returns pre-defined chunks per path."""
    lightrag = MagicMock()
    lightrag.llm_model_func = AsyncMock()

    async def fake_aquery_data(query, param):
        name_map = {
            ("naive",  "dense"):  "naive",
            ("naive",  "bm25"):   "qdrant_sparse",
            ("naive",  "hybrid"): "qdrant_chunks_hybrid",
            ("hybrid", "dense"):  "hybrid",
            ("hybrid", "hybrid"): "qdrant_hybrid",
            ("global", "dense"):  "global_kg",
            ("mix",    "dense"):  "mix",
            ("ppr",    "dense"):  "ppr",
        }
        qdrant_mode = getattr(param, "qdrant_retrieval_mode", "dense")
        key = (param.mode, qdrant_mode)
        path_name = name_map.get(key, param.mode)
        return {
            "status": "success",
            "data": {"chunks": profile_chunks.get(path_name, []), "entities": [], "relations": []},
        }

    lightrag.aquery_data = fake_aquery_data
    return lightrag


async def test_router_returns_chunks_and_trace():
    chunks = [{"chunk_id": f"c{i}", "content": f"text {i}", "file_path": "f.pdf"} for i in range(5)]
    lightrag = _make_router_lightrag({"hybrid": chunks[:3], "naive": chunks[3:]})

    with patch("raganything.retrieval.router.apply_rerank_if_enabled",
               new=AsyncMock(side_effect=lambda **kw: kw["retrieved_docs"])):
        router = RetrievalRouter(lightrag, llm_func=AsyncMock())
        param = QueryParam(mode="hybrid", chunk_top_k=10)
        final_chunks, trace = await router.route("test query", param, profile_name="local")

    assert isinstance(final_chunks, list)
    assert isinstance(trace, dict)
    assert trace["profile"] == "local"
    assert "latency_per_path" in trace
    assert "classifier" in trace["latency_per_path"]


async def test_router_trace_has_all_fields():
    lightrag = _make_router_lightrag({"hybrid": [{"chunk_id": "c1", "content": "hi", "file_path": "f"}]})
    with patch("raganything.retrieval.router.apply_rerank_if_enabled",
               new=AsyncMock(side_effect=lambda **kw: kw["retrieved_docs"])):
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
    lightrag = _make_router_lightrag({"hybrid": [{"chunk_id": "c1", "content": "x", "file_path": "f"}]})
    with patch("raganything.retrieval.router.apply_rerank_if_enabled",
               new=AsyncMock(side_effect=lambda **kw: kw["retrieved_docs"])):
        router = RetrievalRouter(lightrag, llm_func=AsyncMock())
        router._classifier.classify = AsyncMock(side_effect=lambda q: classifier_called.append(q))
        param = QueryParam(mode="hybrid", chunk_top_k=10)
        await router.route("q", param, profile_name="local")

    assert classifier_called == [], "Classifier should not be called when profile_name is explicit"
