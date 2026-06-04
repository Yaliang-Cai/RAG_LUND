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
    param = QueryParam(mode="hybrid")
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


async def test_global_kg_path_uses_global_mode():
    lg = _make_lightrag([{"chunk_id": "c1", "content": "edge evidence", "file_path": "a.pdf"}])
    param = QueryParam(mode="hybrid")
    chunks, latency = await run_path("global_kg", "relationship query", param, lg, overrides={})
    call_param = lg.aquery_data.call_args[0][1]
    assert call_param.mode == "global"
    assert len(chunks) == 1
    assert latency >= 0.0


async def test_overrides_applied():
    lg = _make_lightrag([])
    param = QueryParam(mode="naive")
    await run_path("mix", "test", param, lg, overrides={"kg_chunk_selection_source": "untruncated"})
    call_param = lg.aquery_data.call_args[0][1]
    assert call_param.kg_chunk_selection_source == "untruncated"


async def test_profile_overrides_take_precedence_over_default_qdrant_mode():
    lg = _make_lightrag([])
    param = QueryParam(mode="hybrid", qdrant_retrieval_mode="dense")
    await run_path(
        "local_kg",
        "test",
        param,
        lg,
        overrides={"qdrant_retrieval_mode": "hybrid", "exclude_synonym_edges": True},
    )
    call_param = lg.aquery_data.call_args[0][1]
    assert call_param.mode == "local"
    assert call_param.qdrant_retrieval_mode == "hybrid"
    assert call_param.exclude_synonym_edges is True


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
    from raganything.retrieval.paths import KNOWN_PATHS
    # KNOWN_PATHS includes "gfm" special-case which is not in _PATH_CONFIG
    assert set(_PATH_CONFIG.keys()) | {"gfm"} == KNOWN_PATHS
