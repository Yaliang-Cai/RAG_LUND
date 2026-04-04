from unittest.mock import patch

import pytest

from lightrag.base import QueryParam
from lightrag.utils import apply_rerank_if_enabled, process_chunks_unified


@pytest.mark.asyncio
async def test_apply_rerank_logs_with_item_label():
    async def mock_rerank(query=None, documents=None, top_n=None, **kwargs):
        return [{"index": 0, "relevance_score": 0.9}]

    docs = [{"content": "entity text"}]
    with patch("lightrag.utils.logger.info") as mock_info:
        reranked = await apply_rerank_if_enabled(
            query="q",
            retrieved_docs=docs,
            global_config={"rerank_model_func": mock_rerank},
            enable_rerank=True,
            top_n=1,
            item_label="entities",
        )

    assert len(reranked) == 1
    assert reranked[0]["rerank_score"] == 0.9
    assert any(
        "Successfully reranked: 1 entities from 1 original entities" in str(call.args[0])
        for call in mock_info.call_args_list
    )


@pytest.mark.asyncio
async def test_process_chunks_unified_top_k_scope():
    seen = {}

    async def mock_rerank(query=None, documents=None, top_n=None, **kwargs):
        seen["top_n"] = top_n
        return [
            {"index": i, "relevance_score": 1.0 - i * 0.01}
            for i in range(min(top_n, len(documents)))
        ]

    chunks = [{"content": f"chunk-{i}", "chunk_id": f"c{i}"} for i in range(5)]
    query_param = QueryParam(
        enable_rerank=True,
        chunk_top_k=2,
        rerank_score_scope="top_k",
    )
    rerank_debug = {}

    with patch("lightrag.utils.logger.info") as mock_info:
        result = await process_chunks_unified(
            query="what is this",
            unique_chunks=chunks,
            query_param=query_param,
            global_config={
                "rerank_model_func": mock_rerank,
                "min_rerank_score": 0.0,
            },
            rerank_debug=rerank_debug,
        )

    assert seen["top_n"] == 2
    assert len(result) == 2
    assert rerank_debug["scope"] == "top_k"
    assert len(rerank_debug["scores_all"]) == 2
    assert any(
        "Rerank scores (all reranked chunks):" in str(call.args[0])
        for call in mock_info.call_args_list
    )
    assert any(
        "Rerank scores (final kept chunks):" in str(call.args[0])
        for call in mock_info.call_args_list
    )


@pytest.mark.asyncio
async def test_process_chunks_unified_all_scope_and_invalid_fallback():
    seen = {}

    async def mock_rerank(query=None, documents=None, top_n=None, **kwargs):
        seen["top_n"] = top_n
        return [
            {"index": i, "relevance_score": 0.9 - i * 0.01}
            for i in range(min(top_n, len(documents)))
        ]

    chunks = [{"content": f"chunk-{i}", "chunk_id": f"c{i}"} for i in range(5)]

    # all scope
    query_param_all = QueryParam(
        enable_rerank=True,
        chunk_top_k=2,
        rerank_score_scope="all",
    )
    rerank_debug_all = {}
    await process_chunks_unified(
        query="q",
        unique_chunks=chunks,
        query_param=query_param_all,
        global_config={"rerank_model_func": mock_rerank, "min_rerank_score": 0.0},
        rerank_debug=rerank_debug_all,
    )
    assert seen["top_n"] == 5
    assert rerank_debug_all["scope"] == "all"
    assert rerank_debug_all["count_after_rerank"] == 5

    # invalid scope -> fallback to all
    query_param_invalid = QueryParam(enable_rerank=True, chunk_top_k=2)
    query_param_invalid.rerank_score_scope = "invalid"
    rerank_debug_invalid = {}
    with patch("lightrag.utils.logger.warning") as mock_warning:
        await process_chunks_unified(
            query="q",
            unique_chunks=chunks,
            query_param=query_param_invalid,
            global_config={"rerank_model_func": mock_rerank, "min_rerank_score": 0.0},
            rerank_debug=rerank_debug_invalid,
        )
    assert seen["top_n"] == 5
    assert rerank_debug_invalid["scope"] == "all"
    assert any(
        "Unknown rerank_score_scope" in str(call.args[0])
        for call in mock_warning.call_args_list
    )


@pytest.mark.asyncio
async def test_process_chunks_unified_string_rerank_score_threshold():
    async def mock_rerank(query=None, documents=None, top_n=None, **kwargs):
        return [
            {"index": 0, "relevance_score": "0.95"},
            {"index": 1, "relevance_score": "0.20"},
        ]

    chunks = [{"content": "chunk-a", "chunk_id": "c1"}, {"content": "chunk-b", "chunk_id": "c2"}]
    query_param = QueryParam(enable_rerank=True, chunk_top_k=2, rerank_score_scope="all")
    rerank_debug = {}

    result = await process_chunks_unified(
        query="q",
        unique_chunks=chunks,
        query_param=query_param,
        global_config={"rerank_model_func": mock_rerank, "min_rerank_score": 0.5},
        rerank_debug=rerank_debug,
    )

    assert len(result) == 1
    assert rerank_debug["count_after_threshold"] == 1
    assert rerank_debug["scores_final"] == [0.95]
