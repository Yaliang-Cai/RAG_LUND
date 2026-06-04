# tests/test_agentic_query_mode.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


async def test_agentic_mode_returns_string():
    """mode='agentic' must call AdaptiveAgentGraph and return its result."""
    from raganything.query import QueryMixin

    rag = MagicMock()
    rag.lightrag = MagicMock()
    rag.lightrag.llm_model_func = AsyncMock(return_value="mocked answer")
    rag.logger = MagicMock()
    rag._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})

    with patch("raganything.retrieval.agent_graph.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value="agentic answer")
        result = await QueryMixin.aquery(rag, "test query", mode="agentic")

    assert result == "agentic answer"
    MockGraph.assert_called_once_with(rag.lightrag)
    _, call_kwargs = instance.run.call_args
    assert instance.run.call_args.args == ("test query",)
    assert call_kwargs["return_trace"] is False
    assert "qdrant_retrieval_mode" in call_kwargs


async def test_agentic_mode_passes_return_trace():
    from raganything.query import QueryMixin

    rag = MagicMock()
    rag.lightrag = MagicMock()
    rag.logger = MagicMock()
    rag._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})

    with patch("raganything.retrieval.agent_graph.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value={"answer": "x", "trace": {}})
        result = await QueryMixin.aquery(rag, "test query", mode="agentic", return_trace=True)

    assert result == {"answer": "x", "trace": {}}
    _, call_kwargs = instance.run.call_args
    assert instance.run.call_args.args == ("test query",)
    assert call_kwargs["return_trace"] is True
    assert "qdrant_retrieval_mode" in call_kwargs


async def test_agentic_mode_passes_query_kwargs():
    from raganything.query import QueryMixin

    rag = MagicMock()
    rag.lightrag = MagicMock()
    rag.logger = MagicMock()
    rag._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})

    with patch("raganything.retrieval.agent_graph.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value={"answer": "x", "trace": {}})
        await QueryMixin.aquery(
            rag,
            "test query",
            mode="agentic",
            return_trace=True,
            chunk_top_k=7,
            top_k=9,
            ppr_top_k=11,
        )

    _, call_kwargs = instance.run.call_args
    assert instance.run.call_args.args == ("test query",)
    assert call_kwargs["return_trace"] is True
    assert call_kwargs["chunk_top_k"] == 7
    assert call_kwargs["top_k"] == 9
    assert call_kwargs["ppr_top_k"] == 11
    assert "qdrant_retrieval_mode" in call_kwargs


async def test_agentic_v2_mode_uses_v2_graph_and_passes_query_kwargs():
    from raganything.query import QueryMixin

    rag = MagicMock()
    rag.lightrag = MagicMock()
    rag.logger = MagicMock()
    rag._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})

    with patch("raganything.retrieval.agent_graph_v2.AdaptiveAgentGraphV2") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value={"answer": "x", "trace": {"data": {"chunks": []}}})
        result = await QueryMixin.aquery(
            rag,
            "test query",
            mode="agentic_v2",
            return_trace=True,
            chunk_top_k=7,
            top_k=9,
        )

    assert result == {"answer": "x", "trace": {"data": {"chunks": []}}}
    MockGraph.assert_called_once_with(rag.lightrag)
    assert instance.run.call_args.args == ("test query",)
    _, call_kwargs = instance.run.call_args
    assert call_kwargs["return_trace"] is True
    assert call_kwargs["chunk_top_k"] == 7
    assert call_kwargs["top_k"] == 9
    assert "qdrant_retrieval_mode" in call_kwargs
