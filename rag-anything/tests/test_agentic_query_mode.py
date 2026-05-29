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
    # aquery threads retrieval knobs (top_k / chunk_top_k / enable_rerank /
    # qdrant_retrieval_mode) into the graph constructor since
    # 4798f9b ("thread query params into agentic graph"). The exact kwarg
    # set is verified elsewhere — here we just confirm the lightrag
    # instance is the first positional arg and the call happened once.
    MockGraph.assert_called_once()
    call_args = MockGraph.call_args
    assert call_args.args[0] is rag.lightrag
    instance.run.assert_called_once_with("test query", return_trace=False)


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
    instance.run.assert_called_once_with("test query", return_trace=True)
