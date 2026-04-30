# tests/test_agentic_query_mode.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_rag(vision_model_func=None):
    rag = MagicMock()
    rag.lightrag = MagicMock()
    rag.lightrag.llm_model_func = AsyncMock(return_value="mocked answer")
    rag.logger = MagicMock()
    rag.vision_model_func = vision_model_func
    rag._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})
    return rag


async def test_agentic_mode_returns_string_no_vlm():
    """mode='agentic' without vision_model_func: no vlm_generate_fn passed."""
    from raganything.query import QueryMixin

    rag = _make_rag(vision_model_func=None)

    with patch("raganything.retrieval.agent_graph.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value="agentic answer")
        result = await QueryMixin.aquery(rag, "test query", mode="agentic")

    assert result == "agentic answer"
    call_kwargs = MockGraph.call_args.kwargs
    assert call_kwargs.get("vlm_generate_fn") is None
    instance.run.assert_called_once_with("test query", return_trace=False)


async def test_agentic_mode_passes_vlm_generate_fn_when_vision_available():
    """mode='agentic' with vision_model_func: vlm_generate_fn is a callable."""
    from raganything.query import QueryMixin

    rag = _make_rag(vision_model_func=AsyncMock(return_value="vlm result"))
    rag._process_image_paths_for_vlm = AsyncMock(return_value=("prompt", ["b64img"]))
    rag._build_vlm_messages_with_images = MagicMock(return_value=[])
    rag._call_vlm_with_multimodal_content = AsyncMock(return_value="vlm answer")

    with patch("raganything.retrieval.agent_graph.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value="agentic answer")
        result = await QueryMixin.aquery(rag, "test query", mode="agentic")

    assert result == "agentic answer"
    call_kwargs = MockGraph.call_args.kwargs
    assert callable(call_kwargs.get("vlm_generate_fn"))


async def test_agentic_mode_passes_return_trace():
    from raganything.query import QueryMixin

    rag = _make_rag(vision_model_func=None)

    with patch("raganything.retrieval.agent_graph.AdaptiveAgentGraph") as MockGraph:
        instance = MockGraph.return_value
        instance.run = AsyncMock(return_value={"answer": "x", "trace": {}})
        result = await QueryMixin.aquery(rag, "test query", mode="agentic", return_trace=True)

    assert result == {"answer": "x", "trace": {}}
    instance.run.assert_called_once_with("test query", return_trace=True)
