import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

for _mod in [
    "sentence_transformers",
    "sentence_transformers.cross_encoder",
    "torch",
    "raganything.processor",
    "raganything.batch",
    "raganything.batch_parser",
    "raganything.raganything",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import raganything as _ra
_ra.RAGAnything = MagicMock()
_ra.RAGAnythingConfig = MagicMock()


def _make_service():
    from raganything.services.local_rag import LocalRagService, LocalRagSettings
    settings = MagicMock(spec=LocalRagSettings)
    settings.working_dir = "/tmp/test"
    settings.output_dir = "/tmp/out"
    service = LocalRagService.__new__(LocalRagService)
    service.settings = settings
    service.logger = MagicMock()
    return service


@pytest.mark.asyncio
async def test_stream_query_agentic_passes_conversation_history():
    """conversation_history must be forwarded to query_with_trace in agentic branch."""
    service = _make_service()
    service.query_with_trace = AsyncMock(return_value={
        "answer": "answer",
        "confidence": 0.9,
        "grounded": True,
        "trace": {"profile": "precise", "retrieve_cycles_used": 1, "check_cycles_used": 0},
    })
    history = [{"role": "user", "content": "prev question"},
               {"role": "assistant", "content": "prev answer"}]

    events = []
    async for event in service.stream_query(
        "ws1", "follow-up", mode="agentic", conversation_history=history
    ):
        events.append(event)

    call_kwargs = service.query_with_trace.call_args.kwargs
    assert call_kwargs.get("conversation_history") == history


@pytest.mark.asyncio
async def test_stream_query_vlm_calls_query_with_trace():
    """vlm_enhanced=True must use query_with_trace(vlm_enhanced=True) and yield meta+chunk."""
    service = _make_service()
    service.query_with_trace = AsyncMock(return_value={
        "answer": "VLM answer about the image",
    })

    events = []
    async for event in service.stream_query(
        "ws1", "what is in the image?", mode="hybrid", vlm_enhanced=True
    ):
        events.append(event)

    service.query_with_trace.assert_awaited_once()
    call_kwargs = service.query_with_trace.call_args.kwargs
    assert call_kwargs.get("vlm_enhanced") is True

    assert events[0]["type"] == "meta"
    assert events[1]["type"] == "chunk"
    assert events[1]["text"] == "VLM answer about the image"
    assert len(events) == 2


@pytest.mark.asyncio
async def test_stream_query_vlm_passes_conversation_history():
    """VLM branch must also forward conversation_history."""
    service = _make_service()
    service.query_with_trace = AsyncMock(return_value={"answer": "ok"})
    history = [{"role": "user", "content": "earlier"}]

    async for _ in service.stream_query(
        "ws1", "q", mode="hybrid", vlm_enhanced=True, conversation_history=history
    ):
        pass

    call_kwargs = service.query_with_trace.call_args.kwargs
    assert call_kwargs.get("conversation_history") == history
