import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

# Stub heavy optional deps so local_rag.py can be imported without GPU packages
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
async def test_stream_query_agentic_yields_meta_with_trace():
    service = _make_service()
    fake_result = {
        "answer": "72.3% top-1 accuracy",
        "confidence": 0.91,
        "grounded": True,
        "trace": {
            "profile": "precise",
            "router_cache_hit": False,
            "retrieve_cycles_used": 2,
            "check_cycles_used": 1,
            "rewrite_history": [],
            "sub_questions": None,
        },
    }
    service.query_with_trace = AsyncMock(return_value=fake_result)

    events = []
    async for event in service.stream_query("ws1", "test query", mode="agentic"):
        events.append(event)

    assert events[0]["type"] == "meta"
    trace = events[0]["metadata"]["agentic_trace"]
    assert trace["confidence"] == 0.91
    assert trace["grounded"] is True
    assert trace["profile"] == "precise"
    assert trace["retrieve_cycles_used"] == 2

    assert events[1]["type"] == "chunk"
    assert events[1]["text"] == "72.3% top-1 accuracy"

    assert len(events) == 2


@pytest.mark.asyncio
async def test_stream_query_auto_with_profile_uses_query_with_trace():
    service = _make_service()
    fake_result = {
        "answer": "answer text",
        "confidence": 0.85,
        "grounded": True,
        "trace": {
            "routing": {
                "profile": "multihop",
                "confidence": 0.85,
                "paths_activated": ["ppr", "naive"],
                "chunks_after_rrf": 20,
                "chunks_after_rerank": 10,
                "chunks_after_threshold": 8,
                "latency_per_path": {"ppr": 0.4, "naive": 0.1},
            }
        },
    }
    service.query_with_trace = AsyncMock(return_value=fake_result)

    events = []
    async for event in service.stream_query("ws1", "test", mode="auto", profile="multihop"):
        events.append(event)

    service.query_with_trace.assert_awaited_once()
    call_kwargs = service.query_with_trace.call_args
    assert call_kwargs.kwargs.get("profile") == "multihop" or "multihop" in str(call_kwargs)

    assert events[0]["type"] == "meta"
    assert "routing_trace" in events[0]["metadata"]
    assert events[1]["type"] == "chunk"
