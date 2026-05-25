import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


class _FakeAgentGraph:
    """Captures the retrieval knobs the agentic branch threads in, and replays
    a canned astream_run() so we exercise the AdaptiveAgentGraph-based path."""

    last_init_kwargs: dict = {}

    def __init__(self, lightrag, **kwargs):
        _FakeAgentGraph.last_init_kwargs = kwargs

    async def astream_run(self, query):
        yield "step", "Choosing retrieval profile"
        yield "final", {
            "answer": "72.3% top-1 accuracy",
            "confidence": 0.91,
            "grounded": True,
            "ungrounded_claims": [],
            "chunks": [],
            "trace": {
                "profile": "semantic",
                "router_cache_hit": False,
                "retrieve_cycles_used": 2,
                "check_cycles_used": 1,
                "rewrite_history": [],
                "sub_questions": None,
            },
        }


def _agentic_rag_instance():
    rag_instance = MagicMock()
    rag_instance._ensure_lightrag_initialized = AsyncMock()
    rag_instance.lightrag = MagicMock()
    return rag_instance


@pytest.mark.asyncio
async def test_stream_query_agentic_yields_meta_with_trace():
    service = _make_service()
    service.get_rag = AsyncMock(return_value=_agentic_rag_instance())

    events = []
    with patch(
        "raganything.retrieval.agent_graph.AdaptiveAgentGraph", _FakeAgentGraph
    ):
        async for event in service.stream_query("ws1", "test query", mode="agentic"):
            events.append(event)

    # reasoning steps stream first, then meta, then the answer chunk
    assert any(e["type"] == "reasoning" for e in events)
    meta = next(e for e in events if e["type"] == "meta")
    trace = meta["metadata"]["agentic_trace"]
    assert trace["confidence"] == 0.91
    assert trace["grounded"] is True
    assert trace["profile"] == "semantic"
    assert trace["retrieve_cycles_used"] == 2

    chunk_events = [e for e in events if e["type"] == "chunk"]
    assert chunk_events[0]["text"] == "72.3% top-1 accuracy"


@pytest.mark.asyncio
async def test_stream_query_agentic_threads_retrieval_params():
    """Frontend top_k/chunk_top_k/enable_rerank/qdrant_retrieval_mode must reach
    AdaptiveAgentGraph instead of being dropped (regression: agentic used
    LightRAG's QueryParam defaults 40/20)."""
    service = _make_service()
    service.get_rag = AsyncMock(return_value=_agentic_rag_instance())

    with patch(
        "raganything.retrieval.agent_graph.AdaptiveAgentGraph", _FakeAgentGraph
    ):
        async for _ in service.stream_query(
            "ws1", "q", mode="agentic",
            top_k=10, chunk_top_k=5, enable_rerank=False,
            qdrant_retrieval_mode="bm25",
        ):
            pass

    kw = _FakeAgentGraph.last_init_kwargs
    assert kw["top_k"] == 10
    assert kw["chunk_top_k"] == 5
    assert kw["enable_rerank"] is False
    assert kw["qdrant_retrieval_mode"] == "bm25"


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
