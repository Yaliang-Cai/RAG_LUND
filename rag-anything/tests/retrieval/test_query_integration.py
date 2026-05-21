# tests/retrieval/test_query_integration.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lightrag import QueryParam
from raganything.query import QueryMixin

pytestmark = pytest.mark.asyncio


def _make_mixin(chunks: list[dict]) -> QueryMixin:
    """Build a minimal QueryMixin stand-in."""
    mixin = MagicMock(spec=QueryMixin)
    mixin.lightrag = MagicMock()
    mixin.lightrag.llm_model_func = AsyncMock(return_value="answer text")
    mixin.logger = MagicMock()
    mixin.callback_manager = None

    async def fake_ensure_initialized():
        return {"success": True}

    mixin._ensure_lightrag_initialized = fake_ensure_initialized
    mixin._generate_answer_from_chunks = AsyncMock(return_value="answer text")
    return mixin


async def test_aquery_auto_mode_calls_router():
    mixin = _make_mixin([])
    router_mock = MagicMock()
    router_mock.route = AsyncMock(return_value=(
        [{"chunk_id": "c1", "content": "answer chunk", "file_path": "f.pdf"}],
        {"profile": "semantic", "confidence": 0.9, "reasoning": "r",
         "paths_activated": ["hybrid"], "paths_failed": [],
         "chunks_per_path": {"hybrid": 1}, "chunks_after_rrf": 1,
         "chunks_after_rerank": 1, "chunks_after_threshold": 1,
         "latency_per_path": {"classifier": 0.1, "hybrid": 0.3}},
    ))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        result = await QueryMixin.aquery(mixin, "test query", mode="auto")

    router_mock.route.assert_called_once()
    assert isinstance(result, str)


async def test_aquery_auto_mode_passes_profile_kwarg():
    mixin = _make_mixin([])
    router_mock = MagicMock()
    router_mock.route = AsyncMock(return_value=([], {
        "profile": "semantic", "confidence": 1.0, "reasoning": "",
        "paths_activated": [], "paths_failed": [],
        "chunks_per_path": {}, "chunks_after_rrf": 0,
        "chunks_after_rerank": 0, "chunks_after_threshold": 0,
        "latency_per_path": {"classifier": 0.0},
    }))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        await QueryMixin.aquery(mixin, "CVE-2026-001", mode="auto", profile="semantic")

    _, call_kwargs = router_mock.route.call_args
    assert call_kwargs.get("profile_name") == "semantic"


async def test_aquery_non_auto_mode_unchanged():
    """Non-auto modes must not touch the router at all."""
    mixin = _make_mixin([])
    mixin.lightrag.aquery = AsyncMock(return_value="legacy answer")

    with patch("raganything.query.RetrievalRouter") as router_cls:
        result = await QueryMixin.aquery(mixin, "test", mode="hybrid")

    router_cls.assert_not_called()
    assert result == "legacy answer"


async def test_aquery_vlm_enhanced_auto_mode_uses_router():
    """VLM enhanced + mode=auto: router provides chunks, then image dereference runs."""
    mixin = _make_mixin([])
    mixin.vision_model_func = AsyncMock(return_value="vlm answer")

    test_chunks = [
        {"chunk_id": "c1", "content": "Image Path: /data/img.jpg\nsome text", "file_path": "f.pdf"}
    ]
    router_mock = MagicMock()
    router_mock.route = AsyncMock(return_value=(test_chunks, {
        "profile": "descriptive", "confidence": 0.8, "reasoning": "r",
        "paths_activated": ["mix"], "paths_failed": [],
        "chunks_per_path": {"mix": 1}, "chunks_after_rrf": 1,
        "chunks_after_rerank": 1, "chunks_after_threshold": 1,
        "latency_per_path": {"classifier": 0.2, "mix": 0.5},
    }))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        with patch.object(mixin, "_process_image_paths_for_vlm",
                          new=AsyncMock(return_value=("processed prompt", []))):
            with patch.object(mixin, "_generate_text_answer_from_retrieval_prompt",
                              new=AsyncMock(return_value="text answer")):
                result = await QueryMixin.aquery_vlm_enhanced(
                    mixin, "describe image", mode="auto"
                )

    router_mock.route.assert_called_once()
    assert isinstance(result, str)


async def test_aquery_auto_return_trace_includes_routing():
    mixin = _make_mixin([])
    router_mock = MagicMock()
    routing_trace = {
        "profile": "semantic",
        "confidence": 0.9,
        "reasoning": "factual query",
        "paths_activated": ["hybrid", "naive"],
        "paths_failed": [],
        "chunks_per_path": {"hybrid": 3, "naive": 2},
        "chunks_after_rrf": 4,
        "chunks_after_rerank": 3,
        "chunks_after_threshold": 3,
        "latency_per_path": {"classifier": 0.12, "hybrid": 0.45, "naive": 0.08},
    }
    router_mock.route = AsyncMock(return_value=([], routing_trace))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        result = await QueryMixin.aquery(
            mixin, "test query", mode="auto", return_trace=True
        )

    assert isinstance(result, dict)
    assert "answer" in result
    assert "trace" in result
    assert result["trace"]["routing"]["profile"] == "semantic"
    assert "latency_per_path" in result["trace"]["routing"]
    assert result["trace"]["routing"]["latency_per_path"]["classifier"] == 0.12
