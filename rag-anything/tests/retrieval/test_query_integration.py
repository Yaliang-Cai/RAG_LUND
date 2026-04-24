# tests/retrieval/test_query_integration.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lightrag import QueryParam
from raganything.query import QueryMixin


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
    return mixin


async def test_aquery_auto_mode_calls_router():
    mixin = _make_mixin([])
    router_mock = MagicMock()
    router_mock.route = AsyncMock(return_value=(
        [{"chunk_id": "c1", "content": "answer chunk", "file_path": "f.pdf"}],
        {"profile": "local", "confidence": 0.9, "reasoning": "r",
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
        "profile": "precise", "confidence": 1.0, "reasoning": "",
        "paths_activated": [], "paths_failed": [],
        "chunks_per_path": {}, "chunks_after_rrf": 0,
        "chunks_after_rerank": 0, "chunks_after_threshold": 0,
        "latency_per_path": {"classifier": 0.0},
    }))

    with patch("raganything.query.RetrievalRouter", return_value=router_mock):
        await QueryMixin.aquery(mixin, "CVE-2026-001", mode="auto", profile="precise")

    _, call_kwargs = router_mock.route.call_args
    assert call_kwargs.get("profile_name") == "precise"


async def test_aquery_non_auto_mode_unchanged():
    """Non-auto modes must not touch the router at all."""
    mixin = _make_mixin([])
    mixin.lightrag.aquery = AsyncMock(return_value="legacy answer")

    with patch("raganything.query.RetrievalRouter") as router_cls:
        result = await QueryMixin.aquery(mixin, "test", mode="hybrid")

    router_cls.assert_not_called()
