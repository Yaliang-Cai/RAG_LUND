import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_query_mixin_obj():
    """Build a minimal QueryMixin instance with required attributes mocked."""
    from raganything.query import QueryMixin
    obj = QueryMixin.__new__(QueryMixin)
    obj.lightrag = MagicMock()
    obj.lightrag.text_chunks = AsyncMock()
    obj._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})
    obj._generate_answer_from_chunks = AsyncMock(return_value="France is in Europe.")
    obj.logger = MagicMock()
    obj.callback_manager = None
    obj.vision_model_func = None
    return obj


class TestQueryGFMMode:
    async def test_aquery_gfm_returns_answer_string(self):
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = [
            {"chunk_id": "c1", "content": "France is in Europe.", "score": 0.9}
        ]

        with patch("raganything.query.GFMRetrieverWrapper") as MockCls, \
             patch("raganything.query.GFM_DATA_DIR", "./data"), \
             patch("raganything.query.GFM_DATA_NAME", "graph"), \
             patch("raganything.query.GFM_MODEL_PATH", "model"):
            MockCls.get_instance.return_value = mock_wrapper
            obj = _make_query_mixin_obj()
            result = await obj.aquery("Where is France?", mode="gfm")

        assert result == "France is in Europe."
        obj._generate_answer_from_chunks.assert_called_once()

    async def test_aquery_gfm_return_trace_true(self):
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = [
            {"chunk_id": "c1", "content": "x", "score": 0.5},
            {"chunk_id": "c2", "content": "y", "score": 0.4},
        ]

        with patch("raganything.query.GFMRetrieverWrapper") as MockCls, \
             patch("raganything.query.GFM_DATA_DIR", "./data"), \
             patch("raganything.query.GFM_DATA_NAME", "graph"), \
             patch("raganything.query.GFM_MODEL_PATH", "model"):
            MockCls.get_instance.return_value = mock_wrapper
            obj = _make_query_mixin_obj()
            result = await obj.aquery("query", mode="gfm", return_trace=True)

        assert isinstance(result, dict)
        assert result["answer"] == "France is in Europe."
        assert result["trace"]["mode"] == "gfm"
        assert result["trace"]["chunks_retrieved"] == 2

    async def test_aquery_gfm_passes_chunk_top_k(self):
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = []

        with patch("raganything.query.GFMRetrieverWrapper") as MockCls, \
             patch("raganything.query.GFM_DATA_DIR", "./data"), \
             patch("raganything.query.GFM_DATA_NAME", "graph"), \
             patch("raganything.query.GFM_MODEL_PATH", "model"):
            MockCls.get_instance.return_value = mock_wrapper
            obj = _make_query_mixin_obj()
            await obj.aquery("query", mode="gfm", chunk_top_k=7)

        mock_wrapper.retrieve.assert_called_once()
        call_args = mock_wrapper.retrieve.call_args
        assert call_args[0][1] == 7  # top_k positional arg
