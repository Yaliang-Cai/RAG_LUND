import dataclasses
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@dataclasses.dataclass
class FakeParam:
    mode: str = "hybrid"
    chunk_top_k: int = 10


class TestGFMPath:
    def test_gfm_in_known_paths(self):
        from raganything.retrieval.paths import KNOWN_PATHS
        assert "gfm" in KNOWN_PATHS

    async def test_run_path_gfm_calls_wrapper_retrieve(self):
        from raganything.retrieval.paths import run_path

        mock_chunks = [{"chunk_id": "c1", "content": "hello", "score": 0.9}]
        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = mock_chunks

        mock_lightrag = MagicMock()
        mock_lightrag.text_chunks = AsyncMock()

        with patch("raganything.retrieval.paths.GFMRetrieverWrapper") as MockCls:
            MockCls.get_instance.return_value = mock_wrapper
            with patch("raganything.retrieval.paths.GFM_DATA_DIR", "./data"), \
                 patch("raganything.retrieval.paths.GFM_DATA_NAME", "test_graph"), \
                 patch("raganything.retrieval.paths.GFM_MODEL_PATH", "model"):
                chunks, latency = await run_path(
                    "gfm", "Who is the president?", FakeParam(), mock_lightrag, {}
                )

        assert chunks == mock_chunks
        assert latency >= 0.0
        mock_wrapper.retrieve.assert_called_once_with(
            "Who is the president?", 10, mock_lightrag.text_chunks
        )

    async def test_run_path_gfm_does_not_call_aquery_data(self):
        from raganything.retrieval.paths import run_path

        mock_wrapper = AsyncMock()
        mock_wrapper.retrieve.return_value = []
        mock_lightrag = MagicMock()
        mock_lightrag.text_chunks = AsyncMock()

        with patch("raganything.retrieval.paths.GFMRetrieverWrapper") as MockCls:
            MockCls.get_instance.return_value = mock_wrapper
            with patch("raganything.retrieval.paths.GFM_DATA_DIR", "./data"), \
                 patch("raganything.retrieval.paths.GFM_DATA_NAME", "graph"), \
                 patch("raganything.retrieval.paths.GFM_MODEL_PATH", "model"):
                await run_path("gfm", "query", FakeParam(), mock_lightrag, {})

        mock_lightrag.aquery_data.assert_not_called()
