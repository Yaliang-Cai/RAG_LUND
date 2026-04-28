import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys


@pytest.fixture(autouse=True)
def reset_singleton():
    import raganything.retrieval.gfm_retriever as mod
    mod._instance = None
    yield
    mod._instance = None


class TestGFMRetrieverWrapper:
    async def test_raises_without_data_name(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        with pytest.raises(RuntimeError, match="GFM_DATA_NAME"):
            GFMRetrieverWrapper.get_instance("./data", "", "model")

    async def test_raises_when_gfmrag_not_installed(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        with patch.dict(sys.modules, {"gfmrag": None}):
            with pytest.raises(ImportError, match="gfmrag"):
                GFMRetrieverWrapper.get_instance("./data", "graph", "model")

    async def test_get_instance_returns_singleton(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = MagicMock()
        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            a = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            b = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
        assert a is b
        mock_mod.GFMRetriever.from_index.assert_called_once()

    async def test_retrieve_maps_chunk_ids_to_content(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = {
            "document": [
                {"id": "chunk_abc", "score": 0.9},
                {"id": "chunk_def", "score": 0.7},
            ]
        }
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = mock_inner

        mock_kv = AsyncMock()
        mock_kv.get_by_id.side_effect = lambda cid: {
            "chunk_abc": {"content": "France is a country."},
            "chunk_def": {"content": "Paris is the capital."},
        }.get(cid)

        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            wrapper = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            result = await wrapper.retrieve("Who is president?", top_k=5, text_chunks_kv=mock_kv)

        assert len(result) == 2
        assert result[0] == {"chunk_id": "chunk_abc", "content": "France is a country.", "score": 0.9}
        assert result[1] == {"chunk_id": "chunk_def", "content": "Paris is the capital.", "score": 0.7}

    async def test_retrieve_skips_chunk_ids_missing_from_kv(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = {
            "document": [{"id": "chunk_gone", "score": 0.8}]
        }
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = mock_inner

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = None

        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            wrapper = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            result = await wrapper.retrieve("query", top_k=5, text_chunks_kv=mock_kv)

        assert result == []

    async def test_retrieve_handles_non_dict_chunk_data(self):
        from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper
        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = {
            "document": [{"id": "chunk_str", "score": 0.5}]
        }
        mock_mod = MagicMock()
        mock_mod.GFMRetriever.from_index.return_value = mock_inner

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = "raw string content"

        with patch.dict(sys.modules, {"gfmrag": mock_mod}):
            wrapper = GFMRetrieverWrapper.get_instance("./data", "graph", "model")
            result = await wrapper.retrieve("query", top_k=5, text_chunks_kv=mock_kv)

        assert len(result) == 1
        assert result[0]["content"] == "raw string content"
