from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client import models

from lightrag.base import QueryParam
from lightrag.kg.qdrant_impl import QdrantVectorDBStorage, compute_mdhash_id_for_qdrant
from lightrag.operate import _get_vector_context
from lightrag.utils import EmbeddingFunc


@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 4 for _ in texts])

    return EmbeddingFunc(
        embedding_dim=4,
        func=embed_func,
        model_name="test-model",
    )


@pytest.fixture
def storage(monkeypatch, mock_embedding_func):
    monkeypatch.setenv("QDRANT_ENABLE_SPARSE_BM25", "true")
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }
    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )
    storage._client = MagicMock()
    storage._sparse_vector_name = "bm25"
    storage._enable_sparse_bm25 = True
    storage._search_ef = None
    return storage


def _query_response(*ids: str):
    return SimpleNamespace(
        points=[
            SimpleNamespace(
                payload={"id": point_id, "content": f"content {point_id}"},
                score=1.0 - idx * 0.1,
            )
            for idx, point_id in enumerate(ids)
        ]
    )


def _sparse_embedding():
    return SimpleNamespace(
        indices=np.array([1, 3], dtype=np.int32),
        values=np.array([0.7, 0.2], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_dense_retrieval_mode_uses_dense_query(storage):
    storage._client.query_points.return_value = _query_response("dense-1")

    results = await storage.query(
        "graph search",
        top_k=3,
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        qdrant_retrieval_mode="dense",
    )

    call_kwargs = storage._client.query_points.call_args.kwargs
    assert call_kwargs["query"] == [0.1, 0.2, 0.3, 0.4]
    assert "prefetch" not in call_kwargs
    assert call_kwargs.get("using") is None
    assert results[0]["id"] == "dense-1"
    assert results[0]["distance"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_bm25_retrieval_mode_uses_sparse_query(storage):
    sparse_model = MagicMock()
    sparse_model.query_embed.return_value = [_sparse_embedding()]
    storage._client._get_or_init_sparse_model.return_value = sparse_model
    storage._client.query_points.return_value = _query_response("bm25-1")

    results = await storage.query(
        "graph search",
        top_k=3,
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        qdrant_retrieval_mode="bm25",
    )

    call_kwargs = storage._client.query_points.call_args.kwargs
    assert isinstance(call_kwargs["query"], models.SparseVector)
    assert call_kwargs["query"].indices == [1, 3]
    assert call_kwargs["query"].values == pytest.approx([0.7, 0.2])
    assert call_kwargs["using"] == "bm25"
    assert results[0]["id"] == "bm25-1"


@pytest.mark.asyncio
async def test_hybrid_retrieval_mode_uses_rrf_fusion(storage):
    sparse_model = MagicMock()
    sparse_model.query_embed.return_value = [_sparse_embedding()]
    storage._client._get_or_init_sparse_model.return_value = sparse_model
    storage._client.query_points.return_value = _query_response("hybrid-1")

    results = await storage.query(
        "graph search",
        top_k=3,
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        qdrant_retrieval_mode="hybrid",
    )

    call_kwargs = storage._client.query_points.call_args.kwargs
    assert isinstance(call_kwargs["query"], models.FusionQuery)
    assert call_kwargs["query"].fusion == models.Fusion.RRF
    assert len(call_kwargs["prefetch"]) == 2
    dense_prefetch, sparse_prefetch = call_kwargs["prefetch"]
    assert dense_prefetch.query == [0.1, 0.2, 0.3, 0.4]
    assert dense_prefetch.using is None
    assert isinstance(sparse_prefetch.query, models.SparseVector)
    assert sparse_prefetch.using == "bm25"
    assert results[0]["id"] == "hybrid-1"


@pytest.mark.asyncio
async def test_hybrid_candidate_pool_is_restricted_by_point_ids(storage):
    sparse_model = MagicMock()
    sparse_model.query_embed.return_value = [_sparse_embedding()]
    storage._client._get_or_init_sparse_model.return_value = sparse_model
    storage._client.query_points.return_value = _query_response("hybrid-1")

    await storage.query(
        "graph search",
        top_k=2,
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        qdrant_retrieval_mode="hybrid",
        candidate_ids=["chunk-1", "chunk-2"],
    )

    call_kwargs = storage._client.query_points.call_args.kwargs
    expected_ids = [
        compute_mdhash_id_for_qdrant("chunk-1", prefix=storage.effective_workspace),
        compute_mdhash_id_for_qdrant("chunk-2", prefix=storage.effective_workspace),
    ]

    assert isinstance(call_kwargs["query_filter"], models.Filter)
    has_id_conditions = [
        condition
        for condition in call_kwargs["query_filter"].must
        if isinstance(condition, models.HasIdCondition)
    ]
    assert len(has_id_conditions) == 1
    assert sorted(has_id_conditions[0].has_id) == sorted(expected_ids)

    for prefetch in call_kwargs["prefetch"]:
        prefetch_has_id_conditions = [
            condition
            for condition in prefetch.filter.must
            if isinstance(condition, models.HasIdCondition)
        ]
        assert len(prefetch_has_id_conditions) == 1
        assert sorted(prefetch_has_id_conditions[0].has_id) == sorted(expected_ids)


@pytest.mark.asyncio
async def test_query_param_retrieval_mode_is_forwarded_to_vector_context():
    class QdrantVectorDBStorage:
        cosine_better_than_threshold = 0.2

        def __init__(self):
            self.received_mode = None

        async def query(
            self,
            query,
            top_k,
            query_embedding=None,
            qdrant_retrieval_mode=None,
        ):
            self.received_mode = qdrant_retrieval_mode
            return [{"id": "chunk-1", "content": "content"}]

    chunks_vdb = QdrantVectorDBStorage()
    query_param = QueryParam(
        mode="naive",
        chunk_top_k=1,
        qdrant_retrieval_mode="bm25",
    )

    chunks = await _get_vector_context(
        "graph search",
        chunks_vdb,
        query_param,
        query_embedding=[0.1, 0.2, 0.3, 0.4],
    )

    assert chunks_vdb.received_mode == "bm25"
    assert chunks[0]["chunk_id"] == "chunk-1"
