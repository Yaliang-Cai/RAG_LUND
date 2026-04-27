from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import (
    _apply_token_truncation,
    _build_context_str,
    _build_query_cache_params,
    _merge_all_chunks,
    _keyword_rrf_query_vector_storage,
    _query_vector_storage,
    _resolve_kg_chunk_selection_inputs,
    _rrf_merge_ranked_records,
    extract_keywords_only,
    kg_query,
)
from lightrag.prompt import PROMPTS
from lightrag.utils import pick_by_vector_similarity, process_chunks_unified


class _DummyTokenizer:
    def encode(self, text: str):
        return [ord(ch) for ch in str(text)]


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
        return [{"id": "item-1", "content": "content"}]


@pytest.mark.asyncio
async def test_query_param_accepts_retrieval_ablation_fields():
    param = QueryParam(
        mode="hybrid",
        keyword_fanout_mode="per_keyword_rrf",
        keyword_entity_rrf_k=10,
        keyword_relation_rrf_k=20,
        answer_context_mode="chunk_only_prompt",
        entity_qdrant_retrieval_mode="hybrid",
        chunk_qdrant_retrieval_mode="dense",
        bypass_query_cache=True,
        bypass_keywords_cache=True,
        kg_chunk_selection_source="untruncated",
    )

    assert param.keyword_fanout_mode == "per_keyword_rrf"
    assert param.keyword_entity_rrf_k == 10
    assert param.keyword_relation_rrf_k == 20
    assert param.answer_context_mode == "chunk_only_prompt"
    assert param.entity_qdrant_retrieval_mode == "hybrid"
    assert param.chunk_qdrant_retrieval_mode == "dense"
    assert param.bypass_query_cache is True
    assert param.bypass_keywords_cache is True
    assert param.kg_chunk_selection_source == "untruncated"


@pytest.mark.asyncio
async def test_keyword_rrf_query_vector_storage_uses_store_specific_rrf_k():
    query_param = QueryParam(
        keyword_entity_rrf_k=10,
        keyword_relation_rrf_k=20,
    )

    async def fake_query_vector_storage(
        vector_storage,
        query,
        top_k,
        query_param,
        query_embedding=None,
        *,
        store_kind="generic",
        candidate_ids=None,
    ):
        if store_kind == "entity":
            return [{"entity_id": f"{query}-entity", "entity_name": f"{query}-entity"}]
        return [{"src_id": query, "tgt_id": "Target", "description": "rel"}]

    with patch(
        "lightrag.operate._query_vector_storage",
        new=AsyncMock(side_effect=fake_query_vector_storage),
    ), patch(
        "lightrag.operate._rrf_merge_ranked_records",
        side_effect=lambda ranking_lists, **kwargs: [{"rrf_score": kwargs["rrf_k"]}],
    ) as mock_merge:
        entity_merged, _ = await _keyword_rrf_query_vector_storage(
            SimpleNamespace(),
            ["Alpha", "Beta"],
            5,
            query_param,
            store_kind="entity",
        )
        relation_merged, _ = await _keyword_rrf_query_vector_storage(
            SimpleNamespace(),
            ["uses", "evaluates"],
            5,
            query_param,
            store_kind="relation",
        )

    assert entity_merged == [{"rrf_score": 10}]
    assert relation_merged == [{"rrf_score": 20}]
    assert mock_merge.call_args_list[0].kwargs["rrf_k"] == 10
    assert mock_merge.call_args_list[1].kwargs["rrf_k"] == 20


def test_kg_chunk_selection_source_defaults_to_truncated_inputs():
    raw_entities = [{"entity_name": "raw-entity"}]
    raw_relations = [{"src_id": "raw-src", "tgt_id": "raw-tgt"}]
    truncated_entities = [{"entity_name": "truncated-entity"}]
    truncated_relations = [{"src_id": "truncated-src", "tgt_id": "truncated-tgt"}]

    entities, relations, source = _resolve_kg_chunk_selection_inputs(
        search_result={
            "final_entities": raw_entities,
            "final_relations": raw_relations,
        },
        truncation_result={
            "filtered_entities": truncated_entities,
            "filtered_relations": truncated_relations,
        },
        query_param=QueryParam(),
    )

    assert source == "truncated"
    assert entities is truncated_entities
    assert relations is truncated_relations


def test_kg_chunk_selection_source_can_use_untruncated_inputs():
    raw_entities = [{"entity_name": "raw-entity"}]
    raw_relations = [{"src_id": "raw-src", "tgt_id": "raw-tgt"}]
    truncated_entities = [{"entity_name": "truncated-entity"}]
    truncated_relations = [{"src_id": "truncated-src", "tgt_id": "truncated-tgt"}]

    entities, relations, source = _resolve_kg_chunk_selection_inputs(
        search_result={
            "final_entities": raw_entities,
            "final_relations": raw_relations,
        },
        truncation_result={
            "filtered_entities": truncated_entities,
            "filtered_relations": truncated_relations,
        },
        query_param=QueryParam(kg_chunk_selection_source="untruncated"),
    )

    assert source == "untruncated"
    assert entities is raw_entities
    assert relations is raw_relations


@pytest.mark.asyncio
async def test_chunk_only_prompt_preserves_truncated_kg_selection_inputs():
    search_result = {
        "final_entities": [
            {
                "entity_name": "EntityA",
                "entity_type": "concept",
                "description": "short",
                "file_path": "doc-a",
            },
            {
                "entity_name": "EntityB",
                "entity_type": "concept",
                "description": "x" * 200,
                "file_path": "doc-b",
            },
        ],
        "final_relations": [
            {
                "src_id": "EntityA",
                "tgt_id": "EntityC",
                "description": "short rel",
                "file_path": "doc-r1",
            },
            {
                "src_id": "EntityB",
                "tgt_id": "EntityD",
                "description": "y" * 200,
                "file_path": "doc-r2",
            },
        ],
    }
    global_config = {"tokenizer": _DummyTokenizer()}

    kg_result = await _apply_token_truncation(
        search_result,
        QueryParam(
            mode="hybrid",
            answer_context_mode="kg_prompt",
            max_entity_tokens=80,
            max_relation_tokens=90,
        ),
        global_config,
    )
    chunk_only_result = await _apply_token_truncation(
        search_result,
        QueryParam(
            mode="hybrid",
            answer_context_mode="chunk_only_prompt",
            max_entity_tokens=80,
            max_relation_tokens=90,
        ),
        global_config,
    )

    assert [item["entity_name"] for item in kg_result["filtered_entities"]] == ["EntityA"]
    assert [
        (item["src_id"], item["tgt_id"]) for item in kg_result["filtered_relations"]
    ] == [("EntityA", "EntityC")]
    assert chunk_only_result["filtered_entities"] == kg_result["filtered_entities"]
    assert chunk_only_result["filtered_relations"] == kg_result["filtered_relations"]
    assert chunk_only_result["entities_context"] == []
    assert chunk_only_result["relations_context"] == []


@pytest.mark.asyncio
async def test_merge_all_chunks_defers_ppr_qa_top_k_until_post_rerank():
    ppr_chunks = [
        {"chunk_id": "c1", "content": "chunk-1", "ppr_score": 0.9},
        {"chunk_id": "c2", "content": "chunk-2", "ppr_score": 0.8},
        {"chunk_id": "c3", "content": "chunk-3", "ppr_score": 0.7},
    ]

    merged = await _merge_all_chunks(
        filtered_entities=[],
        filtered_relations=[],
        vector_chunks=[],
        query="q",
        query_param=QueryParam(mode="ppr", ppr_top_k=3, ppr_qa_top_k=2),
        ppr_chunks=ppr_chunks,
    )

    assert [chunk["chunk_id"] for chunk in merged] == ["c1", "c2", "c3"]


@pytest.mark.asyncio
async def test_process_chunks_unified_applies_ppr_qa_top_k_after_rerank():
    candidate_chunks = [
        {"chunk_id": "c1", "content": "chunk-1"},
        {"chunk_id": "c2", "content": "chunk-2"},
        {"chunk_id": "c3", "content": "chunk-3"},
    ]
    reranked_chunks = [
        {"chunk_id": "c3", "content": "chunk-3", "rerank_score": 0.93},
        {"chunk_id": "c1", "content": "chunk-1", "rerank_score": 0.91},
        {"chunk_id": "c2", "content": "chunk-2", "rerank_score": 0.77},
    ]

    async def fake_apply_rerank_if_enabled(**kwargs):
        assert [chunk["chunk_id"] for chunk in kwargs["retrieved_docs"]] == [
            "c1",
            "c2",
            "c3",
        ]
        assert kwargs["top_n"] == 3
        return reranked_chunks

    with patch(
        "lightrag.utils.apply_rerank_if_enabled",
        new=AsyncMock(side_effect=fake_apply_rerank_if_enabled),
    ):
        final_chunks = await process_chunks_unified(
            query="q",
            unique_chunks=candidate_chunks,
            query_param=QueryParam(
                mode="ppr",
                ppr_top_k=3,
                ppr_qa_top_k=2,
                chunk_top_k=0,
                enable_rerank=True,
                rerank_score_scope="all",
            ),
            global_config={"min_rerank_score": 0.0},
            source_type="ppr",
        )

    assert [chunk["chunk_id"] for chunk in final_chunks] == ["c3", "c1"]


def test_query_cache_params_include_kg_chunk_selection_source():
    params = _build_query_cache_params(
        QueryParam(
            kg_chunk_selection_source="untruncated",
            keyword_entity_rrf_k=10,
            keyword_relation_rrf_k=20,
        ),
        user_prompt="",
        system_prompt=None,
        history_signature=None,
        hl_keywords_str="",
        ll_keywords_str="",
    )

    assert params["kg_chunk_selection_source"] == "untruncated"
    assert params["keyword_entity_rrf_k"] == 10
    assert params["keyword_relation_rrf_k"] == 20


@pytest.mark.asyncio
async def test_query_vector_storage_prefers_store_specific_qdrant_mode():
    storage = QdrantVectorDBStorage()
    query_param = QueryParam(
        mode="hybrid",
        qdrant_retrieval_mode="dense",
        entity_qdrant_retrieval_mode="hybrid",
        chunk_qdrant_retrieval_mode="dense",
    )

    await _query_vector_storage(
        storage,
        "graph search",
        3,
        query_param,
        store_kind="entity",
    )
    assert storage.received_mode == "hybrid"

    await _query_vector_storage(
        storage,
        "graph search",
        3,
        query_param,
        store_kind="relation",
    )
    assert storage.received_mode == "hybrid"

    await _query_vector_storage(
        storage,
        "graph search",
        3,
        query_param,
        store_kind="chunk",
    )
    assert storage.received_mode == "dense"


@pytest.mark.asyncio
async def test_extract_keywords_bypass_skips_keywords_cache():
    param = QueryParam(mode="hybrid", bypass_keywords_cache=True)
    global_config = {
        "addon_params": {"language": "English"},
        "enable_keyword_case_normalization": False,
        "entity_uppercase_allowlist": [],
    }
    hashing_kv = SimpleNamespace(global_config={"enable_llm_cache": True})

    async def fake_llm(prompt: str, **kwargs):
        return (
            '{"high_level_keywords":["graph retrieval"],'
            '"low_level_keywords":["LightRAG"]}'
        )

    global_config["llm_model_func"] = fake_llm
    global_config["tokenizer"] = _DummyTokenizer()

    with patch("lightrag.operate.handle_cache", new=AsyncMock()) as mock_cache, patch(
        "lightrag.operate.save_to_cache", new=AsyncMock()
    ) as mock_save:
        hl_keywords, ll_keywords = await extract_keywords_only(
            "How does LightRAG retrieval work?",
            param,
            global_config,
            hashing_kv,
        )

    assert hl_keywords == ["graph retrieval"]
    assert ll_keywords == ["LightRAG"]
    mock_cache.assert_not_called()
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_kg_query_bypass_skips_query_cache():
    query_param = QueryParam(mode="hybrid", bypass_query_cache=True)
    global_config = {
        "llm_model_func": AsyncMock(return_value="fresh answer"),
        "tokenizer": _DummyTokenizer(),
        "enable_image_token_budget": False,
    }
    hashing_kv = SimpleNamespace(global_config={"enable_llm_cache": True})
    empty_storage = SimpleNamespace()

    with patch(
        "lightrag.operate.get_keywords_from_query",
        new=AsyncMock(return_value=(["graph"], ["LightRAG"])),
    ), patch(
        "lightrag.operate._build_query_context",
        new=AsyncMock(
            return_value=QueryContextResult(
                context="Document Chunks only",
                raw_data={"data": {}, "metadata": {}},
            )
        ),
    ), patch(
        "lightrag.operate.handle_cache", new=AsyncMock()
    ) as mock_cache, patch(
        "lightrag.operate.save_to_cache", new=AsyncMock()
    ) as mock_save:
        result = await kg_query(
            "Explain LightRAG retrieval",
            empty_storage,
            empty_storage,
            empty_storage,
            empty_storage,
            query_param,
            global_config,
            hashing_kv=hashing_kv,
            chunks_vdb=empty_storage,
        )

    assert result.content == "fresh answer"
    mock_cache.assert_not_called()
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_build_context_str_uses_chunk_only_context_template():
    query_param = QueryParam(
        mode="hybrid",
        answer_context_mode="chunk_only_prompt",
        enable_rerank=False,
        max_total_tokens=10_000,
    )
    global_config = {
        "tokenizer": _DummyTokenizer(),
        "system_prompt_template": "System {context_data}",
        "enable_image_token_budget": False,
        "min_rerank_score": 0.0,
    }
    entities_context = [
        {"entity": "EntityA", "type": "concept", "description": "entity desc"}
    ]
    relations_context = [
        {
            "entity1": "EntityA",
            "entity2": "EntityB",
            "description": "relation desc",
        }
    ]
    merged_chunks = [
        {
            "id": "DC1",
            "content": "chunk content",
            "file_path": "paper.pdf",
            "chunk_id": "chunk-1",
        }
    ]

    context, raw_data = await _build_context_str(
        entities_context=entities_context,
        relations_context=relations_context,
        merged_chunks=merged_chunks,
        query="What happened?",
        query_param=query_param,
        global_config=global_config,
    )

    assert "Document Chunks" in context
    assert "Knowledge Graph Data (Entity)" not in context
    assert raw_data["data"]["entities"][0]["entity_name"] == "EntityA"


@pytest.mark.asyncio
async def test_apply_token_truncation_chunk_only_prompt_preserves_kg_candidates():
    query_param = QueryParam(
        mode="hybrid",
        answer_context_mode="chunk_only_prompt",
    )
    search_result = {
        "final_entities": [
            {
                "entity_id": "E1",
                "entity_name": "EntityA",
                "description": "entity desc",
                "entity_type": "concept",
            }
        ],
        "final_relations": [
            {
                "src_id": "EntityA",
                "tgt_id": "EntityB",
                "description": "relation desc",
            }
        ],
    }

    result = await _apply_token_truncation(
        search_result,
        query_param,
        {"tokenizer": _DummyTokenizer()},
    )

    assert result["entities_context"] == []
    assert result["relations_context"] == []
    assert result["filtered_entities"] == search_result["final_entities"]
    assert result["filtered_relations"] == search_result["final_relations"]


@pytest.mark.asyncio
async def test_build_context_str_chunk_only_uses_chunk_only_response_prompt_budget():
    original_rag_response = PROMPTS["rag_response"]
    original_naive_response = PROMPTS["naive_rag_response"]

    try:
        PROMPTS["rag_response"] = ("R" * 400) + " {context_data} {response_type} {user_prompt}"
        PROMPTS["naive_rag_response"] = "N {context_data} {response_type} {user_prompt}"

        captured_limits = {}

        async def fake_process_chunks_unified(
            *,
            query,
            unique_chunks,
            query_param,
            global_config,
            source_type="mixed",
            chunk_token_limit=None,
            rerank_debug=None,
        ):
            captured_limits[query_param.answer_context_mode] = chunk_token_limit
            return list(unique_chunks)

        merged_chunks = [
            {
                "id": "DC1",
                "content": "chunk content",
                "file_path": "paper.pdf",
                "chunk_id": "chunk-1",
            }
        ]
        global_config = {
            "tokenizer": _DummyTokenizer(),
            "enable_image_token_budget": False,
            "min_rerank_score": 0.0,
        }

        with patch(
            "lightrag.operate.process_chunks_unified",
            new=fake_process_chunks_unified,
        ):
            await _build_context_str(
                entities_context=[],
                relations_context=[],
                merged_chunks=merged_chunks,
                query="What happened?",
                query_param=QueryParam(
                    mode="hybrid",
                    answer_context_mode="kg_prompt",
                    enable_rerank=False,
                    max_total_tokens=5000,
                ),
                global_config=global_config,
            )
            await _build_context_str(
                entities_context=[],
                relations_context=[],
                merged_chunks=merged_chunks,
                query="What happened?",
                query_param=QueryParam(
                    mode="hybrid",
                    answer_context_mode="chunk_only_prompt",
                    enable_rerank=False,
                    max_total_tokens=5000,
                ),
                global_config=global_config,
            )

        assert (
            captured_limits["chunk_only_prompt"]
            > captured_limits["kg_prompt"]
        )
    finally:
        PROMPTS["rag_response"] = original_rag_response
        PROMPTS["naive_rag_response"] = original_naive_response


def test_rrf_merge_ranked_records_promotes_multi_hit_candidates():
    ranking_lists = [
        [
            {"entity_id": "E1", "entity_name": "Entity 1", "distance": 0.91},
            {"entity_id": "E2", "entity_name": "Entity 2", "distance": 0.90},
        ],
        [
            {"entity_id": "E2", "entity_name": "Entity 2", "distance": 0.95},
            {"entity_id": "E3", "entity_name": "Entity 3", "distance": 0.89},
        ],
    ]

    merged = _rrf_merge_ranked_records(
        ranking_lists,
        item_kind="entity",
        top_k=3,
    )

    assert [item["entity_id"] for item in merged] == ["E2", "E1", "E3"]
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


@pytest.mark.asyncio
async def test_pick_by_vector_similarity_uses_candidate_pool_retriever():
    capture = {}

    async def fake_candidate_retriever(
        *,
        query: str,
        chunk_ids: list[str],
        top_k: int,
        query_embedding=None,
        retrieval_mode: str = "dense",
    ):
        capture["query"] = query
        capture["chunk_ids"] = list(chunk_ids)
        capture["top_k"] = top_k
        capture["query_embedding"] = list(query_embedding or [])
        capture["retrieval_mode"] = retrieval_mode
        return [
            {"id": "chunk-2", "distance": 0.9},
            {"id": "chunk-1", "distance": 0.8},
        ]

    selected = await pick_by_vector_similarity(
        query="graph query",
        text_chunks_storage=SimpleNamespace(),
        chunks_vdb=SimpleNamespace(),
        num_of_chunks=2,
        entity_info=[
            {"sorted_chunks": ["chunk-1", "chunk-2"]},
            {"sorted_chunks": ["chunk-2", "chunk-3"]},
        ],
        embedding_func=AsyncMock(),
        query_embedding=[0.1, 0.2],
        retrieval_mode="hybrid",
        candidate_retriever=fake_candidate_retriever,
    )

    assert selected == ["chunk-2", "chunk-1"]
    assert capture == {
        "query": "graph query",
        "chunk_ids": ["chunk-1", "chunk-2", "chunk-3"],
        "top_k": 2,
        "query_embedding": [0.1, 0.2],
        "retrieval_mode": "hybrid",
    }
