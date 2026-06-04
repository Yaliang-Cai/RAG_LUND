import pytest

from lightrag.operate import (
    _merge_relation_keywords,
    _merge_edges_then_upsert,
    _normalize_entity_surface,
    _normalize_high_level_keyword,
    _normalize_keyword_list,
    _process_extraction_result,
)
from lightrag.constants import (
    DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION,
    DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION,
    DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH,
)


@pytest.mark.asyncio
async def test_process_extraction_result_surface_normalization_toggle():
    tuple_delimiter = "<|#|>"
    completion_delimiter = "<|COMPLETE|>"
    extraction = "\n".join(
        [
            f"entity{tuple_delimiter}llm application{tuple_delimiter}work{tuple_delimiter}System for customer support",
            f"entity{tuple_delimiter}rag pipeline{tuple_delimiter}process{tuple_delimiter}Retrieval pipeline",
            f"relation{tuple_delimiter}llm application{tuple_delimiter}rag pipeline{tuple_delimiter}integration{tuple_delimiter}The system integrates the pipeline",
            completion_delimiter,
        ]
    )

    raw_nodes, raw_edges = await _process_extraction_result(
        extraction,
        chunk_key="chunk-1",
        timestamp=1,
        enable_entity_surface_normalization=False,
        entity_uppercase_allowlist=["LLM", "RAG"],
    )
    assert "llm application" in raw_nodes
    assert ("llm application", "rag pipeline") in raw_edges

    normalized_nodes, normalized_edges = await _process_extraction_result(
        extraction,
        chunk_key="chunk-1",
        timestamp=1,
        enable_entity_surface_normalization=True,
        entity_uppercase_allowlist=["LLM", "RAG"],
    )
    assert "LLM Application" in normalized_nodes
    assert "RAG Pipeline" in normalized_nodes
    assert ("LLM Application", "RAG Pipeline") in normalized_edges


def test_normalize_entity_surface_preserves_existing_uppercase():
    assert _normalize_entity_surface("OpenAI API", {"api"}) == "OpenAI API"
    assert _normalize_entity_surface("BERT", {"bert"}) == "BERT"
    assert _normalize_entity_surface("Machine learning", {"llm"}) == "Machine Learning"
    assert _normalize_entity_surface("llm application", {"llm"}) == "LLM Application"
    assert _normalize_entity_surface("iPhone", {"api"}) == "iPhone"


def test_keyword_case_normalization_helpers():
    allowlist = {"api", "bert", "6g"}
    assert (
        _normalize_high_level_keyword("Retrieval Architecture", allowlist)
        == "retrieval architecture"
    )
    assert (
        _normalize_high_level_keyword("OpenAI API", allowlist)
        == "OpenAI API"
    )
    assert _normalize_keyword_list(
        ["openai api", "Machine learning", "machine learning"],
        keyword_kind="low_level",
        uppercase_allowlist=allowlist,
    ) == ["Openai API", "Machine Learning"]
    assert (
        _merge_relation_keywords(
            ["Retrieval, retrieval,OpenAI API,openai api"],
            uppercase_allowlist=allowlist,
            enable_case_normalization=True,
        )
        == "OpenAI API,retrieval"
    )
    assert (
        _merge_relation_keywords(
            ["Retrieval, retrieval"],
            uppercase_allowlist=allowlist,
            enable_case_normalization=False,
        )
        == "Retrieval,retrieval"
    )


def test_new_switch_defaults_are_enabled():
    assert DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION is True
    assert DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION is True
    assert DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH is True


class _GraphStrictSkipStub:
    def __init__(self):
        self.remove_edges_calls = []
        self.upsert_edge_calls = []

    async def has_node(self, node_id: str) -> bool:
        return False

    async def remove_edges(self, edges):
        self.remove_edges_calls.append(list(edges))

    async def upsert_edge(self, *_args, **_kwargs):
        self.upsert_edge_calls.append(True)


class _VectorDeleteStub:
    def __init__(self):
        self.delete_calls = []

    async def delete(self, ids):
        self.delete_calls.append(list(ids))


@pytest.mark.asyncio
async def test_merge_relation_strict_endpoint_match_skips_and_cleans():
    graph = _GraphStrictSkipStub()
    rel_vdb = _VectorDeleteStub()
    result = await _merge_edges_then_upsert(
        src_id="A",
        tgt_id="B",
        edges_data=[
            {
                "source_id": "chunk-1",
                "weight": 1.0,
                "description": "A relates to B",
                "keywords": "relation",
                "file_path": "doc.md",
                "timestamp": 1,
            }
        ],
        knowledge_graph_inst=graph,
        relationships_vdb=rel_vdb,
        entity_vdb=None,
        global_config={
            "strict_relation_endpoint_entity_match": True,
            "source_ids_limit_method": "FIFO",
            "max_source_ids_per_relation": 300,
            "max_file_paths": 100,
            "file_path_more_placeholder": "truncated",
        },
    )
    assert result is None
    assert graph.remove_edges_calls == [[("A", "B")]]
    assert rel_vdb.delete_calls
    assert not graph.upsert_edge_calls
