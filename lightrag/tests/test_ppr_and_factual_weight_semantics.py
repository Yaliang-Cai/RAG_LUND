import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAG_ANYTHING_ROOT = PROJECT_ROOT.parent / "rag-anything"
if str(RAG_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ANYTHING_ROOT))

from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    initialize_pipeline_status,
    initialize_share_data,
    set_default_workspace,
)
from lightrag.operate import merge_nodes_and_edges
from lightrag.ppr import personalized_pagerank
from lightrag.utils import compute_entity_id
from tests.test_doc_write_delete_consistency_mocked import (
    _DummyProcessor,
    _InMemoryGraphStorage,
    _InMemoryKVStorage,
    _InMemoryVectorStorage,
    _build_global_config,
)


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_multimodal_belongs_to_weight_is_normalized_to_one():
    processor = _DummyProcessor(SimpleNamespace())
    chunk_id = "chunk-mm-1"
    modal_entity_name = "DOC_TABLE_MAIN"

    chunk_results = [
        (
            {
                "E1": [{"source_id": chunk_id}],
                modal_entity_name: [{"source_id": chunk_id}],
            },
            {},
        )
    ]
    enhanced = await processor._batch_add_belongs_to_relations_by_chunk_mapping(
        chunk_results=chunk_results,
        chunk_to_modal_entity={chunk_id: modal_entity_name},
        chunk_to_file_path={chunk_id: "doc-mm.pdf"},
    )

    _, maybe_edges = enhanced[0]
    relation = maybe_edges.get(("E1", modal_entity_name), [{}])[0]
    assert relation.get("weight") == 1.0


def test_ppr_synonym_weight_mode_changes_chunk_ranking():
    entity_nodes = [
        {"entity_id": "A"},
        {"entity_id": "B"},
        {"entity_id": "C"},
    ]
    entity_edges = [
        {
            "src": "A",
            "tgt": "B",
            "weight": 1.0,
            "edge_type": "FACTUAL",
            "provenance": "relation_extraction",
            "source_id": "chunk-fact",
        },
        {
            "src": "A",
            "tgt": "C",
            "weight": 0.9,
            "edge_type": "SYNONYM",
            "provenance": "synonym_detection",
            "source_id": "",
        },
    ]
    chunk_nodes = [{"chunk_id": "chunk-b"}, {"chunk_id": "chunk-c"}]
    chunk_entity_edges = [
        {"chunk_id": "chunk-b", "entity_id": "B"},
        {"chunk_id": "chunk-c", "entity_id": "C"},
    ]
    entity_seed_weights = {"A": 1.0}
    chunk_seed_weights = {}

    raw_ranked = personalized_pagerank(
        entity_nodes=entity_nodes,
        entity_edges=entity_edges,
        chunk_nodes=chunk_nodes,
        chunk_entity_edges=chunk_entity_edges,
        entity_seed_weights=entity_seed_weights,
        chunk_seed_weights=chunk_seed_weights,
        ppr_synonym_weight_mode="raw",
        top_k=2,
    )
    plus_one_ranked = personalized_pagerank(
        entity_nodes=entity_nodes,
        entity_edges=entity_edges,
        chunk_nodes=chunk_nodes,
        chunk_entity_edges=chunk_entity_edges,
        entity_seed_weights=entity_seed_weights,
        chunk_seed_weights=chunk_seed_weights,
        ppr_synonym_weight_mode="plus_one",
        top_k=2,
    )

    assert raw_ranked and plus_one_ranked
    assert raw_ranked[0][0] == "chunk-b"
    assert plus_one_ranked[0][0] == "chunk-c"


@pytest.mark.asyncio
async def test_factual_weight_raw_is_idempotent_for_same_source_reverse_edges():
    workspace = "test-factual-weight-raw-idempotent"
    initialize_share_data(workers=1)
    set_default_workspace(workspace)
    await initialize_pipeline_status(workspace=workspace)
    pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)
    pipeline_status_lock = get_pipeline_status_lock(workspace=workspace)

    graph = _InMemoryGraphStorage()
    entities_vdb = _InMemoryVectorStorage()
    relationships_vdb = _InMemoryVectorStorage()
    full_entities = _InMemoryKVStorage()
    full_relations = _InMemoryKVStorage()
    entity_chunks = _InMemoryKVStorage()
    relation_chunks = _InMemoryKVStorage()
    global_config = _build_global_config(workspace)

    src_name, src_type = "SRC", "ORG"
    tgt_name, tgt_type = "TGT", "ORG"
    src_id = compute_entity_id(src_name, src_type, True)
    tgt_id = compute_entity_id(tgt_name, tgt_type, True)
    chunk_id = "chunk-one"

    maybe_nodes = {
        src_name: [
            {
                "entity_name": src_name,
                "entity_type": src_type,
                "description": "src desc",
                "source_id": chunk_id,
                "file_path": "doc.pdf",
                "timestamp": 1,
            }
        ],
        tgt_name: [
            {
                "entity_name": tgt_name,
                "entity_type": tgt_type,
                "description": "tgt desc",
                "source_id": chunk_id,
                "file_path": "doc.pdf",
                "timestamp": 1,
            }
        ],
    }
    maybe_edges = {
        tuple(sorted((src_name, tgt_name))): [
            {
                "description": "SRC relates to TGT",
                "keywords": "k1",
                "weight": 1.0,
                "source_id": chunk_id,
                "file_path": "doc.pdf",
                "timestamp": 1,
            },
            {
                "description": "TGT relates to SRC",
                "keywords": "k2",
                "weight": 2.0,
                "source_id": chunk_id,
                "file_path": "doc.pdf",
                "timestamp": 2,
            },
        ]
    }

    await merge_nodes_and_edges(
        chunk_results=[(maybe_nodes, maybe_edges)],
        knowledge_graph_inst=graph,
        entity_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        global_config=global_config,
        full_entities_storage=full_entities,
        full_relations_storage=full_relations,
        doc_id="doc-1",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=None,
        entity_chunks_storage=entity_chunks,
        relation_chunks_storage=relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path="doc.pdf",
    )

    edge = await graph.get_edge(src_id, tgt_id)
    assert edge is not None
    assert edge.get("weight_raw") == pytest.approx(2.0, rel=1e-6, abs=1e-6)
    assert edge.get("weight") == pytest.approx(math.log1p(2.0), rel=1e-6, abs=1e-6)

    # Replay identical evidence from the same chunk -> no extra increment.
    await merge_nodes_and_edges(
        chunk_results=[(maybe_nodes, maybe_edges)],
        knowledge_graph_inst=graph,
        entity_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        global_config=global_config,
        full_entities_storage=full_entities,
        full_relations_storage=full_relations,
        doc_id="doc-1",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=None,
        entity_chunks_storage=entity_chunks,
        relation_chunks_storage=relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path="doc.pdf",
    )

    edge_after_replay = await graph.get_edge(src_id, tgt_id)
    assert edge_after_replay is not None
    assert edge_after_replay.get("weight_raw") == pytest.approx(2.0, rel=1e-6, abs=1e-6)
    assert edge_after_replay.get("weight") == pytest.approx(
        math.log1p(2.0), rel=1e-6, abs=1e-6
    )
