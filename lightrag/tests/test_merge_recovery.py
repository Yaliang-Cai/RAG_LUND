from __future__ import annotations

import pytest

from lightrag.constants import (
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
)
from lightrag.operate import _merge_edges_then_upsert, _merge_nodes_then_upsert
from lightrag.utils import make_relation_chunk_key


class _GraphStorage:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str], dict] = {}

    async def get_node(self, node_id: str):
        node = self.nodes.get(node_id)
        return dict(node) if node is not None else None

    async def has_node(self, node_id: str):
        return node_id in self.nodes

    async def upsert_node(self, node_id: str, node_data: dict):
        self.nodes[node_id] = dict(node_data)

    async def has_edge(self, src_id: str, tgt_id: str):
        return (src_id, tgt_id) in self.edges

    async def get_edge(self, src_id: str, tgt_id: str):
        edge = self.edges.get((src_id, tgt_id))
        return dict(edge) if edge is not None else None

    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: dict):
        self.edges[(src_id, tgt_id)] = dict(edge_data)


class _KVStorage:
    def __init__(self):
        self.data: dict[str, dict] = {}

    async def get_by_id(self, key: str):
        value = self.data.get(key)
        return dict(value) if value is not None else None

    async def upsert(self, payload: dict[str, dict]):
        for key, value in payload.items():
            self.data[key] = dict(value)

    async def index_done_callback(self):
        return None


class _VectorStorage:
    def __init__(self):
        self.data: dict[str, dict] = {}
        self.deleted: list[list[str]] = []

    async def upsert(self, payload: dict[str, dict]):
        for key, value in payload.items():
            self.data[key] = dict(value)

    async def delete(self, ids: list[str]):
        self.deleted.append(list(ids))
        for item_id in ids:
            self.data.pop(item_id, None)


def _global_config() -> dict:
    return {
        "enable_entity_disambiguation": False,
        "file_path_more_placeholder": DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
        "force_llm_summary_on_merge": 99999,
        "max_file_paths": 99999,
        "max_source_ids_per_entity": 99999,
        "max_source_ids_per_relation": 99999,
        "source_ids_limit_method": SOURCE_IDS_LIMIT_METHOD_FIFO,
        "strict_relation_endpoint_entity_match": False,
        "summary_context_size": 99999,
        "summary_max_tokens": 99999,
    }


@pytest.mark.asyncio
async def test_merge_node_recovers_missing_graph_node_from_entity_chunks_replay():
    graph = _GraphStorage()
    entity_chunks = _KVStorage()
    entity_vdb = _VectorStorage()
    entity_name = "Main Receiver (MR)"
    await entity_chunks.upsert({entity_name: {"chunk_ids": ["chunk-a"], "count": 1}})

    result = await _merge_nodes_then_upsert(
        entity_name,
        [
            {
                "entity_name": entity_name,
                "entity_type": "TECHNICAL_TERM",
                "description": "Main receiver wakes for low-power paging.",
                "source_id": "chunk-a",
                "file_path": "R2-2600929_6GR_CP_paging.doc",
            }
        ],
        graph,
        entity_vdb,
        _global_config(),
        entity_chunks_storage=entity_chunks,
    )

    assert result["_changed"] is True
    assert graph.nodes[entity_name]["description"] == (
        "Main receiver wakes for low-power paging."
    )
    assert graph.nodes[entity_name]["source_id"] == "chunk-a"
    assert (await entity_chunks.get_by_id(entity_name)) == {
        "chunk_ids": ["chunk-a"],
        "count": 1,
    }
    assert any(
        payload["entity_id"] == entity_name for payload in entity_vdb.data.values()
    )


@pytest.mark.asyncio
async def test_merge_edge_recovers_missing_graph_edge_from_relation_chunks_replay():
    graph = _GraphStorage()
    relation_chunks = _KVStorage()
    relationships_vdb = _VectorStorage()
    entity_vdb = _VectorStorage()
    storage_key = make_relation_chunk_key("Paging Occasion", "Paging Frame")
    await relation_chunks.upsert({storage_key: {"chunk_ids": ["chunk-b"], "count": 1}})

    result = await _merge_edges_then_upsert(
        "Paging Occasion",
        "Paging Frame",
        [
            {
                "src_id": "Paging Occasion",
                "tgt_id": "Paging Frame",
                "description": "Paging occasions are grouped into paging frames.",
                "keywords": "paging",
                "weight": 1.0,
                "source_id": "chunk-b",
                "file_path": "R2-2600929_6GR_CP_paging.doc",
            }
        ],
        graph,
        relationships_vdb,
        entity_vdb,
        _global_config(),
        relation_chunks_storage=relation_chunks,
    )

    assert result["description"] == "Paging occasions are grouped into paging frames."
    assert graph.edges[("Paging Occasion", "Paging Frame")]["source_id"] == "chunk-b"
    assert (await relation_chunks.get_by_id(storage_key)) == {
        "chunk_ids": ["chunk-b"],
        "count": 1,
    }
    assert relationships_vdb.data


@pytest.mark.asyncio
async def test_merge_node_idempotent_replay_repairs_entity_chunks_and_vector():
    graph = _GraphStorage()
    entity_chunks = _KVStorage()
    entity_vdb = _VectorStorage()
    entity_name = "LP-WUS Occasion (LO)"
    graph.nodes[entity_name] = {
        "entity_id": entity_name,
        "entity_type": "TECHNICAL_TERM",
        "description": "Low-power wake-up signal occasion.",
        "source_id": "chunk-c",
        "file_path": "doc.pdf",
        "created_at": 1,
        "truncate": "",
    }

    result = await _merge_nodes_then_upsert(
        entity_name,
        [
            {
                "entity_name": entity_name,
                "entity_type": "TECHNICAL_TERM",
                "description": "Low-power wake-up signal occasion.",
                "source_id": "chunk-c",
                "file_path": "doc.pdf",
            }
        ],
        graph,
        entity_vdb,
        _global_config(),
        entity_chunks_storage=entity_chunks,
    )

    assert result["_changed"] is False
    assert (await entity_chunks.get_by_id(entity_name)) == {
        "chunk_ids": ["chunk-c"],
        "count": 1,
    }
    assert any(
        payload["entity_id"] == entity_name for payload in entity_vdb.data.values()
    )


@pytest.mark.asyncio
async def test_merge_edge_idempotent_replay_repairs_relation_chunks_and_vector():
    graph = _GraphStorage()
    relation_chunks = _KVStorage()
    relationships_vdb = _VectorStorage()
    entity_vdb = _VectorStorage()
    graph.edges[("Paging Occasion", "Paging Frame")] = {
        "weight": 1.0,
        "weight_raw": 1.0,
        "description": "Paging occasions are grouped into paging frames.",
        "keywords": "paging",
        "source_id": "chunk-d",
        "file_path": "doc.pdf",
        "created_at": 1,
        "truncate": "",
        "edge_type": "factual",
        "provenance": "extracted",
    }

    result = await _merge_edges_then_upsert(
        "Paging Occasion",
        "Paging Frame",
        [
            {
                "src_id": "Paging Occasion",
                "tgt_id": "Paging Frame",
                "description": "Paging occasions are grouped into paging frames.",
                "keywords": "paging",
                "weight": 1.0,
                "source_id": "chunk-d",
                "file_path": "doc.pdf",
            }
        ],
        graph,
        relationships_vdb,
        entity_vdb,
        _global_config(),
        relation_chunks_storage=relation_chunks,
    )

    storage_key = make_relation_chunk_key("Paging Occasion", "Paging Frame")
    assert result["source_id"] == "chunk-d"
    assert (await relation_chunks.get_by_id(storage_key)) == {
        "chunk_ids": ["chunk-d"],
        "count": 1,
    }
    assert relationships_vdb.data
