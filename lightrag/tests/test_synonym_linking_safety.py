from pathlib import Path

import pytest

from lightrag.kg.shared_storage import initialize_share_data
from lightrag.synonym_linking import build_synonym_edges, clear_synonym_edges
from lightrag.utils import compute_entity_vdb_id


class FakeEntityVDB:
    def __init__(self) -> None:
        self._vectors = {
            compute_entity_vdb_id("Alpha", "", False): [1.0, 0.0],
            compute_entity_vdb_id("Beta", "", False): [1.0, 0.0],
        }

    async def get_vectors_by_ids(self, ids):
        return {item_id: self._vectors[item_id] for item_id in ids if item_id in self._vectors}


class RaceGraph:
    workspace = "synonym-race-test"

    def __init__(self, existing_on_locked_check=None) -> None:
        self.get_edges_calls = 0
        self.existing_on_locked_check = existing_on_locked_check or {}
        self.upserted_edges = []

    async def get_all_labels(self):
        return ["Alpha", "Beta"]

    async def get_edges_batch(self, edge_pairs):
        self.get_edges_calls += 1
        if self.get_edges_calls >= 2:
            return self.existing_on_locked_check
        return {}

    async def upsert_edges_batch(self, edges):
        self.upserted_edges.extend(edges)


class ClearGraph:
    def __init__(self) -> None:
        self.removed_edges = []

    async def get_all_edges(self):
        return [
            {
                "source": "Alpha",
                "target": "Beta",
                "edge_type": "SYNONYM",
                "provenance": "synonym_detection",
                "keywords": "synonym,alias",
            },
            {
                "source": "Beta",
                "target": "Gamma",
                "edge_type": "FACTUAL",
                "provenance": "relation_extraction",
                "keywords": "supports",
                "source_id": "chunk-1",
            },
        ]

    async def remove_edges(self, edges):
        self.removed_edges.extend(edges)


@pytest.mark.asyncio
async def test_synonym_linking_rechecks_under_graph_lock_before_upsert():
    initialize_share_data()
    factual_edge = {
        ("Alpha", "Beta"): {
            "source_id": "chunk-1",
            "edge_type": "FACTUAL",
            "provenance": "extraction",
            "keywords": "supports",
            "weight": 1.0,
        }
    }
    graph = RaceGraph(existing_on_locked_check=factual_edge)

    created = await build_synonym_edges(
        entities_vdb=FakeEntityVDB(),
        knowledge_graph_inst=graph,
        new_entity_ids=["Alpha"],
        synonymy_threshold=0.8,
        min_entity_len=0,
        enable_entity_disambiguation=False,
    )

    assert created == 0
    assert graph.upserted_edges == []
    assert graph.get_edges_calls == 2


@pytest.mark.asyncio
async def test_synonym_linking_writes_when_locked_recheck_is_still_empty():
    initialize_share_data()
    graph = RaceGraph()

    created = await build_synonym_edges(
        entities_vdb=FakeEntityVDB(),
        knowledge_graph_inst=graph,
        new_entity_ids=["Alpha"],
        synonymy_threshold=0.8,
        min_entity_len=0,
        enable_entity_disambiguation=False,
    )

    assert created == 1
    assert len(graph.upserted_edges) == 1
    src_id, tgt_id, edge_data = graph.upserted_edges[0]
    assert (src_id, tgt_id) == ("Alpha", "Beta")
    assert edge_data["edge_type"] == "SYNONYM"


@pytest.mark.asyncio
async def test_clear_synonym_edges_only_removes_synonym_pairs():
    graph = ClearGraph()

    removed = await clear_synonym_edges(graph)

    assert removed == 1
    assert graph.removed_edges == [("Alpha", "Beta")]


def test_neo4j_batch_synonym_upsert_has_factual_edge_guard():
    source = (
        Path(__file__).resolve().parents[1] / "lightrag" / "kg" / "neo4j_impl.py"
    ).read_text(encoding="utf-8")

    assert "incoming_is_synonym" in source
    assert "current_is_synonym" in source
    assert "current_is_empty" in source
    assert (
        "WHERE NOT incoming_is_synonym OR current_is_synonym OR current_is_empty"
        in source
    )
