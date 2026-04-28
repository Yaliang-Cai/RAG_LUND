import csv
import json
import io
from pathlib import Path
import pytest


# ── helpers: import the functions under test ──────────────────────────────────
def _import():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "export_lightrag_to_gfm",
        Path(__file__).parents[2] / "scripts" / "export_lightrag_to_gfm.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _import()


class TestWriteNodesCsv:
    def test_document_nodes_use_chunk_id_as_name(self, mod, tmp_path):
        chunks = {"chunk_abc": "France is a country."}
        entities = []
        mod.write_nodes_csv(tmp_path, chunks, entities)
        rows = list(csv.DictReader((tmp_path / "nodes.csv").open()))
        assert rows[0]["name"] == "chunk_abc"
        assert rows[0]["type"] == "document"
        attrs = json.loads(rows[0]["attributes"])
        assert attrs["content"] == "France is a country."

    def test_entity_nodes_prefixed_with_entity_(self, mod, tmp_path):
        chunks = {}
        entities = [{"name": "France", "description": "A country", "entity_id": "e1", "source_ids": ""}]
        mod.write_nodes_csv(tmp_path, chunks, entities)
        rows = list(csv.DictReader((tmp_path / "nodes.csv").open()))
        assert rows[0]["name"] == "entity_France"
        assert rows[0]["type"] == "entity"

    def test_header_is_name_type_attributes(self, mod, tmp_path):
        mod.write_nodes_csv(tmp_path, {}, [])
        with open(tmp_path / "nodes.csv") as f:
            header = f.readline().strip()
        assert header == "name,type,attributes"


class TestWriteEdgesCsv:
    def test_mentioned_in_edges_for_entity_source_ids(self, mod, tmp_path):
        chunks = {"chunk_abc": "content"}
        entities = [
            {"name": "France", "entity_id": "e1", "description": "", "source_ids": "chunk_abc"}
        ]
        mod.write_edges_csv(tmp_path, entities, [], chunks)
        rows = list(csv.DictReader((tmp_path / "edges.csv").open()))
        assert any(
            r["source"] == "entity_France"
            and r["relation"] == "mentioned_in"
            and r["target"] == "chunk_abc"
            for r in rows
        )

    def test_entity_entity_edges_from_relations(self, mod, tmp_path):
        chunks = {}
        entities = [
            {"name": "France", "entity_id": "e1", "description": "", "source_ids": ""},
            {"name": "Paris", "entity_id": "e2", "description": "", "source_ids": ""},
        ]
        relations = [{"src": "e1", "relation": "capital_of", "tgt": "e2"}]
        mod.write_edges_csv(tmp_path, entities, relations, chunks)
        rows = list(csv.DictReader((tmp_path / "edges.csv").open()))
        assert any(
            r["source"] == "entity_France"
            and r["relation"] == "capital_of"
            and r["target"] == "entity_Paris"
            for r in rows
        )

    def test_header_is_source_relation_target_attributes(self, mod, tmp_path):
        mod.write_edges_csv(tmp_path, [], [], {})
        with open(tmp_path / "edges.csv") as f:
            header = f.readline().strip()
        assert header == "source,relation,target,attributes"


class TestWriteRelationsCsv:
    def test_includes_mentioned_in_always(self, mod, tmp_path):
        mod.write_relations_csv(tmp_path, [])
        rows = list(csv.DictReader((tmp_path / "relations.csv").open()))
        names = [r["name"] for r in rows]
        assert "mentioned_in" in names

    def test_includes_relation_types_from_input(self, mod, tmp_path):
        relations = [{"src": "a", "relation": "capital_of", "tgt": "b"}]
        mod.write_relations_csv(tmp_path, relations)
        rows = list(csv.DictReader((tmp_path / "relations.csv").open()))
        names = [r["name"] for r in rows]
        assert "capital_of" in names

    def test_header_is_name_attributes(self, mod, tmp_path):
        mod.write_relations_csv(tmp_path, [])
        with open(tmp_path / "relations.csv") as f:
            header = f.readline().strip()
        assert header == "name,attributes"


class TestWriteDocumentsJson:
    def test_writes_chunk_id_to_content_mapping(self, mod, tmp_path):
        chunks = {"chunk_abc": "France is a country.", "chunk_def": "Paris is the capital."}
        mod.write_documents_json(tmp_path, chunks)
        data = json.loads((tmp_path / "documents.json").read_text())
        assert data == chunks
