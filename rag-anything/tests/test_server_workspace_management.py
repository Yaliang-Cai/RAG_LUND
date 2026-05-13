from pathlib import Path
import sys
import types
from types import SimpleNamespace

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTRAG_ROOT))

if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.CrossEncoder = object
    sentence_transformers_stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers_stub

if "python_multipart" not in sys.modules:
    python_multipart_stub = types.ModuleType("python_multipart")
    python_multipart_stub.__version__ = "0.0.13"
    sys.modules["python_multipart"] = python_multipart_stub
if "multipart.multipart" not in sys.modules:
    multipart_pkg_stub = types.ModuleType("multipart")
    multipart_mod_stub = types.ModuleType("multipart.multipart")
    multipart_mod_stub.parse_options_header = lambda value: (value, {})
    multipart_pkg_stub.multipart = multipart_mod_stub
    sys.modules["multipart"] = multipart_pkg_stub
    sys.modules["multipart.multipart"] = multipart_mod_stub

import raganything as raganything_pkg

if not hasattr(raganything_pkg, "RAGAnything"):
    raganything_pkg.RAGAnything = object
if not hasattr(raganything_pkg, "RAGAnythingConfig"):
    raganything_pkg.RAGAnythingConfig = object

from server import app as server_app


class _FakeDocStatus:
    def __init__(self, docs):
        self.docs = docs

    async def get_docs_paginated(
        self,
        status_filter=None,
        page=1,
        page_size=200,
        sort_field="file_path",
        sort_direction="asc",
    ):
        rows = []
        for doc_id, payload in self.docs.items():
            rows.append((doc_id, SimpleNamespace(**payload)))
        return rows, len(rows)

    async def get_by_id(self, doc_id):
        return self.docs.get(doc_id)


class _FakeGraph:
    def __init__(self):
        self.nodes = {
            "ALPHA|ORG": {
                "entity_name": "ALPHA",
                "entity_type": "ORG",
                "description": "Alpha description",
                "source_id": "chunk-1",
                "file_path": "doc1.pdf",
            }
        }

    async def get_all_labels(self):
        return list(self.nodes)

    async def get_nodes_batch(self, node_ids):
        return {node_id: self.nodes[node_id] for node_id in node_ids if node_id in self.nodes}

    async def get_all_edges(self):
        return [
            {
                "source": "ALPHA|ORG",
                "target": "BETA|ORG",
                "description": "related",
            }
        ]


class _FakeRagWrapper:
    def __init__(self, lightrag):
        self.lightrag = lightrag

    async def _ensure_lightrag_initialized(self):
        return {"success": True}


class _FakeStorage:
    def __init__(self, name, dropped):
        self.name = name
        self.dropped = dropped

    async def drop(self):
        self.dropped.append(self.name)


class _FakeDeleteWorkspaceService:
    def __init__(self, tmp_path: Path, lightrag):
        self.settings = SimpleNamespace(
            output_dir=str(tmp_path / "output"),
            working_dir_root=str(tmp_path / "rag_workspace"),
        )
        self._rag_instances = {"ws": object()}
        self._warmed_workspaces = {"ws"}
        self.lightrag = lightrag

    async def get_rag(self, workspace_id):
        return _FakeRagWrapper(self.lightrag)


class _FakeService:
    def __init__(self, tmp_path: Path):
        self.docs = {
            "doc-1": {
                "status": "processed",
                "file_path": "doc1.pdf",
                "chunks_count": 3,
                "chunks_list": ["chunk-1", "chunk-mm-1", "chunk-mm-2"],
                "multimodal_processed": False,
                "multimodal_stage": "chunks_stored",
                "multimodal_failed_items": [{"index": 1, "error": "timeout"}],
                "multimodal_chunk_ids": ["chunk-mm-1", "chunk-mm-2"],
                "created_at": "2026-05-13T00:00:00+00:00",
                "updated_at": "2026-05-13T01:00:00+00:00",
            },
            "doc-2": {
                "status": "processed",
                "file_path": "doc2.pdf",
                "chunks_count": 1,
                "chunks_list": ["chunk-2"],
                "multimodal_processed": True,
                "multimodal_stage": "completed",
                "multimodal_failed_items": [],
                "multimodal_chunk_ids": [],
            },
        }
        self.settings = SimpleNamespace(
            output_dir=str(tmp_path / "output"),
            working_dir_root=str(tmp_path / "rag_workspace"),
            enable_synonym_linking=True,
        )
        self.lightrag = SimpleNamespace(
            doc_status=_FakeDocStatus(self.docs),
            chunk_entity_relation_graph=_FakeGraph(),
        )
        self.deleted = []
        self.synonym_calls = []

    async def get_rag(self, workspace_id):
        return _FakeRagWrapper(self.lightrag)

    async def lightrag_adelete_by_doc_id(
        self,
        workspace_id,
        doc_id,
        *,
        delete_llm_cache=False,
    ):
        self.deleted.append((workspace_id, doc_id, delete_llm_cache))
        return SimpleNamespace(
            status="success",
            doc_id=doc_id,
            message="deleted",
            status_code=200,
            file_path=self.docs[doc_id]["file_path"],
        )

    async def finalize_workspace_synonyms(self, workspace_id, *, force=False, reset_existing=True):
        self.synonym_calls.append((workspace_id, force, reset_existing))
        return {"success": True, "cleared_edges": 1, "created_edges": 2}


def _client_with_fake_service(monkeypatch, tmp_path):
    fake_service = _FakeService(tmp_path)
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(server_app, "UPLOADS_DIR", uploads)
    server_app.app.dependency_overrides[server_app.get_service] = lambda: fake_service
    client = TestClient(server_app.app)
    return client, fake_service, uploads


def teardown_function():
    server_app.app.dependency_overrides.clear()


def test_workspace_documents_exports_multimodal_status(monkeypatch, tmp_path):
    client, _, _ = _client_with_fake_service(monkeypatch, tmp_path)

    response = client.get("/workspace/ws/documents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace_id"] == "ws"
    assert payload["count"] == 2
    assert payload["documents"][0]["doc_id"] == "doc-1"
    assert payload["documents"][0]["file_path"] == "doc1.pdf"
    assert payload["documents"][0]["chunks_count"] == 3
    assert payload["documents"][0]["multimodal_processed"] is False
    assert payload["documents"][0]["multimodal_stage"] == "chunks_stored"
    assert payload["documents"][0]["multimodal_chunk_ids"] == ["chunk-mm-1", "chunk-mm-2"]
    assert payload["documents"][0]["multimodal_failed_items"] == [
        {"index": 1, "error": "timeout"}
    ]


def test_delete_document_removes_artifacts_after_storage_delete_and_rebuilds_synonyms(
    monkeypatch,
    tmp_path,
):
    client, fake_service, uploads = _client_with_fake_service(monkeypatch, tmp_path)
    upload_dir = uploads / "ws"
    output_root = Path(fake_service.settings.output_dir) / "ws"
    output_dir = output_root / "doc1" / "hybrid_auto"
    other_output_dir = output_root / "doc2" / "hybrid_auto"
    upload_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    other_output_dir.mkdir(parents=True)
    (upload_dir / "doc1.pdf").write_text("raw", encoding="utf-8")
    (upload_dir / "doc2.pdf").write_text("raw", encoding="utf-8")
    (output_dir / "doc1.md").write_text("md", encoding="utf-8")
    (output_root / "doc1" / "images").mkdir()
    (output_root / "doc1" / "images" / "page.png").write_text("image", encoding="utf-8")
    (other_output_dir / "doc2.md").write_text("md", encoding="utf-8")

    response = client.delete("/workspace/ws/documents/doc-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["delete_result"]["status"] == "success"
    assert fake_service.deleted == [("ws", "doc-1", False)]
    assert fake_service.synonym_calls == [("ws", True, True)]
    assert not (upload_dir / "doc1.pdf").exists()
    assert (upload_dir / "doc2.pdf").exists()
    assert not (output_root / "doc1").exists()
    assert (other_output_dir / "doc2.md").exists()
    assert "synonym_rebuild" in payload
    assert payload["artifacts_deleted"]


def test_graph_entities_exports_audit_metadata(monkeypatch, tmp_path):
    client, _, _ = _client_with_fake_service(monkeypatch, tmp_path)

    response = client.get("/graph/ws/entities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace_id"] == "ws"
    assert payload["count"] == 1
    entity = payload["entities"][0]
    assert entity["label"] == "ALPHA|ORG"
    assert entity["entity_name"] == "ALPHA"
    assert entity["entity_type"] == "ORG"
    assert entity["source_id"] == "chunk-1"
    assert entity["file_path"] == "doc1.pdf"
    assert entity["description"] == "Alpha description"


def test_graph_stats_prefers_lightrag_graph_storage(monkeypatch, tmp_path):
    client, _, _ = _client_with_fake_service(monkeypatch, tmp_path)

    response = client.get("/graph/ws/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["entity_count"] == 1
    assert payload["relation_count"] == 1
    assert payload["source"] == "graph_storage"


def test_delete_workspace_drops_graph_vector_and_kv_storages(monkeypatch, tmp_path):
    dropped = []
    storage_names = [
        "text_chunks",
        "full_docs",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
        "doc_status",
    ]
    lightrag = SimpleNamespace(
        **{name: _FakeStorage(name, dropped) for name in storage_names}
    )
    fake_service = _FakeDeleteWorkspaceService(tmp_path, lightrag)

    uploads = tmp_path / "uploads"
    for path in (
        uploads / "ws",
        Path(fake_service.settings.output_dir) / "ws",
        Path(fake_service.settings.working_dir_root) / "ws",
    ):
        path.mkdir(parents=True)
        (path / "marker.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(server_app, "UPLOADS_DIR", uploads)
    server_app.app.dependency_overrides[server_app.get_service] = lambda: fake_service
    client = TestClient(server_app.app)

    response = client.delete("/workspace/ws")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["drop_errors"] == []
    assert set(dropped) == set(storage_names)
    assert not (uploads / "ws").exists()
    assert not (Path(fake_service.settings.output_dir) / "ws").exists()
    assert not (Path(fake_service.settings.working_dir_root) / "ws").exists()
