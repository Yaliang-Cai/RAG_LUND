import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")
    stub.CrossEncoder = object
    stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = stub

from raganything.services.local_rag import LocalRagService


class _DummyLightRAG:
    def __init__(self) -> None:
        self.calls = 0

    async def rebuild_synonym_edges(self, *, reset_existing: bool = True):
        self.calls += 1
        return {
            "success": True,
            "skipped": False,
            "cleared_edges": 3,
            "created_edges": 7,
            "reset_existing": bool(reset_existing),
        }


@pytest.mark.asyncio
async def test_finalize_workspace_synonyms_uses_manifest_and_reruns_on_threshold_change(
    tmp_path: Path,
):
    service = object.__new__(LocalRagService)
    service.settings = SimpleNamespace(
        enable_synonym_linking=True,
        working_dir_root=str(tmp_path),
        enable_entity_disambiguation=True,
        synonymy_threshold=0.8,
        synonymy_topk=2048,
        synonymy_min_entity_len=2,
    )
    service.logger = logging.getLogger("test_synonym_stage3_service")
    service._workspace_synonym_locks = {}
    service._workspace_synonym_ready = set()

    dummy_lightrag = _DummyLightRAG()
    dummy_rag = SimpleNamespace(lightrag=dummy_lightrag)

    async def _get_rag(workspace_id: str):
        return dummy_rag

    async def _ensure_workspace_warmed(workspace_id: str):
        return None

    service.get_rag = _get_rag
    service._ensure_workspace_warmed = _ensure_workspace_warmed

    first = await service.finalize_workspace_synonyms("ws")
    second = await service.finalize_workspace_synonyms("ws")

    assert first["created_edges"] == 7
    assert second["skipped"] is True
    assert second["reason"] == "up_to_date"
    assert dummy_lightrag.calls == 1

    service.settings.synonymy_threshold = 0.81
    third = await service.finalize_workspace_synonyms("ws")

    assert third["created_edges"] == 7
    assert dummy_lightrag.calls == 2

    manifest_path = tmp_path / "ws" / "synonym_linking_manifest.json"
    assert manifest_path.exists()
    payload = manifest_path.read_text(encoding="utf-8")
    assert '"synonymy_threshold": 0.81' in payload
