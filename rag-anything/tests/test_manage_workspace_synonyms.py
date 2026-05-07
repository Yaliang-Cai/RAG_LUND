from pathlib import Path
import sys
import asyncio
import types
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

from scripts.manage_workspace_synonyms import (
    WorkspaceSnapshot,
    _build_safety_issues,
    _open_service_and_rag,
    _parse_args,
    _resolve_workspace_context,
    _split_edge_signature_sets,
)


def test_resolve_workspace_context_requires_leaf_workspace_dir(tmp_path: Path):
    workspace_dir = tmp_path / "docbench_shared_graphbm25_20260504_v0"
    workspace_dir.mkdir()
    (workspace_dir / "kv_store_full_docs.json").write_text("{}", encoding="utf-8")

    context = _resolve_workspace_context(str(workspace_dir))

    assert context.workspace_id == "docbench_shared_graphbm25_20260504_v0"
    assert context.working_dir_root == tmp_path
    assert context.manifest_path == workspace_dir / "synonym_linking_manifest.json"


def test_resolve_workspace_context_accepts_workspace_id_override(tmp_path: Path):
    workspace_dir = tmp_path / "hotpotqa"
    workspace_dir.mkdir()
    (workspace_dir / "kv_store_full_docs.json").write_text("{}", encoding="utf-8")

    context = _resolve_workspace_context(
        str(workspace_dir),
        workspace_id_raw="hotpotqa_hr2_v0",
    )

    assert context.workspace_id == "hotpotqa_hr2_v0"
    assert context.workspace_path == workspace_dir
    assert context.working_dir_root == tmp_path
    assert context.manifest_path == workspace_dir / "synonym_linking_manifest.json"


def test_resolve_workspace_context_accepts_workspace_id_override_parent_dir(
    tmp_path: Path,
):
    parent_dir = tmp_path / "hotpotqa"
    workspace_dir = parent_dir / "hotpotqa_hr2_v0"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "kv_store_full_docs.json").write_text("{}", encoding="utf-8")

    context = _resolve_workspace_context(
        str(parent_dir),
        workspace_id_raw="hotpotqa_hr2_v0",
    )

    assert context.workspace_id == "hotpotqa_hr2_v0"
    assert context.workspace_path == workspace_dir
    assert context.working_dir_root == parent_dir
    assert context.manifest_path == workspace_dir / "synonym_linking_manifest.json"


def test_resolve_workspace_context_accepts_single_leaf_parent(tmp_path: Path):
    parent_dir = tmp_path / "rag_workspaces"
    workspace_dir = parent_dir / "docbench_shared_graphbm25_20260504_v0"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "kv_store_full_docs.json").write_text("{}", encoding="utf-8")

    context = _resolve_workspace_context(str(parent_dir))

    assert context.workspace_id == "docbench_shared_graphbm25_20260504_v0"
    assert context.workspace_path == workspace_dir
    assert context.working_dir_root == parent_dir


def test_open_service_uses_working_dir_root_not_leaf_workspace(
    tmp_path: Path,
    monkeypatch,
):
    parent_dir = tmp_path / "hotpotqa"
    workspace_dir = parent_dir / "hotpotqa_hr2_v0"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "kv_store_full_docs.json").write_text("{}", encoding="utf-8")
    context = _resolve_workspace_context(
        str(parent_dir),
        workspace_id_raw="hotpotqa_hr2_v0",
    )
    calls: dict[str, str | None] = {}

    class FakeSettings(SimpleNamespace):
        @classmethod
        def from_env(cls):
            return cls()

    class FakeService:
        def __init__(self, settings):
            self.settings = settings

        async def get_rag(self, workspace_id, working_dir=None):
            calls["workspace_id"] = workspace_id
            calls["working_dir"] = working_dir
            return SimpleNamespace(lightrag=object())

        async def _ensure_workspace_warmed(self, workspace_id):
            calls["warmed_workspace_id"] = workspace_id

    fake_local_rag = types.ModuleType("raganything.services.local_rag")
    fake_local_rag.LocalRagSettings = FakeSettings
    fake_local_rag.LocalRagService = FakeService
    monkeypatch.setitem(
        sys.modules,
        "raganything.services.local_rag",
        fake_local_rag,
    )

    asyncio.run(_open_service_and_rag(context))

    assert calls["workspace_id"] == "hotpotqa_hr2_v0"
    assert calls["working_dir"] == str(parent_dir)
    assert calls["warmed_workspace_id"] == "hotpotqa_hr2_v0"


def test_parse_args_keeps_workspace_id_optional_for_apply(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_workspace_synonyms.py",
            "apply",
            "--workspace-path",
            "/tmp/hotpotqa",
            "--workspace-id",
            "hotpotqa_hr2_v0",
            "--synonymy-threshold",
            "0.8",
        ],
    )

    args = _parse_args()

    assert args.command == "apply"
    assert args.workspace_path == "/tmp/hotpotqa"
    assert args.workspace_id == "hotpotqa_hr2_v0"
    assert args.synonymy_threshold == 0.8


def test_resolve_workspace_context_rejects_parent_dir(tmp_path: Path):
    parent_dir = tmp_path / "rag_workspaces"
    parent_dir.mkdir()

    with pytest.raises(ValueError, match="leaf workspace directory"):
        _resolve_workspace_context(str(parent_dir))


def test_split_edge_signature_sets_deduplicates_reversed_edges():
    edges = [
        {
            "source": "B",
            "target": "A",
            "weight": 1.0,
            "description": "factual",
            "source_id": "chunk-1",
        },
        {
            "source": "A",
            "target": "B",
            "weight": 1.0,
            "description": "factual",
            "source_id": "chunk-1",
        },
        {
            "source": "C",
            "target": "D",
            "weight": 0.91,
            "description": "Synonym: C <-> D",
            "keywords": "synonym,alias",
            "provenance": "synonym_detection",
            "edge_type": "SYNONYM",
        },
        {
            "source": "D",
            "target": "C",
            "weight": 0.91,
            "description": "Synonym: C <-> D",
            "keywords": "synonym,alias",
            "provenance": "synonym_detection",
            "edge_type": "SYNONYM",
        },
    ]

    factual, synonym = _split_edge_signature_sets(edges)

    assert len(factual) == 1
    assert len(synonym) == 1


def test_build_safety_issues_detects_only_real_mutations():
    before = WorkspaceSnapshot(
        node_count=10,
        factual_edge_count=3,
        synonym_edge_count=1,
        factual_edge_signatures=frozenset({"edge-a", "edge-b", "edge-c"}),
        entities_vdb_count=4,
        relationships_vdb_count=5,
        chunks_vdb_count=6,
        profile_sha256="abc",
    )
    after_same = WorkspaceSnapshot(
        node_count=10,
        factual_edge_count=3,
        synonym_edge_count=7,
        factual_edge_signatures=frozenset({"edge-a", "edge-b", "edge-c"}),
        entities_vdb_count=4,
        relationships_vdb_count=5,
        chunks_vdb_count=6,
        profile_sha256="abc",
    )
    after_changed = WorkspaceSnapshot(
        node_count=11,
        factual_edge_count=2,
        synonym_edge_count=7,
        factual_edge_signatures=frozenset({"edge-a", "edge-b", "edge-x"}),
        entities_vdb_count=4,
        relationships_vdb_count=8,
        chunks_vdb_count=6,
        profile_sha256="def",
    )

    assert _build_safety_issues(before, after_same) == []

    issues = _build_safety_issues(before, after_changed)
    fields = {issue["field"] for issue in issues}
    assert "node_count" in fields
    assert "factual_edge_count" in fields
    assert "factual_edge_signatures" in fields
    assert "relationships_vdb_count" in fields
    assert "profile_sha256" in fields
