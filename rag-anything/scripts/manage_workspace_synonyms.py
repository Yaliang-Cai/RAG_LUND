#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manage SYNONYM edges for an existing workspace without touching factual graph data.

Usage examples:

python scripts/manage_workspace_synonyms.py status \
  --workspace-path /data/.../rag_workspaces/docbench_shared_graphbm25_20260504_v0

python scripts/manage_workspace_synonyms.py apply \
  --workspace-path /data/.../rag_workspaces/docbench_shared_graphbm25_20260504_v0 \
  --synonymy-threshold 0.82

python scripts/manage_workspace_synonyms.py clear \
  --workspace-path /data/.../rag_workspaces/docbench_shared_graphbm25_20260504_v0
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Allow running from repo root without installing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

load_dotenv()

from lightrag.synonym_linking import _is_synonym_edge, clear_synonym_edges


EXPECTED_WORKSPACE_FILES = (
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_doc_status.json",
)


@dataclass(frozen=True)
class WorkspaceContext:
    workspace_path: Path
    workspace_id: str
    working_dir_root: Path
    manifest_path: Path
    profile_path: Path


@dataclass(frozen=True)
class WorkspaceSnapshot:
    node_count: int
    factual_edge_count: int
    synonym_edge_count: int
    factual_edge_signatures: frozenset[str]
    entities_vdb_count: int
    relationships_vdb_count: int
    chunks_vdb_count: int
    profile_sha256: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely manage SYNONYM edges for an existing workspace."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("status", "clear", "apply"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--workspace-path",
            required=True,
            help="Absolute path to the leaf workspace directory.",
        )
        subparser.add_argument(
            "--workspace-id",
            default=None,
            help=(
                "Workspace id stored in graph/vector payloads. Defaults to the "
                "workspace-path directory name for backwards compatibility."
            ),
        )
        if command == "apply":
            subparser.add_argument(
                "--synonymy-threshold",
                type=float,
                required=True,
                help="Cosine threshold used when rebuilding SYNONYM edges.",
            )
            subparser.add_argument(
                "--synonymy-topk",
                type=int,
                default=None,
                help="Synonym top-k setting forwarded to the rebuild runtime.",
            )
            subparser.add_argument(
                "--synonymy-min-entity-len",
                type=int,
                default=None,
                help="Minimum alnum/CJK entity length used on the query side.",
            )

    return parser.parse_args()


def _resolve_workspace_context(
    workspace_path_raw: str,
    workspace_id_raw: str | None = None,
) -> WorkspaceContext:
    workspace_path = Path(workspace_path_raw).expanduser().resolve()
    if not workspace_path.is_dir():
        raise ValueError(f"Workspace path does not exist or is not a directory: {workspace_path}")

    def _is_leaf_workspace(path: Path) -> bool:
        return any((path / name).exists() for name in EXPECTED_WORKSPACE_FILES)

    workspace_id_override = (
        str(workspace_id_raw).strip() if workspace_id_raw is not None else None
    )
    if not _is_leaf_workspace(workspace_path):
        nested_candidates: list[Path] = []
        if workspace_id_override:
            nested_candidates.append(workspace_path / workspace_id_override)
        nested_candidates.append(workspace_path / workspace_path.name)
        leaf_children = [
            child
            for child in workspace_path.iterdir()
            if child.is_dir() and _is_leaf_workspace(child)
        ]
        if len(leaf_children) == 1:
            nested_candidates.append(leaf_children[0])

        for nested in nested_candidates:
            if nested.is_dir() and _is_leaf_workspace(nested):
                workspace_path = nested
                break
        else:
            tried = ", ".join(
                str(path)
                for path in dict.fromkeys(nested_candidates)
                if path != workspace_path
            )
            raise ValueError(
                "workspace-path must point to the leaf workspace directory or its "
                f"single-workspace parent: {workspace_path}"
                + (f"; tried nested candidates: {tried}" if tried else "")
            )

    workspace_id = (
        workspace_id_override if workspace_id_override is not None else workspace_path.name
    )
    if not workspace_id:
        raise ValueError("workspace-id must not be empty")
    working_dir_root = workspace_path.parent
    return WorkspaceContext(
        workspace_path=workspace_path,
        workspace_id=workspace_id,
        working_dir_root=working_dir_root,
        manifest_path=workspace_path / "synonym_linking_manifest.json",
        profile_path=workspace_path / ".ablation_index_profile.json",
    )


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _edge_signature(edge: dict[str, Any]) -> str | None:
    source = str(edge.get("source", "") or "").strip()
    target = str(edge.get("target", "") or "").strip()
    if not source or not target:
        return None

    src_id, tgt_id = (source, target) if source < target else (target, source)
    properties = {k: v for k, v in edge.items() if k not in {"source", "target"}}
    return _stable_json({"pair": [src_id, tgt_id], "properties": properties})


def _split_edge_signature_sets(
    edges: list[dict[str, Any]],
) -> tuple[frozenset[str], frozenset[str]]:
    factual_signatures: set[str] = set()
    synonym_signatures: set[str] = set()
    for edge in edges:
        signature = _edge_signature(edge)
        if signature is None:
            continue
        if _is_synonym_edge(edge):
            synonym_signatures.add(signature)
        else:
            factual_signatures.add(signature)
    return frozenset(factual_signatures), frozenset(synonym_signatures)


def _sha256_bytes(raw_bytes: bytes | None) -> str | None:
    if raw_bytes is None:
        return None
    return hashlib.sha256(raw_bytes).hexdigest()


def _read_profile_sha256(profile_path: Path) -> str | None:
    if not profile_path.exists():
        return None
    return _sha256_bytes(profile_path.read_bytes())


def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _snapshot_diff(before: frozenset[str], after: frozenset[str], limit: int = 5) -> dict[str, list[str]]:
    removed = sorted(before - after)[:limit]
    added = sorted(after - before)[:limit]
    return {"removed": removed, "added": added}


def _build_safety_issues(
    before: WorkspaceSnapshot, after: WorkspaceSnapshot
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if before.node_count != after.node_count:
        issues.append(
            {
                "field": "node_count",
                "before": before.node_count,
                "after": after.node_count,
            }
        )
    if before.factual_edge_count != after.factual_edge_count:
        issues.append(
            {
                "field": "factual_edge_count",
                "before": before.factual_edge_count,
                "after": after.factual_edge_count,
            }
        )
    if before.factual_edge_signatures != after.factual_edge_signatures:
        issues.append(
            {
                "field": "factual_edge_signatures",
                "diff": _snapshot_diff(
                    before.factual_edge_signatures,
                    after.factual_edge_signatures,
                ),
            }
        )
    for field_name in (
        "entities_vdb_count",
        "relationships_vdb_count",
        "chunks_vdb_count",
        "profile_sha256",
    ):
        if getattr(before, field_name) != getattr(after, field_name):
            issues.append(
                {
                    "field": field_name,
                    "before": getattr(before, field_name),
                    "after": getattr(after, field_name),
                }
            )
    return issues


async def _count_workspace_points(vdb: Any) -> int:
    from qdrant_client import models
    from lightrag.kg.qdrant_impl import workspace_filter_condition

    if not hasattr(vdb, "_client") or not hasattr(vdb, "final_namespace"):
        raise RuntimeError(
            f"Unsupported vector storage for count verification: {type(vdb).__name__}"
        )

    result = await vdb._run_client_call_with_timeout(
        vdb._client.count,
        collection_name=vdb.final_namespace,
        count_filter=models.Filter(
            must=[workspace_filter_condition(vdb.effective_workspace)]
        ),
        exact=True,
    )
    return int(result.count)


async def _collect_workspace_snapshot(
    rag: Any,
    context: WorkspaceContext,
) -> WorkspaceSnapshot:
    lightrag = rag.lightrag
    if lightrag is None:
        raise RuntimeError("LightRAG runtime not initialized")

    graph = lightrag.chunk_entity_relation_graph
    nodes = await graph.get_all_nodes()
    edges = await graph.get_all_edges()
    factual_signatures, synonym_signatures = _split_edge_signature_sets(edges)

    entities_vdb_count, relationships_vdb_count, chunks_vdb_count = await asyncio.gather(
        _count_workspace_points(lightrag.entities_vdb),
        _count_workspace_points(lightrag.relationships_vdb),
        _count_workspace_points(lightrag.chunks_vdb),
    )

    return WorkspaceSnapshot(
        node_count=len(nodes),
        factual_edge_count=len(factual_signatures),
        synonym_edge_count=len(synonym_signatures),
        factual_edge_signatures=factual_signatures,
        entities_vdb_count=entities_vdb_count,
        relationships_vdb_count=relationships_vdb_count,
        chunks_vdb_count=chunks_vdb_count,
        profile_sha256=_read_profile_sha256(context.profile_path),
    )


def _settings_for_workspace(
    context: WorkspaceContext,
    *,
    synonymy_threshold: float | None,
    synonymy_topk: int | None,
    synonymy_min_entity_len: int | None,
):
    from raganything.services.local_rag import LocalRagSettings

    settings = LocalRagSettings.from_env()
    settings.working_dir_root = str(context.working_dir_root)
    settings.enable_synonym_linking = True
    settings.enable_entity_disambiguation = False
    if synonymy_threshold is not None:
        settings.synonymy_threshold = float(synonymy_threshold)
    if synonymy_topk is not None:
        settings.synonymy_topk = int(synonymy_topk)
    if synonymy_min_entity_len is not None:
        settings.synonymy_min_entity_len = int(synonymy_min_entity_len)
    return settings


async def _open_service_and_rag(
    context: WorkspaceContext,
    *,
    synonymy_threshold: float | None = None,
    synonymy_topk: int | None = None,
    synonymy_min_entity_len: int | None = None,
) -> tuple[Any, Any]:
    from raganything.services.local_rag import LocalRagService

    service = LocalRagService(
        _settings_for_workspace(
            context,
            synonymy_threshold=synonymy_threshold,
            synonymy_topk=synonymy_topk,
            synonymy_min_entity_len=synonymy_min_entity_len,
        )
    )
    rag = await service.get_rag(
        context.workspace_id,
        working_dir=str(context.working_dir_root),
    )
    await service._ensure_workspace_warmed(context.workspace_id)
    if getattr(rag, "lightrag", None) is None:
        raise RuntimeError("LightRAG runtime not initialized after workspace warmup")
    return service, rag


def _status_payload(
    context: WorkspaceContext,
    snapshot: WorkspaceSnapshot,
) -> dict[str, Any]:
    return {
        "workspace_path": str(context.workspace_path),
        "workspace_id": context.workspace_id,
        "manifest_path": str(context.manifest_path),
        "manifest": _load_manifest(context.manifest_path),
        "node_count": snapshot.node_count,
        "factual_edge_count": snapshot.factual_edge_count,
        "synonym_edge_count": snapshot.synonym_edge_count,
        "entities_vdb_count": snapshot.entities_vdb_count,
        "relationships_vdb_count": snapshot.relationships_vdb_count,
        "chunks_vdb_count": snapshot.chunks_vdb_count,
        "profile_path": str(context.profile_path),
        "profile_sha256": snapshot.profile_sha256,
    }


def _write_synonym_manifest(
    context: WorkspaceContext,
    service: Any,
    *,
    cleared_edges: int,
    created_edges: int,
) -> None:
    payload = service._current_synonym_manifest(context.workspace_id)
    payload.update(
        {
            "status": "completed",
            "cleared_edges": int(cleared_edges),
            "created_edges": int(created_edges),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    context.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    context.manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _discard_synonym_manifest(context: WorkspaceContext, service: Any) -> None:
    ready = getattr(service, "_workspace_synonym_ready", None)
    if ready is not None:
        ready.discard(context.workspace_id)
    if context.manifest_path.exists():
        context.manifest_path.unlink()


async def _run_status(context: WorkspaceContext) -> dict[str, Any]:
    service = None
    try:
        service, rag = await _open_service_and_rag(context)
        snapshot = await _collect_workspace_snapshot(rag, context)
        return {"success": True, "command": "status", **_status_payload(context, snapshot)}
    finally:
        if service is not None:
            await service.cleanup_workspace_instance(context.workspace_id)


async def _run_clear(context: WorkspaceContext) -> dict[str, Any]:
    service = None
    try:
        service, rag = await _open_service_and_rag(context)
        before = await _collect_workspace_snapshot(rag, context)
        cleared_edges = await clear_synonym_edges(rag.lightrag.chunk_entity_relation_graph)
        _discard_synonym_manifest(context, service)
        after = await _collect_workspace_snapshot(rag, context)
        issues = _build_safety_issues(before, after)
        if issues:
            raise RuntimeError(
                "Detected factual graph or storage mutation while clearing synonym edges: "
                + json.dumps(issues, ensure_ascii=False, indent=2)
            )
        return {
            "success": True,
            "command": "clear",
            "cleared_edges": int(cleared_edges),
            **_status_payload(context, after),
        }
    finally:
        if service is not None:
            await service.cleanup_workspace_instance(context.workspace_id)


async def _run_apply(context: WorkspaceContext, args: argparse.Namespace) -> dict[str, Any]:
    service = None
    try:
        service, rag = await _open_service_and_rag(
            context,
            synonymy_threshold=args.synonymy_threshold,
            synonymy_topk=args.synonymy_topk,
            synonymy_min_entity_len=args.synonymy_min_entity_len,
        )
        before = await _collect_workspace_snapshot(rag, context)
        result = await rag.lightrag.rebuild_synonym_edges(reset_existing=True)
        cleared_edges = int(result.get("cleared_edges", 0))
        created_edges = int(result.get("created_edges", 0))
        result = dict(result)
        result["manifest_path"] = str(context.manifest_path)
        after = await _collect_workspace_snapshot(rag, context)
        issues = _build_safety_issues(before, after)
        if issues:
            raise RuntimeError(
                "Detected factual graph or storage mutation while rebuilding synonym edges: "
                + json.dumps(issues, ensure_ascii=False, indent=2)
            )
        _write_synonym_manifest(
            context,
            service,
            cleared_edges=cleared_edges,
            created_edges=created_edges,
        )
        ready = getattr(service, "_workspace_synonym_ready", None)
        if ready is not None:
            ready.add(context.workspace_id)
        return {
            "success": True,
            "command": "apply",
            "requested_threshold": float(args.synonymy_threshold),
            "requested_topk": int(service.settings.synonymy_topk),
            "requested_min_entity_len": int(service.settings.synonymy_min_entity_len),
            **result,
            **_status_payload(context, after),
        }
    finally:
        if service is not None:
            await service.cleanup_workspace_instance(context.workspace_id)


async def _amain(args: argparse.Namespace) -> dict[str, Any]:
    context = _resolve_workspace_context(
        args.workspace_path,
        workspace_id_raw=getattr(args, "workspace_id", None),
    )
    if args.command == "status":
        return await _run_status(context)
    if args.command == "clear":
        return await _run_clear(context)
    if args.command == "apply":
        return await _run_apply(context, args)
    raise ValueError(f"Unsupported command: {args.command}")


def main() -> int:
    args = _parse_args()
    try:
        payload = asyncio.run(_amain(args))
    except Exception as exc:
        error_payload = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
