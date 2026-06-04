"""Read + write endpoints for in-place graph property editing.

Backed by LightRAG's storage interface (upsert_node / upsert_edge).
All writes are gated by GovernanceService.ensure_writable and audited.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

# Server-managed keys that callers may not set/overwrite.
RESERVED_NODE_KEYS = frozenset({"source_id", "file_path", "created_at", "entity_id"})
RESERVED_EDGE_KEYS = frozenset({"source_id", "file_path", "created_at"})

router = APIRouter(tags=["graph-edit"])


class PropertyUpdate(BaseModel):
    properties: dict[str, Any] = Field(default_factory=dict)


def get_service(request: Request):
    """Resolved via app.dependency_overrides in app.py; tests override directly."""
    return request.app.state.service


def get_gov(request: Request):
    return request.app.state.gov


@router.get("/graph/{workspace_id}/nodes/{node_id}")
async def get_node(workspace_id: str, node_id: str, service=Depends(get_service)):
    rag = await service.get_rag(workspace_id)
    node = await rag.chunk_entity_relation_graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return {"id": node_id, "properties": dict(node)}
