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
