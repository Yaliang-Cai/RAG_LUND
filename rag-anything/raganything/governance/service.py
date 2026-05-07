"""GovernanceService — single class, all PostgreSQL access.

Public methods are organized by concern:
  - workspaces: ensure, freeze, unfreeze, get
  - documents:  upsert, mark_status, get, list_by_workspace
  - provenance: insert_chunks, backfill_entities_relations, lookup, delete_for_doc
  - jobs:       create, mark_running, mark_done, mark_failed, get, list, cancel_pending
  - audit:      record, list_by_workspace
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import asyncpg

from raganything.governance.models import WorkspaceRow

logger = logging.getLogger(__name__)


class WorkspaceFrozenError(Exception):
    """Raised when a write op targets a frozen workspace."""


class GovernanceService:
    def __init__(self, pool: asyncpg.Pool, rag_service=None):
        self._pool = pool
        self._rag = rag_service  # injected later; not used by all methods

    # --- workspaces -----------------------------------------------------------

    async def ensure_workspace(self, workspace_id: str, *, owner: Optional[str] = None) -> None:
        """Insert workspace row if missing. No-op if it already exists."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workspaces (workspace_id, owner)
                VALUES ($1, $2)
                ON CONFLICT (workspace_id) DO NOTHING
            """, workspace_id, owner)

    async def get_workspace(self, workspace_id: str) -> Optional[WorkspaceRow]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workspaces WHERE workspace_id = $1", workspace_id
            )
        if row is None:
            return None
        # Convert asyncpg Record to dict, handling JSONB fields
        row_dict = dict(row)
        # Ensure metadata is a dict (asyncpg handles JSONB correctly, but be defensive)
        if isinstance(row_dict.get("metadata"), str):
            row_dict["metadata"] = json.loads(row_dict["metadata"])
        return WorkspaceRow.model_validate(row_dict)

    async def set_frozen(self, workspace_id: str, frozen: bool) -> bool:
        """Set the frozen flag. Returns True if the row existed and was updated."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE workspaces SET frozen = $2 WHERE workspace_id = $1",
                workspace_id, frozen,
            )
        # asyncpg execute returns "UPDATE n"
        return result.endswith(" 1")

    async def ensure_writable(self, workspace_id: str) -> None:
        """Raise WorkspaceFrozenError if the workspace exists and is frozen."""
        async with self._pool.acquire() as conn:
            frozen = await conn.fetchval(
                "SELECT frozen FROM workspaces WHERE workspace_id = $1", workspace_id
            )
        if frozen is True:
            raise WorkspaceFrozenError(f"workspace '{workspace_id}' is frozen")
