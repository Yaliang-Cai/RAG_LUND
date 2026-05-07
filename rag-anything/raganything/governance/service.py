"""GovernanceService — single class, all PostgreSQL access.

Public methods are organized by concern:
  - workspaces: ensure, freeze, unfreeze, get
  - documents:  upsert, mark_status, get, list_by_workspace
  - provenance: insert_chunks, backfill_entities_relations, lookup, delete_for_doc
  - jobs:       create, mark_running, mark_done, mark_failed, get, list, cancel_pending
  - audit:      record, list_by_workspace
"""
from __future__ import annotations

import logging
from typing import Optional

import asyncpg

from raganything.governance.models import DocumentRow, DocumentStatus, WorkspaceRow

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
        return WorkspaceRow.model_validate(dict(row)) if row else None

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

    # --- documents -------------------------------------------------------

    async def upsert_document(
        self,
        workspace_id: str,
        filename: str,
        file_hash: str,
        size_bytes: int,
        *,
        force: bool = False,
    ) -> tuple["UUID", bool]:  # type: ignore[name-defined]
        """Insert a document; return (doc_id, is_duplicate).

        - If (workspace_id, file_hash) already exists and force=False, return existing
          doc_id with is_duplicate=True.
        - If force=True, atomically delete any existing document with the same hash and
          insert a fresh row. Cascade FK on `provenance.doc_id` deletes provenance for
          the old doc — this is the intended "re-ingest from scratch" semantic.
        """
        async with self._pool.acquire() as conn:
            if force:
                async with conn.transaction():
                    await conn.execute(
                        "DELETE FROM documents WHERE workspace_id = $1 AND file_hash = $2",
                        workspace_id, file_hash,
                    )
                    row = await conn.fetchrow(
                        """
                        INSERT INTO documents (workspace_id, filename, file_hash, size_bytes, status)
                        VALUES ($1, $2, $3, $4, 'pending')
                        RETURNING doc_id
                        """,
                        workspace_id, filename, file_hash, size_bytes,
                    )
                return row["doc_id"], False

            row = await conn.fetchrow("""
                INSERT INTO documents (workspace_id, filename, file_hash, size_bytes, status)
                VALUES ($1, $2, $3, $4, 'pending')
                ON CONFLICT (workspace_id, file_hash) DO NOTHING
                RETURNING doc_id
            """, workspace_id, filename, file_hash, size_bytes)
            if row is not None:
                return row["doc_id"], False

            existing = await conn.fetchval("""
                SELECT doc_id FROM documents
                 WHERE workspace_id = $1 AND file_hash = $2
            """, workspace_id, file_hash)
            return existing, True

    async def mark_document_status(
        self, doc_id, status: DocumentStatus, *, error: Optional[str] = None, finished: bool = False,
    ) -> None:
        async with self._pool.acquire() as conn:
            if finished:
                await conn.execute("""
                    UPDATE documents
                       SET status = $2, error = $3, finished_at = NOW()
                     WHERE doc_id = $1
                """, doc_id, status, error)
            else:
                await conn.execute("""
                    UPDATE documents
                       SET status = $2, error = COALESCE($3, error)
                     WHERE doc_id = $1
                """, doc_id, status, error)

    async def get_document(self, doc_id):
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM documents WHERE doc_id = $1", doc_id)
        return DocumentRow.model_validate(dict(row)) if row else None

    async def list_documents(self, workspace_id: str, *, status: Optional[DocumentStatus] = None):
        async with self._pool.acquire() as conn:
            if status:
                rows = await conn.fetch("""
                    SELECT * FROM documents
                     WHERE workspace_id = $1 AND status = $2
                     ORDER BY ingested_at DESC
                """, workspace_id, status)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM documents
                     WHERE workspace_id = $1
                     ORDER BY ingested_at DESC
                """, workspace_id)
        return [DocumentRow.model_validate(dict(r)) for r in rows]

    # --- provenance -----------------------------------------------------------

    async def insert_provenance(
        self, workspace_id: str, doc_id, kind: str, ref_ids: list[str]
    ) -> None:
        """Bulk insert provenance rows. ON CONFLICT DO NOTHING (idempotent)."""
        if not ref_ids:
            return
        records = [(workspace_id, doc_id, kind, r) for r in ref_ids]
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany("""
                    INSERT INTO provenance (workspace_id, doc_id, kind, ref_id)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                """, records)

    async def get_provenance_for_doc(self, doc_id) -> dict[str, list[str]]:
        """Return {kind: [ref_id, ...]} for the given doc."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT kind, ref_id FROM provenance WHERE doc_id = $1", doc_id
            )
        out: dict[str, list[str]] = {"chunk": [], "entity": [], "relation": []}
        for r in rows:
            out.setdefault(r["kind"], []).append(r["ref_id"])
        return out

    async def find_doc_exclusive_refs(
        self, workspace_id: str, doc_id, kind: str, ref_ids: list[str]
    ) -> list[str]:
        """Return the subset of ref_ids whose ONLY provenance source is this doc.

        Used by per-doc deletion to avoid removing entities/relations shared with other docs.
        """
        if not ref_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT ref_id
                  FROM provenance
                 WHERE workspace_id = $1 AND kind = $2 AND ref_id = ANY($3::text[])
                 GROUP BY ref_id
                HAVING COUNT(DISTINCT doc_id) = 1
                   AND (array_agg(DISTINCT doc_id))[1] = $4
            """, workspace_id, kind, ref_ids, doc_id)
        return [r["ref_id"] for r in rows]

    async def delete_provenance_for_doc(self, doc_id) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM provenance WHERE doc_id = $1", doc_id)
