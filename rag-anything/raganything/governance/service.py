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

    async def backfill_legacy_workspaces(self, workspace_ids: list[str]) -> int:
        """Insert workspace rows for any workspace_id not already in PG.

        Used at startup to register filesystem-only workspaces created before
        the governance layer existed. Returns count of new rows inserted.
        """
        if not workspace_ids:
            return 0
        async with self._pool.acquire() as conn:
            existing = {
                r["workspace_id"]
                for r in await conn.fetch(
                    "SELECT workspace_id FROM workspaces WHERE workspace_id = ANY($1::text[])",
                    workspace_ids,
                )
            }
            new = [w for w in workspace_ids if w not in existing]
            if not new:
                return 0
            await conn.executemany("""
                INSERT INTO workspaces (workspace_id, metadata)
                VALUES ($1, $2)
                ON CONFLICT (workspace_id) DO NOTHING
            """, [(w, {"legacy": True}) for w in new])
        return len(new)

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
                   AND bool_and(doc_id = $4)
            """, workspace_id, kind, ref_ids, doc_id)
        return [r["ref_id"] for r in rows]

    async def delete_provenance_for_doc(self, doc_id) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM provenance WHERE doc_id = $1", doc_id)

    # --- jobs -----------------------------------------------------------------

    async def create_job(self, workspace_id: str, doc_ids: list) -> "UUID":  # type: ignore
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO ingest_jobs (workspace_id, doc_ids, status)
                VALUES ($1, $2, 'queued')
                RETURNING job_id
            """, workspace_id, doc_ids)
        return row["job_id"]

    async def mark_job_running(self, job_id) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingest_jobs SET status = 'running' WHERE job_id = $1", job_id
            )

    async def mark_job_done(self, job_id) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE ingest_jobs
                   SET status = 'done', finished_at = NOW()
                 WHERE job_id = $1
            """, job_id)

    async def mark_job_failed(self, job_id, error: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE ingest_jobs
                   SET status = 'failed', error = $2, finished_at = NOW()
                 WHERE job_id = $1
            """, job_id, error)

    async def update_job_progress(self, job_id, progress: dict) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingest_jobs SET progress = $2 WHERE job_id = $1",
                job_id, progress,
            )

    async def get_job(self, job_id):
        from raganything.governance.models import JobRow
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM ingest_jobs WHERE job_id = $1", job_id)
        return JobRow.model_validate(dict(row)) if row else None

    async def list_jobs(
        self, workspace_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ):
        from raganything.governance.models import JobRow
        clauses, args = [], []
        if workspace_id:
            args.append(workspace_id)
            clauses.append(f"workspace_id = ${len(args)}")
        if status:
            args.append(status)
            clauses.append(f"status = ${len(args)}")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        args.append(limit)
        sql = f"SELECT * FROM ingest_jobs {where} ORDER BY started_at DESC LIMIT ${len(args)}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [JobRow.model_validate(dict(r)) for r in rows]

    # --- audit ----------------------------------------------------------------

    async def record_audit(
        self,
        workspace_id: str,
        action: str,
        *,
        doc_id=None,
        actor: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ingest_audit (workspace_id, doc_id, action, actor, details)
                VALUES ($1, $2, $3, $4, $5)
            """, workspace_id, doc_id, action, actor, details or {})

    async def list_audit(
        self,
        workspace_id: Optional[str] = None,
        *,
        action: Optional[str] = None,
        limit: int = 100,
    ):
        from raganything.governance.models import AuditRow
        clauses, args = [], []
        if workspace_id:
            args.append(workspace_id)
            clauses.append(f"workspace_id = ${len(args)}")
        if action:
            args.append(action)
            clauses.append(f"action = ${len(args)}")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        args.append(limit)
        sql = f"SELECT * FROM ingest_audit {where} ORDER BY timestamp DESC LIMIT ${len(args)}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [AuditRow.model_validate(dict(r)) for r in rows]

    # --- ingest orchestration -------------------------------------------------

    async def run_ingest(
        self,
        job_id,
        doc_id,
        workspace_id: str,
        file_path: str,
        *,
        chunking_strategy: Optional[str] = None,
    ) -> None:
        """Background coroutine launched by JobRunner.

        Captures chunk provenance by:
          1. Registering IngestProvenanceCallback to capture LightRAG's
             internal `doc-<hash>` id via the on_document_complete hook.
          2. After ingest, querying lr.text_chunks for entries whose
             `full_doc_id` matches the captured LightRAG id.

        Falls back to a before/after snapshot diff if the callback didn't
        fire (e.g., on errors or unsupported LightRAG versions).
        """
        from raganything.governance.callbacks import (
            IngestProvenanceCallback, JobProgressCallback,
        )
        from pathlib import Path as _P

        if self._rag is None:
            raise RuntimeError("GovernanceService.run_ingest requires rag_service injection")

        prov_cb = IngestProvenanceCallback(workspace_id, doc_id)
        prog_cb = JobProgressCallback(self, job_id)
        self._rag.register_callback(prov_cb)
        self._rag.register_callback(prog_cb)

        await self.mark_document_status(doc_id, "parsing")
        try:
            rag_inner = await self._rag.get_rag(workspace_id)
            await rag_inner._ensure_lightrag_initialized()
            lr = rag_inner.lightrag

            # Snapshot BEFORE in case the doc_id callback path doesn't fire.
            chunks_before = await self._snapshot_chunk_ids(lr)

            workspace_output = str(_P(self._rag.settings.output_dir) / workspace_id)
            await self._rag.ingest(
                file_path,
                workspace_id=workspace_id,
                output_dir=workspace_output,
                chunking_strategy=chunking_strategy or None,
            )

            # Primary path: filter text_chunks by full_doc_id captured from
            # on_document_complete. This is robust against concurrent ingests
            # in a way snapshot-diff is not.
            new_chunk_ids: list[str] = []
            if prov_cb.lightrag_doc_id:
                new_chunk_ids = await self._find_chunks_for_lightrag_doc(
                    lr, prov_cb.lightrag_doc_id,
                )
                if new_chunk_ids:
                    logger.info(
                        "run_ingest %s/%s captured %d chunks via full_doc_id=%s",
                        workspace_id, doc_id, len(new_chunk_ids), prov_cb.lightrag_doc_id,
                    )

            # Fallback: snapshot diff
            if not new_chunk_ids:
                chunks_after = await self._snapshot_chunk_ids(lr)
                new_chunk_ids = sorted(chunks_after - chunks_before)
                logger.info(
                    "run_ingest %s/%s captured %d new chunks via snapshot diff",
                    workspace_id, doc_id, len(new_chunk_ids),
                )

            await self.insert_provenance(workspace_id, doc_id, "chunk", new_chunk_ids)
            await self._backfill_entity_relation_provenance(
                workspace_id, doc_id, new_chunk_ids,
            )
            await self.mark_document_status(doc_id, "done", finished=True)
            await self.record_audit(
                workspace_id, "ingest",
                doc_id=doc_id,
                details={
                    "chunk_count": len(new_chunk_ids),
                    "filename": _P(file_path).name,
                    "lightrag_doc_id": prov_cb.lightrag_doc_id,
                },
            )
            await prog_cb.flush()
        except Exception as exc:
            await self.mark_document_status(
                doc_id, "failed", error=str(exc), finished=True,
            )
            await self.record_audit(
                workspace_id, "ingest_failed",
                doc_id=doc_id, details={"error": str(exc)},
            )
            raise
        finally:
            self._rag.unregister_callback(prov_cb)
            self._rag.unregister_callback(prog_cb)

    async def _find_chunks_for_lightrag_doc(self, lr, lightrag_doc_id: str) -> list[str]:
        """Query LightRAG's text_chunks KV and return chunk_ids whose full_doc_id matches.

        Returns [] on failure so callers fall back to snapshot-diff.
        """
        if lr is None or not lightrag_doc_id:
            return []
        tc = getattr(lr, "text_chunks", None)
        if tc is None:
            return []
        try:
            # 1. JsonKVStorage in-memory dict (multiprocessing shared)
            inner = getattr(tc, "_data", None)
            if isinstance(inner, dict):
                return sorted(
                    cid for cid, data in inner.items()
                    if isinstance(data, dict) and data.get("full_doc_id") == lightrag_doc_id
                )

            # 2. PGKVStorage direct query
            db = getattr(tc, "db", None)
            if db is not None and callable(getattr(db, "query", None)):
                workspace = getattr(tc, "workspace", "") or ""
                rows = await db.query(
                    "SELECT id FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND full_doc_id=$2",
                    [workspace, lightrag_doc_id],
                    multirows=True,
                ) or []
                return sorted(r["id"] for r in rows)

            # 3. Future-proof: get_all() returning dict[id, value]
            if hasattr(tc, "get_all"):
                data = await tc.get_all()
                if isinstance(data, dict):
                    return sorted(
                        cid for cid, v in data.items()
                        if isinstance(v, dict) and v.get("full_doc_id") == lightrag_doc_id
                    )
        except Exception:
            logger.exception(
                "_find_chunks_for_lightrag_doc failed for %s; will fall back to snapshot diff",
                lightrag_doc_id,
            )
        return []

    async def _snapshot_chunk_ids(self, lr) -> set[str]:
        """Return the current set of chunk IDs from LightRAG's text_chunks KV.

        Tries storage APIs in order of preference:
          1. JsonKVStorage: reads ._data dict directly (in-memory shared state).
          2. PGKVStorage: issues a direct SELECT id FROM LIGHTRAG_DOC_CHUNKS.
          3. Any KV with a get_all() -> dict[id, value] method (future-proof).
          4. Any KV with an all_keys() method.
        Returns an empty set on any failure so ingest still succeeds.
        Safe under JobRunner(max_concurrent=1) — no other writer is active.
        """
        if lr is None:
            return set()
        tc = getattr(lr, "text_chunks", None)
        if tc is None:
            return set()
        try:
            # 1. JsonKVStorage: _data is a shared-memory proxy dict populated after initialize()
            inner = getattr(tc, "_data", None)
            if isinstance(inner, dict):
                return set(inner.keys())

            # 2. PGKVStorage: use the injected db connection pool directly
            db = getattr(tc, "db", None)
            if db is not None and callable(getattr(db, "query", None)):
                workspace = getattr(tc, "workspace", "") or ""
                rows = await db.query(
                    "SELECT id FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1",
                    [workspace],
                    multirows=True,
                ) or []
                return {r["id"] for r in rows}

            # 3. Future-proof: get_all() returning dict[id, value]
            if hasattr(tc, "get_all"):
                data = await tc.get_all()
                if isinstance(data, dict):
                    return set(data.keys())

            # 4. Future-proof: all_keys()
            if hasattr(tc, "all_keys"):
                keys = await tc.all_keys()
                return set(keys) if keys else set()

        except Exception:
            logger.exception("_snapshot_chunk_ids failed; provenance will be empty")
        return set()

    async def delete_document(self, workspace_id: str, doc_id) -> dict:
        """Five-step orchestrated delete. Returns a cleanup report dict."""
        report: dict = {"qdrant": "skipped", "neo4j": "skipped",
                        "kv": "skipped", "fs": "skipped"}
        prov = await self.get_provenance_for_doc(doc_id)
        chunk_ids = prov.get("chunk", [])
        entity_ids = prov.get("entity", [])
        relation_ids = prov.get("relation", [])

        await self.mark_document_status(doc_id, "deleting")

        rag = await self._rag.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        lr = rag.lightrag

        # 3a. Qdrant — delete points by id from the three collections.
        try:
            if chunk_ids and lr.chunks_vdb is not None:
                await lr.chunks_vdb.delete(chunk_ids)
            if entity_ids and lr.entities_vdb is not None:
                await lr.entities_vdb.delete(entity_ids)
            if relation_ids and lr.relationships_vdb is not None:
                await lr.relationships_vdb.delete(relation_ids)
            report["qdrant"] = "ok"
        except Exception as exc:
            report["qdrant"] = f"failed: {exc}"
            logger.exception("delete_document: qdrant cleanup failed")

        # 3b. Neo4j — only delete entities/relations whose only provenance is this doc.
        try:
            ent_exclusive = await self.find_doc_exclusive_refs(
                workspace_id, doc_id, "entity", entity_ids,
            )
            rel_exclusive = await self.find_doc_exclusive_refs(
                workspace_id, doc_id, "relation", relation_ids,
            )
            graph = lr.chunk_entity_relation_graph
            if graph is not None:
                for ent in ent_exclusive:
                    try:
                        await graph.delete_node(ent)
                    except Exception:
                        logger.warning("neo4j delete_node(%s) failed", ent)
                for rel_key in rel_exclusive:
                    if "::" in rel_key:
                        src, tgt = rel_key.split("::", 1)
                        try:
                            await graph.delete_edge(src, tgt)
                        except Exception:
                            logger.warning("neo4j delete_edge(%s) failed", rel_key)
            report["neo4j"] = "ok"
        except Exception as exc:
            report["neo4j"] = f"failed: {exc}"
            logger.exception("delete_document: neo4j cleanup failed")

        # 3c. LightRAG KV — chunks and full_docs.
        try:
            if chunk_ids and lr.text_chunks is not None:
                for cid in chunk_ids:
                    try:
                        await lr.text_chunks.delete(cid)
                    except Exception:
                        pass
            if lr.full_docs is not None:
                try:
                    await lr.full_docs.delete(str(doc_id))
                except Exception:
                    pass
            report["kv"] = "ok"
        except Exception as exc:
            report["kv"] = f"failed: {exc}"
            logger.exception("delete_document: kv cleanup failed")

        # 3d. Filesystem — only remove uploads/{ws}/{filename} if no other doc shares it.
        try:
            doc = await self.get_document(doc_id)
            if doc is not None:
                async with self._pool.acquire() as conn:
                    other = await conn.fetchval("""
                        SELECT COUNT(*) FROM documents
                         WHERE workspace_id = $1 AND filename = $2 AND doc_id <> $3
                    """, workspace_id, doc.filename, doc_id)
                if not other:
                    from raganything.constants import DEFAULT_UPLOADS_DIR
                    import os as _os
                    uploads = _os.environ.get("RAGANYTHING_UPLOADS_DIR", DEFAULT_UPLOADS_DIR)
                    from pathlib import Path as _P
                    target = _P(uploads).resolve() / workspace_id / doc.filename
                    if target.exists():
                        target.unlink()
            report["fs"] = "ok"
        except Exception as exc:
            report["fs"] = f"failed: {exc}"
            logger.exception("delete_document: fs cleanup failed")

        # 4. Provenance + status update.
        await self.delete_provenance_for_doc(doc_id)
        await self.mark_document_status(doc_id, "deleted", finished=True)

        # 5. Audit.
        await self.record_audit(
            workspace_id, "delete_doc", doc_id=doc_id, details={"cleanup": report},
        )
        return report

    async def _backfill_entity_relation_provenance(
        self, workspace_id: str, doc_id, chunk_ids: list[str]
    ) -> None:
        if not chunk_ids:
            return
        rag = await self._rag.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        lr = rag.lightrag
        ent_set: set[str] = set()
        rel_set: set[str] = set()
        for cid in chunk_ids:
            try:
                ents = await lr.entity_chunks.get(cid) or []
                rels = await lr.relation_chunks.get(cid) or []
            except Exception:
                logger.exception("backfill: failed to read provenance for chunk %s", cid)
                continue
            ent_set.update(ents if isinstance(ents, list) else [])
            rel_set.update(rels if isinstance(rels, list) else [])
        await self.insert_provenance(workspace_id, doc_id, "entity", sorted(ent_set))
        await self.insert_provenance(workspace_id, doc_id, "relation", sorted(rel_set))
