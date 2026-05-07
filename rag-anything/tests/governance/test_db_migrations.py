import pytest

from raganything.governance.db import mark_orphaned_jobs_crashed, run_migrations

from .conftest import pytestmark_pg

pytestmark = [pytestmark_pg, pytest.mark.asyncio]


async def test_migrations_create_tables(pg_pool):
    await run_migrations(pg_pool)
    async with pg_pool.acquire() as conn:
        tables = {
            r["table_name"]
            for r in await conn.fetch(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = current_schema()
                """
            )
        }
    assert {
        "workspaces",
        "documents",
        "provenance",
        "ingest_jobs",
        "ingest_audit",
        "schema_version",
    } <= tables


async def test_migrations_idempotent(pg_pool):
    await run_migrations(pg_pool)
    await run_migrations(pg_pool)  # second run is a no-op
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("SELECT version FROM schema_version ORDER BY version")
    assert [r["version"] for r in rows] == [1]


async def test_mark_orphaned_jobs_crashed(pg_pool):
    await run_migrations(pg_pool)
    async with pg_pool.acquire() as conn:
        await conn.execute("INSERT INTO workspaces (workspace_id) VALUES ('w1')")
        await conn.execute(
            """
            INSERT INTO documents (workspace_id, filename, file_hash, size_bytes, status)
            VALUES ('w1', 'a.pdf', 'h1', 100, 'parsing')
            """
        )
        await conn.execute(
            "INSERT INTO ingest_jobs (workspace_id, status) VALUES ('w1', 'running')"
        )
    n = await mark_orphaned_jobs_crashed(pg_pool)
    assert n == 1
    async with pg_pool.acquire() as conn:
        job_status = await conn.fetchval("SELECT status FROM ingest_jobs LIMIT 1")
        doc_status = await conn.fetchval("SELECT status FROM documents LIMIT 1")
    assert job_status == "crashed"
    assert doc_status == "failed"
