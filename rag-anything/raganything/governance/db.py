"""asyncpg pool factory and migration runner."""
from __future__ import annotations

import json
import logging
from importlib import resources

import asyncpg

from raganything.governance.settings import GovernanceSettings

logger = logging.getLogger(__name__)


async def create_pool(settings: GovernanceSettings) -> asyncpg.Pool:
    """Create an asyncpg pool. Caller is responsible for closing."""
    async def _init(conn):
        """Register JSONB codec on connection creation."""
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog',
        )

    pool = await asyncpg.create_pool(
        settings.pg_dsn,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
        command_timeout=settings.pg_command_timeout,
        init=_init,
    )
    if pool is None:
        raise RuntimeError(f"asyncpg.create_pool returned None for dsn={settings.pg_dsn}")
    logger.info(
        "governance: PG pool created (min=%d max=%d)",
        settings.pg_pool_min,
        settings.pg_pool_max,
    )
    return pool


def _list_migration_files() -> list[tuple[int, str, str]]:
    """Return [(version, filename, sql_text), ...] sorted by version."""
    migrations: list[tuple[int, str, str]] = []
    pkg = resources.files("raganything.governance.migrations")
    for entry in pkg.iterdir():
        name = entry.name
        if not name.endswith(".sql"):
            continue
        try:
            version = int(name.split("_", 1)[0])
        except ValueError:
            continue
        sql = entry.read_text(encoding="utf-8")
        migrations.append((version, name, sql))
    migrations.sort(key=lambda x: x[0])
    return migrations


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Apply any unapplied migrations. Idempotent."""
    async with pool.acquire() as conn:
        # Bootstrap schema_version so we can read it before applying any migration.
        # The first migration also creates this table; CREATE TABLE IF NOT EXISTS makes it safe.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version    INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        applied = {
            r["version"]
            for r in await conn.fetch("SELECT version FROM schema_version")
        }
        for version, name, sql in _list_migration_files():
            if version in applied:
                logger.debug("governance: migration %s already applied", name)
                continue
            logger.info("governance: applying migration %s", name)
            try:
                async with conn.transaction():
                    await conn.execute(sql)
            except Exception:
                logger.exception("governance: migration %s failed", name)
                raise


async def mark_orphaned_jobs_crashed(pool: asyncpg.Pool) -> int:
    """On startup, mark jobs left running by a previous process as crashed.

    Also marks documents in interim states as failed. Returns count of jobs touched.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            jobs_touched = await conn.execute(
                """
                UPDATE ingest_jobs
                   SET status = 'crashed',
                       error = COALESCE(error, 'process restart'),
                       finished_at = NOW()
                 WHERE status = 'running'
                """
            )
            await conn.execute(
                """
                UPDATE documents
                   SET status = 'failed',
                       error = COALESCE(error, 'process restart'),
                       finished_at = NOW()
                 WHERE status IN ('parsing', 'indexing', 'pending')
                """
            )
    # asyncpg's execute returns "UPDATE n" — extract n
    try:
        return int(jobs_touched.split()[-1])
    except (ValueError, IndexError):
        return 0
