"""Integration test fixtures: full GovernanceService over real PG, RAG mocked."""
import os
import uuid
import json
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
import pytest_asyncio

from raganything.governance.db import run_migrations
from raganything.governance.jobs import JobRunner
from raganything.governance.service import GovernanceService

PG_TEST_DSN = os.getenv("RAGANYTHING_PG_TEST_DSN",
                       "postgresql://localhost:5432/raganything_test")


def _pg_available() -> bool:
    import socket
    from urllib.parse import urlparse
    p = urlparse(PG_TEST_DSN)
    try:
        with socket.create_connection((p.hostname or "localhost", p.port or 5432), timeout=1):
            return True
    except OSError:
        return False


pytestmark_pg = pytest.mark.skipif(not _pg_available(), reason="PG not reachable")


@pytest_asyncio.fixture
async def pg_pool():
    schema = f"itest_{uuid.uuid4().hex[:8]}"
    boot = await asyncpg.connect(PG_TEST_DSN)
    try:
        await boot.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await boot.close()

    async def _setup(conn):
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog',
        )

    pool = await asyncpg.create_pool(PG_TEST_DSN, min_size=1, max_size=4, setup=_setup)
    try:
        await run_migrations(pool)
        yield pool
    finally:
        await pool.close()
        cleanup = await asyncpg.connect(PG_TEST_DSN)
        try: await cleanup.execute(f'DROP SCHEMA "{schema}" CASCADE')
        finally: await cleanup.close()


@pytest_asyncio.fixture
async def fake_rag():
    """Mock LocalRagService that records calls without doing anything."""
    rag = MagicMock()
    rag.settings = MagicMock(output_dir="/tmp")
    rag.ingest = AsyncMock(return_value="w1")
    rag.register_callback = MagicMock()
    rag.unregister_callback = MagicMock()
    inner_rag = MagicMock()
    inner_rag._ensure_lightrag_initialized = AsyncMock()
    inner_rag.lightrag = MagicMock()
    inner_rag.lightrag.entity_chunks = MagicMock()
    inner_rag.lightrag.entity_chunks.get = AsyncMock(return_value=[])
    inner_rag.lightrag.relation_chunks = MagicMock()
    inner_rag.lightrag.relation_chunks.get = AsyncMock(return_value=[])
    # text_chunks: simulate an empty JsonKVStorage so _snapshot_chunk_ids returns set()
    inner_rag.lightrag.text_chunks = MagicMock()
    inner_rag.lightrag.text_chunks._data = {}
    inner_rag.lightrag.text_chunks.db = None
    rag.get_rag = AsyncMock(return_value=inner_rag)
    return rag


@pytest_asyncio.fixture
async def gov(pg_pool, fake_rag):
    return GovernanceService(pg_pool, fake_rag)


@pytest_asyncio.fixture
async def runner(gov):
    r = JobRunner(gov, max_concurrent=2)
    await r.start()
    try:
        yield r
    finally:
        await r.stop(grace_period=2)
