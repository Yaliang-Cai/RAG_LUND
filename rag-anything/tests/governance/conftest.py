"""Shared PG fixtures for governance tests.

Each test gets an ephemeral schema (CREATE SCHEMA test_<uuid>; SET search_path)
so tests don't trample each other and a single PG instance can run the full suite.
"""
from __future__ import annotations

import json
import os
import uuid

import asyncpg
import pytest
import pytest_asyncio

PG_TEST_DSN = os.getenv(
    "RAGANYTHING_PG_TEST_DSN",
    "postgresql://localhost:5432/raganything_test",
)


def _pg_available() -> bool:
    """Quick check: is the test PG reachable?"""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(PG_TEST_DSN)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


pytestmark_pg = pytest.mark.skipif(
    not _pg_available(),
    reason=f"PG not reachable at {PG_TEST_DSN}; set RAGANYTHING_PG_TEST_DSN or start local PG",
)


@pytest_asyncio.fixture
async def pg_pool():
    """Yield a pool whose default search_path is an ephemeral schema."""
    schema = f"test_{uuid.uuid4().hex[:12]}"
    bootstrap = await asyncpg.connect(PG_TEST_DSN)
    try:
        await bootstrap.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await bootstrap.close()

    async def _setup(conn):
        # `setup` runs on every acquire (asyncpg resets session state on release).
        await conn.execute(f'SET search_path TO "{schema}"')
        # Register JSONB codec (idempotent, persists on connection)
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog',
        )

    pool = await asyncpg.create_pool(
        PG_TEST_DSN, min_size=1, max_size=4, setup=_setup
    )
    try:
        yield pool
    finally:
        await pool.close()
        cleanup = await asyncpg.connect(PG_TEST_DSN)
        try:
            await cleanup.execute(f'DROP SCHEMA "{schema}" CASCADE')
        finally:
            await cleanup.close()
