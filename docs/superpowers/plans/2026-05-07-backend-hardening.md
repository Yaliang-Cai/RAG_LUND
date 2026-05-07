# Backend Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PostgreSQL-backed governance layer over the existing RAG-Anything FastAPI service, refactor onto FastAPI lifespan, and convert ingest to a job-based API with status polling — without touching Neo4j, Qdrant, or LightRAG-internal storage.

**Architecture:** Five-phase rollout from spec §10. Phase 1 introduces the `raganything/governance/` module and PG schema with no route changes. Phase 2 refactors lifespan + DI. Phase 3 backfills existing workspaces. Phase 4 converts `/ingest` to job-based. Phase 5 adds new governance endpoints (freeze, per-doc delete, audit log).

**Tech Stack:** FastAPI, asyncpg, PostgreSQL ≥ 13 (uses `gen_random_uuid()` from `pgcrypto`), pydantic v2, pytest + pytest-asyncio, existing LightRAG/Neo4j/Qdrant stack untouched.

**Spec:** `docs/superpowers/specs/2026-05-07-backend-hardening-design.md` (commit 9d38af2)

---

## File structure

### Created files

```
rag-anything/raganything/governance/
  __init__.py             # public exports: GovernanceService, JobRunner, GovernanceSettings
  settings.py             # GovernanceSettings.from_env()
  db.py                   # asyncpg pool factory, run_migrations(), mark_orphaned_jobs_crashed()
  models.py               # pydantic schemas: DocumentRow, JobRow, AuditRow, IngestResponse
  service.py              # GovernanceService — single class, all PG access
  jobs.py                 # JobRunner (in-process asyncio task tracker)
  callbacks.py            # IngestProvenanceCallback, JobProgressCallback
  migrations/
    __init__.py
    001_init.sql          # all five tables + indexes
```

```
rag-anything/tests/governance/
  __init__.py
  conftest.py             # ephemeral PG schema fixture
  test_settings.py
  test_db_migrations.py
  test_service_workspaces.py
  test_service_documents.py
  test_service_provenance.py
  test_service_audit.py
  test_jobs.py
  test_callbacks.py
```

```
rag-anything/tests/integration/governance/
  __init__.py
  conftest.py
  test_ingest_flow.py
  test_delete_doc.py
  test_freeze.py
```

```
rag-anything/tests/manual/
  governance.http         # REST Client format scripts for ops reference
```

### Modified files

| File | Change |
|---|---|
| `rag-anything/raganything/constants.py` | Add 6 governance defaults (DSN, pool sizes, job concurrency, etc.) |
| `rag-anything/raganything/services/local_rag.py` | Add `LocalRagService.aclose()` |
| `rag-anything/server/app.py` | Lifespan + `app.state` DI refactor; new endpoints; `/ingest` becomes job-based |
| `rag-anything/server/SERVER_ARCH.md` | Document new endpoints + governance layer |
| `rag-anything/pyproject.toml` | Add `asyncpg>=0.29` dependency |

### File responsibility boundaries

- **`db.py`** owns the asyncpg pool factory and migration runner. Pure infrastructure — no business logic.
- **`service.py`** is the only place outside `db.py` that executes SQL. Routes never touch PG directly.
- **`jobs.py`** owns the in-process task lifecycle. Talks to `service.py` for state updates, never to PG directly.
- **`callbacks.py`** holds the `ProcessingCallback` subclasses that bridge LightRAG events into governance state.
- **`server/app.py`** routes are thin: validate input → call `gov.x()` or `rag.x()` → return.

---

# Phase 1 — PG infrastructure and governance module (no route changes)

This phase ships zero behavior changes. Existing endpoints work identically. Phase 1 lets you bring up PostgreSQL, run migrations, exercise `GovernanceService` and `JobRunner` from tests in isolation, and verify the schema before any route touches it.

## Task 1: Add governance dependencies

**Files:**
- Modify: `rag-anything/pyproject.toml`

- [ ] **Step 1: Add asyncpg to project dependencies**

Open `rag-anything/pyproject.toml`. Find the `[project] dependencies = [...]` list. Add `"asyncpg>=0.29"` to the list.

- [ ] **Step 2: Install in current env**

```bash
cd rag-anything && pip install -e .
```

Expected: `asyncpg` installed without error.

- [ ] **Step 3: Smoke-import**

```bash
python -c "import asyncpg; print(asyncpg.__version__)"
```

Expected: prints version `>= 0.29`.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/pyproject.toml
git commit -m "deps: add asyncpg for governance layer"
```

---

## Task 2: Add governance constants

**Files:**
- Modify: `rag-anything/raganything/constants.py`

- [ ] **Step 1: Append governance defaults**

Open `rag-anything/raganything/constants.py`. At the end of the file (after the last existing constant), append:

```python
# ---------------------------------------------------------------------------
# Governance layer (PostgreSQL-backed)
# ---------------------------------------------------------------------------

DEFAULT_PG_DSN: str = "postgresql://localhost:5432/raganything"
DEFAULT_PG_POOL_MIN: int = 2
DEFAULT_PG_POOL_MAX: int = 10
DEFAULT_PG_COMMAND_TIMEOUT_SECONDS: int = 30

DEFAULT_JOB_MAX_CONCURRENT: int = 1
DEFAULT_JOB_PROGRESS_INTERVAL_SECONDS: int = 5
DEFAULT_JOB_PROGRESS_CHUNK_INTERVAL: int = 10
DEFAULT_JOB_SHUTDOWN_GRACE_SECONDS: int = 30
```

- [ ] **Step 2: Verify imports still work**

```bash
python -c "from raganything.constants import DEFAULT_PG_DSN, DEFAULT_JOB_MAX_CONCURRENT; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/raganything/constants.py
git commit -m "feat(governance): add governance defaults to constants"
```

---

## Task 3: Create governance package skeleton

**Files:**
- Create: `rag-anything/raganything/governance/__init__.py`
- Create: `rag-anything/raganything/governance/settings.py`

- [ ] **Step 1: Write `settings.py`**

Create `rag-anything/raganything/governance/settings.py`:

```python
"""Governance layer settings loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from raganything.constants import (
    DEFAULT_PG_DSN,
    DEFAULT_PG_POOL_MIN,
    DEFAULT_PG_POOL_MAX,
    DEFAULT_PG_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_JOB_MAX_CONCURRENT,
    DEFAULT_JOB_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_JOB_PROGRESS_CHUNK_INTERVAL,
    DEFAULT_JOB_SHUTDOWN_GRACE_SECONDS,
)


@dataclass(frozen=True)
class GovernanceSettings:
    pg_dsn: str = DEFAULT_PG_DSN
    pg_pool_min: int = DEFAULT_PG_POOL_MIN
    pg_pool_max: int = DEFAULT_PG_POOL_MAX
    pg_command_timeout: int = DEFAULT_PG_COMMAND_TIMEOUT_SECONDS
    job_max_concurrent: int = DEFAULT_JOB_MAX_CONCURRENT
    job_progress_interval: int = DEFAULT_JOB_PROGRESS_INTERVAL_SECONDS
    job_progress_chunk_interval: int = DEFAULT_JOB_PROGRESS_CHUNK_INTERVAL
    job_shutdown_grace: int = DEFAULT_JOB_SHUTDOWN_GRACE_SECONDS

    @classmethod
    def from_env(cls) -> "GovernanceSettings":
        return cls(
            pg_dsn=os.getenv("RAGANYTHING_PG_DSN", DEFAULT_PG_DSN),
            pg_pool_min=int(os.getenv("RAGANYTHING_PG_POOL_MIN", DEFAULT_PG_POOL_MIN)),
            pg_pool_max=int(os.getenv("RAGANYTHING_PG_POOL_MAX", DEFAULT_PG_POOL_MAX)),
            pg_command_timeout=int(os.getenv(
                "RAGANYTHING_PG_COMMAND_TIMEOUT", DEFAULT_PG_COMMAND_TIMEOUT_SECONDS
            )),
            job_max_concurrent=int(os.getenv(
                "RAGANYTHING_JOB_MAX_CONCURRENT", DEFAULT_JOB_MAX_CONCURRENT
            )),
            job_progress_interval=int(os.getenv(
                "RAGANYTHING_JOB_PROGRESS_INTERVAL", DEFAULT_JOB_PROGRESS_INTERVAL_SECONDS
            )),
            job_progress_chunk_interval=int(os.getenv(
                "RAGANYTHING_JOB_PROGRESS_CHUNK_INTERVAL", DEFAULT_JOB_PROGRESS_CHUNK_INTERVAL
            )),
            job_shutdown_grace=int(os.getenv(
                "RAGANYTHING_JOB_SHUTDOWN_GRACE", DEFAULT_JOB_SHUTDOWN_GRACE_SECONDS
            )),
        )
```

- [ ] **Step 2: Write `__init__.py`**

Create `rag-anything/raganything/governance/__init__.py`:

```python
"""PostgreSQL-backed governance layer for RAG-Anything."""
from raganything.governance.settings import GovernanceSettings

__all__ = ["GovernanceSettings"]
```

- [ ] **Step 3: Write the failing test**

Create `rag-anything/tests/governance/__init__.py` (empty file).

Create `rag-anything/tests/governance/test_settings.py`:

```python
import os
from unittest.mock import patch

from raganything.governance.settings import GovernanceSettings


def test_defaults_loaded_when_env_unset():
    with patch.dict(os.environ, {}, clear=False):
        # Strip all RAGANYTHING_* vars to test defaults
        for key in list(os.environ):
            if key.startswith("RAGANYTHING_PG_") or key.startswith("RAGANYTHING_JOB_"):
                os.environ.pop(key, None)
        s = GovernanceSettings.from_env()
        assert s.pg_dsn == "postgresql://localhost:5432/raganything"
        assert s.pg_pool_min == 2
        assert s.pg_pool_max == 10
        assert s.job_max_concurrent == 1


def test_env_overrides_applied():
    with patch.dict(os.environ, {
        "RAGANYTHING_PG_DSN": "postgresql://test:5433/db",
        "RAGANYTHING_PG_POOL_MAX": "20",
        "RAGANYTHING_JOB_MAX_CONCURRENT": "4",
    }):
        s = GovernanceSettings.from_env()
        assert s.pg_dsn == "postgresql://test:5433/db"
        assert s.pg_pool_max == 20
        assert s.job_max_concurrent == 4
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd rag-anything && pytest tests/governance/test_settings.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/governance/ rag-anything/tests/governance/__init__.py rag-anything/tests/governance/test_settings.py
git commit -m "feat(governance): scaffold package with GovernanceSettings"
```

---

## Task 4: Write the schema migration

**Files:**
- Create: `rag-anything/raganything/governance/migrations/__init__.py`
- Create: `rag-anything/raganything/governance/migrations/001_init.sql`

- [ ] **Step 1: Create migration file**

Create `rag-anything/raganything/governance/migrations/__init__.py` (empty).

Create `rag-anything/raganything/governance/migrations/001_init.sql`:

```sql
-- 001_init: initial governance schema
-- Idempotent: can be re-run safely.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id   TEXT PRIMARY KEY,
    frozen         BOOLEAN NOT NULL DEFAULT FALSE,
    owner          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id   TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    filename       TEXT NOT NULL,
    file_hash      TEXT NOT NULL,
    size_bytes     BIGINT NOT NULL,
    status         TEXT NOT NULL,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ,
    error          TEXT,
    UNIQUE (workspace_id, file_hash)
);
CREATE INDEX IF NOT EXISTS idx_documents_ws_status ON documents(workspace_id, status);

CREATE TABLE IF NOT EXISTS provenance (
    workspace_id   TEXT NOT NULL,
    doc_id         UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,
    ref_id         TEXT NOT NULL,
    PRIMARY KEY (workspace_id, kind, ref_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_prov_doc ON provenance(doc_id);
CREATE INDEX IF NOT EXISTS idx_prov_ws_kind ON provenance(workspace_id, kind);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id   TEXT NOT NULL,
    doc_ids        UUID[] NOT NULL DEFAULT '{}',
    status         TEXT NOT NULL,
    progress       JSONB NOT NULL DEFAULT '{}'::jsonb,
    error          TEXT,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_jobs_ws_status ON ingest_jobs(workspace_id, status);

CREATE TABLE IF NOT EXISTS ingest_audit (
    id             BIGSERIAL PRIMARY KEY,
    workspace_id   TEXT NOT NULL,
    doc_id         UUID,
    action         TEXT NOT NULL,
    actor          TEXT,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details        JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_audit_ws_time ON ingest_audit(workspace_id, timestamp DESC);

INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
```

- [ ] **Step 2: Commit (no test yet — runner comes in Task 5)**

```bash
git add rag-anything/raganything/governance/migrations/
git commit -m "feat(governance): add initial schema migration"
```

---

## Task 5: Implement `db.py` — pool + migration runner

**Files:**
- Create: `rag-anything/raganything/governance/db.py`

- [ ] **Step 1: Write `db.py`**

```python
"""asyncpg pool factory and migration runner."""
from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Optional

import asyncpg

from raganything.governance.settings import GovernanceSettings

logger = logging.getLogger(__name__)


async def create_pool(settings: GovernanceSettings) -> asyncpg.Pool:
    """Create an asyncpg pool. Caller is responsible for closing."""
    pool = await asyncpg.create_pool(
        settings.pg_dsn,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
        command_timeout=settings.pg_command_timeout,
    )
    if pool is None:
        raise RuntimeError(f"asyncpg.create_pool returned None for dsn={settings.pg_dsn}")
    logger.info("governance: PG pool created (min=%d max=%d)",
                settings.pg_pool_min, settings.pg_pool_max)
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
        # Bootstrap schema_version if not present (the first migration creates it,
        # but we need to read it to know whether to skip).
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version    INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        applied = {r["version"] for r in await conn.fetch("SELECT version FROM schema_version")}
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
            jobs_touched = await conn.execute("""
                UPDATE ingest_jobs
                   SET status = 'crashed',
                       error = COALESCE(error, 'process restart'),
                       finished_at = NOW()
                 WHERE status = 'running'
            """)
            await conn.execute("""
                UPDATE documents
                   SET status = 'failed',
                       error = COALESCE(error, 'process restart'),
                       finished_at = NOW()
                 WHERE status IN ('parsing', 'indexing', 'pending')
            """)
    # asyncpg's execute returns "UPDATE n" — extract n
    try:
        return int(jobs_touched.split()[-1])
    except (ValueError, IndexError):
        return 0
```

- [ ] **Step 2: Write the failing test**

Create `rag-anything/tests/governance/conftest.py`:

```python
"""Shared PG fixtures for governance tests.

Each test gets an ephemeral schema (CREATE SCHEMA test_<uuid>; SET search_path)
so tests don't trample each other and a single PG instance can run the full suite.
"""
from __future__ import annotations

import os
import uuid
import pytest
import pytest_asyncio
import asyncpg

PG_TEST_DSN = os.getenv("RAGANYTHING_PG_TEST_DSN", "postgresql://localhost:5432/raganything_test")


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
    # Use a one-shot connection to create the schema
    bootstrap = await asyncpg.connect(PG_TEST_DSN)
    try:
        await bootstrap.execute(f'CREATE SCHEMA "{schema}"')
    finally:
        await bootstrap.close()

    async def _init(conn):
        await conn.execute(f'SET search_path TO "{schema}"')

    pool = await asyncpg.create_pool(PG_TEST_DSN, min_size=1, max_size=4, init=_init)
    try:
        yield pool
    finally:
        await pool.close()
        cleanup = await asyncpg.connect(PG_TEST_DSN)
        try:
            await cleanup.execute(f'DROP SCHEMA "{schema}" CASCADE')
        finally:
            await cleanup.close()
```

Create `rag-anything/tests/governance/test_db_migrations.py`:

```python
import pytest
from raganything.governance.db import run_migrations, mark_orphaned_jobs_crashed
from .conftest import pytestmark_pg

pytestmark = [pytestmark_pg, pytest.mark.asyncio]


async def test_migrations_create_tables(pg_pool):
    await run_migrations(pg_pool)
    async with pg_pool.acquire() as conn:
        tables = {r["table_name"] for r in await conn.fetch("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = current_schema()
        """)}
    assert {"workspaces", "documents", "provenance", "ingest_jobs", "ingest_audit",
            "schema_version"} <= tables


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
        await conn.execute("""
            INSERT INTO documents (workspace_id, filename, file_hash, size_bytes, status)
            VALUES ('w1', 'a.pdf', 'h1', 100, 'parsing')
        """)
        await conn.execute("""
            INSERT INTO ingest_jobs (workspace_id, status) VALUES ('w1', 'running')
        """)
    n = await mark_orphaned_jobs_crashed(pg_pool)
    assert n == 1
    async with pg_pool.acquire() as conn:
        job_status = await conn.fetchval("SELECT status FROM ingest_jobs LIMIT 1")
        doc_status = await conn.fetchval("SELECT status FROM documents LIMIT 1")
    assert job_status == "crashed"
    assert doc_status == "failed"
```

- [ ] **Step 3: Run tests**

```bash
cd rag-anything && pytest tests/governance/test_db_migrations.py -v
```

Expected: 3 passed (or skipped if no PG running — which is fine for now).

- [ ] **Step 4: Commit**

```bash
git add rag-anything/raganything/governance/db.py rag-anything/tests/governance/conftest.py rag-anything/tests/governance/test_db_migrations.py
git commit -m "feat(governance): add asyncpg pool + idempotent migration runner"
```

---

## Task 6: Models (pydantic schemas)

**Files:**
- Create: `rag-anything/raganything/governance/models.py`

- [ ] **Step 1: Write `models.py`**

```python
"""Pydantic schemas for the governance API surface."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

DocumentStatus = Literal[
    "pending", "parsing", "indexing", "done", "failed", "deleting", "deleted",
]
JobStatus = Literal["queued", "running", "done", "failed", "crashed"]


class WorkspaceRow(BaseModel):
    workspace_id: str
    frozen: bool
    owner: Optional[str] = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRow(BaseModel):
    doc_id: UUID
    workspace_id: str
    filename: str
    file_hash: str
    size_bytes: int
    status: DocumentStatus
    ingested_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class JobRow(BaseModel):
    job_id: UUID
    workspace_id: str
    doc_ids: list[UUID] = Field(default_factory=list)
    status: JobStatus
    progress: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None


class AuditRow(BaseModel):
    id: int
    workspace_id: str
    doc_id: Optional[UUID] = None
    action: str
    actor: Optional[str] = None
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    job_id: Optional[UUID]
    doc_id: UUID
    status: str  # "queued" | "duplicate"
    duplicate: bool
```

- [ ] **Step 2: Smoke test**

```bash
python -c "from raganything.governance.models import DocumentRow, JobRow, IngestResponse; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/raganything/governance/models.py
git commit -m "feat(governance): add pydantic models for governance API"
```

---

## Task 7: GovernanceService — workspaces

**Files:**
- Create: `rag-anything/raganything/governance/service.py` (initial: workspace ops only)
- Create: `rag-anything/tests/governance/test_service_workspaces.py`

- [ ] **Step 1: Write `service.py` (workspace ops)**

```python
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
```

- [ ] **Step 2: Write the failing test**

Create `rag-anything/tests/governance/test_service_workspaces.py`:

```python
import pytest
from raganything.governance.db import run_migrations
from raganything.governance.service import GovernanceService, WorkspaceFrozenError
from .conftest import pytestmark_pg

pytestmark = [pytestmark_pg, pytest.mark.asyncio]


@pytest.fixture
async def gov(pg_pool):
    await run_migrations(pg_pool)
    return GovernanceService(pg_pool)


async def test_ensure_workspace_idempotent(gov):
    await gov.ensure_workspace("w1", owner="alice")
    await gov.ensure_workspace("w1", owner="bob")  # no-op
    ws = await gov.get_workspace("w1")
    assert ws is not None
    assert ws.workspace_id == "w1"
    assert ws.owner == "alice"  # original owner preserved
    assert ws.frozen is False


async def test_set_frozen_toggles_flag(gov):
    await gov.ensure_workspace("w1")
    assert await gov.set_frozen("w1", True) is True
    ws = await gov.get_workspace("w1")
    assert ws.frozen is True
    assert await gov.set_frozen("w1", False) is True
    ws = await gov.get_workspace("w1")
    assert ws.frozen is False


async def test_set_frozen_returns_false_for_missing_workspace(gov):
    assert await gov.set_frozen("nope", True) is False


async def test_ensure_writable_raises_when_frozen(gov):
    await gov.ensure_workspace("w1")
    await gov.set_frozen("w1", True)
    with pytest.raises(WorkspaceFrozenError):
        await gov.ensure_writable("w1")


async def test_ensure_writable_passes_when_unfrozen_or_missing(gov):
    await gov.ensure_writable("never_seen")  # no row → not frozen
    await gov.ensure_workspace("w2")
    await gov.ensure_writable("w2")  # exists, frozen=False
```

- [ ] **Step 3: Run tests**

```bash
cd rag-anything && pytest tests/governance/test_service_workspaces.py -v
```

Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/raganything/governance/service.py rag-anything/tests/governance/test_service_workspaces.py
git commit -m "feat(governance): GovernanceService workspace ops"
```

---

## Task 8: GovernanceService — documents

**Files:**
- Modify: `rag-anything/raganything/governance/service.py` (append document methods)
- Create: `rag-anything/tests/governance/test_service_documents.py`

- [ ] **Step 1: Append document methods to `service.py`**

Add to `service.py` after the workspace methods, inside the class:

```python
    # --- documents ------------------------------------------------------------

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

        - If (workspace_id, file_hash) already exists and force=False, return existing doc_id with is_duplicate=True.
        - If force=True, always insert (filename can be reused; no UNIQUE on (workspace_id, filename)).
        """
        async with self._pool.acquire() as conn:
            if force:
                row = await conn.fetchrow("""
                    INSERT INTO documents (workspace_id, filename, file_hash, size_bytes, status)
                    VALUES ($1, $2, $3, $4, 'pending')
                    RETURNING doc_id
                """, workspace_id, filename, file_hash, size_bytes)
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
        self, doc_id, status: str, *, error: Optional[str] = None, finished: bool = False
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
        from raganything.governance.models import DocumentRow
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM documents WHERE doc_id = $1", doc_id)
        return DocumentRow.model_validate(dict(row)) if row else None

    async def list_documents(self, workspace_id: str, *, status: Optional[str] = None):
        from raganything.governance.models import DocumentRow
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
```

- [ ] **Step 2: Write the failing test**

Create `rag-anything/tests/governance/test_service_documents.py`:

```python
import pytest
from raganything.governance.db import run_migrations
from raganything.governance.service import GovernanceService
from .conftest import pytestmark_pg

pytestmark = [pytestmark_pg, pytest.mark.asyncio]


@pytest.fixture
async def gov(pg_pool):
    await run_migrations(pg_pool)
    g = GovernanceService(pg_pool)
    await g.ensure_workspace("w1")
    return g


async def test_upsert_document_first_time(gov):
    doc_id, dup = await gov.upsert_document("w1", "a.pdf", "hash1", 1024)
    assert dup is False
    doc = await gov.get_document(doc_id)
    assert doc.filename == "a.pdf"
    assert doc.status == "pending"


async def test_upsert_document_duplicate_returns_existing(gov):
    doc1, _ = await gov.upsert_document("w1", "a.pdf", "hash1", 1024)
    doc2, dup = await gov.upsert_document("w1", "a.pdf", "hash1", 1024)
    assert dup is True
    assert doc1 == doc2


async def test_upsert_document_force_creates_new_row(gov):
    doc1, _ = await gov.upsert_document("w1", "a.pdf", "hash1", 1024)
    doc2, dup = await gov.upsert_document("w1", "a.pdf", "hash1", 1024, force=True)
    assert dup is False
    assert doc1 != doc2


async def test_mark_document_status_transitions(gov):
    doc_id, _ = await gov.upsert_document("w1", "a.pdf", "h", 1)
    await gov.mark_document_status(doc_id, "parsing")
    assert (await gov.get_document(doc_id)).status == "parsing"
    await gov.mark_document_status(doc_id, "done", finished=True)
    doc = await gov.get_document(doc_id)
    assert doc.status == "done"
    assert doc.finished_at is not None


async def test_list_documents_filters_by_status(gov):
    d1, _ = await gov.upsert_document("w1", "a.pdf", "h1", 1)
    d2, _ = await gov.upsert_document("w1", "b.pdf", "h2", 1)
    await gov.mark_document_status(d1, "done", finished=True)
    done = await gov.list_documents("w1", status="done")
    assert {d.doc_id for d in done} == {d1}
    all_docs = await gov.list_documents("w1")
    assert {d.doc_id for d in all_docs} == {d1, d2}
```

- [ ] **Step 3: Run tests**

```bash
cd rag-anything && pytest tests/governance/test_service_documents.py -v
```

Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/raganything/governance/service.py rag-anything/tests/governance/test_service_documents.py
git commit -m "feat(governance): GovernanceService document ops with idempotency"
```

---

## Task 9: GovernanceService — provenance

**Files:**
- Modify: `rag-anything/raganything/governance/service.py` (append provenance methods)
- Create: `rag-anything/tests/governance/test_service_provenance.py`

- [ ] **Step 1: Append provenance methods to `service.py`**

Add inside the `GovernanceService` class:

```python
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
                   AND MAX(doc_id) = $4
            """, workspace_id, kind, ref_ids, doc_id)
        return [r["ref_id"] for r in rows]

    async def delete_provenance_for_doc(self, doc_id) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM provenance WHERE doc_id = $1", doc_id)
```

- [ ] **Step 2: Write the failing test**

Create `rag-anything/tests/governance/test_service_provenance.py`:

```python
import pytest
from raganything.governance.db import run_migrations
from raganything.governance.service import GovernanceService
from .conftest import pytestmark_pg

pytestmark = [pytestmark_pg, pytest.mark.asyncio]


@pytest.fixture
async def gov(pg_pool):
    await run_migrations(pg_pool)
    g = GovernanceService(pg_pool)
    await g.ensure_workspace("w1")
    return g


async def test_insert_and_get_provenance(gov):
    d, _ = await gov.upsert_document("w1", "a.pdf", "h", 1)
    await gov.insert_provenance("w1", d, "chunk", ["c1", "c2"])
    await gov.insert_provenance("w1", d, "entity", ["E.Apple"])
    prov = await gov.get_provenance_for_doc(d)
    assert sorted(prov["chunk"]) == ["c1", "c2"]
    assert prov["entity"] == ["E.Apple"]
    assert prov["relation"] == []


async def test_insert_provenance_idempotent(gov):
    d, _ = await gov.upsert_document("w1", "a.pdf", "h", 1)
    await gov.insert_provenance("w1", d, "chunk", ["c1"])
    await gov.insert_provenance("w1", d, "chunk", ["c1"])  # no-op
    prov = await gov.get_provenance_for_doc(d)
    assert prov["chunk"] == ["c1"]


async def test_find_doc_exclusive_refs_isolates_unique(gov):
    d1, _ = await gov.upsert_document("w1", "a.pdf", "h1", 1)
    d2, _ = await gov.upsert_document("w1", "b.pdf", "h2", 1)
    await gov.insert_provenance("w1", d1, "entity", ["E.Shared", "E.OnlyD1"])
    await gov.insert_provenance("w1", d2, "entity", ["E.Shared", "E.OnlyD2"])
    exclusive = await gov.find_doc_exclusive_refs(
        "w1", d1, "entity", ["E.Shared", "E.OnlyD1"]
    )
    assert exclusive == ["E.OnlyD1"]


async def test_delete_provenance_for_doc(gov):
    d, _ = await gov.upsert_document("w1", "a.pdf", "h", 1)
    await gov.insert_provenance("w1", d, "chunk", ["c1", "c2"])
    await gov.delete_provenance_for_doc(d)
    prov = await gov.get_provenance_for_doc(d)
    assert prov == {"chunk": [], "entity": [], "relation": []}
```

- [ ] **Step 3: Run tests**

```bash
cd rag-anything && pytest tests/governance/test_service_provenance.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/raganything/governance/service.py rag-anything/tests/governance/test_service_provenance.py
git commit -m "feat(governance): GovernanceService provenance ops with shared-entity protection"
```

---

## Task 10: GovernanceService — jobs and audit

**Files:**
- Modify: `rag-anything/raganything/governance/service.py` (append job + audit methods)
- Create: `rag-anything/tests/governance/test_service_audit.py`

- [ ] **Step 1: Append job + audit methods**

Add inside `GovernanceService`:

```python
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
        import json as _json
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingest_jobs SET progress = $2 WHERE job_id = $1",
                job_id, _json.dumps(progress),
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
        import json as _json
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ingest_audit (workspace_id, doc_id, action, actor, details)
                VALUES ($1, $2, $3, $4, $5)
            """, workspace_id, doc_id, action, actor, _json.dumps(details or {}))

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
```

- [ ] **Step 2: Write the failing tests**

Create `rag-anything/tests/governance/test_service_audit.py`:

```python
import pytest
from raganything.governance.db import run_migrations
from raganything.governance.service import GovernanceService
from .conftest import pytestmark_pg

pytestmark = [pytestmark_pg, pytest.mark.asyncio]


@pytest.fixture
async def gov(pg_pool):
    await run_migrations(pg_pool)
    g = GovernanceService(pg_pool)
    await g.ensure_workspace("w1")
    return g


async def test_job_lifecycle(gov):
    d, _ = await gov.upsert_document("w1", "a.pdf", "h", 1)
    job_id = await gov.create_job("w1", [d])
    j = await gov.get_job(job_id)
    assert j.status == "queued"
    await gov.mark_job_running(job_id)
    assert (await gov.get_job(job_id)).status == "running"
    await gov.update_job_progress(job_id, {"parsed": 5, "indexed": 3, "total": 10})
    assert (await gov.get_job(job_id)).progress == {"parsed": 5, "indexed": 3, "total": 10}
    await gov.mark_job_done(job_id)
    j = await gov.get_job(job_id)
    assert j.status == "done"
    assert j.finished_at is not None


async def test_mark_job_failed_records_error(gov):
    d, _ = await gov.upsert_document("w1", "a.pdf", "h", 1)
    job_id = await gov.create_job("w1", [d])
    await gov.mark_job_failed(job_id, "boom")
    j = await gov.get_job(job_id)
    assert j.status == "failed"
    assert j.error == "boom"


async def test_list_jobs_filters(gov):
    d, _ = await gov.upsert_document("w1", "a.pdf", "h", 1)
    j1 = await gov.create_job("w1", [d])
    j2 = await gov.create_job("w1", [d])
    await gov.mark_job_done(j1)
    done = await gov.list_jobs(workspace_id="w1", status="done")
    assert {j.job_id for j in done} == {j1}
    queued = await gov.list_jobs(workspace_id="w1", status="queued")
    assert {j.job_id for j in queued} == {j2}


async def test_record_and_list_audit(gov):
    await gov.record_audit("w1", "ingest", actor="alice", details={"chunk_count": 5})
    await gov.record_audit("w1", "delete_doc", actor="bob")
    rows = await gov.list_audit(workspace_id="w1")
    assert len(rows) == 2
    actions = [r.action for r in rows]
    # most recent first
    assert actions == ["delete_doc", "ingest"]
    ingest_only = await gov.list_audit(workspace_id="w1", action="ingest")
    assert len(ingest_only) == 1
    assert ingest_only[0].details == {"chunk_count": 5}
```

- [ ] **Step 3: Run tests**

```bash
cd rag-anything && pytest tests/governance/test_service_audit.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/raganything/governance/service.py rag-anything/tests/governance/test_service_audit.py
git commit -m "feat(governance): GovernanceService job + audit ops"
```

---

## Task 11: JobRunner — in-process task tracker

**Files:**
- Create: `rag-anything/raganything/governance/jobs.py`
- Create: `rag-anything/tests/governance/test_jobs.py`

- [ ] **Step 1: Write `jobs.py`**

```python
"""In-process JobRunner — one asyncio.Task per job, state delegated to GovernanceService.

The submit/cancel/stop interface is the swap point for a future ARQ + Redis backend.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict
from uuid import UUID

logger = logging.getLogger(__name__)

CoroFactory = Callable[[], Awaitable[None]]


class JobRunner:
    def __init__(self, gov, max_concurrent: int = 1):
        self._gov = gov
        self._sem = asyncio.Semaphore(max_concurrent)
        self._tasks: Dict[UUID, asyncio.Task] = {}
        self._stopping = False

    async def start(self) -> None:
        """Reserved for symmetry with future Redis-backed runners. No-op today."""
        logger.info("JobRunner started (max_concurrent=%d)", self._sem._value)

    async def submit(self, job_id: UUID, coro_factory: CoroFactory) -> None:
        if self._stopping:
            raise RuntimeError("JobRunner is shutting down; refusing new submissions")
        task = asyncio.create_task(self._run(job_id, coro_factory))
        self._tasks[job_id] = task

    async def _run(self, job_id: UUID, coro_factory: CoroFactory) -> None:
        async with self._sem:
            try:
                await self._gov.mark_job_running(job_id)
                await coro_factory()
                await self._gov.mark_job_done(job_id)
            except asyncio.CancelledError:
                await self._gov.mark_job_failed(job_id, "cancelled")
                raise
            except Exception as exc:
                logger.exception("job %s failed", job_id)
                await self._gov.mark_job_failed(job_id, str(exc))
            finally:
                self._tasks.pop(job_id, None)

    async def cancel(self, job_id: UUID) -> bool:
        t = self._tasks.get(job_id)
        if t and not t.done():
            t.cancel()
            return True
        return False

    async def stop(self, grace_period: int) -> None:
        self._stopping = True
        if not self._tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=grace_period,
            )
        except asyncio.TimeoutError:
            logger.warning("JobRunner.stop: %d tasks exceeded grace; cancelling",
                           len(self._tasks))
            for t in list(self._tasks.values()):
                t.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
```

- [ ] **Step 2: Write the failing test**

Create `rag-anything/tests/governance/test_jobs.py`:

```python
import asyncio
import pytest
from uuid import uuid4

from raganything.governance.jobs import JobRunner

pytestmark = pytest.mark.asyncio


class FakeGov:
    def __init__(self):
        self.events = []
    async def mark_job_running(self, jid): self.events.append(("run", jid))
    async def mark_job_done(self, jid): self.events.append(("done", jid))
    async def mark_job_failed(self, jid, err): self.events.append(("fail", jid, err))


async def test_submit_runs_to_completion():
    gov = FakeGov()
    runner = JobRunner(gov, max_concurrent=2)
    await runner.start()
    jid = uuid4()
    done = asyncio.Event()
    async def work():
        done.set()
    await runner.submit(jid, work)
    await asyncio.wait_for(done.wait(), timeout=2)
    # Allow the task to flush its done callback
    await asyncio.sleep(0)
    assert ("run", jid) in gov.events
    assert ("done", jid) in gov.events


async def test_submit_records_failure():
    gov = FakeGov()
    runner = JobRunner(gov)
    jid = uuid4()
    async def boom():
        raise ValueError("nope")
    await runner.submit(jid, boom)
    for _ in range(20):
        await asyncio.sleep(0.05)
        if any(e[0] == "fail" for e in gov.events):
            break
    fails = [e for e in gov.events if e[0] == "fail"]
    assert fails and fails[0][2] == "nope"


async def test_cancel_in_flight():
    gov = FakeGov()
    runner = JobRunner(gov)
    jid = uuid4()
    started = asyncio.Event()
    async def slow():
        started.set()
        await asyncio.sleep(10)
    await runner.submit(jid, slow)
    await started.wait()
    cancelled = await runner.cancel(jid)
    assert cancelled is True
    for _ in range(20):
        await asyncio.sleep(0.05)
        if any(e[0] == "fail" for e in gov.events):
            break
    fails = [e for e in gov.events if e[0] == "fail"]
    assert fails and fails[0][2] == "cancelled"


async def test_stop_waits_for_in_flight_then_cancels():
    gov = FakeGov()
    runner = JobRunner(gov)
    jid = uuid4()
    started = asyncio.Event()
    async def slow():
        started.set()
        await asyncio.sleep(5)
    await runner.submit(jid, slow)
    await started.wait()
    await runner.stop(grace_period=1)
    assert any(e[0] == "fail" for e in gov.events)
```

- [ ] **Step 3: Run tests**

```bash
cd rag-anything && pytest tests/governance/test_jobs.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/raganything/governance/jobs.py rag-anything/tests/governance/test_jobs.py
git commit -m "feat(governance): JobRunner with cancel and graceful stop"
```

---

## Task 12: Provenance + progress callbacks

**Files:**
- Create: `rag-anything/raganything/governance/callbacks.py`
- Create: `rag-anything/tests/governance/test_callbacks.py`

- [ ] **Step 1: Inspect existing callback contract**

Read `rag-anything/raganything/callbacks.py` to find the `ProcessingCallback` base class and its event hooks. The relevant methods to override are typically `on_chunk_indexed` (or equivalent) and `on_progress`.

```bash
grep -n "class ProcessingCallback\|async def on_" rag-anything/raganything/callbacks.py
```

Note the exact method names — the implementation in step 2 must match.

- [ ] **Step 2: Write `callbacks.py`**

```python
"""Bridge LightRAG callbacks into governance state.

Method names mirror raganything.callbacks.ProcessingCallback. If your local copy
has differently named hooks, override the matching methods instead.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional
from uuid import UUID

from raganything.callbacks import ProcessingCallback

logger = logging.getLogger(__name__)


class IngestProvenanceCallback(ProcessingCallback):
    """Capture chunk_ids as they are indexed for a single document.

    The owning code (GovernanceService.run_ingest) reads `chunk_ids` after
    LocalRagService.ingest() returns and persists provenance rows.
    """

    def __init__(self, workspace_id: str, doc_id: UUID):
        super().__init__()
        self.workspace_id = workspace_id
        self.doc_id = doc_id
        self.chunk_ids: list[str] = []

    async def on_chunk_indexed(self, chunk_id: str, **_: object) -> None:  # type: ignore[override]
        self.chunk_ids.append(chunk_id)


class JobProgressCallback(ProcessingCallback):
    """Debounced writer of `{parsed, indexed, total}` to ingest_jobs.progress."""

    def __init__(self, gov, job_id: UUID, *, interval_s: int = 5, chunk_interval: int = 10):
        super().__init__()
        self._gov = gov
        self._job_id = job_id
        self._interval_s = interval_s
        self._chunk_interval = chunk_interval
        self._state = {"parsed": 0, "indexed": 0, "total": 0}
        self._last_flush = 0.0
        self._chunks_since_flush = 0
        self._lock = asyncio.Lock()

    async def _maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or self._chunks_since_flush >= self._chunk_interval \
                 or (now - self._last_flush) >= self._interval_s:
            async with self._lock:
                await self._gov.update_job_progress(self._job_id, dict(self._state))
            self._last_flush = now
            self._chunks_since_flush = 0

    async def on_progress(self, **kw: object) -> None:  # type: ignore[override]
        # Accepts {"parsed": int, "indexed": int, "total": int} — extras ignored
        for key in ("parsed", "indexed", "total"):
            if key in kw and isinstance(kw[key], int):
                self._state[key] = kw[key]  # type: ignore[assignment]
        await self._maybe_flush()

    async def on_chunk_indexed(self, chunk_id: str, **_: object) -> None:  # type: ignore[override]
        self._state["indexed"] = self._state.get("indexed", 0) + 1
        self._chunks_since_flush += 1
        await self._maybe_flush()

    async def flush(self) -> None:
        await self._maybe_flush(force=True)
```

> **Note for the implementer:** if `ProcessingCallback` does not declare `on_chunk_indexed` or `on_progress`, check the class and adjust to whichever hook fires per chunk and on parse-progress events. The test in step 3 stubs the gov layer and exercises only the override surface — adjust method names there too if needed.

- [ ] **Step 3: Write the failing test**

Create `rag-anything/tests/governance/test_callbacks.py`:

```python
import asyncio
import pytest
from uuid import uuid4

from raganything.governance.callbacks import IngestProvenanceCallback, JobProgressCallback

pytestmark = pytest.mark.asyncio


async def test_provenance_callback_collects_chunk_ids():
    cb = IngestProvenanceCallback("w1", uuid4())
    await cb.on_chunk_indexed("c1")
    await cb.on_chunk_indexed("c2")
    assert cb.chunk_ids == ["c1", "c2"]


class FakeGov:
    def __init__(self):
        self.writes: list[dict] = []
    async def update_job_progress(self, jid, progress):
        self.writes.append(dict(progress))


async def test_progress_callback_debounces_by_chunk_count():
    gov = FakeGov()
    cb = JobProgressCallback(gov, uuid4(), interval_s=999, chunk_interval=3)
    await cb.on_chunk_indexed("a")
    await cb.on_chunk_indexed("b")
    assert gov.writes == []  # not yet
    await cb.on_chunk_indexed("c")
    assert len(gov.writes) == 1
    assert gov.writes[0]["indexed"] == 3


async def test_progress_callback_flush_forces_write():
    gov = FakeGov()
    cb = JobProgressCallback(gov, uuid4(), interval_s=999, chunk_interval=999)
    await cb.on_progress(parsed=5, total=10)
    assert gov.writes == []
    await cb.flush()
    assert len(gov.writes) == 1
    assert gov.writes[0]["parsed"] == 5
    assert gov.writes[0]["total"] == 10
```

- [ ] **Step 4: Run tests**

```bash
cd rag-anything && pytest tests/governance/test_callbacks.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/governance/callbacks.py rag-anything/tests/governance/test_callbacks.py
git commit -m "feat(governance): provenance + progress callbacks"
```

---

## Task 13: Update `__init__.py` exports

**Files:**
- Modify: `rag-anything/raganything/governance/__init__.py`

- [ ] **Step 1: Re-export public names**

Replace the contents of `rag-anything/raganything/governance/__init__.py`:

```python
"""PostgreSQL-backed governance layer for RAG-Anything."""
from raganything.governance.callbacks import (
    IngestProvenanceCallback,
    JobProgressCallback,
)
from raganything.governance.db import (
    create_pool,
    mark_orphaned_jobs_crashed,
    run_migrations,
)
from raganything.governance.jobs import JobRunner
from raganything.governance.models import (
    AuditRow,
    DocumentRow,
    IngestResponse,
    JobRow,
    WorkspaceRow,
)
from raganything.governance.service import GovernanceService, WorkspaceFrozenError
from raganything.governance.settings import GovernanceSettings

__all__ = [
    "AuditRow",
    "DocumentRow",
    "GovernanceService",
    "GovernanceSettings",
    "IngestProvenanceCallback",
    "IngestResponse",
    "JobProgressCallback",
    "JobRow",
    "JobRunner",
    "WorkspaceFrozenError",
    "WorkspaceRow",
    "create_pool",
    "mark_orphaned_jobs_crashed",
    "run_migrations",
]
```

- [ ] **Step 2: Smoke test**

```bash
python -c "from raganything.governance import GovernanceService, JobRunner, GovernanceSettings; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/raganything/governance/__init__.py
git commit -m "feat(governance): public re-exports from package root"
```

---

# Phase 2 — Lifespan and DI refactor

This phase changes `app.py` to use FastAPI's `lifespan` context manager and `app.state` for all long-lived dependencies. **No user-facing endpoint behavior changes.** Existing routes work identically; they just resolve their dependencies through `Depends(...)` instead of module globals.

## Task 14: `LocalRagService.aclose()`

**Files:**
- Modify: `rag-anything/raganything/services/local_rag.py`

- [ ] **Step 1: Add `aclose()` method**

Find the existing `cleanup_workspace_instance` method in `local_rag.py` (currently at line ~1795 — verify with grep before editing). Add this new method **immediately after** `cleanup_workspace_instance`:

```python
    async def aclose(self) -> None:
        """Release all long-lived resources held by this service.

        Called from the FastAPI lifespan shutdown hook. Idempotent.
        """
        # 1. Finalize each cached LightRAG instance (closes Neo4j sessions, Qdrant clients).
        workspace_ids = list(self._rag_instances.keys())
        for wid in workspace_ids:
            try:
                await self.cleanup_workspace_instance(wid)
            except Exception:
                logger.exception("aclose: cleanup_workspace_instance(%s) failed", wid)

        # 2. Close httpx-backed OpenAI clients.
        for attr in ("text_client", "vision_client"):
            client = getattr(self, attr, None)
            if client is None:
                continue
            try:
                await client.close()
            except Exception:
                logger.exception("aclose: %s.close() failed", attr)

        # 3. Drop GPU-resident model cache so a subsequent process can reload cleanly.
        try:
            _MODEL_CACHE.clear()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("aclose: model cache cleanup failed")
```

- [ ] **Step 2: Verify import**

```bash
python -c "from raganything.services.local_rag import LocalRagService; assert hasattr(LocalRagService, 'aclose'); print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/raganything/services/local_rag.py
git commit -m "feat(local_rag): add aclose() for lifespan-managed cleanup"
```

---

## Task 15: Lifespan + app.state DI in `server/app.py`

**Files:**
- Modify: `rag-anything/server/app.py`

This task is the largest single edit in Phase 2. Apply it carefully.

- [ ] **Step 1: Add imports and lifespan to `app.py`**

In `rag-anything/server/app.py`, **near the top of the file with the other imports** (after the existing FastAPI/pydantic imports, around line 12), add:

```python
from contextlib import asynccontextmanager

from raganything.governance import (
    GovernanceService,
    GovernanceSettings,
    JobRunner,
    create_pool,
    mark_orphaned_jobs_crashed,
    run_migrations,
)
```

- [ ] **Step 2: Replace the `app = FastAPI(...)` line and the lazy `_service` global**

Find the block (currently lines 72–74):

```python
app = FastAPI(title="RAGAnything Local Service")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
_service: Optional[LocalRagService] = None
```

Replace with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_settings = LocalRagSettings.from_env()
    gov_settings = GovernanceSettings.from_env()

    pg_pool = await create_pool(gov_settings)
    await run_migrations(pg_pool)
    crashed = await mark_orphaned_jobs_crashed(pg_pool)
    if crashed:
        logger.warning("lifespan: marked %d orphaned jobs as crashed", crashed)

    rag_service = LocalRagService(rag_settings)
    gov_service = GovernanceService(pg_pool, rag_service)
    job_runner = JobRunner(gov_service, max_concurrent=gov_settings.job_max_concurrent)
    await job_runner.start()

    app.state.pg_pool = pg_pool
    app.state.rag = rag_service
    app.state.gov = gov_service
    app.state.jobs = job_runner
    app.state.gov_settings = gov_settings

    logger.info("lifespan: startup complete")
    try:
        yield
    finally:
        await job_runner.stop(grace_period=gov_settings.job_shutdown_grace)
        await rag_service.aclose()
        await pg_pool.close()
        logger.info("lifespan: shutdown complete")


app = FastAPI(title="RAGAnything Local Service", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
```

Delete the `_service: Optional[LocalRagService] = None` line entirely.

- [ ] **Step 3: Replace the `get_service()` function**

Find the existing function (currently around line 141):

```python
def get_service() -> LocalRagService:
    global _service
    if _service is None:
        settings = LocalRagSettings.from_env()
        _service = LocalRagService(settings)
    return _service
```

Replace with:

```python
def get_service(request: Request) -> LocalRagService:
    """Backward-compat alias for get_rag(). Prefer Depends(get_rag) in new code."""
    return request.app.state.rag


def get_rag(request: Request) -> LocalRagService:
    return request.app.state.rag


def get_gov(request: Request) -> GovernanceService:
    return request.app.state.gov


def get_jobs(request: Request) -> JobRunner:
    return request.app.state.jobs
```

- [ ] **Step 4: Run the existing test suite — nothing should break**

```bash
cd rag-anything && pytest tests/ -x --ignore=tests/governance --ignore=tests/integration -q
```

Expected: same pass/fail as before this task. (Skip if no PG; we're verifying we didn't break legacy behavior.)

- [ ] **Step 5: Manually start the server and confirm lifespan logs**

```bash
cd rag-anything && uvicorn server.app:app --host 127.0.0.1 --port 9621 --log-level info
```

Expected log lines: `governance: PG pool created (...)` then `lifespan: startup complete`. Hit `Ctrl+C`. Expected: `lifespan: shutdown complete`.

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): FastAPI lifespan + app.state DI"
```

---

# Phase 3 — Backfill existing workspaces

After Phase 2, the lifespan creates the schema and starts the service. But existing workspaces on disk have no rows in `workspaces` or `documents`. This phase scans the filesystem on startup and inserts placeholder rows so `GET /workspaces` and the frozen-flag check work for legacy workspaces.

## Task 16: Workspace backfill on startup

**Files:**
- Modify: `rag-anything/raganything/governance/service.py` (add `backfill_legacy_workspaces`)
- Modify: `rag-anything/server/app.py` (call from lifespan)

- [ ] **Step 1: Add `backfill_legacy_workspaces` to `GovernanceService`**

Append to `GovernanceService` class:

```python
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
                VALUES ($1, $2::jsonb)
                ON CONFLICT (workspace_id) DO NOTHING
            """, [(w, '{"legacy": true}') for w in new])
        return len(new)
```

- [ ] **Step 2: Wire into lifespan**

In `app.py`, inside the `lifespan` function, **after** `await job_runner.start()` but **before** assigning `app.state`, add:

```python
    # Discover existing workspaces on disk and register them.
    working_root = Path(rag_settings.working_dir_root).resolve()
    output_root = Path(rag_settings.output_dir).resolve()
    discovered: set[str] = set()
    for root in (working_root, output_root):
        if root.exists():
            for d in root.iterdir():
                if d.is_dir():
                    discovered.add(d.name)
    if discovered:
        new_count = await gov_service.backfill_legacy_workspaces(sorted(discovered))
        if new_count:
            logger.info("lifespan: backfilled %d legacy workspaces", new_count)
```

- [ ] **Step 3: Write the test**

Append to `rag-anything/tests/governance/test_service_workspaces.py`:

```python
async def test_backfill_legacy_workspaces(gov):
    inserted = await gov.backfill_legacy_workspaces(["w1", "w2"])
    assert inserted == 2
    again = await gov.backfill_legacy_workspaces(["w1", "w2", "w3"])
    assert again == 1  # only w3 was new
    rows = [await gov.get_workspace(w) for w in ("w1", "w2", "w3")]
    assert all(r is not None for r in rows)
    assert rows[0].metadata == {"legacy": True}
```

- [ ] **Step 4: Run test**

```bash
cd rag-anything && pytest tests/governance/test_service_workspaces.py::test_backfill_legacy_workspaces -v
```

Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/governance/service.py rag-anything/server/app.py rag-anything/tests/governance/test_service_workspaces.py
git commit -m "feat(governance): backfill legacy workspaces on startup"
```

---

# Phase 4 — Job-based ingest

This is the breaking API change. Existing `/ingest` callers will see a different response shape. The WebUI must be updated to poll `/jobs/{id}` — covered separately in the frontend spec.

## Task 17: GovernanceService.run_ingest orchestrator

**Files:**
- Modify: `rag-anything/raganything/governance/service.py` (add `run_ingest`)

- [ ] **Step 1: Add `run_ingest` method**

Append to `GovernanceService`:

```python
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

        Wraps LocalRagService.ingest() with status updates, provenance capture,
        and audit logging.
        """
        from raganything.governance.callbacks import (
            IngestProvenanceCallback, JobProgressCallback,
        )
        if self._rag is None:
            raise RuntimeError("GovernanceService.run_ingest requires rag_service injection")

        prov_cb = IngestProvenanceCallback(workspace_id, doc_id)
        prog_cb = JobProgressCallback(self, job_id)
        self._rag.register_callback(prov_cb)
        self._rag.register_callback(prog_cb)

        await self.mark_document_status(doc_id, "parsing")
        try:
            from pathlib import Path as _P
            workspace_output = str(_P(self._rag.settings.output_dir) / workspace_id)
            await self._rag.ingest(
                file_path,
                workspace_id=workspace_id,
                output_dir=workspace_output,
                chunking_strategy=chunking_strategy or None,
            )
            # Persist chunk provenance gathered by the callback.
            await self.insert_provenance(workspace_id, doc_id, "chunk", prov_cb.chunk_ids)
            # Backfill entity / relation provenance from LightRAG's per-chunk maps.
            await self._backfill_entity_relation_provenance(
                workspace_id, doc_id, prov_cb.chunk_ids,
            )
            await self.mark_document_status(doc_id, "done", finished=True)
            await self.record_audit(
                workspace_id, "ingest",
                doc_id=doc_id,
                details={
                    "chunk_count": len(prov_cb.chunk_ids),
                    "filename": _P(file_path).name,
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
```

> **Implementer note:** the exact return shape of `lr.entity_chunks.get(cid)` may differ. If it returns a dict or a different structure in your LightRAG version, adapt the union step accordingly — the contract is "given chunk_id, return iterable of entity names."

- [ ] **Step 2: Commit (no isolated test — exercised by integration test in Task 23)**

```bash
git add rag-anything/raganything/governance/service.py
git commit -m "feat(governance): run_ingest orchestrator with provenance backfill"
```

---

## Task 18: Convert `POST /ingest` to job-based

**Files:**
- Modify: `rag-anything/server/app.py`

- [ ] **Step 1: Replace the `/ingest` route**

Find the existing `@app.post("/ingest")` route in `app.py` (currently lines 238–304). **Replace the entire function body** with:

```python
@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(default=None),
    chunking_strategy: Optional[str] = Form(default=None),
    force: bool = Query(default=False),
    _auth: None = Depends(verify_api_key),
    rag_service: LocalRagService = Depends(get_rag),
    gov: GovernanceService = Depends(get_gov),
    jobs: JobRunner = Depends(get_jobs),
):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    if chunking_strategy and chunking_strategy not in VALID_CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunking_strategy '{chunking_strategy}'.",
        )

    file_stem = Path(file.filename).stem
    final_workspace_id = (
        workspace_id.strip() if workspace_id and workspace_id.strip()
        else _compute_workspace_id(file_stem)
    )
    _validate_workspace_id(final_workspace_id)

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")
    file_hash = hashlib.sha256(content).hexdigest()

    # Workspace + frozen check
    await gov.ensure_workspace(final_workspace_id)
    try:
        await gov.ensure_writable(final_workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Idempotency check
    original_filename = Path(file.filename).name
    if not original_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    doc_id, duplicate = await gov.upsert_document(
        final_workspace_id, original_filename, file_hash, len(content), force=force,
    )
    if duplicate:
        return {
            "job_id": None,
            "doc_id": str(doc_id),
            "workspace_id": final_workspace_id,
            "status": "duplicate",
            "duplicate": True,
        }

    # Save the original to uploads/{ws}/{filename}
    upload_dir = UPLOADS_DIR / final_workspace_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / original_filename
    try:
        upload_path.resolve().relative_to(upload_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    upload_path.write_bytes(content)

    # Enqueue the job
    job_id = await gov.create_job(final_workspace_id, [doc_id])

    async def _run():
        await gov.run_ingest(
            job_id, doc_id, final_workspace_id,
            str(upload_path),
            chunking_strategy=chunking_strategy,
        )

    await jobs.submit(job_id, _run)
    return {
        "job_id": str(job_id),
        "doc_id": str(doc_id),
        "workspace_id": final_workspace_id,
        "status": "queued",
        "duplicate": False,
    }
```

Make sure the file imports `WorkspaceFrozenError` — add to the existing import block:

```python
from raganything.governance import (
    GovernanceService,
    GovernanceSettings,
    JobRunner,
    WorkspaceFrozenError,
    create_pool,
    mark_orphaned_jobs_crashed,
    run_migrations,
)
```

- [ ] **Step 2: Manual smoke test**

Start the server. Upload any small `.md` file:

```bash
curl -F "file=@README.md" -F "workspace_id=smoke" http://127.0.0.1:9621/ingest
```

Expected: `{"job_id": "...", "doc_id": "...", "status": "queued", ...}` returns immediately.

Re-upload the same file:

```bash
curl -F "file=@README.md" -F "workspace_id=smoke" http://127.0.0.1:9621/ingest
```

Expected: `{"job_id": null, "status": "duplicate", "duplicate": true, ...}`.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): convert /ingest to job-based with idempotency"
```

---

## Task 19: Convert `/ingest/batch` and `/retry/{ws}` to job-based

**Files:**
- Modify: `rag-anything/server/app.py`

- [ ] **Step 1: Replace `/ingest/batch`**

Find `@app.post("/ingest/batch")` (currently at line 307). Replace its body to use the same enqueue pattern: validate each file, hash + upsert each into `documents`, save to `uploads/`, then create one job whose `doc_ids` is the list of new (non-duplicate) doc_ids and whose coroutine runs `gov.run_ingest` once per non-duplicate doc.

Replacement body:

```python
@app.post("/ingest/batch")
async def ingest_batch(
    files: List[UploadFile] = File(...),
    workspace_id: Optional[str] = Form(default=None),
    chunking_strategy: Optional[str] = Form(default=None),
    force: bool = Query(default=False),
    _auth: None = Depends(verify_api_key),
    rag_service: LocalRagService = Depends(get_rag),
    gov: GovernanceService = Depends(get_gov),
    jobs: JobRunner = Depends(get_jobs),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    for f in files:
        if Path(f.filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400,
                                detail=f"Unsupported file type: {f.filename}")
    if chunking_strategy and chunking_strategy not in VALID_CHUNKING_STRATEGIES:
        raise HTTPException(status_code=400, detail="Invalid chunking_strategy")

    first_stem = Path(files[0].filename).stem
    final_workspace_id = (
        workspace_id.strip() if workspace_id and workspace_id.strip()
        else _compute_workspace_id(first_stem)
    )
    _validate_workspace_id(final_workspace_id)

    await gov.ensure_workspace(final_workspace_id)
    try:
        await gov.ensure_writable(final_workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    upload_dir = UPLOADS_DIR / final_workspace_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    new_doc_specs: list[tuple] = []   # (doc_id, upload_path)
    duplicates: list[str] = []

    for f in files:
        content = await f.read()
        file_hash = hashlib.sha256(content).hexdigest()
        name = Path(f.filename).name
        if not name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        upload_path = upload_dir / name
        try:
            upload_path.resolve().relative_to(upload_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")
        doc_id, duplicate = await gov.upsert_document(
            final_workspace_id, name, file_hash, len(content), force=force,
        )
        if duplicate:
            duplicates.append(str(doc_id))
            continue
        upload_path.write_bytes(content)
        new_doc_specs.append((doc_id, upload_path))

    if not new_doc_specs:
        return {
            "job_id": None,
            "workspace_id": final_workspace_id,
            "status": "duplicate",
            "duplicate_doc_ids": duplicates,
        }

    job_id = await gov.create_job(final_workspace_id, [d for d, _ in new_doc_specs])

    async def _run():
        for d, p in new_doc_specs:
            try:
                await gov.run_ingest(job_id, d, final_workspace_id, str(p),
                                     chunking_strategy=chunking_strategy)
            except Exception:
                logger.exception("batch ingest doc %s failed", d)

    await jobs.submit(job_id, _run)
    return {
        "job_id": str(job_id),
        "workspace_id": final_workspace_id,
        "status": "queued",
        "doc_ids": [str(d) for d, _ in new_doc_specs],
        "duplicate_doc_ids": duplicates,
    }
```

- [ ] **Step 2: Replace `/retry/{workspace_id}`**

Find `@app.post("/retry/{workspace_id}")` (currently around line 386). Replace body:

```python
@app.post("/retry/{workspace_id}")
async def retry_ingest(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    rag_service: LocalRagService = Depends(get_rag),
    gov: GovernanceService = Depends(get_gov),
    jobs: JobRunner = Depends(get_jobs),
):
    _validate_workspace_id(workspace_id)
    upload_dir = UPLOADS_DIR / workspace_id
    if not upload_dir.exists():
        raise HTTPException(status_code=404,
                            detail=f"No uploads for workspace '{workspace_id}'")
    files = sorted(p for p in upload_dir.iterdir() if p.is_file())
    if not files:
        raise HTTPException(status_code=404,
                            detail=f"No files in workspace '{workspace_id}'")

    await gov.ensure_workspace(workspace_id)
    try:
        await gov.ensure_writable(workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    doc_specs: list[tuple] = []
    for fp in files:
        content = fp.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        # force=True so repeated retries always re-process; the user explicitly asked to retry
        doc_id, _ = await gov.upsert_document(
            workspace_id, fp.name, file_hash, len(content), force=True,
        )
        doc_specs.append((doc_id, fp))

    job_id = await gov.create_job(workspace_id, [d for d, _ in doc_specs])

    async def _run():
        for d, fp in doc_specs:
            try:
                await gov.run_ingest(job_id, d, workspace_id, str(fp))
            except Exception:
                logger.exception("retry ingest doc %s failed", d)

    await jobs.submit(job_id, _run)
    return {"job_id": str(job_id), "workspace_id": workspace_id, "status": "queued"}
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): convert /ingest/batch and /retry to job-based"
```

---

## Task 20: Job query/management endpoints

**Files:**
- Modify: `rag-anything/server/app.py`

- [ ] **Step 1: Append job endpoints**

Add new endpoints **after the `/retry/{workspace_id}` block**:

```python
# =========================================================================
# Jobs
# =========================================================================

@app.get("/jobs/{job_id}")
async def get_job_endpoint(
    job_id: str,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    from uuid import UUID
    try:
        jid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    j = await gov.get_job(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return j.model_dump(mode="json")


@app.get("/jobs")
async def list_jobs_endpoint(
    workspace_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    rows = await gov.list_jobs(
        workspace_id=workspace_id, status=status, limit=max(1, min(limit, 200)),
    )
    return {"jobs": [r.model_dump(mode="json") for r in rows]}


@app.delete("/jobs/{job_id}")
async def cancel_job_endpoint(
    job_id: str,
    _auth: None = Depends(verify_api_key),
    jobs: JobRunner = Depends(get_jobs),
):
    from uuid import UUID
    try:
        jid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    cancelled = await jobs.cancel(jid)
    return {"job_id": job_id, "cancelled": cancelled}
```

- [ ] **Step 2: Manual smoke test**

```bash
# Returns the job from the previous /ingest call
curl http://127.0.0.1:9621/jobs/<job_id_from_earlier>
curl http://127.0.0.1:9621/jobs?workspace_id=smoke
```

Expected: JSON job rows.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): add /jobs endpoints for status polling and cancel"
```

---

# Phase 5 — Governance endpoints (freeze, per-doc delete, audit)

## Task 21: Freeze / unfreeze workspace endpoints

**Files:**
- Modify: `rag-anything/server/app.py`

- [ ] **Step 1: Append endpoints**

Add after the existing `/workspace/{workspace_id}/stats` endpoint:

```python
@app.patch("/workspace/{workspace_id}/freeze")
async def freeze_workspace(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    await gov.ensure_workspace(workspace_id)
    ws_before = await gov.get_workspace(workspace_id)
    prev = bool(ws_before.frozen) if ws_before else False
    ok = await gov.set_frozen(workspace_id, True)
    if not ok:
        raise HTTPException(status_code=404, detail="Workspace not found")
    await gov.record_audit(workspace_id, "freeze", details={"previous_state": prev})
    return {"workspace_id": workspace_id, "frozen": True}


@app.patch("/workspace/{workspace_id}/unfreeze")
async def unfreeze_workspace(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    ws_before = await gov.get_workspace(workspace_id)
    if ws_before is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ok = await gov.set_frozen(workspace_id, False)
    await gov.record_audit(workspace_id, "unfreeze",
                           details={"previous_state": ws_before.frozen})
    return {"workspace_id": workspace_id, "frozen": False}
```

- [ ] **Step 2: Add frozen check to existing `DELETE /workspace/{ws}`**

Find the existing `@app.delete("/workspace/{workspace_id}")` route. **Immediately after `_validate_workspace_id(workspace_id)`** add:

```python
    gov: GovernanceService = request.app.state.gov  # type: ignore
    try:
        await gov.ensure_writable(workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
```

(Add `request: Request` to the function signature if not already there.) After the existing delete logic finishes, append:

```python
    await gov.record_audit(workspace_id, "delete_workspace",
                           details={"deleted": deleted, "drop_errors": drop_errors})
```

- [ ] **Step 3: Manual smoke test**

```bash
curl -X PATCH http://127.0.0.1:9621/workspace/smoke/freeze
# now ingest fails:
curl -F "file=@small.md" -F "workspace_id=smoke" http://127.0.0.1:9621/ingest
# Expected: 409 "workspace 'smoke' is frozen"
curl -X PATCH http://127.0.0.1:9621/workspace/smoke/unfreeze
```

- [ ] **Step 4: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): freeze/unfreeze workspace endpoints + audit"
```

---

## Task 22: Per-document delete endpoint

**Files:**
- Modify: `rag-anything/raganything/governance/service.py` (add `delete_document`)
- Modify: `rag-anything/server/app.py` (add route)

- [ ] **Step 1: Add `delete_document` to `GovernanceService`**

Append to `GovernanceService`:

```python
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
                    # rel_key encoding is LightRAG-specific (often "src::tgt")
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
```

> **Implementer note:** the `delete_node` / `delete_edge` / `delete` method names on LightRAG's storage objects vary by version. If they differ (e.g., `remove_node`), update accordingly. The five-step structure is what matters — the exact LightRAG API may need adjustment.

- [ ] **Step 2: Add the route**

Append to `app.py`:

```python
@app.delete("/workspace/{workspace_id}/document/{doc_id}")
async def delete_document_endpoint(
    workspace_id: str,
    doc_id: str,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    from uuid import UUID
    try:
        did = UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid doc_id")
    try:
        await gov.ensure_writable(workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    doc = await gov.get_document(did)
    if doc is None or doc.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Document not found in workspace")
    report = await gov.delete_document(workspace_id, did)
    return {"doc_id": doc_id, "workspace_id": workspace_id, "cleanup": report}
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/raganything/governance/service.py rag-anything/server/app.py
git commit -m "feat(governance): per-document deletion with shared-entity protection"
```

---

## Task 23: Audit endpoints

**Files:**
- Modify: `rag-anything/server/app.py`

- [ ] **Step 1: Append audit endpoints**

```python
@app.get("/workspace/{workspace_id}/audit")
async def workspace_audit(
    workspace_id: str,
    action: Optional[str] = None,
    limit: int = 100,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    rows = await gov.list_audit(workspace_id=workspace_id, action=action,
                                limit=max(1, min(limit, 500)))
    return {"audit": [r.model_dump(mode="json") for r in rows]}


@app.get("/admin/audit")
async def admin_audit(
    action: Optional[str] = None,
    limit: int = 100,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    rows = await gov.list_audit(action=action, limit=max(1, min(limit, 500)))
    return {"audit": [r.model_dump(mode="json") for r in rows]}
```

- [ ] **Step 2: Manual smoke test**

```bash
curl http://127.0.0.1:9621/workspace/smoke/audit
curl http://127.0.0.1:9621/admin/audit?limit=20
```

Expected: JSON arrays of audit events.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): audit log endpoints"
```

---

# Phase 6 — Integration tests, manual scripts, docs

## Task 24: Integration test — full ingest flow

**Files:**
- Create: `rag-anything/tests/integration/governance/__init__.py`
- Create: `rag-anything/tests/integration/governance/conftest.py`
- Create: `rag-anything/tests/integration/governance/test_ingest_flow.py`

- [ ] **Step 1: Conftest — shared fixtures**

Create `rag-anything/tests/integration/governance/__init__.py` (empty).

Create `rag-anything/tests/integration/governance/conftest.py`:

```python
"""Integration test fixtures: full GovernanceService over real PG, RAG mocked."""
import os
import uuid
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

    async def _init(conn): await conn.execute(f'SET search_path TO "{schema}"')
    pool = await asyncpg.create_pool(PG_TEST_DSN, min_size=1, max_size=4, init=_init)
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
```

- [ ] **Step 2: Test full ingest happy path**

Create `rag-anything/tests/integration/governance/test_ingest_flow.py`:

```python
import asyncio
import pytest
from .conftest import pytestmark_pg

pytestmark = [pytestmark_pg, pytest.mark.asyncio]


async def test_full_ingest_path_to_done(gov, runner, fake_rag, tmp_path):
    await gov.ensure_workspace("w1")
    file = tmp_path / "demo.md"
    file.write_text("hello")
    doc_id, dup = await gov.upsert_document("w1", "demo.md", "hash-1", 5)
    assert dup is False
    job_id = await gov.create_job("w1", [doc_id])

    async def _do():
        await gov.run_ingest(job_id, doc_id, "w1", str(file))

    await runner.submit(job_id, _do)

    for _ in range(40):
        await asyncio.sleep(0.05)
        j = await gov.get_job(job_id)
        if j.status in ("done", "failed", "crashed"):
            break

    j = await gov.get_job(job_id)
    assert j.status == "done"
    doc = await gov.get_document(doc_id)
    assert doc.status == "done"
    audit = await gov.list_audit("w1", action="ingest")
    assert len(audit) == 1
    assert audit[0].doc_id == doc_id


async def test_ingest_failure_marks_doc_and_audits(gov, runner, fake_rag, tmp_path):
    fake_rag.ingest.side_effect = RuntimeError("parse exploded")
    await gov.ensure_workspace("w1")
    f = tmp_path / "boom.md"
    f.write_text("x")
    did, _ = await gov.upsert_document("w1", "boom.md", "h2", 1)
    jid = await gov.create_job("w1", [did])
    await runner.submit(jid, lambda: gov.run_ingest(jid, did, "w1", str(f)))

    for _ in range(40):
        await asyncio.sleep(0.05)
        j = await gov.get_job(jid)
        if j.status in ("done", "failed", "crashed"):
            break
    assert (await gov.get_job(jid)).status == "failed"
    assert (await gov.get_document(did)).status == "failed"
    audit = await gov.list_audit("w1", action="ingest_failed")
    assert len(audit) == 1
```

- [ ] **Step 3: Run tests**

```bash
cd rag-anything && pytest tests/integration/governance/test_ingest_flow.py -v
```

Expected: 2 passed (or skipped if no PG).

- [ ] **Step 4: Commit**

```bash
git add rag-anything/tests/integration/governance/
git commit -m "test(governance): integration test for ingest job flow"
```

---

## Task 25: Manual REST scripts and architecture doc

**Files:**
- Create: `rag-anything/tests/manual/governance.http`
- Modify: `rag-anything/server/SERVER_ARCH.md`

- [ ] **Step 1: Manual REST scripts**

Create `rag-anything/tests/manual/governance.http`:

```
### Variables (edit before running)
@baseUrl = http://127.0.0.1:9621
@apiKey = your-api-key-or-leave-empty
@workspace = smoke
@docId = paste-doc-id-from-ingest-response
@jobId = paste-job-id-from-ingest-response

### Ingest a file (job-based)
POST {{baseUrl}}/ingest
X-Api-Key: {{apiKey}}
Content-Type: multipart/form-data; boundary=----X

------X
Content-Disposition: form-data; name="file"; filename="demo.md"
Content-Type: text/markdown

# Hello
------X
Content-Disposition: form-data; name="workspace_id"

{{workspace}}
------X--

### Force re-ingest of a duplicate
POST {{baseUrl}}/ingest?force=true

### Poll a job
GET {{baseUrl}}/jobs/{{jobId}}
X-Api-Key: {{apiKey}}

### List jobs for a workspace
GET {{baseUrl}}/jobs?workspace_id={{workspace}}&limit=10
X-Api-Key: {{apiKey}}

### Cancel a running job
DELETE {{baseUrl}}/jobs/{{jobId}}
X-Api-Key: {{apiKey}}

### Freeze
PATCH {{baseUrl}}/workspace/{{workspace}}/freeze
X-Api-Key: {{apiKey}}

### Unfreeze
PATCH {{baseUrl}}/workspace/{{workspace}}/unfreeze
X-Api-Key: {{apiKey}}

### Delete a document
DELETE {{baseUrl}}/workspace/{{workspace}}/document/{{docId}}
X-Api-Key: {{apiKey}}

### Audit log for a workspace
GET {{baseUrl}}/workspace/{{workspace}}/audit?limit=50
X-Api-Key: {{apiKey}}

### Cross-workspace audit
GET {{baseUrl}}/admin/audit?limit=50
X-Api-Key: {{apiKey}}
```

- [ ] **Step 2: Update `SERVER_ARCH.md`**

Open `rag-anything/server/SERVER_ARCH.md`. Append a new section before the final "目录布局" block:

```markdown
---

## Governance layer (Phase 1)

The service uses PostgreSQL as a metadata layer for document governance. It does NOT replace Neo4j, Qdrant, or LightRAG's KV — it sits alongside them.

### New tables (in `raganything.governance.migrations/001_init.sql`)

| Table | Purpose |
|---|---|
| `workspaces` | Per-workspace flags (frozen, owner, metadata) |
| `documents` | One row per ingested file. `UNIQUE (workspace_id, file_hash)` enforces idempotency |
| `provenance` | Maps `(workspace_id, doc_id) → (chunk_id | entity_name | relation_key)` for surgical deletion |
| `ingest_jobs` | Background job state with progress JSONB |
| `ingest_audit` | Append-only governance audit log |

### New endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/jobs?workspace_id=&status=` | List jobs |
| `DELETE` | `/jobs/{job_id}` | Cancel a queued/running job |
| `PATCH` | `/workspace/{ws}/freeze` | Mark workspace read-only |
| `PATCH` | `/workspace/{ws}/unfreeze` | Re-enable writes |
| `DELETE` | `/workspace/{ws}/document/{doc_id}` | Per-document delete with shared-entity protection |
| `GET` | `/workspace/{ws}/audit` | Per-workspace audit log |
| `GET` | `/admin/audit` | Cross-workspace audit log |

### Behavior changes

- `POST /ingest` returns `{job_id, doc_id, status: "queued"}` immediately and runs ingest in the background. Re-uploads of the same file return `{status: "duplicate"}`.
- `POST /ingest/batch` and `POST /retry/{ws}` follow the same job-based contract.
- `DELETE /workspace/{ws}` rejects with 409 if the workspace is frozen.

### Lifespan management

`server/app.py` uses an `@asynccontextmanager lifespan` that creates the PG pool, runs migrations, marks orphaned jobs as crashed, and exposes everything via `app.state`. On shutdown, in-flight jobs get a 30s grace period before cancellation; LightRAG storages and httpx clients are closed cleanly.

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `RAGANYTHING_PG_DSN` | `postgresql://localhost:5432/raganything` | PG connection string |
| `RAGANYTHING_PG_POOL_MIN` / `_MAX` | `2` / `10` | asyncpg pool sizing |
| `RAGANYTHING_JOB_MAX_CONCURRENT` | `1` | parallel jobs across all workspaces |
| `RAGANYTHING_JOB_PROGRESS_INTERVAL` | `5` | seconds between progress flushes |
| `RAGANYTHING_JOB_SHUTDOWN_GRACE` | `30` | seconds to wait for in-flight jobs at shutdown |
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/tests/manual/ rag-anything/server/SERVER_ARCH.md
git commit -m "docs(governance): manual REST scripts and SERVER_ARCH update"
```

---

# Final verification

## Task 26: Run the full governance test suite

- [ ] **Step 1: Run all governance tests**

```bash
cd rag-anything && pytest tests/governance/ tests/integration/governance/ -v
```

Expected: all tests pass (skipped if PG not running — set `RAGANYTHING_PG_TEST_DSN`).

- [ ] **Step 2: Run the existing test suite to confirm no regressions**

```bash
cd rag-anything && pytest tests/ --ignore=tests/governance --ignore=tests/integration -q
```

Expected: same pass/fail as before this work began.

- [ ] **Step 3: Final manual end-to-end smoke**

Walk through the manual REST scripts in `tests/manual/governance.http` against a running server:

1. Ingest a small file → returns `{status: "queued"}`.
2. Re-upload same file → returns `{status: "duplicate"}`.
3. Poll job → eventually `done`.
4. Freeze → ingest fails with 409.
5. Unfreeze → ingest works again.
6. Delete the document → response shows cleanup `{qdrant: ok, neo4j: ok, kv: ok, fs: ok}`.
7. Audit log shows all five events.

Expected: all behaviors match the spec's acceptance criteria (§13).

---

# Acceptance criteria recap (from spec §13)

A v1 implementation is complete when:

- `pytest tests/governance/ tests/integration/governance/` is green with PG running.
- Lifespan startup logs report success; shutdown is clean.
- The seven manual scenarios in §13 of the spec all pass — verified above in Task 26 step 3.
