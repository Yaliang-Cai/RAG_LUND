# Backend Hardening — Design Spec

**Date:** 2026-05-07
**Scope:** `rag-anything/server/app.py` + new `raganything/governance/` module
**Status:** Approved for implementation planning

## 1. Problem & goals

The current FastAPI backend (`rag-anything/server/app.py`) is functional for single-user PoC use but lacks the data-governance and lifecycle controls needed to scale to a small team or production. Specifically:

- No FastAPI `lifespan` hooks: Neo4j/Qdrant connections are created lazily on first request and never explicitly closed; reload leaks GPU memory held by SentenceTransformer/CrossEncoder.
- `_service` and `_rag_instances` are module-level singletons — incompatible with multi-worker uvicorn/Gunicorn deployment without rework.
- `/ingest` blocks the HTTP response for the entire parse-plus-index cycle (minutes for large PDFs); only `/retry` uses `BackgroundTasks`, and that path has no status surface.
- No per-document deletion — `DELETE /workspace/{ws}` is the only removal mechanism, which destroys the entire knowledge graph.
- No idempotency — re-uploading the same file creates duplicate chunks and entities.
- No provenance tracking — cannot answer "which entities/chunks came from doc X?"
- No audit log, no read-only flag.

**Goal:** Add a governance layer that delivers the four highest-value gaps — per-document deletion with provenance (A), file-hash idempotency (B), append-only audit log (E), and a per-workspace frozen flag (F) — backed by a new PostgreSQL service. Refactor the route layer onto FastAPI `lifespan` and `app.state` so multi-worker is a future swap, not a rewrite. Convert ingest to a job-based API with status polling.

**Out of goals (Phase 2+):** Redis-backed distributed jobs and locks, multi-worker deployment, snapshot/rollback, staging-to-prod promotion, RAGAS evaluation, Phoenix tracing, frontend modernization.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  FastAPI process (single uvicorn worker)                    │
│                                                             │
│  ┌────────────┐   ┌─────────────┐   ┌──────────────────┐    │
│  │  Routes    │──▶│ GovService  │──▶│ LocalRagService  │    │
│  │ (app.py)   │   │ (new)       │   │ (existing)       │    │
│  └────────────┘   └──────┬──────┘   └────────┬─────────┘    │
│         │                │                   │              │
│         │           ┌────▼─────┐             │              │
│         │           │ JobRunner│             │              │
│         │           │ (asyncio │             │              │
│         │           │   tasks) │             │              │
│         │           └────┬─────┘             │              │
│         ▼                ▼                   ▼              │
│   ┌──────────────────────────────────────────────────┐      │
│   │  lifespan-managed clients (singletons)           │      │
│   │  • asyncpg pool   • Neo4j driver                 │      │
│   │  • Qdrant client  • SentenceTransformer/CrossEnc │      │
│   └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
        │                │                │           │
   ┌────▼────┐     ┌─────▼────┐     ┌─────▼────┐  ┌───▼────┐
   │PostgreSQL│    │  Neo4j   │     │  Qdrant  │  │  FS    │
   │governance│    │  graph   │     │ vectors  │  │uploads │
   │  layer   │    │          │     │          │  │ output │
   └─────────┘     └──────────┘     └──────────┘  └────────┘
```

### Storage role assignment (no overlap)

| Store                 | Owns                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| **PostgreSQL** *(new)* | Workspace metadata, document registry, provenance map, job state, audit log                       |
| **Neo4j** *(unchanged)* | Knowledge graph: entities, relations, multi-hop traversal                                        |
| **Qdrant** *(unchanged)* | Dense + sparse vectors for chunks, entities, relations                                          |
| **Filesystem** *(unchanged)* | `uploads/`, `output/` (parsed MD + images), LightRAG KV JSON files (`text_chunks`, `doc_status`, etc.) |

### New module layout

```
raganything/governance/
  __init__.py
  db.py                  # asyncpg pool + run_migrations()
  service.py             # GovernanceService — only place that touches PG
  jobs.py                # JobRunner (in-process asyncio tasks)
  models.py              # pydantic schemas for API surface
  callbacks.py           # IngestProvenanceCallback, JobProgressCallback
  migrations/
    001_init.sql         # all five tables
    002_*.sql            # future
```

`LocalRagService` is unchanged except for one new method (`aclose()`) and one new optional callback hook (chunk-id capture during ingest). All PG access flows through `GovernanceService`. Routes are thin — they call `gov.x()` for governance ops, `rag.x()` for legacy paths.

## 3. PostgreSQL schema

```sql
-- 1. Workspaces: governance flags + ownership
CREATE TABLE workspaces (
    workspace_id   TEXT PRIMARY KEY,
    frozen         BOOLEAN NOT NULL DEFAULT FALSE,
    owner          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- 2. Documents: one row per ingested file
CREATE TABLE documents (
    doc_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id   TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    filename       TEXT NOT NULL,
    file_hash      TEXT NOT NULL,                  -- sha256 of file bytes
    size_bytes     BIGINT NOT NULL,
    status         TEXT NOT NULL,                  -- pending|parsing|indexing|done|failed|deleting|deleted
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ,
    error          TEXT,
    UNIQUE (workspace_id, file_hash)
);
CREATE INDEX idx_documents_ws_status ON documents(workspace_id, status);

-- 3. Provenance: which LightRAG objects came from which document
CREATE TABLE provenance (
    workspace_id   TEXT NOT NULL,
    doc_id         UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,                  -- chunk|entity|relation
    ref_id         TEXT NOT NULL,                  -- LightRAG-side id
    PRIMARY KEY (workspace_id, kind, ref_id, doc_id)
);
CREATE INDEX idx_prov_doc ON provenance(doc_id);
CREATE INDEX idx_prov_ws_kind ON provenance(workspace_id, kind);

-- 4. Ingest jobs: one row per /ingest call
CREATE TABLE ingest_jobs (
    job_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id   TEXT NOT NULL,
    doc_ids        UUID[] NOT NULL DEFAULT '{}',
    status         TEXT NOT NULL,                  -- queued|running|done|failed|crashed
    progress       JSONB NOT NULL DEFAULT '{}',    -- {parsed, indexed, total}
    error          TEXT,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ
);
CREATE INDEX idx_jobs_ws_status ON ingest_jobs(workspace_id, status);

-- 5. Audit log: append-only
CREATE TABLE ingest_audit (
    id             BIGSERIAL PRIMARY KEY,
    workspace_id   TEXT NOT NULL,
    doc_id         UUID,
    action         TEXT NOT NULL,                  -- ingest|delete_doc|delete_workspace|freeze|unfreeze|ingest_forced|ingest_failed
    actor          TEXT,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details        JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_audit_ws_time ON ingest_audit(workspace_id, timestamp DESC);
```

### Schema design notes

- **`provenance` is wide-but-thin.** Three rows per chunk/entity/relation × N docs is well within PG's comfort zone given the indexes.
- **Idempotency** comes from `UNIQUE (workspace_id, file_hash)` plus `INSERT … ON CONFLICT DO NOTHING RETURNING doc_id`.
- **Batch ingest** uses `ingest_jobs.doc_ids` as an array — avoids a join table.
- **Migrations:** flat numbered SQL files, applied in order; a `schema_version (version int PRIMARY KEY, applied_at timestamptz)` table tracks state. Idempotent. No Alembic.
- **Crash recovery** runs in lifespan startup: `UPDATE ingest_jobs SET status='crashed' WHERE status='running'` and `UPDATE documents SET status='failed' WHERE status IN ('parsing','indexing')`.

## 4. Lifespan & dependency injection

Replace lazy module-level init with an explicit `lifespan` context manager. All long-lived clients become `app.state` attributes, accessed via FastAPI `Depends`.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = LocalRagSettings.from_env()
    gov_settings = GovernanceSettings.from_env()

    pg_pool = await asyncpg.create_pool(
        gov_settings.pg_dsn, min_size=2, max_size=10, command_timeout=30,
    )
    await run_migrations(pg_pool)
    await mark_orphaned_jobs_crashed(pg_pool)

    rag_service = LocalRagService(settings)
    gov_service = GovernanceService(pg_pool, rag_service)
    job_runner  = JobRunner(gov_service)
    await job_runner.start()

    app.state.pg_pool = pg_pool
    app.state.rag     = rag_service
    app.state.gov     = gov_service
    app.state.jobs    = job_runner

    yield

    await job_runner.stop(grace_period=30)
    await rag_service.aclose()
    await pg_pool.close()

app = FastAPI(title="RAGAnything Local Service", lifespan=lifespan)
```

### `LocalRagService.aclose()` (new)

- Iterate `self._rag_instances`, call `lightrag.finalize_storages()` on each (closes Neo4j sessions, Qdrant clients held by LightRAG).
- Clear `_MODEL_CACHE` (frees GPU memory for SentenceTransformer/CrossEncoder).
- `await client.aclose()` on all httpx clients owned by `LocalRagService`.

### Dependency injection switch

```python
def get_gov(request: Request) -> GovernanceService:
    return request.app.state.gov

def get_rag(request: Request) -> LocalRagService:
    return request.app.state.rag

def get_jobs(request: Request) -> JobRunner:
    return request.app.state.jobs
```

The existing `get_service()` function becomes an alias for `get_rag()` during the migration window, then is removed.

## 5. Job lifecycle

In-process `asyncio.Task` per job, state in PostgreSQL, no Redis. The `JobRunner` interface (`submit`, `cancel`, `stop`) is the seam to swap in ARQ + Redis when going multi-worker.

### Endpoint changes

| Before                                                  | After                                                                                  |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `POST /ingest` blocks until done                        | `POST /ingest` enqueues, returns `{job_id, doc_id, status: "queued", duplicate: bool}` |
| `POST /ingest/batch` blocks until done                  | `POST /ingest/batch` enqueues N docs under one job_id                                  |
| `POST /retry/{ws}` BackgroundTasks fire-and-forget      | `POST /retry/{ws}` enqueues, returns job_id                                            |
| *(none)*                                                | `GET /jobs/{job_id}` returns full job row from PG                                      |
| *(none)*                                                | `GET /jobs?workspace_id=...&status=...` list with filters                              |
| *(none)*                                                | `DELETE /jobs/{job_id}` cancels a queued/running job                                   |

### `JobRunner` (in-process)

```python
class JobRunner:
    def __init__(self, gov: GovernanceService, max_concurrent: int = 1):
        self._gov = gov
        self._sem = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[UUID, asyncio.Task] = {}

    async def submit(self, job_id: UUID, coro_factory) -> None:
        task = asyncio.create_task(self._run(job_id, coro_factory))
        self._tasks[job_id] = task

    async def _run(self, job_id, coro_factory):
        async with self._sem:
            try:
                await self._gov.mark_job_running(job_id)
                await coro_factory()
                await self._gov.mark_job_done(job_id)
            except asyncio.CancelledError:
                await self._gov.mark_job_failed(job_id, "cancelled")
                raise
            except Exception as exc:
                await self._gov.mark_job_failed(job_id, str(exc))
            finally:
                self._tasks.pop(job_id, None)

    async def cancel(self, job_id: UUID) -> bool: ...
    async def stop(self, grace_period: int): ...
```

`max_concurrent=1` matches today's workspace-serialized behavior. Configurable via `GovernanceSettings` later if needed.

### `/ingest` flow end-to-end

1. Client `POST /ingest` (file, optional `workspace_id`, optional `force=true`).
2. Route handler:
   1. Validate file extension and `workspace_id` (existing `_validate_workspace_id` rules).
   2. Compute `sha256(file_bytes)`.
   3. `gov.ensure_workspace(workspace_id, frozen_check=True)` → 409 if frozen.
   4. `gov.upsert_document(...)` — `ON CONFLICT DO NOTHING RETURNING doc_id`. Empty result means duplicate; return `{doc_id: existing, status: "duplicate", duplicate: true}` with no new job (unless `force=true`).
   5. Save file to `uploads/{ws}/{filename}` (existing logic, unchanged).
   6. `gov.create_job(workspace_id, doc_ids=[doc_id])`.
   7. `jobs.submit(job_id, lambda: gov.run_ingest(job_id, doc_id, file_path))`.
   8. Return `{job_id, doc_id, status: "queued", duplicate: false}`.
3. Background `gov.run_ingest`:
   1. `UPDATE documents SET status='parsing'`.
   2. Call `rag_service.ingest(...)` with an `IngestProvenanceCallback` injected.
   3. Backfill entity/relation provenance (see §6).
   4. `UPDATE documents SET status='done', finished_at=NOW()`.
   5. `INSERT INTO ingest_audit (action='ingest', ...)`.
4. Client polls `GET /jobs/{job_id}` → `{status, progress: {parsed, indexed, total}, error}`.

### Cancellation semantics

- **Queued (not yet running):** `task.cancel()` → job `failed (cancelled)`. Document stays at `pending`. User may retry.
- **Running:** `CancelledError` propagates; job and doc go to `failed`. Provenance rows already inserted are kept. Partial Neo4j/Qdrant state is left in place — re-running is idempotent at the chunk-id level (LightRAG dedups on chunk hash). Full cleanup of partial ingests is a Phase 1.5 concern.
- **Process crash:** lifespan startup recovery marks `running → crashed`, `parsing/indexing → failed`. Same partial-state caveat.

### Progress reporting

A new `JobProgressCallback` plugs into the existing `MetricsCallback` / `ProcessingCallback` plumbing. It writes `{parsed, indexed, total}` to the `ingest_jobs.progress` JSONB column, debounced to every 5 s or every 10 chunks (whichever first) to keep PG write rate sane.

## 6. Provenance & per-document deletion

### Provenance recording

Three mappings per ingested doc:

1. `doc_id → chunk_ids[]` — captured directly via `IngestProvenanceCallback.on_chunks_indexed`.
2. `doc_id → entity_names[]` — derived after ingest from LightRAG's `entity_chunks` KV (`chunk_id → entity_names`).
3. `doc_id → relation_keys[]` — derived after ingest from LightRAG's `relation_chunks` KV.

LightRAG itself is not modified. Provenance backfill runs in `gov.run_ingest` step 3:

```python
async def _backfill_entity_relation_provenance(
    self, workspace_id: str, doc_id: UUID, chunk_ids: list[str]
) -> None:
    lr = (await self._rag.get_rag(workspace_id)).lightrag
    rows = []
    for cid in chunk_ids:
        for ent in (await lr.entity_chunks.get(cid) or []):
            rows.append((workspace_id, doc_id, "entity", ent))
        for rel in (await lr.relation_chunks.get(cid) or []):
            rows.append((workspace_id, doc_id, "relation", rel))
    # bulk INSERT … ON CONFLICT DO NOTHING
```

### `DELETE /workspace/{ws}/document/{doc_id}` — five-step orchestration

1. PG: `SELECT` all provenance rows for `doc_id` → `chunk_ids`, `entity_names`, `relation_keys`.
2. PG: `UPDATE documents SET status='deleting' WHERE doc_id=?` (locks against concurrent ops).
3. Storage cleanup (best-effort, errors logged into audit `details` but not fatal):
   - **Qdrant:** delete points by `chunk_ids` from `chunks_vdb`, by `entity_names` from `entities_vdb`, by `relation_keys` from `relationships_vdb`.
   - **Neo4j:** delete only entities/relations whose sole provenance is this doc (shared-entity protection — see SQL below).
   - **LightRAG KV:** `lr.text_chunks.delete(chunk_ids)`, `lr.full_docs.delete([doc_id])`, prune `entity_chunks` / `relation_chunks` references.
   - **Filesystem:** remove `uploads/{ws}/{filename}` only if no other doc shares the same `filename`.
4. PG: `DELETE FROM provenance WHERE doc_id=?`; `UPDATE documents SET status='deleted'` (kept as tombstone for audit continuity).
5. PG: `INSERT INTO ingest_audit (action='delete_doc', details=cleanup_summary)`.

### Shared-entity protection

```sql
SELECT ref_id
FROM provenance
WHERE workspace_id = $1 AND kind = 'entity' AND ref_id = ANY($2::text[])
GROUP BY ref_id
HAVING COUNT(DISTINCT doc_id) = 1
   AND MAX(doc_id) = $3
```

Only entities whose *only* provenance source is the doc being deleted are removed from Neo4j. Same logic for relations. This is the dividing line between "delete a doc" and "destroy your knowledge graph."

### Failure handling

The five steps are not a single atomic transaction (Neo4j and Qdrant are not in PG's transactional scope). Mitigation:

- Forward progress with logging — if step 3a fails, log + continue to 3b. Provenance row stays in PG so a future repair pass can finish the work.
- Partial-delete observability via the audit row's `details` field: `{qdrant: ok, neo4j: failed, kv: ok, fs: ok}`.
- A repair endpoint (`/admin/repair-doc/{doc_id}`) is explicitly out of scope here; documented as a Phase 1.5 follow-up.

## 7. Idempotency, frozen flag, audit log

### Idempotency (file-hash dedup)

- `documents.file_hash = sha256(file_bytes)` at upload time.
- `UNIQUE (workspace_id, file_hash)` enforces dedup.
- `INSERT … ON CONFLICT DO NOTHING RETURNING doc_id` — empty result means duplicate.
- Response: `{job_id: null, doc_id: <existing>, status: "duplicate", duplicate: true}`.
- **Override:** `?force=true` skips the check (audit logged as `ingest_forced`). Used for "re-ingest after parser upgrade."
- **Edge cases:** same file under different filenames → still detected; same file in different workspaces → not deduped (workspace isolation by design); SHA-256 collisions → not handled (unrealistic).

### Frozen flag

```
PATCH /workspace/{ws}/freeze    → workspaces.frozen = TRUE
PATCH /workspace/{ws}/unfreeze  → workspaces.frozen = FALSE
GET   /workspace/{ws}           → returns frozen along with stats
```

`GovernanceService.ensure_writable(workspace_id)` is called at the top of every write endpoint:

| Endpoint                                              | If frozen                                              |
| ----------------------------------------------------- | ------------------------------------------------------ |
| `POST /ingest`, `/ingest/batch`, `/retry/{ws}`        | 409 Conflict, `"workspace is frozen"`                  |
| `DELETE /workspace/{ws}`                              | 409 Conflict (must unfreeze first — guards against accidental wipes) |
| `DELETE /workspace/{ws}/document/{d}`                 | 409 Conflict                                           |
| `PATCH /workspace/{ws}/unfreeze`                      | always allowed                                         |
| All `/query*`, `/graph/*`, `/files/*`, `/content/*`   | always allowed (read paths bypass the check)           |

One indexed PG lookup per write request — negligible overhead.

### Audit log

| Action            | Triggered by                      | Recorded `details`                                                        |
| ----------------- | --------------------------------- | ------------------------------------------------------------------------- |
| `ingest`          | successful ingest completion      | `{filename, hash, chunk_count, entity_count, relation_count, duration_ms}` |
| `ingest_forced`   | `?force=true` ingest              | same as above + `{forced: true}`                                          |
| `ingest_failed`   | terminal job failure              | `{error, stage}`                                                          |
| `delete_doc`      | per-document delete               | `{cleanup: {qdrant, neo4j, kv, fs}}`                                      |
| `delete_workspace` | full workspace delete            | `{deleted_dirs, drop_errors}`                                             |
| `freeze`/`unfreeze` | flag change                     | `{previous_state}`                                                        |

`actor` derivation: `"key:{first_8_chars}"` from API key when set, `"system"` for crash-recovery, `"unknown"` when no key.

Read endpoints:

```
GET /workspace/{ws}/audit?limit=100&action=...&since=...
GET /admin/audit?limit=100   (cross-workspace, requires API key)
```

No retention policy in v1. A pruning cron (`DELETE FROM ingest_audit WHERE timestamp < NOW() - INTERVAL '90 days'`) is a Phase 1.5 concern.

## 8. Error handling

| Failure                              | Behavior                                                   | User-visible                                              |
| ------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------- |
| PG unreachable at startup            | lifespan raises, uvicorn exits non-zero                    | hard fail (correct — no governance = no service)          |
| PG dies mid-request                  | asyncpg pool retries once; then 503                        | request not lost; client retries                          |
| Neo4j/Qdrant down during query       | existing LightRAG error path; 500 with cause               | unchanged from today                                      |
| Neo4j/Qdrant down during ingest job  | job `failed`, document `failed`, partial provenance kept    | client sees `failed` via `/jobs/{id}`                     |
| Process crash mid-job                | startup: `running → crashed`, `parsing/indexing → failed`   | client poll returns `crashed`; can retry                  |
| Per-doc delete partial failure       | each storage cleanup wrapped in `try/except`; logged        | delete completes best-effort; audit `details` shows state |
| Cancellation during ingest           | `CancelledError` propagates; `failed (cancelled)`           | client may re-issue ingest                                |
| Schema migration fails               | startup raises with failed migration filename              | hard fail; operator must intervene                        |

**Logging conventions:** governance ops log with `extra={"workspace_id": ..., "doc_id": ..., "job_id": ...}`. New `governance` logger is a child of root.

**No retries inside request handlers.** asyncpg pool covers transient PG glitches; LightRAG's `CircuitBreaker` + `async_retry` already cover LLM/embed flakiness. No third retry layer.

## 9. Testing strategy

**1. Unit tests (`tests/governance/`)** — pytest + pytest-asyncio, no external services.

- Schema migration is idempotent (run twice, no error).
- `GovernanceService.upsert_document` returns existing `doc_id` on hash collision.
- `JobRunner.cancel` marks job + document correctly.
- Frozen workspace rejects writes; unfreeze restores access.
- Provenance backfill computes correct sets from a mock LightRAG KV.

**2. Integration tests (`tests/integration/governance/`)** — require PG running.

- Ephemeral PG schema per test (`CREATE SCHEMA test_<uuid>`), torn down after.
- `LocalRagService` mocked at the `ingest()` boundary (no real Neo4j/Qdrant in this layer).
- Cover full job lifecycle: enqueue → progress → done; enqueue → cancel.
- Cover delete-doc orchestration with mocked storage clients.

**3. End-to-end smoke (`tests/e2e/`)** — require all services running.

- One scripted scenario: ingest small file → poll job to done → query → delete-doc → re-query → confirm gone.
- Marked `@pytest.mark.e2e`, skipped by default in CI; run manually before releases.

**4. Manual API testing** — `tests/manual/governance.http` (REST Client format) — curl/httpie scripts kept in repo for ops reference.

**Out of scope:** load testing, chaos testing, multi-worker concurrency tests (Phase 2).

## 10. Migration plan

Five-step rollout, each independently shippable and revertible:

1. **Add `governance/` module + PG infra.** Schema, migrations, `GovernanceService`, `JobRunner`. No route changes. Tests pass.
2. **Add lifespan, refactor route DI.** Module globals → `app.state`. Existing endpoints behave identically. `LocalRagService.aclose()` added.
3. **Backfill existing workspaces into PG.** Startup scans `working_dir_root/`, inserts `workspaces` rows for each existing dir (status `legacy`, no provenance). Existing workspaces remain queryable; only newly ingested docs get provenance.
4. **Switch `/ingest`, `/ingest/batch`, `/retry` to job-based.** Old blocking behavior gone. Frontend must update in lockstep — covered in the frontend spec.
5. **Add new endpoints.** `/jobs/*`, `/workspace/*/freeze`, `/workspace/*/document/*`, `/workspace/*/audit`, `/admin/audit`.

Steps 1–3 ship without breaking anything and let the governance layer be tested in isolation. Step 4 is the breaking change; it's intentionally last so the WebUI rollout can be coordinated.

## 11. Configuration additions

New environment variables (read in `GovernanceSettings.from_env()`):

| Variable                             | Default                                          | Purpose                                  |
| ------------------------------------ | ------------------------------------------------ | ---------------------------------------- |
| `RAGANYTHING_PG_DSN`                 | `postgresql://localhost:5432/raganything`        | PG connection string                     |
| `RAGANYTHING_PG_POOL_MIN`            | `2`                                              | asyncpg pool min size                    |
| `RAGANYTHING_PG_POOL_MAX`            | `10`                                             | asyncpg pool max size                    |
| `RAGANYTHING_JOB_MAX_CONCURRENT`     | `1`                                              | `JobRunner` concurrency cap              |
| `RAGANYTHING_JOB_PROGRESS_INTERVAL`  | `5`                                              | Seconds between progress writes          |
| `RAGANYTHING_JOB_SHUTDOWN_GRACE`     | `30`                                             | Seconds for in-flight jobs at shutdown   |

All keys are also added to `raganything/constants.py` per the existing convention.

## 12. Explicit non-goals (Phase 2+)

- Multi-worker / Gunicorn deployment (needs Redis for distributed locks + ARQ for jobs + SSE pub/sub fanout).
- Snapshot/rollback of Neo4j+Qdrant+FS (requires coordinated dump tooling).
- Staging→prod workspace promotion.
- Soft-delete with grace period.
- Audit log retention/rotation cron.
- RAGAS evaluation + Phoenix tracing — separate spec.
- Frontend modernization — separate spec.
- Repair tool for orphaned partial-ingest state — Phase 1.5.

These are deferred. Approach 2 → Approach 3 is a clean upgrade path because all the seams are explicit (`JobRunner` swap, lock backend swap, KV migration is a LightRAG config change).

## 13. Acceptance criteria

A v1 implementation is complete when:

- `pytest tests/governance/ tests/integration/governance/` is green with PG running.
- `docker-compose-free` startup script: `pg_ctl start && uvicorn server.app:app` brings the service up clean; lifespan logs report startup success.
- Manual scenarios verified:
  - Upload same file twice → second upload returns `duplicate: true`, no new job created.
  - Upload, poll job to done, query — same answer as today.
  - Upload, cancel job → `/jobs/{id}` returns `failed (cancelled)`, doc shows `failed`.
  - Freeze workspace, attempt ingest → 409. Unfreeze, retry → succeeds.
  - Ingest two docs sharing one entity, delete one doc → entity remains in graph; provenance shows only the surviving doc's reference.
  - Kill the process during ingest, restart → job is `crashed`, doc is `failed`.
  - `GET /workspace/{ws}/audit` returns chronologically ordered events for all of the above.

## 14. Open questions / known gaps

- **Repair endpoint** for orphaned partial state from cancelled/crashed ingests is deferred to Phase 1.5. In v1, operators can manually clean up via SQL + LightRAG admin tooling.
- **Audit retention** is unbounded in v1. Manual `DELETE FROM ingest_audit WHERE timestamp < NOW() - INTERVAL '90 days'` is the obvious follow-up.
- **Backfill of existing workspaces** in step 3 assigns `status='legacy'` and no provenance — those workspaces cannot use per-doc deletion until re-ingested. Documented; not solved automatically.
