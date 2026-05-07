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
