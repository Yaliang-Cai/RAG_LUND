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
