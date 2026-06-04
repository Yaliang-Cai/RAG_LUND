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
