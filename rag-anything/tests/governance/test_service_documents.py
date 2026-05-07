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
