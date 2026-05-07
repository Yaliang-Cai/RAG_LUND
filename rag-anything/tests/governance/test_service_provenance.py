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
