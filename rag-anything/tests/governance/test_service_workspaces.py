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


async def test_backfill_legacy_workspaces(gov):
    inserted = await gov.backfill_legacy_workspaces(["w1", "w2"])
    assert inserted == 2
    again = await gov.backfill_legacy_workspaces(["w1", "w2", "w3"])
    assert again == 1  # only w3 was new
    rows = [await gov.get_workspace(w) for w in ("w1", "w2", "w3")]
    assert all(r is not None for r in rows)
    assert rows[0].metadata == {"legacy": True}


async def test_delete_workspace_cascades_documents_and_provenance(gov):
    await gov.ensure_workspace("w1")
    doc_id, _ = await gov.upsert_document("w1", "a.pdf", "h1", 1)
    await gov.insert_provenance("w1", doc_id, "chunk", ["chunk-1"])

    assert await gov.delete_workspace("w1") is True

    assert await gov.get_workspace("w1") is None
    assert await gov.list_documents("w1") == []
    assert await gov.get_provenance_for_doc(doc_id) == {
        "chunk": [],
        "entity": [],
        "relation": [],
    }
