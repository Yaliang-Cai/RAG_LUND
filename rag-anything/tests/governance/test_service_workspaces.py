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
