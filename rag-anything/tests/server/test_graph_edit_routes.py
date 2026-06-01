import pytest
from tests.governance.conftest import pytestmark_pg  # type: ignore

# Skip all tests in this module when PG is not reachable.
pytestmark = pytestmark_pg


@pytest.mark.asyncio
async def test_get_node_returns_full_properties(app_client):
    r = await app_client.get("/graph/ws1/nodes/Alice")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "Alice"
    assert body["properties"]["description"] == "engineer"
    assert body["properties"]["entity_type"] == "person"


@pytest.mark.asyncio
async def test_get_node_404_when_missing(app_client):
    r = await app_client.get("/graph/ws1/nodes/NoOne")
    assert r.status_code == 404
