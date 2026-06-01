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


from urllib.parse import quote


@pytest.mark.asyncio
async def test_get_edge_returns_properties(app_client):
    edge_id = quote("Alice|Acme", safe="")
    r = await app_client.get(f"/graph/ws1/edges/{edge_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "Alice|Acme"
    assert body["properties"]["description"] == "works at"
    assert float(body["properties"]["weight"]) == 1.0


@pytest.mark.asyncio
async def test_get_edge_404_when_missing(app_client):
    r = await app_client.get(f"/graph/ws1/edges/{quote('Alice|Nope', safe='')}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_edge_400_when_malformed(app_client):
    r = await app_client.get("/graph/ws1/edges/no-separator")
    assert r.status_code == 400
