"""Test fixtures for graph_edit_routes: NetworkX storage + PG governance."""
from __future__ import annotations

import tempfile
from typing import AsyncIterator

import pytest_asyncio
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

# Re-export pg_pool fixture + the skip marker for PG-dependent tests
from tests.governance.conftest import pg_pool, pytestmark_pg  # type: ignore  # noqa: F401

# Apply PG-availability skip to every test in this module
pytestmark = pytestmark_pg

from raganything.governance import GovernanceService
from raganything.governance.db import run_migrations
from server.graph_edit_routes import router as graph_edit_router
from server import graph_edit_routes as ger


class _FakeLightRAG:
    def __init__(self, kg):
        self.chunk_entity_relation_graph = kg


class _FakeRagService:
    def __init__(self, kg):
        self._kg = kg

    async def get_rag(self, workspace_id: str):  # noqa: ARG002
        return _FakeLightRAG(self._kg)


@pytest_asyncio.fixture
async def networkx_kg() -> AsyncIterator:
    """Empty NetworkX KG with a couple of seeded nodes/edges."""
    from lightrag.kg.networkx_impl import NetworkXStorage

    with tempfile.TemporaryDirectory() as tmp:
        kg = NetworkXStorage(
            namespace="test",
            workspace="",
            global_config={"working_dir": tmp},
            embedding_func=None,
        )
        await kg.initialize()
        await kg.upsert_node("Alice", {"description": "engineer", "entity_type": "person"})
        await kg.upsert_node("Acme", {"description": "company", "entity_type": "organization"})
        await kg.upsert_edge(
            "Alice", "Acme",
            {"description": "works at", "weight": 1.0, "keywords": "employment"},
        )
        try:
            yield kg
        finally:
            await kg.finalize()


@pytest_asyncio.fixture
async def app_client(pg_pool, networkx_kg) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    await run_migrations(pg_pool)
    gov = GovernanceService(pg_pool)
    rag_service = _FakeRagService(networkx_kg)
    await gov.ensure_workspace("ws1")

    app.state.service = rag_service
    app.state.gov = gov

    def get_service(request: Request):
        return request.app.state.service

    def get_gov(request: Request):
        return request.app.state.gov

    app.dependency_overrides[ger.get_service] = get_service
    app.dependency_overrides[ger.get_gov] = get_gov

    app.include_router(graph_edit_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c
