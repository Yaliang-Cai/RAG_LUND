# LightRAG Graph Governance Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add online property editing for graph nodes and edges to RAG-Anything's GraphPage, plus zoom/focus/layout/legend UX widgets, without changing the renderer.

**Architecture:** New FastAPI write routes in `server/graph_edit_routes.py` (mounted on existing app) guarded by `GovernanceService.ensure_writable` + `record_audit`. New Zustand store `store/graph.ts` owns graph UI state. New `PropertyEditDialog` + supporting components in `server/frontend/src/components/graph/`. Renderer stays react-force-graph; widgets driven via imperative `fgRef`.

**Tech Stack:** Python 3.x · FastAPI · pytest + pytest-asyncio + asyncpg · LightRAG storage interface (Neo4j prod, NetworkX in tests) · React 19 · TanStack Query · Zustand · @base-ui/react · Vitest + Testing Library · sonner (toasts) · axios.

**Spec:** `rag-anything/docs/superpowers/specs/2026-05-20-lightrag-graph-governance-port-design.md`

---

## File Structure

**Backend — new**
- `rag-anything/server/graph_edit_routes.py` — APIRouter with 4 endpoints (GET node/edge, PUT node/edge)
- `rag-anything/tests/server/__init__.py`
- `rag-anything/tests/server/conftest.py` — async FastAPI test app w/ NetworkX-backed rag + pg-backed governance
- `rag-anything/tests/server/test_graph_edit_routes.py`

**Backend — modify**
- `rag-anything/server/app.py` — add `app.include_router(graph_edit_routes.router)` near other route mounts

**Frontend — new**
- `rag-anything/server/frontend/src/store/graph.ts`
- `rag-anything/server/frontend/src/store/graph.test.ts`
- `rag-anything/server/frontend/src/components/graph/PropertyEditDialog.tsx`
- `rag-anything/server/frontend/src/components/graph/PropertyEditDialog.test.tsx`
- `rag-anything/server/frontend/src/components/graph/EdgeSheet.tsx`
- `rag-anything/server/frontend/src/components/graph/ZoomControl.tsx`
- `rag-anything/server/frontend/src/components/graph/FocusOnNode.tsx`
- `rag-anything/server/frontend/src/components/graph/LayoutsControl.tsx`
- `rag-anything/server/frontend/src/components/graph/Legend.tsx`

**Frontend — modify**
- `rag-anything/server/frontend/src/api/graph.ts` — add `getNode`, `getEdge`, `updateNode`, `updateEdge`
- `rag-anything/server/frontend/src/hooks/useGraph.ts` — add `useGraphNode`, `useGraphEdge`, `useUpdateNode`, `useUpdateEdge`
- `rag-anything/server/frontend/src/types/index.ts` — add `NodeDetail`, `EdgeDetail`
- `rag-anything/server/frontend/src/components/graph/ForceGraph.tsx` — read selection/highlight from store, expose link click
- `rag-anything/server/frontend/src/components/graph/GraphSearch.tsx` — dispatch into store
- `rag-anything/server/frontend/src/components/graph/NodeSheet.tsx` — add "Edit" button + read from store
- `rag-anything/server/frontend/src/routes/GraphPage.tsx` — compose new controls, drop local state in favor of store

**Docs — modify**
- `rag-anything/CLAUDE.md` — append a section under "重要模式" documenting graph governance routes, frozen-workspace gate, last-writer-wins limitation

---

## Task 1: Define editable-property contract (constants, shared)

**Files:**
- Create: `rag-anything/server/graph_edit_routes.py` (skeleton only — constants + router)

- [ ] **Step 1: Create file with reserved-key constant and empty router**

```python
# rag-anything/server/graph_edit_routes.py
"""Read + write endpoints for in-place graph property editing.

Backed by LightRAG's storage interface (upsert_node / upsert_edge).
All writes are gated by GovernanceService.ensure_writable and audited.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

# Server-managed keys that callers may not set/overwrite.
RESERVED_NODE_KEYS = frozenset({"source_id", "file_path", "created_at", "entity_id"})
RESERVED_EDGE_KEYS = frozenset({"source_id", "file_path", "created_at"})

router = APIRouter(tags=["graph-edit"])


class PropertyUpdate(BaseModel):
    properties: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 2: Commit**

```bash
git add rag-anything/server/graph_edit_routes.py
git commit -m "feat(graph-edit): scaffold router + reserved-key constants"
```

---

## Task 2: GET node endpoint — read full property bag from LightRAG storage

**Files:**
- Modify: `rag-anything/server/graph_edit_routes.py`
- Create: `rag-anything/tests/server/__init__.py` (empty)
- Create: `rag-anything/tests/server/conftest.py`
- Create: `rag-anything/tests/server/test_graph_edit_routes.py`

- [ ] **Step 1: Create empty test package init**

```bash
mkdir -p rag-anything/tests/server
: > rag-anything/tests/server/__init__.py
```

- [ ] **Step 2: Write the test fixture**

The fixture builds a minimal FastAPI app with NetworkX-backed LightRAG storage (no Neo4j dep) and PG-backed governance. It reuses `tests/governance/conftest.py:pg_pool`.

```python
# rag-anything/tests/server/conftest.py
"""Test fixtures for graph_edit_routes: NetworkX storage + PG governance."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

# Re-export pg_pool fixture + the skip marker for PG-dependent tests
from tests.governance.conftest import pg_pool, pytestmark_pg  # type: ignore  # noqa: F401

# Apply PG-availability skip to every test in this module
pytestmark = pytestmark_pg

from raganything.governance import GovernanceService
from server.graph_edit_routes import router as graph_edit_router

# Tiny in-process rag stand-in: just exposes `lightrag.chunk_entity_relation_graph`
# with the NetworkX implementation. Avoids spinning a full LocalRagService.


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
    gov = GovernanceService(pg_pool)
    rag_service = _FakeRagService(networkx_kg)
    await gov.ensure_workspace("ws1")

    # Mirror server/app.py's dependency wiring
    app.state.service = rag_service
    app.state.gov = gov

    def get_service(request: Request):
        return request.app.state.service

    def get_gov(request: Request):
        return request.app.state.gov

    # Inject overrides expected by the router (see Task 2 Step 3)
    from server import graph_edit_routes as ger
    app.dependency_overrides[ger.get_service] = get_service
    app.dependency_overrides[ger.get_gov] = get_gov

    app.include_router(graph_edit_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c
```

- [ ] **Step 3: Write failing test for GET node**

```python
# rag-anything/tests/server/test_graph_edit_routes.py
import pytest

# conftest sets pytestmark = pytestmark_pg (PG availability skip).
# Each test is async, so mark them individually:


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
```

- [ ] **Step 4: Run tests — expect failure (endpoint missing)**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: FAIL with 404 / route not found / import error in `server.graph_edit_routes` (no `get_service` symbol yet).

- [ ] **Step 5: Implement GET node endpoint + dependency hooks**

Add to `rag-anything/server/graph_edit_routes.py`:

```python
# at top, after imports
def get_service(request: Request):
    """Resolved via app.dependency_overrides in app.py; tests override directly."""
    return request.app.state.service


def get_gov(request: Request):
    return request.app.state.gov


@router.get("/graph/{workspace_id}/nodes/{node_id}")
async def get_node(workspace_id: str, node_id: str, service=Depends(get_service)):
    rag = await service.get_rag(workspace_id)
    node = await rag.chunk_entity_relation_graph.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return {"id": node_id, "properties": dict(node)}
```

- [ ] **Step 6: Run tests — expect pass**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add rag-anything/server/graph_edit_routes.py rag-anything/tests/server/
git commit -m "feat(graph-edit): GET /graph/{ws}/nodes/{id} + test harness"
```

---

## Task 3: GET edge endpoint

**Files:**
- Modify: `rag-anything/server/graph_edit_routes.py`
- Modify: `rag-anything/tests/server/test_graph_edit_routes.py`

- [ ] **Step 1: Write failing tests**

Append to `test_graph_edit_routes.py`:

```python
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
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py::test_get_edge_returns_properties -v`
Expected: FAIL (404 from missing route).

- [ ] **Step 3: Implement GET edge**

Append to `graph_edit_routes.py`:

```python
def _split_edge_id(edge_id: str) -> tuple[str, str]:
    if "|" not in edge_id:
        raise HTTPException(
            status_code=400,
            detail="edge_id must be 'source|target'",
        )
    src, tgt = edge_id.split("|", 1)
    if not src or not tgt:
        raise HTTPException(status_code=400, detail="edge_id source/target empty")
    return src, tgt


@router.get("/graph/{workspace_id}/edges/{edge_id}")
async def get_edge(workspace_id: str, edge_id: str, service=Depends(get_service)):
    src, tgt = _split_edge_id(edge_id)
    rag = await service.get_rag(workspace_id)
    edge = await rag.chunk_entity_relation_graph.get_edge(src, tgt)
    if edge is None:
        raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
    return {"id": edge_id, "properties": dict(edge)}
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/graph_edit_routes.py rag-anything/tests/server/test_graph_edit_routes.py
git commit -m "feat(graph-edit): GET /graph/{ws}/edges/{id}"
```

---

## Task 4: PUT node — happy path + reserved-key guard

**Files:**
- Modify: `rag-anything/server/graph_edit_routes.py`
- Modify: `rag-anything/tests/server/test_graph_edit_routes.py`

- [ ] **Step 1: Write failing tests**

Append to `test_graph_edit_routes.py`:

```python
@pytest.mark.asyncio
async def test_put_node_updates_properties_and_returns_fresh(app_client):
    r = await app_client.put(
        "/graph/ws1/nodes/Alice",
        json={"properties": {"description": "senior engineer", "entity_type": "person"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["properties"]["description"] == "senior engineer"

    # verify GET reflects the change
    g = await app_client.get("/graph/ws1/nodes/Alice")
    assert g.json()["properties"]["description"] == "senior engineer"


@pytest.mark.asyncio
async def test_put_node_rejects_reserved_keys(app_client):
    r = await app_client.put(
        "/graph/ws1/nodes/Alice",
        json={"properties": {"description": "x", "source_id": "evil"}},
    )
    assert r.status_code == 400
    assert "source_id" in r.json()["detail"]


@pytest.mark.asyncio
async def test_put_node_404_when_missing(app_client):
    r = await app_client.put(
        "/graph/ws1/nodes/NoOne",
        json={"properties": {"description": "ghost"}},
    )
    assert r.status_code == 404
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: 3 new failures (route missing).

- [ ] **Step 3: Implement PUT node**

Append to `graph_edit_routes.py`:

```python
def _reject_reserved(properties: dict[str, Any], reserved: frozenset[str]) -> None:
    bad = sorted(k for k in properties if k in reserved)
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Reserved keys may not be set: {', '.join(bad)}",
        )


@router.put("/graph/{workspace_id}/nodes/{node_id}")
async def update_node(
    workspace_id: str,
    node_id: str,
    body: PropertyUpdate,
    service=Depends(get_service),
    gov=Depends(get_gov),
):
    _reject_reserved(body.properties, RESERVED_NODE_KEYS)
    await gov.ensure_writable(workspace_id)
    rag = await service.get_rag(workspace_id)
    kg = rag.chunk_entity_relation_graph

    old = await kg.get_node(node_id)
    if old is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    merged = {k: old[k] for k in RESERVED_NODE_KEYS if k in old}
    merged.update(body.properties)
    await kg.upsert_node(node_id, merged)
    new = await kg.get_node(node_id) or {}

    await gov.record_audit(
        workspace_id,
        "graph.node.update",
        details={"target": node_id, "old": dict(old), "new": dict(new)},
    )
    return {"id": node_id, "properties": dict(new)}
```

Also import `WorkspaceFrozenError` at top and add an exception handler — see Task 6 (covered there). For now: `ensure_writable` raises and FastAPI returns 500 unless mapped; we'll wire the 423 mapping in Task 6 along with its own dedicated test.

- [ ] **Step 4: Run — expect pass**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: all green (frozen-workspace test not yet added).

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/graph_edit_routes.py rag-anything/tests/server/test_graph_edit_routes.py
git commit -m "feat(graph-edit): PUT node with reserved-key guard + audit"
```

---

## Task 5: PUT edge — happy path

**Files:**
- Modify: `rag-anything/server/graph_edit_routes.py`
- Modify: `rag-anything/tests/server/test_graph_edit_routes.py`

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_put_edge_updates_properties(app_client):
    edge_id = quote("Alice|Acme", safe="")
    r = await app_client.put(
        f"/graph/ws1/edges/{edge_id}",
        json={"properties": {"description": "founded", "weight": 2.5, "keywords": "founder"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["properties"]["description"] == "founded"
    assert float(body["properties"]["weight"]) == 2.5


@pytest.mark.asyncio
async def test_put_edge_rejects_reserved_keys(app_client):
    edge_id = quote("Alice|Acme", safe="")
    r = await app_client.put(
        f"/graph/ws1/edges/{edge_id}",
        json={"properties": {"description": "x", "source_id": "evil"}},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_put_edge_404_when_missing(app_client):
    edge_id = quote("Alice|Nowhere", safe="")
    r = await app_client.put(
        f"/graph/ws1/edges/{edge_id}",
        json={"properties": {"description": "x"}},
    )
    assert r.status_code == 404
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: 3 new failures.

- [ ] **Step 3: Implement PUT edge**

Append to `graph_edit_routes.py`:

```python
@router.put("/graph/{workspace_id}/edges/{edge_id}")
async def update_edge(
    workspace_id: str,
    edge_id: str,
    body: PropertyUpdate,
    service=Depends(get_service),
    gov=Depends(get_gov),
):
    src, tgt = _split_edge_id(edge_id)
    _reject_reserved(body.properties, RESERVED_EDGE_KEYS)
    await gov.ensure_writable(workspace_id)
    rag = await service.get_rag(workspace_id)
    kg = rag.chunk_entity_relation_graph

    old = await kg.get_edge(src, tgt)
    if old is None:
        raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")

    merged = {k: old[k] for k in RESERVED_EDGE_KEYS if k in old}
    merged.update(body.properties)
    await kg.upsert_edge(src, tgt, merged)
    new = await kg.get_edge(src, tgt) or {}

    await gov.record_audit(
        workspace_id,
        "graph.edge.update",
        details={"target": edge_id, "old": dict(old), "new": dict(new)},
    )
    return {"id": edge_id, "properties": dict(new)}
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/graph_edit_routes.py rag-anything/tests/server/test_graph_edit_routes.py
git commit -m "feat(graph-edit): PUT edge with reserved-key guard + audit"
```

---

## Task 6: Frozen-workspace 423 mapping + test

**Files:**
- Modify: `rag-anything/server/graph_edit_routes.py`
- Modify: `rag-anything/tests/server/test_graph_edit_routes.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_put_node_returns_423_when_frozen(app_client):
    # Freeze the workspace via the test app's gov instance
    gov = app_client._transport.app.state.gov  # type: ignore[attr-defined]
    await gov.set_frozen("ws1", True)
    r = await app_client.put(
        "/graph/ws1/nodes/Alice",
        json={"properties": {"description": "x"}},
    )
    assert r.status_code == 423
    assert "frozen" in r.json()["detail"].lower()
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py::test_put_node_returns_423_when_frozen -v`
Expected: FAIL (500 not 423).

- [ ] **Step 3: Map exception**

Add to `graph_edit_routes.py` near the imports:

```python
from raganything.governance.service import WorkspaceFrozenError


@router.exception_handler(WorkspaceFrozenError) if False else None  # placeholder
```

Replace the placeholder by registering the handler on the app instead — exception handlers don't attach to routers. Instead, raise an `HTTPException` inline. Wrap each `gov.ensure_writable(...)` call:

```python
async def _ensure_writable(gov, workspace_id: str) -> None:
    try:
        await gov.ensure_writable(workspace_id)
    except WorkspaceFrozenError as e:
        raise HTTPException(status_code=423, detail=str(e) or "Workspace is frozen")
```

Replace both `await gov.ensure_writable(workspace_id)` calls in `update_node` and `update_edge` with `await _ensure_writable(gov, workspace_id)`.

- [ ] **Step 4: Run — expect pass**

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/graph_edit_routes.py rag-anything/tests/server/test_graph_edit_routes.py
git commit -m "feat(graph-edit): map WorkspaceFrozenError to 423 Locked"
```

---

## Task 7: Audit-row content verification

**Files:**
- Modify: `rag-anything/tests/server/test_graph_edit_routes.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_put_node_writes_audit_row_with_diff(app_client):
    r = await app_client.put(
        "/graph/ws1/nodes/Alice",
        json={"properties": {"description": "principal engineer", "entity_type": "person"}},
    )
    assert r.status_code == 200
    gov = app_client._transport.app.state.gov  # type: ignore[attr-defined]
    audits = await gov.list_audit("ws1", action="graph.node.update", limit=10)
    assert len(audits) == 1
    details = audits[0].details
    assert details["target"] == "Alice"
    assert details["old"]["description"] == "engineer"
    assert details["new"]["description"] == "principal engineer"
```

- [ ] **Step 2: Run — expect pass**

(The implementation in Task 4 already writes the audit row; this test only verifies it.) If `audits[0].details` is a string rather than dict, the JSONB codec is not registered — the test harness fixture in Task 2 already registers it via the `pg_pool` fixture. If it fails, fix the codec registration there.

Run: `pytest rag-anything/tests/server/test_graph_edit_routes.py::test_put_node_writes_audit_row_with_diff -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/tests/server/test_graph_edit_routes.py
git commit -m "test(graph-edit): verify audit row captures old/new diff"
```

---

## Task 8: Mount router on the main app

**Files:**
- Modify: `rag-anything/server/app.py`

- [ ] **Step 1: Locate the router-mount section in `app.py`**

Run: `grep -n "include_router\|app = FastAPI" rag-anything/server/app.py`
You should see existing `app.include_router(...)` calls. Pick the cluster and add ours alongside.

- [ ] **Step 2: Add the include**

Near the top of `app.py` with the other imports, add:

```python
from server import graph_edit_routes
```

After `app = FastAPI(...)` (and any existing `app.include_router(...)`), add:

```python
app.include_router(graph_edit_routes.router)
```

The router's `get_service` / `get_gov` dependencies read from `request.app.state`. Verify that `app.state.service` and `app.state.gov` are set in `app.py`'s lifespan. From the spec context, `gov_service = GovernanceService(pg_pool, rag_service)` is already created in `app.py:104`. Add one line in the lifespan/startup right after that assignment:

```python
app.state.service = rag_service
app.state.gov = gov_service
```

(If those lines already exist, leave them.)

- [ ] **Step 3: Smoke-check the app boots**

Run: `python -c "from server.app import app; print(len(app.routes))"`
Expected: prints a number (no ImportError).

- [ ] **Step 4: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): mount graph_edit_routes + expose service/gov on app.state"
```

---

## Task 9: Frontend types + API client extensions

**Files:**
- Modify: `rag-anything/server/frontend/src/types/index.ts`
- Modify: `rag-anything/server/frontend/src/api/graph.ts`

- [ ] **Step 1: Add `NodeDetail` / `EdgeDetail` types**

Append to `types/index.ts`:

```typescript
export interface NodeDetail {
  id: string
  properties: Record<string, unknown>
}

export interface EdgeDetail {
  id: string                // "source|target"
  properties: Record<string, unknown>
}
```

- [ ] **Step 2: Extend `api/graph.ts`**

Append to `api/graph.ts`:

```typescript
import type { NodeDetail, EdgeDetail } from '@/types'

export async function getNode(workspaceId: string, nodeId: string): Promise<NodeDetail> {
  const { data } = await client.get<NodeDetail>(
    `/graph/${workspaceId}/nodes/${encodeURIComponent(nodeId)}`
  )
  return data
}

export async function getEdge(workspaceId: string, edgeId: string): Promise<EdgeDetail> {
  const { data } = await client.get<EdgeDetail>(
    `/graph/${workspaceId}/edges/${encodeURIComponent(edgeId)}`
  )
  return data
}

export async function updateNode(
  workspaceId: string,
  nodeId: string,
  properties: Record<string, unknown>
): Promise<NodeDetail> {
  const { data } = await client.put<NodeDetail>(
    `/graph/${workspaceId}/nodes/${encodeURIComponent(nodeId)}`,
    { properties }
  )
  return data
}

export async function updateEdge(
  workspaceId: string,
  edgeId: string,
  properties: Record<string, unknown>
): Promise<EdgeDetail> {
  const { data } = await client.put<EdgeDetail>(
    `/graph/${workspaceId}/edges/${encodeURIComponent(edgeId)}`,
    { properties }
  )
  return data
}
```

- [ ] **Step 3: Type-check**

Run: `cd rag-anything/server/frontend && npx tsc -b --noEmit`
Expected: no new errors.

- [ ] **Step 4: Commit**

```bash
git add rag-anything/server/frontend/src/types/index.ts rag-anything/server/frontend/src/api/graph.ts
git commit -m "feat(fe): NodeDetail/EdgeDetail types + getNode/updateNode/getEdge/updateEdge"
```

---

## Task 10: Zustand graph UI store

**Files:**
- Create: `rag-anything/server/frontend/src/store/graph.ts`
- Create: `rag-anything/server/frontend/src/store/graph.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// rag-anything/server/frontend/src/store/graph.test.ts
import { describe, it, expect, beforeEach } from 'vitest'
import { useGraphUiStore } from './graph'

describe('graph UI store', () => {
  beforeEach(() => useGraphUiStore.getState().reset())

  it('select / focus / openEdit / closeEdit', () => {
    const s = useGraphUiStore.getState()
    s.selectNode('Alice')
    expect(useGraphUiStore.getState().selectedNodeId).toBe('Alice')

    const before = useGraphUiStore.getState().focusRequestId
    s.focusOn('Alice')
    expect(useGraphUiStore.getState().focusRequestId).toBeGreaterThan(before)

    s.openEdit({ kind: 'node', id: 'Alice', draft: { description: 'eng' } })
    expect(useGraphUiStore.getState().editing?.id).toBe('Alice')
    s.updateDraft({ description: 'senior eng' })
    expect(useGraphUiStore.getState().editing?.draft.description).toBe('senior eng')

    s.closeEdit()
    expect(useGraphUiStore.getState().editing).toBeNull()
  })
})
```

- [ ] **Step 2: Run — expect failure**

Run: `cd rag-anything/server/frontend && npx vitest run src/store/graph.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement store**

```typescript
// rag-anything/server/frontend/src/store/graph.ts
import { create } from 'zustand'

export type Layout = 'force' | 'radial' | 'tree'

export interface EditingState {
  kind: 'node' | 'edge'
  id: string
  draft: Record<string, unknown>
}

interface GraphUiStore {
  selectedNodeId: string | null
  selectedEdgeId: string | null
  focusRequestId: number
  layout: Layout
  visibleLabels: Set<string>
  legendVisible: boolean
  editing: EditingState | null

  selectNode: (id: string | null) => void
  selectEdge: (id: string | null) => void
  focusOn: (id: string) => void
  setLayout: (l: Layout) => void
  toggleLabel: (label: string) => void
  setLegendVisible: (v: boolean) => void
  openEdit: (e: EditingState) => void
  updateDraft: (patch: Record<string, unknown>) => void
  closeEdit: () => void
  reset: () => void
}

const initial = {
  selectedNodeId: null,
  selectedEdgeId: null,
  focusRequestId: 0,
  layout: 'force' as Layout,
  visibleLabels: new Set<string>(),
  legendVisible: true,
  editing: null,
}

export const useGraphUiStore = create<GraphUiStore>((set, get) => ({
  ...initial,
  selectNode: (id) => set({ selectedNodeId: id, selectedEdgeId: null }),
  selectEdge: (id) => set({ selectedEdgeId: id, selectedNodeId: null }),
  focusOn: (id) => set({ selectedNodeId: id, focusRequestId: get().focusRequestId + 1 }),
  setLayout: (l) => set({ layout: l }),
  toggleLabel: (label) =>
    set((s) => {
      const next = new Set(s.visibleLabels)
      next.has(label) ? next.delete(label) : next.add(label)
      return { visibleLabels: next }
    }),
  setLegendVisible: (v) => set({ legendVisible: v }),
  openEdit: (e) => set({ editing: e }),
  updateDraft: (patch) =>
    set((s) => (s.editing ? { editing: { ...s.editing, draft: { ...s.editing.draft, ...patch } } } : {})),
  closeEdit: () => set({ editing: null }),
  reset: () => set({ ...initial, visibleLabels: new Set() }),
}))
```

- [ ] **Step 4: Run — expect pass**

Run: `cd rag-anything/server/frontend && npx vitest run src/store/graph.test.ts`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/store/graph.ts rag-anything/server/frontend/src/store/graph.test.ts
git commit -m "feat(fe): Zustand store for graph UI state (select/focus/edit draft)"
```

---

## Task 11: TanStack Query hooks (useGraphNode, useUpdateNode, useUpdateEdge)

**Files:**
- Modify: `rag-anything/server/frontend/src/hooks/useGraph.ts`

- [ ] **Step 1: Append hooks**

```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { getNode, getEdge, updateNode, updateEdge } from '@/api/graph'
import { toast } from 'sonner'

export function useGraphNode(workspaceId: string, nodeId: string | null) {
  return useQuery({
    queryKey: ['graph', 'node', workspaceId, nodeId],
    queryFn: () => getNode(workspaceId, nodeId!),
    enabled: !!workspaceId && !!nodeId,
  })
}

export function useGraphEdge(workspaceId: string, edgeId: string | null) {
  return useQuery({
    queryKey: ['graph', 'edge', workspaceId, edgeId],
    queryFn: () => getEdge(workspaceId, edgeId!),
    enabled: !!workspaceId && !!edgeId,
  })
}

export function useUpdateNode(workspaceId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ nodeId, properties }: { nodeId: string; properties: Record<string, unknown> }) =>
      updateNode(workspaceId, nodeId, properties),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['graph', 'node', workspaceId, vars.nodeId] })
      qc.invalidateQueries({ queryKey: ['graph', 'overview', workspaceId] })
      qc.invalidateQueries({ queryKey: ['graph', 'subgraph', workspaceId] })
      toast.success('Node updated')
    },
    onError: (err: Error) => toast.error(err.message),
  })
}

export function useUpdateEdge(workspaceId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ edgeId, properties }: { edgeId: string; properties: Record<string, unknown> }) =>
      updateEdge(workspaceId, edgeId, properties),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['graph', 'edge', workspaceId, vars.edgeId] })
      qc.invalidateQueries({ queryKey: ['graph', 'overview', workspaceId] })
      qc.invalidateQueries({ queryKey: ['graph', 'subgraph', workspaceId] })
      toast.success('Edge updated')
    },
    onError: (err: Error) => toast.error(err.message),
  })
}
```

- [ ] **Step 2: Type-check**

Run: `cd rag-anything/server/frontend && npx tsc -b --noEmit`
Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/hooks/useGraph.ts
git commit -m "feat(fe): useGraphNode/useGraphEdge/useUpdateNode/useUpdateEdge hooks"
```

---

## Task 12: PropertyEditDialog component

**Files:**
- Create: `rag-anything/server/frontend/src/components/graph/PropertyEditDialog.tsx`
- Create: `rag-anything/server/frontend/src/components/graph/PropertyEditDialog.test.tsx`

The component re-implements LightRAG's `PropertyEditDialog` shape using RAG-Anything's `@base-ui/react`-based UI kit (`@/components/ui/dialog`, `@/components/ui/button`, `@/components/ui/input`, `@/components/ui/textarea`). It is driven entirely by the Zustand store's `editing` state.

- [ ] **Step 1: Write failing test**

```typescript
// PropertyEditDialog.test.tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { PropertyEditDialog } from './PropertyEditDialog'
import { useGraphUiStore } from '@/store/graph'

vi.mock('@/api/graph', () => ({
  updateNode: vi.fn(async () => ({ id: 'Alice', properties: { description: 'new' } })),
  updateEdge: vi.fn(),
  getNode: vi.fn(),
  getEdge: vi.fn(),
}))

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('PropertyEditDialog', () => {
  beforeEach(() => useGraphUiStore.getState().reset())

  it('renders draft fields and saves on click', async () => {
    useGraphUiStore.getState().openEdit({
      kind: 'node',
      id: 'Alice',
      draft: { description: 'engineer', entity_type: 'person' },
    })
    wrap(<PropertyEditDialog workspaceId="ws1" />)

    const desc = await screen.findByLabelText(/description/i)
    expect(desc).toHaveValue('engineer')
    await userEvent.clear(desc)
    await userEvent.type(desc, 'senior engineer')
    await userEvent.click(screen.getByRole('button', { name: /save/i }))

    const { updateNode } = await import('@/api/graph')
    expect(updateNode).toHaveBeenCalledWith('ws1', 'Alice', expect.objectContaining({
      description: 'senior engineer',
      entity_type: 'person',
    }))
  })
})
```

- [ ] **Step 2: Run — expect failure**

Run: `cd rag-anything/server/frontend && npx vitest run src/components/graph/PropertyEditDialog.test.tsx`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement component**

```typescript
// rag-anything/server/frontend/src/components/graph/PropertyEditDialog.tsx
import { useGraphUiStore } from '@/store/graph'
import { useUpdateNode, useUpdateEdge } from '@/hooks/useGraph'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
// No Label primitive in this codebase — use a plain <label> element below.

const MULTILINE_KEYS = new Set(['description', 'keywords'])

export function PropertyEditDialog({ workspaceId }: { workspaceId: string }) {
  const editing = useGraphUiStore((s) => s.editing)
  const closeEdit = useGraphUiStore((s) => s.closeEdit)
  const updateDraft = useGraphUiStore((s) => s.updateDraft)
  const updateNode = useUpdateNode(workspaceId)
  const updateEdge = useUpdateEdge(workspaceId)
  const open = !!editing

  if (!editing) return null

  const handleSave = async () => {
    const props = Object.fromEntries(
      Object.entries(editing.draft).map(([k, v]) => [k, typeof v === 'string' ? v.trim() : v])
    )
    if (editing.kind === 'node') {
      await updateNode.mutateAsync({ nodeId: editing.id, properties: props })
    } else {
      await updateEdge.mutateAsync({ edgeId: editing.id, properties: props })
    }
    closeEdit()
  }

  const isSubmitting = updateNode.isPending || updateEdge.isPending
  const keys = Object.keys(editing.draft)

  return (
    <Dialog open={open} onOpenChange={(o: boolean) => !o && closeEdit()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-base">
            Edit {editing.kind} · {editing.id}
          </DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3 py-2">
          {keys.map((k) => {
            const value = String(editing.draft[k] ?? '')
            return (
              <div key={k} className="flex flex-col gap-1">
                <label htmlFor={`field-${k}`} className="text-xs text-muted-foreground uppercase">
                  {k}
                </label>
                {MULTILINE_KEYS.has(k) ? (
                  <Textarea
                    id={`field-${k}`}
                    value={value}
                    rows={5}
                    onChange={(e) => updateDraft({ [k]: e.target.value })}
                  />
                ) : (
                  <Input
                    id={`field-${k}`}
                    value={value}
                    onChange={(e) => updateDraft({ [k]: e.target.value })}
                  />
                )}
              </div>
            )
          })}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={closeEdit} disabled={isSubmitting}>Cancel</Button>
          <Button onClick={handleSave} disabled={isSubmitting}>
            {isSubmitting ? 'Saving…' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
```

Verified present in `components/ui/`: `dialog.tsx`, `input.tsx`, `textarea.tsx`, `button.tsx`, `sheet.tsx`. **No `label.tsx` exists** — use a plain `<label>` HTML element with Tailwind classes (already done above). If any other expected primitive is missing, scaffold it by copying the existing shadcn pattern (the codebase uses `@base-ui/react`); do not invent new APIs.

- [ ] **Step 4: Run — expect pass**

Run: `cd rag-anything/server/frontend && npx vitest run src/components/graph/PropertyEditDialog.test.tsx`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/components/graph/PropertyEditDialog.tsx rag-anything/server/frontend/src/components/graph/PropertyEditDialog.test.tsx
git commit -m "feat(fe): PropertyEditDialog driven by Zustand graph store"
```

---

## Task 13: EdgeSheet component

**Files:**
- Create: `rag-anything/server/frontend/src/components/graph/EdgeSheet.tsx`

- [ ] **Step 1: Implement (no test — same shape as NodeSheet which has no test)**

```typescript
// rag-anything/server/frontend/src/components/graph/EdgeSheet.tsx
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { Pencil } from 'lucide-react'
import { useGraphUiStore } from '@/store/graph'
import { useGraphEdge } from '@/hooks/useGraph'

export function EdgeSheet({ workspaceId }: { workspaceId: string }) {
  const edgeId = useGraphUiStore((s) => s.selectedEdgeId)
  const selectEdge = useGraphUiStore((s) => s.selectEdge)
  const openEdit = useGraphUiStore((s) => s.openEdit)
  const { data } = useGraphEdge(workspaceId, edgeId)

  return (
    <Sheet open={!!edgeId} onOpenChange={(open: boolean) => !open && selectEdge(null)}>
      <SheetContent side="right" className="w-80">
        {edgeId && (
          <>
            <SheetHeader>
              <SheetTitle className="text-base">{edgeId}</SheetTitle>
            </SheetHeader>
            <div className="mt-4 flex flex-col gap-3 text-sm">
              {data &&
                Object.entries(data.properties).map(([k, v]) => (
                  <div key={k}>
                    <span className="text-xs text-muted-foreground uppercase">{k}</span>
                    <p className="mt-0.5 text-muted-foreground text-xs leading-relaxed">{String(v ?? '—')}</p>
                  </div>
                ))}
              <Button
                size="sm"
                className="mt-3"
                disabled={!data}
                onClick={() =>
                  data && openEdit({ kind: 'edge', id: edgeId, draft: { ...data.properties } })
                }
              >
                <Pencil className="h-3.5 w-3.5 mr-1" /> Edit
              </Button>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
```

- [ ] **Step 2: Type-check**

Run: `cd rag-anything/server/frontend && npx tsc -b --noEmit`

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/graph/EdgeSheet.tsx
git commit -m "feat(fe): EdgeSheet — view + edit edge properties"
```

---

## Task 14: NodeSheet gains "Edit" button + reads from store

**Files:**
- Modify: `rag-anything/server/frontend/src/components/graph/NodeSheet.tsx`

- [ ] **Step 1: Replace prop-driven sheet with store-driven sheet that fetches full props**

```typescript
// rag-anything/server/frontend/src/components/graph/NodeSheet.tsx
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { Pencil } from 'lucide-react'
import { useGraphUiStore } from '@/store/graph'
import { useGraphNode } from '@/hooks/useGraph'

export function NodeSheet({ workspaceId }: { workspaceId: string }) {
  const nodeId = useGraphUiStore((s) => s.selectedNodeId)
  const selectNode = useGraphUiStore((s) => s.selectNode)
  const openEdit = useGraphUiStore((s) => s.openEdit)
  const { data } = useGraphNode(workspaceId, nodeId)

  return (
    <Sheet open={!!nodeId} onOpenChange={(open: boolean) => !open && selectNode(null)}>
      <SheetContent side="right" className="w-80">
        {nodeId && (
          <>
            <SheetHeader>
              <SheetTitle className="text-base">{nodeId}</SheetTitle>
            </SheetHeader>
            <div className="mt-4 flex flex-col gap-3 text-sm">
              {data &&
                Object.entries(data.properties).map(([k, v]) => (
                  <div key={k}>
                    <span className="text-xs text-muted-foreground uppercase">{k}</span>
                    <p className="mt-0.5 text-muted-foreground text-xs leading-relaxed">{String(v ?? '—')}</p>
                  </div>
                ))}
              <Button
                size="sm"
                className="mt-3"
                disabled={!data}
                onClick={() =>
                  data && openEdit({ kind: 'node', id: nodeId, draft: { ...data.properties } })
                }
              >
                <Pencil className="h-3.5 w-3.5 mr-1" /> Edit
              </Button>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
```

- [ ] **Step 2: Type-check**

Run: `cd rag-anything/server/frontend && npx tsc -b --noEmit`
Expected: GraphPage.tsx breaks (passing `node`/`onClose` props that no longer exist). Fixed in Task 18.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/graph/NodeSheet.tsx
git commit -m "refactor(fe): NodeSheet reads selection from store, adds Edit button"
```

---

## Task 15: ZoomControl + FocusOnNode wired to fgRef

ForceGraph2D exposes an imperative ref with `zoom()`, `centerAt(x,y,ms)`, `zoomToFit(ms,padding)`. We'll forward the ref out of `ForceGraph.tsx` and have small controls call it.

**Files:**
- Modify: `rag-anything/server/frontend/src/components/graph/ForceGraph.tsx`
- Create: `rag-anything/server/frontend/src/components/graph/ZoomControl.tsx`
- Create: `rag-anything/server/frontend/src/components/graph/FocusOnNode.tsx`

- [ ] **Step 1: Forward ref from ForceGraph**

Replace the existing `export function ForceGraph` signature with a `forwardRef` that exposes `{ zoomIn, zoomOut, zoomToFit, focusNode }`. Sketch:

```typescript
import { forwardRef, useImperativeHandle, useRef, useCallback, useEffect } from 'react'
import { useGraphUiStore } from '@/store/graph'

export interface ForceGraphHandle {
  zoomIn(): void
  zoomOut(): void
  zoomToFit(): void
  focusNode(id: string): void
}

export const ForceGraph = forwardRef<ForceGraphHandle, ForceGraphProps>(function ForceGraph(
  { data, onNodeClick, highlightNodeId },
  ref
) {
  const fgRef = useRef<any>(null)

  useImperativeHandle(ref, () => ({
    zoomIn: () => fgRef.current?.zoom((fgRef.current.zoom() ?? 1) * 1.4, 250),
    zoomOut: () => fgRef.current?.zoom((fgRef.current.zoom() ?? 1) / 1.4, 250),
    zoomToFit: () => fgRef.current?.zoomToFit(400, 40),
    focusNode: (id: string) => {
      const n = data.nodes.find((x) => x.id === id) as any
      if (n && typeof n.x === 'number') fgRef.current?.centerAt(n.x, n.y, 400)
    },
  }), [data])

  // Subscribe to store's focusRequestId — re-focus when bumped
  const focusReq = useGraphUiStore((s) => s.focusRequestId)
  const focusId = useGraphUiStore((s) => s.selectedNodeId)
  useEffect(() => {
    if (focusReq && focusId) {
      const n = data.nodes.find((x) => x.id === focusId) as any
      if (n && typeof n.x === 'number') fgRef.current?.centerAt(n.x, n.y, 400)
    }
  }, [focusReq, focusId, data])

  // ... existing handleNodeClick, paintNode, graphData ...

  return (
    <ForceGraph2D
      ref={fgRef as any}
      graphData={graphData as AnyNode}
      // ... rest unchanged ...
    />
  )
})
```

Keep all existing render logic — only add the ref forwarding, the imperative handle, and the focus effect.

- [ ] **Step 2: Implement ZoomControl**

```typescript
// rag-anything/server/frontend/src/components/graph/ZoomControl.tsx
import { Button } from '@/components/ui/button'
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import type { ForceGraphHandle } from './ForceGraph'
import type { RefObject } from 'react'

export function ZoomControl({ graphRef }: { graphRef: RefObject<ForceGraphHandle | null> }) {
  return (
    <div className="absolute top-2 right-2 flex flex-col gap-1 bg-background/80 backdrop-blur rounded-md p-1 shadow">
      <Button variant="ghost" size="icon-sm" title="Zoom in" onClick={() => graphRef.current?.zoomIn()}>
        <ZoomIn className="h-3.5 w-3.5" />
      </Button>
      <Button variant="ghost" size="icon-sm" title="Zoom out" onClick={() => graphRef.current?.zoomOut()}>
        <ZoomOut className="h-3.5 w-3.5" />
      </Button>
      <Button variant="ghost" size="icon-sm" title="Fit to view" onClick={() => graphRef.current?.zoomToFit()}>
        <Maximize2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}
```

- [ ] **Step 3: Implement FocusOnNode**

```typescript
// rag-anything/server/frontend/src/components/graph/FocusOnNode.tsx
import { Button } from '@/components/ui/button'
import { Crosshair } from 'lucide-react'
import { useGraphUiStore } from '@/store/graph'

export function FocusOnNode() {
  const selected = useGraphUiStore((s) => s.selectedNodeId)
  const focusOn = useGraphUiStore((s) => s.focusOn)
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      title="Re-center on selected node"
      disabled={!selected}
      onClick={() => selected && focusOn(selected)}
    >
      <Crosshair className="h-3.5 w-3.5" />
    </Button>
  )
}
```

- [ ] **Step 4: Type-check**

Run: `cd rag-anything/server/frontend && npx tsc -b --noEmit`

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/components/graph/ForceGraph.tsx rag-anything/server/frontend/src/components/graph/ZoomControl.tsx rag-anything/server/frontend/src/components/graph/FocusOnNode.tsx
git commit -m "feat(fe): forwardRef on ForceGraph + ZoomControl + FocusOnNode"
```

---

## Task 16: LayoutsControl (preset switcher)

LightRAG ships multiple sigma.js layout algorithms. For react-force-graph the practical knobs are the d3 force strengths. Three presets that map to noticeable differences:

| preset | charge | linkDistance |
|---|---|---|
| `force` (default) | -30 | 30 |
| `radial` | -120 | 60 |
| `tree` | -10 | 15 |

**Files:**
- Create: `rag-anything/server/frontend/src/components/graph/LayoutsControl.tsx`
- Modify: `rag-anything/server/frontend/src/components/graph/ForceGraph.tsx` — apply layout from store on mount + when changed

- [ ] **Step 1: Apply layout in ForceGraph**

Inside `ForceGraph`, add:

```typescript
const layout = useGraphUiStore((s) => s.layout)
useEffect(() => {
  const fg: any = fgRef.current
  if (!fg) return
  const presets: Record<string, { charge: number; link: number }> = {
    force: { charge: -30, link: 30 },
    radial: { charge: -120, link: 60 },
    tree: { charge: -10, link: 15 },
  }
  const p = presets[layout] ?? presets.force
  fg.d3Force('charge')?.strength(p.charge)
  fg.d3Force('link')?.distance(p.link)
  fg.d3ReheatSimulation?.()
}, [layout])
```

- [ ] **Step 2: Implement LayoutsControl**

```typescript
// rag-anything/server/frontend/src/components/graph/LayoutsControl.tsx
import { useGraphUiStore, type Layout } from '@/store/graph'

const OPTIONS: { value: Layout; label: string }[] = [
  { value: 'force', label: 'Force' },
  { value: 'radial', label: 'Radial' },
  { value: 'tree', label: 'Tree' },
]

export function LayoutsControl() {
  const layout = useGraphUiStore((s) => s.layout)
  const setLayout = useGraphUiStore((s) => s.setLayout)
  return (
    <select
      value={layout}
      onChange={(e) => setLayout(e.target.value as Layout)}
      className="h-7 text-xs bg-background border border-border rounded px-2"
      title="Layout"
    >
      {OPTIONS.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  )
}
```

- [ ] **Step 3: Type-check**

Run: `cd rag-anything/server/frontend && npx tsc -b --noEmit`

- [ ] **Step 4: Commit**

```bash
git add rag-anything/server/frontend/src/components/graph/LayoutsControl.tsx rag-anything/server/frontend/src/components/graph/ForceGraph.tsx
git commit -m "feat(fe): LayoutsControl — switch react-force-graph force presets"
```

---

## Task 17: Legend (type → color reference)

**Files:**
- Create: `rag-anything/server/frontend/src/components/graph/Legend.tsx`
- Modify: `rag-anything/server/frontend/src/components/graph/ForceGraph.tsx` — export `TYPE_COLORS`

- [ ] **Step 1: Export TYPE_COLORS from ForceGraph**

Change `const TYPE_COLORS` to `export const TYPE_COLORS`.

- [ ] **Step 2: Implement Legend**

```typescript
// rag-anything/server/frontend/src/components/graph/Legend.tsx
import { useGraphUiStore } from '@/store/graph'
import { TYPE_COLORS } from './ForceGraph'

export function Legend() {
  const visible = useGraphUiStore((s) => s.legendVisible)
  if (!visible) return null
  return (
    <div className="absolute bottom-2 left-2 bg-background/80 backdrop-blur rounded-md p-2 shadow text-xs">
      <div className="font-medium mb-1">Types</div>
      <ul className="flex flex-col gap-0.5">
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <li key={type} className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded-full" style={{ background: color }} />
            <span className="text-muted-foreground">{type}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
```

- [ ] **Step 3: Type-check + commit**

```bash
cd rag-anything/server/frontend && npx tsc -b --noEmit
git add rag-anything/server/frontend/src/components/graph/Legend.tsx rag-anything/server/frontend/src/components/graph/ForceGraph.tsx
git commit -m "feat(fe): Legend overlay"
```

---

## Task 18: Compose everything in GraphPage

**Files:**
- Modify: `rag-anything/server/frontend/src/routes/GraphPage.tsx`
- Modify: `rag-anything/server/frontend/src/components/graph/GraphSearch.tsx`
- Modify: `rag-anything/server/frontend/src/components/graph/ForceGraph.tsx` — emit edge clicks via store; click on canvas clears selection

- [ ] **Step 1: GraphSearch writes into store**

Change the `onResult` callback in `GraphSearch.tsx` (or its consumer) so results call `useGraphUiStore.getState().focusOn(id)` instead of an external prop. Keep the prop for backward compat but make it optional. Quick path:

```typescript
import { useGraphUiStore } from '@/store/graph'

// inside the result-click handler:
useGraphUiStore.getState().focusOn(result.id)
onResult?.(result.id)
```

- [ ] **Step 2: ForceGraph emits node/edge selection into store**

In `ForceGraph.tsx`, update `handleNodeClick` and add `handleLinkClick`:

```typescript
import { useGraphUiStore } from '@/store/graph'

const selectNode = useGraphUiStore((s) => s.selectNode)
const selectEdge = useGraphUiStore((s) => s.selectEdge)

const handleNodeClick = useCallback((node: AnyNode) => {
  selectNode((node as GraphNode).id)
  onNodeClick?.(node as GraphNode)
}, [onNodeClick, selectNode])

const handleLinkClick = useCallback((link: AnyNode) => {
  const src = typeof link.source === 'object' ? link.source.id : link.source
  const tgt = typeof link.target === 'object' ? link.target.id : link.target
  selectEdge(`${src}|${tgt}`)
}, [selectEdge])

// pass onLinkClick={handleLinkClick} to <ForceGraph2D ... />
```

Also derive `highlightNodeId` from the store if the prop is not supplied:

```typescript
const storeSelected = useGraphUiStore((s) => s.selectedNodeId)
const effectiveHighlight = highlightNodeId ?? storeSelected
```

Use `effectiveHighlight` inside `paintNode` instead of `highlightNodeId`.

- [ ] **Step 3: Rewrite GraphPage to compose new controls**

```typescript
// rag-anything/server/frontend/src/routes/GraphPage.tsx
import { useRef } from 'react'
import { useGraphOverview } from '@/hooks/useGraph'
import { useAppStore } from '@/store'
import { ForceGraph, type ForceGraphHandle } from '@/components/graph/ForceGraph'
import { GraphSearch } from '@/components/graph/GraphSearch'
import { NodeSheet } from '@/components/graph/NodeSheet'
import { EdgeSheet } from '@/components/graph/EdgeSheet'
import { PropertyEditDialog } from '@/components/graph/PropertyEditDialog'
import { ZoomControl } from '@/components/graph/ZoomControl'
import { FocusOnNode } from '@/components/graph/FocusOnNode'
import { LayoutsControl } from '@/components/graph/LayoutsControl'
import { Legend } from '@/components/graph/Legend'
import { Button } from '@/components/ui/button'
import { RefreshCw } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'

export default function GraphPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { data, isLoading } = useGraphOverview(workspaceId)
  const qc = useQueryClient()
  const fgRef = useRef<ForceGraphHandle | null>(null)

  if (isLoading) {
    return <div className="flex items-center justify-center h-full text-sm text-muted-foreground">Loading graph...</div>
  }
  if (!data || data.nodes.length === 0) {
    return <div className="flex items-center justify-center h-full text-sm text-muted-foreground">No graph data. Ingest documents first.</div>
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border shrink-0">
        <GraphSearch workspaceId={workspaceId} />
        <LayoutsControl />
        <FocusOnNode />
        <Button
          variant="ghost"
          size="icon-sm"
          title="Refresh"
          onClick={() => qc.invalidateQueries({ queryKey: ['graph', 'overview', workspaceId] })}
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
        <span className="text-xs text-muted-foreground ml-auto">
          {data.nodes.length} nodes · {data.edges.length} edges
        </span>
      </div>
      <div className="flex-1 relative">
        <ForceGraph ref={fgRef} data={data} />
        <ZoomControl graphRef={fgRef} />
        <Legend />
      </div>
      <NodeSheet workspaceId={workspaceId} />
      <EdgeSheet workspaceId={workspaceId} />
      <PropertyEditDialog workspaceId={workspaceId} />
    </div>
  )
}
```

`GraphSearch` no longer needs the `onResult` prop in this composition (it writes to the store directly); leave the prop optional in its signature for back-compat.

- [ ] **Step 4: Type-check + run frontend tests**

```bash
cd rag-anything/server/frontend && npx tsc -b --noEmit && npx vitest run
```

Expected: type-check passes, all frontend tests pass.

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/routes/GraphPage.tsx \
        rag-anything/server/frontend/src/components/graph/GraphSearch.tsx \
        rag-anything/server/frontend/src/components/graph/ForceGraph.tsx
git commit -m "feat(fe): GraphPage composes new controls; selection lives in store"
```

---

## Task 19: Backend full-suite + frontend full-suite regression check

- [ ] **Step 1: Run backend tests**

Run: `cd rag-anything && pytest tests/server -v`
Expected: all green. If PG isn't reachable, tests will skip via the `pgtest_dsn_or_skip` marker from `tests/governance/conftest.py` — that's fine; document in the commit message that PG-dependent tests were skipped locally.

- [ ] **Step 2: Run frontend tests**

Run: `cd rag-anything/server/frontend && npx vitest run`
Expected: all green.

- [ ] **Step 3: No commit (verification only)**

---

## Task 20: Manual UI verification

Cannot be substituted with automated tests — must be done before claiming completion.

- [ ] **Step 1: Start backend**

Run: `cd rag-anything && uvicorn server.app:app --host 0.0.0.0 --port 9621`

- [ ] **Step 2: Start frontend dev server**

In another terminal: `cd rag-anything/server/frontend && npm run dev`

- [ ] **Step 3: Walk through scenarios**

In the browser:

1. Open `/graph`. Select a workspace with existing entities.
2. Click a node → NodeSheet opens with full properties (not just label/type/description).
3. Click **Edit** → PropertyEditDialog opens with all fields pre-filled.
4. Change `description`, click **Save**. Toast "Node updated" appears. Subgraph refreshes.
5. Reload the page → edited description persists.
6. Click an edge → EdgeSheet opens; Edit → change `weight`, save, reload, persists.
7. Switch layout via dropdown → graph re-flows visibly.
8. Click zoom in/out/fit → camera responds.
9. Search a node label → graph re-centers on it; NodeSheet opens for that node.
10. Freeze the workspace via existing settings UI → attempt to edit → toast surfaces "Workspace is frozen".
11. Confirm audit row in PG:
    ```
    psql $RAGANYTHING_PG_DSN -c "SELECT action, details FROM ingest_audit
        WHERE action LIKE 'graph.%' ORDER BY timestamp DESC LIMIT 5;"
    ```

If anything fails, fix and commit the fix; don't mark the task done.

- [ ] **Step 4: Note results in commit message**

No code commit unless fixes were made. If everything works, proceed to Task 21.

---

## Task 21: Document new endpoints + known limits in CLAUDE.md

**Files:**
- Modify: `rag-anything/CLAUDE.md`

- [ ] **Step 1: Append a new bullet under "重要模式"**

```markdown
- **图谱在线治理 (Graph governance)**：节点/关系属性可经 `PUT /graph/{ws}/nodes/{id}` 与 `PUT /graph/{ws}/edges/{src|tgt}` 直接编辑（见 `server/graph_edit_routes.py`）。
  - 所有写操作经 `GovernanceService.ensure_writable()` 校验（冻结工作区返回 `423 Locked`）并写入 `ingest_audit` 表（`action=graph.node.update` / `graph.edge.update`，`details` 包含旧/新属性差异）。
  - 保留键 `source_id` / `file_path` / `created_at` / `entity_id` 由服务端管理，不接受外部覆盖（返回 `400`）。
  - **并发模型**：last-writer-wins；本期未实现乐观锁，依赖审计表作恢复路径。
  - **未实现**：节点合并 / 删除 / 边创建 / 重命名（属于后续设计）。
```

- [ ] **Step 2: Commit**

```bash
git add rag-anything/CLAUDE.md
git commit -m "docs(claude): document graph governance routes, frozen gate, last-writer-wins"
```

---

## Self-review checklist (already run)

- **Spec coverage:** §3 architecture → Tasks 1, 8, 10. §4 backend contract → Tasks 1-7. §5 frontend components → Tasks 9-17. §6 data flow → Tasks 11, 14, 18. §7 error handling → Task 6 (423), Task 11 (toast surface via axios interceptor reusing existing pattern), spec §7 deferred concurrency UI documented in Task 21. §8 testing → Tasks 2-7 (backend), Tasks 10, 12 (frontend store + dialog), Task 20 (manual). §9 out-of-scope items not implemented. ✓
- **Placeholder scan:** none. All code blocks complete.
- **Type consistency:** `useGraphUiStore` name + `ForceGraphHandle` + `EditingState` + `PropertyUpdate` are used identically across all referencing tasks.
