# LightRAG Graph Governance Port — Design

**Date:** 2026-05-20
**Status:** Draft (awaiting user review)
**Scope tier:** B — read-only UX upgrades + property editing on nodes and edges (no node merging, no rename, no renderer swap)

## 1. Goal

Bring LightRAG WebUI's interactive graph UX and online property editing for nodes and edges into RAG-Anything's `GraphPage.tsx`, without changing the renderer (keep `react-force-graph`) and without breaking the multi-workspace + frozen-workspace model.

Explicitly deferred to future designs: node merging, node deletion, edge creation, rename/relabel, settings panel, fullscreen, sigma.js renderer swap.

## 2. Context (verified)

- RAG-Anything's graph storage is **Neo4j** via LightRAG's `Neo4JStorage` (configured in `raganything/services/local_rag.py:2044`); GraphML on disk is a fallback used by the read-only `/graph/{ws}/stats` and `/graph/{ws}/labels` endpoints.
- LightRAG's storage interface already exposes `upsert_node`, `upsert_edge`, `delete_node` (see `lightrag/base.py:766`, `lightrag/kg/neo4j_impl.py:1076`). No upstream change required.
- LightRAG ships a FastAPI router (`lightrag/api/routers/graph_routes.py`) with `/graph/entity/edit`, `/graph/relation/edit`, `/graph/entities/merge`, but it binds to a single rag instance — incompatible with RAG-Anything's per-workspace rag construction in `service.get_or_create_workspace`. We do **not** mount it.
- `raganything/governance/service.py` already provides `GovernanceService.ensure_writable(workspace_id)` (raises `WorkspaceFrozenError`) and `record_audit(...)`. New write endpoints will use both.
- Frontend already has `api/graph.ts`, `routes/GraphPage.tsx`, `components/graph/{ForceGraph,GraphSearch,NodeSheet}.tsx`. No Zustand store yet for graph UI — `store/index.ts` is unrelated.

## 3. Architecture

Two layers:

**Backend** — new module `server/graph_edit_routes.py`, mounted on the main FastAPI app via `app.include_router(...)`. Reuses the existing `GovernanceService` dependency and the per-workspace `rag = service.get_or_create_workspace(ws)` pattern from `app.py`.

**Frontend** — port LightRAG WebUI's editing dialog and a subset of its control widgets into `server/frontend/src/components/graph/`, plus a new Zustand store `store/graph.ts` for graph-UI state (selection, focus, layout, edit draft). Server state continues to flow through TanStack Query mutations in `api/graph.ts`.

Renderer stays **react-force-graph**. LightRAG WebUI uses sigma.js + graphology, so most of its widgets are renderer-coupled and cannot be copied verbatim — we port the interaction patterns and UI shells and rewire them to react-force-graph's imperative `fgRef` API.

## 4. Backend contract

### Endpoints

```
GET    /graph/{workspace_id}/nodes/{node_id}
GET    /graph/{workspace_id}/edges/{edge_id}       # edge_id = url-encoded "{src}|{tgt}"
PUT    /graph/{workspace_id}/nodes/{node_id}
PUT    /graph/{workspace_id}/edges/{edge_id}
```

### Request body (both PUTs)

```json
{ "properties": { "<key>": "<value>", ... } }
```

`properties` is a **full replacement** of the editable property bag, matching `upsert_node(node_data=...)` semantics. The server strips reserved keys (`source_id`, `file_path`, `created_at`, `entity_id`) before upsert; an unknown reserved key in the request returns 400.

`rename_to` is intentionally **not** part of the schema in this iteration. Adding it later is additive.

### Response (both PUTs)

```json
{ "id": "...", "properties": { ... }, "updated_at": "2026-05-20T..." }
```

### Status codes

- `200` success
- `400` malformed body / unknown reserved key
- `404` workspace, node, or edge not found
- `423 Locked` workspace frozen (`WorkspaceFrozenError`)
- `500` upstream storage error

### Server-side flow (per write)

```
ensure_writable(workspace_id)            # may raise WorkspaceFrozenError → 423
rag = service.get_or_create_workspace(workspace_id)
old = await rag.lightrag.chunk_entity_relation_graph.get_node(node_id)
# Preserve reserved keys from old, overlay request properties (full replacement of editable bag)
merged = {**{k: old[k] for k in RESERVED_KEYS if k in old}, **request.properties}
await rag.lightrag.chunk_entity_relation_graph.upsert_node(node_id, merged)
new = await rag.lightrag.chunk_entity_relation_graph.get_node(node_id)
await governance.record_audit(
    workspace_id=ws,
    action="graph.node.update",         # or "graph.edge.update"
    target=node_id,                      # or edge_id
    payload={"old": old, "new": new},
)
return {"id": node_id, "properties": new, "updated_at": ...}
```

### Concurrency

Last-writer-wins. No optimistic locking. The audit trail is the safety net. Documented as a known limitation in CLAUDE.md.

## 5. Frontend components

Files added to `server/frontend/src/`:

```
api/graph.ts                       # extended: getNode, getEdge, updateNode, updateEdge
store/graph.ts                     # NEW Zustand store (UI state only)
components/graph/
  ForceGraph.tsx                   # existing — reads selection/focus from store
  GraphSearch.tsx                  # existing — writes selection into store
  NodeSheet.tsx                    # existing — gains "Edit" button → opens PropertyEditDialog
  EdgeSheet.tsx                    # NEW sibling for edges
  PropertyEditDialog.tsx           # NEW (ported from lightrag_webui)
  EditablePropertyRow.tsx          # NEW (ported)
  PropertyRowComponents.tsx        # NEW (ported)
  LayoutsControl.tsx               # NEW (ported shell, rewired to RFG d3Force presets)
  ZoomControl.tsx                  # NEW (ported shell, rewired to fgRef.zoom)
  FocusOnNode.tsx                  # NEW (ported shell, rewired to fgRef.centerAt)
  Legend.tsx                       # NEW (ported, renderer-agnostic)
```

### Zustand store — `store/graph.ts`

```ts
type GraphStore = {
  selectedNodeId: string | null
  selectedEdgeId: string | null
  focusRequestId: number              // bumped to trigger fgRef.centerAt
  layout: 'force' | 'radial' | 'tree'
  visibleLabels: Set<string>          // empty = show all
  legendVisible: boolean
  editing: { kind: 'node'|'edge', id: string, draft: Record<string, any> } | null

  selectNode, selectEdge, focusOn, setLayout, toggleLabel,
  openEdit, updateDraft, closeEdit
}
```

UI state only. Server state stays in TanStack Query; the dialog dispatches mutations via `api/graph.ts` so cache invalidation of `getOverview` / `getSubgraph` / `getNode` happens centrally.

### Component reuse rules

- `EditablePropertyRow`, `PropertyRowComponents`, `PropertyEditDialog`, `Legend` — port JSX + Tailwind classes, strip sigma-specific imports, replace LightRAG api-client calls with the new `api/graph.ts` functions.
- `LayoutsControl`, `ZoomControl`, `FocusOnNode` — port the UI shell (buttons, icons, hotkeys); rewrite handlers against `fgRef`.

## 6. Data flow (node edit, end to end)

```
1. User clicks node in ForceGraph
2. ForceGraph → store.selectNode(id)
3. NodeSheet observes store, useQuery(['node', ws, id], () => getNode(ws, id))
4. User clicks "Edit" → store.openEdit({kind:'node', id, draft: <fetched props>})
5. PropertyEditDialog renders rows; edits mutate store.editing.draft
6. Save → useMutation(updateNode) fires
       backend: ensure_writable → upsert_node → record_audit → return fresh node
7. onSuccess: queryClient.invalidateQueries(['node', ws, id])
               queryClient.invalidateQueries(['subgraph', ws, ...])
               store.closeEdit()
8. ForceGraph re-renders with refreshed subgraph
```

Focus / zoom / layout flows are pure client-side: store action → `useEffect` in `ForceGraph` reads new state and calls the imperative `fgRef` method. No server roundtrip.

## 7. Error handling

| Failure | Surface |
|---|---|
| 423 Locked (frozen workspace) | Toast: "Workspace is frozen. Unfreeze in Settings to edit." Save disabled on retry. |
| 404 (concurrent deletion) | Toast + auto-close dialog + invalidate subgraph. |
| 400 (validation, reserved key) | Inline error under offending row; Save stays disabled. |
| 500 / network | Toast with retry button; draft preserved in store. |
| Concurrent edits | Last-writer-wins; no UI resolution this iteration. Audit log is recovery path. |

Backend uses `HTTPException`. Frontend extends the existing axios interceptor in `api/client.ts` to map status → toast category — no per-call duplication.

## 8. Testing

**Backend (`tests/server/test_graph_edit_routes.py`)** — pytest + httpx AsyncClient against a test FastAPI app backed by LightRAG's `NetworkXStorage` (keeps CI dependency-free; no Neo4j needed):

- happy path PUT node, PUT edge → verify upsert + audit row written
- frozen workspace → 423
- missing node/edge → 404
- reserved keys stripped or rejected
- audit payload contains `{old, new}` diff

**Frontend (Vitest + React Testing Library)**

- `PropertyEditDialog.test.tsx` — open with seeded draft, type, Save calls mutation with expected body
- `store/graph.test.ts` — reducer-level tests for select/focus/openEdit/closeEdit
- Optional Playwright smoke (`e2e/graph-edit.spec.ts`) — open GraphPage, click node, edit description, assert persisted value after reload. Skip and note as follow-up if Playwright isn't wired.

**Manual verification before declaring done** — start dev server, edit one node property end-to-end against a real Neo4j workspace, confirm audit row in Postgres.

## 9. Out of scope (this iteration)

- `MergeDialog` and node merging — needs entity-embedding consistency design.
- Node deletion, edge creation, rename/relabel.
- Settings panel, FullScreenControl, GraphControl (sigma renderer host).
- Optimistic locking / conflict UI.
- Bulk edits.
