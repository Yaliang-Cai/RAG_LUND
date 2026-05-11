# Frontend Modernization — Design Spec

**Date:** 2026-05-11
**Branch:** `frontend-modernization` (based on `backend-hardening` @ 3ae4ed2)
**Status:** Approved for implementation planning

## 1. Problem & Goals

The current frontend (`rag-anything/server/templates/index.html`) is a 2587-line monolithic Jinja2 + vanilla JS single-file SPA. It has no build toolchain, no type safety, no component boundaries, and cannot grow alongside the new backend API surface introduced in `backend-hardening`:

- No job-status polling UI (`/jobs/*` endpoints exist but are unused)
- Knowledge graph rendered via legacy pyvis iframe (`/graph/{ws}/html`) — not composable
- No per-document delete or freeze toggle in the UI
- No ingest progress feedback (old blocking `/ingest` UX)
- No PDF viewer — only parsed Markdown preview

**Goal:** Replace the single-file frontend with a React + Tailwind + shadcn/ui + Vite + Bun SPA that covers all v1 features, is statically served by FastAPI from `static/dist/`, and is fully adapted to the `backend-hardening` API contract.

**Out of scope (v2):** Audit log dedicated page (placeholder tab only), PDF.js annotation support, E2E tests (Playwright), multi-worker backend deployment.

## 2. Architecture

### Project Layout

```
rag-anything/server/
  frontend/              ← Vite project root
    package.json         # scripts: dev / build / lint
    vite.config.ts       # outDir: ../static/dist, dev proxy
    tailwind.config.ts
    tsconfig.json
    index.html           # Vite entry (not Jinja2)
    src/
      main.tsx           # ReactDOM.createRoot + providers
      App.tsx            # Router + ThemeProvider + QueryClientProvider
      routes/            # Page-level components (lazy-loaded)
        ChatPage.tsx
        DocumentsPage.tsx
        GraphPage.tsx
        JobsPage.tsx
      components/
        layout/          # TopNav, WorkspaceSwitcher, ThemeToggle
        chat/            # MessageList, MessageBubble, StreamingMessage, ReasoningTrace
        documents/       # FileList, FileUpload, MarkdownViewer, PdfViewer
        graph/           # ForceGraph, GraphControls, NodeTooltip, GraphSearch
        jobs/            # JobCard, JobList, ProgressBar
        ui/              # shadcn/ui re-exports
      hooks/             # Custom React Query hooks
      store/             # Zustand store
      api/               # Fetch/axios wrappers (one file per endpoint group)
      types/             # TypeScript interfaces aligned to backend Pydantic models
  static/
    dist/                ← bun run build output; FastAPI mounts this
  app.py
```

### Runtime Architecture

```
Browser
  └── React SPA (:5173 dev / :9621 prod via FastAPI)
        ├── TanStack Query  →  REST API (/files, /jobs, /graph/*, /workspace/*)
        ├── Zustand store   →  workspaceId, theme, selectedFileId
        ├── fetch ReadableStream  →  POST /query/stream (SSE)
        └── react-force-graph-2d  →  /graph/{ws}/subgraph + /graph/{ws}/overview
```

### FastAPI Changes (Minimal)

Remove Jinja2 template machinery from `app.py`:
- Delete `TEMPLATES`, `_USE_LOCAL_STATIC`, and the `GET /` route
- Remove `from fastapi.templating import Jinja2Templates`
- After all API routes, add as the **last** statement:

```python
app.mount("/", StaticFiles(directory=str(APP_ROOT / "static/dist"), html=True), name="spa")
```

`html=True` causes FastAPI to serve `index.html` for all unmatched paths, enabling React Router client-side routing.

## 3. Visual Design

**Direction:** Modern tool aesthetic (Linear/Notion family). shadcn/ui default theme with minimal customization.

| Token | Value |
|-------|-------|
| Background | `#0f172a` (slate-900) |
| Surface | `#1e293b` (slate-800) |
| Accent / active | `#6366f1` (indigo-500) |
| Text primary | `#f1f5f9` (slate-100) |
| Text muted | `#64748b` (slate-500) |
| Font | system-ui / Inter |
| Theme | Dark default; light toggle via shadcn `ThemeProvider` |

## 4. Routing & Layout

**Router:** React Router v6, lazy-loaded pages.

```
/          → redirect → /chat
/chat      → ChatPage
/documents → DocumentsPage
/graph     → GraphPage
/jobs      → JobsPage
```

**AppShell** wraps all pages with a fixed top navigation bar. The shell does not remount on page switches.

**TopNav:**
```
[RAGAnything]   Chat  Documents  Graph  Jobs      [ws: default ▾]  [🌙]
```
- Active tab: indigo bottom border
- WorkspaceSwitcher: `DropdownMenu` + `Command` (searchable), calls `GET /workspaces`
- ThemeToggle: icon button, persisted in Zustand + localStorage

**Global Job Notifications (AppShell responsibility):**

`AppShell` runs `useJobs(workspaceId)` at the shell level (2s polling when any job is running). It tracks the previous jobs snapshot via `useRef` and diffs on each poll result:
- If a job transitions from `running` → `failed`: fire Sonner error toast with file name and a "View Jobs" link (`navigate('/jobs')`)
- If a job transitions from `running` → `done`: fire Sonner success toast (optional, can be silenced if noisy)

The Jobs tab in `TopNav` shows a badge count (`RunningJobsCount`) when any jobs are active, so users always have ambient awareness regardless of which page they are on. This badge is derived from the same `useJobs` query result — no extra fetch.

## 5. Pages

### Chat (`/chat`)

**Layout:** Toolbar (query mode selector, settings, new-conversation button) → scrollable message list → fixed input bar.

**Key behaviors:**
- SSE via `fetch` + `ReadableStream` (not `EventSource`, since the endpoint is `POST /query/stream`)
- Reasoning trace rendered in a `Collapsible` (collapsed by default), `IBM Plex Mono` font
- Auto-scroll to bottom on new chunks; user scroll-up pauses auto-scroll
- Query mode: `hybrid` default; `Select` component with options `naive / local / global / hybrid`

**Citation jump (PDF deep-link):**

When the stream completes, the response payload includes a `source_nodes` array. Each node must contain:
```ts
{ doc_id: string; filename: string; page_num: number; excerpt: string }
```
These are rendered as inline citation chips below the AI answer (e.g., `[paper_01.pdf p.4]`). Clicking a chip:
1. Sets `selectedFileId` and `pendingPageNum` in Zustand store
2. Navigates to `/documents`
3. `DocumentsPage` reads `pendingPageNum` from the store on mount, passes it to `<PdfViewer pageNumber={pendingPageNum} />`
4. `react-pdf` renders directly to the target page; `PdfViewer` clears `pendingPageNum` after mount

**Backend contract requirement:** `POST /query/stream` SSE events must include a terminal `source_nodes` event (or embed `page_num` in existing source metadata). This requires a backend verification step before the PdfViewer feature can be activated — if `page_num` is absent from the current response, the citation chips degrade gracefully to filename-only (no page jump).

### Documents (`/documents`)

**Layout:** Fixed-width file list panel (left) + preview area (right) with `[Markdown] [PDF]` tab switcher.

**File list actions:**
- Click file → load preview
- Delete button → `AlertDialog` confirm → `DELETE /workspace/{ws}/document/{doc_id}` → invalidate files query
- Freeze toggle → `Switch` → `POST /workspace/{ws}/freeze` or `/unfreeze`; frozen workspace disables upload and delete

**Upload:**
- Drag-and-drop + click-to-select; validates against `SUPPORTED_EXTENSIONS` before POST
- `POST /ingest` (job-based) → on success, Sonner toast "Ingest job created" + `navigate('/jobs')`

**PDF Viewer:** `react-pdf` `<Document>` + `<Page>`, toolbar with prev/next/zoom/page-number. Source URL: `GET /uploads/{workspaceId}/{filename}`.

**Markdown Viewer:** `react-markdown` + `rehype-highlight` + `rehype-katex` + `remark-math`.

### Graph (`/graph`)

**Initial load:** `GET /graph/{ws}/overview` → renders full overview graph.

**Interactions:**
- Search bar → `GET /graph/{ws}/search?query=...` → highlight matching nodes, zoom-to-fit
- Click node → `Sheet` (right-side drawer) with entity details: name, type, source documents
- Right-click node → expand subgraph via `GET /graph/{ws}/subgraph?seed={nodeId}&depth=2`
- Depth selector (`Select`) controls subgraph expansion depth

**Node coloring:** Fixed palette keyed on entity type (`Concept`, `Person`, `Organization`, `Location`, `Other`).

**Library:** `react-force-graph-2d` (Canvas rendering). Node color/size via `nodeColor` / `nodeVal` props. Edge labels via `linkLabel`.

### Jobs (`/jobs`)

**Polling:** `TanStack Query` `refetchInterval: (data) => data?.some(j => j.status === 'running') ? 2000 : false`.

**Sections:**
- Running: `Progress` bar + cancel button (`DELETE /jobs/{job_id}`)
- Completed: timestamp, file name
- Failed: file name + retry button (calls `POST /retry/{workspaceId}`)
- Audit log: `Collapsible` at bottom (placeholder; v2 gets dedicated page)

## 6. State Management

### TanStack Query (server state)

One custom hook per resource:

| Hook | Endpoint | Notes |
|------|----------|-------|
| `useWorkspaces()` | `GET /workspaces` | staleTime: 30s |
| `useFiles(wsId)` | `GET /files/{ws}` | key: `['files', wsId]` |
| `useJobs(wsId)` | `GET /jobs?workspace_id={ws}` | polling interval dynamic |
| `useGraph(wsId)` | `GET /graph/{ws}/overview` | staleTime: 60s |
| `useSubgraph(wsId, seed, depth)` | `GET /graph/{ws}/subgraph` | enabled when seed set |

Workspace switch: `queryClient.invalidateQueries()` clears all caches, triggering refetch for the new workspace.

### Zustand Store (UI state)

```ts
interface AppStore {
  workspaceId: string
  setWorkspace: (id: string) => void
  theme: 'dark' | 'light'
  toggleTheme: () => void
  selectedFileId: string | null
  setSelectedFile: (id: string | null) => void
  pendingPageNum: number | null           // set by citation chip click
  setPendingPageNum: (n: number | null) => void
}
```

### SSE Stream Hook

`useStreamQuery()` manages `fetch` + `ReadableStream` directly (not React Query):
- State: `chunks: string[]`, `reasoning: string`, `status: 'idle' | 'streaming' | 'done' | 'error'`
- On send: clears previous state, opens stream, appends chunks as they arrive
- On component unmount: aborts the fetch via `AbortController`

## 7. API Layer

`src/api/client.ts` — axios instance; `baseURL` is empty string (same-origin in prod, proxied in dev via Vite).

`src/api/` files:

| File | Functions |
|------|-----------|
| `workspace.ts` | `getWorkspaces`, `freezeWorkspace`, `unfreezeWorkspace`, `deleteWorkspace` |
| `files.ts` | `getFiles`, `getFileContent`, `uploadFile`, `deleteDocument` |
| `jobs.ts` | `getJobs`, `getJob`, `cancelJob` |
| `graph.ts` | `getOverview`, `getSubgraph`, `searchGraph`, `getLabels`, `getStats` |
| `query.ts` | `postQuery`, `openQueryStream` (returns `Response` for streaming) |
| `config.ts` | `getConfig` |

## 8. Error Handling

| Layer | Strategy |
|-------|----------|
| React Query failure | `onError` callback → Sonner toast with backend `detail` field |
| SSE disconnect | Set status `'error'`, show inline "Connection lost — retry" |
| Upload wrong type | Client-side pre-validation, toast before POST |
| `423 Locked` (frozen ws) | Detect status code, toast "Workspace is frozen" |
| Component crash | `ErrorBoundary` per page, renders shadcn `Alert` + reload button |

## 9. Build & Toolchain

**Vite config:**
```ts
build: { outDir: '../static/dist', emptyOutDir: true }
server: {
  port: 5173,
  proxy: {
    '/ingest': 'http://localhost:9621',
    '/query':  'http://localhost:9621',
    '/jobs':   'http://localhost:9621',
    '/graph':  'http://localhost:9621',
    '/workspace':  'http://localhost:9621',
    '/workspaces': 'http://localhost:9621',
    '/files':  'http://localhost:9621',
    '/uploads':'http://localhost:9621',
    '/output': 'http://localhost:9621',
    '/config': 'http://localhost:9621',
  }
}
```

**Dev workflow:**
```bash
# Terminal 1
uvicorn server.app:app --host 0.0.0.0 --port 9621 --reload

# Terminal 2
cd rag-anything/server/frontend
bun run dev   # http://localhost:5173
```

**Key dependencies:**

| Package | Purpose |
|---------|---------|
| `react` + `react-dom` ^18 | UI framework |
| `react-router-dom` ^6 | Client-side routing |
| `@tanstack/react-query` ^5 | Server state |
| `zustand` ^4 | UI state |
| `react-force-graph-2d` ^1 | Knowledge graph |
| `react-pdf` ^9 | PDF viewer |
| `react-markdown` + `rehype-*` + `remark-math` | Markdown + KaTeX |
| `sonner` ^1 | Toast notifications |
| `axios` ^1 | HTTP client |
| shadcn/ui components | UI primitives (installed via `bunx shadcn@latest add`) |
| `vitest` + `@testing-library/react` | Unit tests |

**`.gitignore` additions:**
```
rag-anything/server/frontend/node_modules/
rag-anything/server/static/dist/
.superpowers/
```

## 10. Testing

- **Unit tests:** `vitest` + `@testing-library/react`; cover custom hooks (`useStreamQuery`, `useJobs` polling logic) and API data-transform functions.
- **Type safety:** `strict: true` TypeScript; `src/types/` interfaces aligned to backend Pydantic models catch contract drift at compile time.
- **E2E:** Out of scope for v1.

## 11. Implementation Notes

- `POST /query/stream` uses server-sent events over HTTP. The implementation must use `fetch` + `ReadableStream` rather than `EventSource` (which only supports `GET`).
- `static/dist/` is not committed to git; it is generated at build time.
- The `app.mount("/", StaticFiles(...), name="spa")` call must be the **last** statement in `app.py` after all API routes are registered, or FastAPI will route API calls to the static handler first.
- Legacy `/graph/{ws}/html` endpoint (pyvis iframe) remains in `app.py` but is not surfaced in the new frontend. It can be removed in a future cleanup PR.
- **`page_num` availability:** Before implementing citation jump, verify that `POST /query/stream` (or the underlying LightRAG retrieval) returns `page_num` in `source_nodes`. If the field is absent, the plan must include a backend task to surface it. Citation chips must degrade gracefully (filename-only) if `page_num` is `null`.
- **Global job diff logic:** `AppShell`'s job-diff ref must handle the case where the component remounts (e.g., HMR in dev) without re-firing stale failure toasts. Store the last-seen job snapshot in Zustand (persisted across remounts) rather than a plain `useRef`.
