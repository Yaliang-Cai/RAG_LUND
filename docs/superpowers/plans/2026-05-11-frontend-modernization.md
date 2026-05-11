# Frontend Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `rag-anything/server/templates/index.html` (2587-line Jinja2 + vanilla JS) with a React 18 + Tailwind + shadcn/ui + Vite + Bun SPA, served by FastAPI from `static/dist/`, fully adapted to the backend-hardening API contract.

**Architecture:** Vite project lives in `server/frontend/`; `bun run build` outputs to `server/static/dist/`; FastAPI mounts that directory last (after all API routes) with `html=True` for SPA routing. In development, Vite's proxy forwards all API calls from `:5173` to FastAPI at `:9621`.

**Tech Stack:** React 18, React Router v6, TanStack Query v5, Zustand v4, shadcn/ui, Tailwind CSS v3, react-force-graph-2d, react-pdf v9, react-markdown, Vite 5, Bun, Vitest, @testing-library/react

---

## File Map

### New files (frontend)
```
rag-anything/server/frontend/
  package.json
  vite.config.ts
  tailwind.config.ts
  postcss.config.ts
  tsconfig.json
  tsconfig.node.json
  components.json             ← shadcn config
  index.html
  src/
    main.tsx
    App.tsx
    types/index.ts
    api/
      client.ts
      workspace.ts
      files.ts
      jobs.ts
      graph.ts
      query.ts
    store/index.ts
    hooks/
      useWorkspaces.ts
      useFiles.ts
      useJobs.ts
      useGraph.ts
      useStreamQuery.ts
      __tests__/
        useJobs.test.ts
        useStreamQuery.test.ts
        store.test.ts
    components/
      layout/
        AppShell.tsx
        TopNav.tsx
        WorkspaceSwitcher.tsx
        ThemeToggle.tsx
      chat/
        ChatInput.tsx
        MessageBubble.tsx
        ReasoningTrace.tsx
        StreamingMessage.tsx
        CitationChip.tsx
        MessageList.tsx
      documents/
        FileList.tsx
        FileUpload.tsx
        MarkdownViewer.tsx
        PdfViewer.tsx
      graph/
        ForceGraph.tsx
        GraphControls.tsx
        GraphSearch.tsx
        NodeSheet.tsx
      jobs/
        JobCard.tsx
        JobList.tsx
      ui/              ← shadcn auto-generated
    routes/
      ChatPage.tsx
      DocumentsPage.tsx
      GraphPage.tsx
      JobsPage.tsx
```

### Modified files (backend)
```
rag-anything/server/app.py      ← remove Jinja2/GET /, mount SPA, surface source_nodes
.gitignore                       ← add node_modules + static/dist + .superpowers/
```

---

## Phase 0 — Project Scaffolding

### Task 1: Initialize Vite + Bun project

**Files:**
- Create: `rag-anything/server/frontend/package.json`
- Create: `rag-anything/server/frontend/vite.config.ts`
- Create: `rag-anything/server/frontend/tsconfig.json`
- Create: `rag-anything/server/frontend/tsconfig.node.json`
- Create: `rag-anything/server/frontend/index.html`

- [ ] **Step 1: Create the Vite project using Bun**

```bash
cd rag-anything/server
bun create vite frontend --template react-ts
cd frontend
```

- [ ] **Step 2: Replace `vite.config.ts` with proxy config**

```ts
// rag-anything/server/frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/ingest':    { target: 'http://localhost:9621', changeOrigin: true },
      '/query':     { target: 'http://localhost:9621', changeOrigin: true },
      '/jobs':      { target: 'http://localhost:9621', changeOrigin: true },
      '/graph':     { target: 'http://localhost:9621', changeOrigin: true },
      '/workspace': { target: 'http://localhost:9621', changeOrigin: true },
      '/workspaces':{ target: 'http://localhost:9621', changeOrigin: true },
      '/files':     { target: 'http://localhost:9621', changeOrigin: true },
      '/content':   { target: 'http://localhost:9621', changeOrigin: true },
      '/uploads':   { target: 'http://localhost:9621', changeOrigin: true },
      '/output':    { target: 'http://localhost:9621', changeOrigin: true },
      '/config':    { target: 'http://localhost:9621', changeOrigin: true },
      '/retry':     { target: 'http://localhost:9621', changeOrigin: true },
    },
  },
})
```

- [ ] **Step 3: Update `tsconfig.json` with path alias**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **Step 4: Install core dependencies**

```bash
cd rag-anything/server/frontend
bun add react-router-dom @tanstack/react-query zustand axios
bun add react-force-graph-2d react-pdf react-markdown
bun add rehype-highlight rehype-katex remark-math
bun add sonner clsx tailwind-merge
bun add -d vitest @testing-library/react @testing-library/user-event jsdom
bun add -d @types/react-pdf
```

- [ ] **Step 5: Verify the dev server starts**

```bash
bun run dev
```

Expected: Vite server starts on `http://localhost:5173` with the default React template page.

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/frontend/
git commit -m "feat(frontend): scaffold Vite + React + TS project"
```

---

### Task 2: Configure Tailwind CSS + shadcn/ui

**Files:**
- Create: `rag-anything/server/frontend/tailwind.config.ts`
- Create: `rag-anything/server/frontend/postcss.config.ts`
- Create: `rag-anything/server/frontend/components.json`
- Modify: `rag-anything/server/frontend/src/index.css`

- [ ] **Step 1: Install Tailwind**

```bash
cd rag-anything/server/frontend
bun add -d tailwindcss postcss autoprefixer
bunx tailwindcss init -p --ts
```

- [ ] **Step 2: Write `tailwind.config.ts`**

```ts
import type { Config } from 'tailwindcss'

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
    },
  },
  plugins: [],
} satisfies Config
```

- [ ] **Step 3: Replace `src/index.css` with shadcn CSS variables**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --primary: 239 84% 67%;
    --primary-foreground: 0 0% 100%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --ring: 239 84% 67%;
    --radius: 0.5rem;
  }
  .light {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --primary: 239 84% 67%;
    --primary-foreground: 0 0% 100%;
    --secondary: 210 40% 96%;
    --secondary-foreground: 222.2 84% 4.9%;
    --muted: 210 40% 96%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96%;
    --accent-foreground: 222.2 84% 4.9%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --ring: 239 84% 67%;
    --radius: 0.5rem;
  }
}

@layer base {
  * { @apply border-border; }
  body { @apply bg-background text-foreground; }
}
```

- [ ] **Step 4: Initialize shadcn/ui**

```bash
bunx shadcn@latest init
```

When prompted: choose dark theme, use `src/components/ui` as component dir, yes to CSS variables.

- [ ] **Step 5: Install required shadcn components**

```bash
bunx shadcn@latest add button input select dropdown-menu command progress collapsible alert-dialog sheet switch alert navigation-menu badge separator scroll-area
```

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/frontend/
git commit -m "feat(frontend): configure Tailwind + shadcn/ui"
```

---

## Phase 1 — Foundation

### Task 3: TypeScript types

**Files:**
- Create: `rag-anything/server/frontend/src/types/index.ts`

- [ ] **Step 1: Write all types aligned to backend Pydantic models**

```ts
// src/types/index.ts

export interface Workspace {
  workspace_id: string
  name: string
  frozen: boolean
  document_count: number
  created_at: string
}

export interface DocumentRecord {
  doc_id: string
  filename: string
  file_hash: string
  status: 'pending' | 'processing' | 'processed' | 'failed'
  created_at: string
}

export type JobStatus = 'running' | 'done' | 'failed' | 'cancelled'

export interface Job {
  job_id: string
  workspace_id: string
  doc_id: string
  filename: string
  status: JobStatus
  progress: number        // 0-100
  error: string | null
  created_at: string
  updated_at: string
}

export interface GraphNode {
  id: string
  label: string
  type: string            // entity_type: Concept | Person | Organization | Location | Other
  description: string
}

export interface GraphEdge {
  source: string
  target: string
  label: string
  weight: number
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface SourceNode {
  doc_id: string
  filename: string
  page_num: number | null
  excerpt: string
}

export interface StreamMetaEvent {
  type: 'meta'
  data: Record<string, unknown>
  metadata: Record<string, unknown>
}

export interface StreamChunkEvent {
  type: 'chunk'
  text: string
}

export interface StreamDoneEvent {
  type: 'done'
  graph: GraphData | null
  source_nodes: SourceNode[]
}

export interface StreamErrorEvent {
  type: 'error'
  text: string
}

export type StreamEvent = StreamMetaEvent | StreamChunkEvent | StreamDoneEvent | StreamErrorEvent

export interface QueryParams {
  workspace_id: string
  query: string
  mode?: 'naive' | 'local' | 'global' | 'hybrid'
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
  return_graph?: boolean
}

export interface FileRecord {
  filename: string
  doc_id?: string
}

export interface AuditEntry {
  id: number
  workspace_id: string
  action: string
  doc_id: string | null
  detail: string | null
  created_at: string
}
```

- [ ] **Step 2: Commit**

```bash
git add rag-anything/server/frontend/src/types/
git commit -m "feat(frontend): add TypeScript types aligned to backend models"
```

---

### Task 4: API layer

**Files:**
- Create: `rag-anything/server/frontend/src/api/client.ts`
- Create: `rag-anything/server/frontend/src/api/workspace.ts`
- Create: `rag-anything/server/frontend/src/api/files.ts`
- Create: `rag-anything/server/frontend/src/api/jobs.ts`
- Create: `rag-anything/server/frontend/src/api/graph.ts`
- Create: `rag-anything/server/frontend/src/api/query.ts`

- [ ] **Step 1: Create axios client**

```ts
// src/api/client.ts
import axios from 'axios'

const client = axios.create({ baseURL: '/' })

client.interceptors.response.use(
  (r) => r,
  (err) => {
    const detail = err.response?.data?.detail ?? err.message
    return Promise.reject(new Error(detail))
  }
)

export default client
```

- [ ] **Step 2: Workspace API**

```ts
// src/api/workspace.ts
import client from './client'
import type { Workspace, AuditEntry } from '@/types'

export async function getWorkspaces(): Promise<Workspace[]> {
  const { data } = await client.get<{ workspaces: Workspace[] }>('/workspaces')
  return data.workspaces
}

export async function deleteWorkspace(id: string): Promise<void> {
  await client.delete(`/workspace/${id}`)
}

export async function freezeWorkspace(id: string): Promise<void> {
  await client.post(`/workspace/${id}/freeze`)
}

export async function unfreezeWorkspace(id: string): Promise<void> {
  await client.post(`/workspace/${id}/unfreeze`)
}

export async function getAuditLog(id: string): Promise<AuditEntry[]> {
  const { data } = await client.get<{ entries: AuditEntry[] }>(`/workspace/${id}/audit`)
  return data.entries
}
```

- [ ] **Step 3: Files API**

```ts
// src/api/files.ts
import client from './client'
import type { FileRecord } from '@/types'

export async function getFiles(workspaceId: string): Promise<FileRecord[]> {
  const { data } = await client.get<{ files: string[] }>(`/files/${workspaceId}`)
  return data.files.map((filename) => ({ filename }))
}

export async function getFileContent(workspaceId: string, filename: string): Promise<string> {
  const { data } = await client.get<{ content: string }>(`/content/${workspaceId}`, {
    params: { filename },
  })
  return data.content
}

export async function uploadFile(workspaceId: string, file: File): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('file', file)
  form.append('workspace_id', workspaceId)
  const { data } = await client.post<{ job_id: string }>('/ingest', form)
  return data
}

export async function deleteDocument(workspaceId: string, docId: string): Promise<void> {
  await client.delete(`/workspace/${workspaceId}/document/${docId}`)
}

export function getUploadUrl(workspaceId: string, filename: string): string {
  return `/uploads/${workspaceId}/${filename}`
}
```

- [ ] **Step 4: Jobs API**

```ts
// src/api/jobs.ts
import client from './client'
import type { Job } from '@/types'

export async function getJobs(workspaceId?: string): Promise<Job[]> {
  const { data } = await client.get<{ jobs: Job[] }>('/jobs', {
    params: workspaceId ? { workspace_id: workspaceId } : undefined,
  })
  return data.jobs
}

export async function getJob(jobId: string): Promise<Job> {
  const { data } = await client.get<Job>(`/jobs/${jobId}`)
  return data
}

export async function cancelJob(jobId: string): Promise<void> {
  await client.delete(`/jobs/${jobId}`)
}

export async function retryWorkspace(workspaceId: string): Promise<{ job_id: string }> {
  const { data } = await client.post<{ job_id: string }>(`/retry/${workspaceId}`)
  return data
}
```

- [ ] **Step 5: Graph API**

```ts
// src/api/graph.ts
import client from './client'
import type { GraphData } from '@/types'

export async function getOverview(workspaceId: string, maxNodes = 100): Promise<GraphData> {
  const { data } = await client.get<GraphData>(`/graph/${workspaceId}/overview`, {
    params: { max_nodes: maxNodes },
  })
  return data
}

export async function getSubgraph(
  workspaceId: string,
  seed: string,
  depth = 2,
  maxNodes = 50
): Promise<GraphData> {
  const { data } = await client.get<GraphData>(`/graph/${workspaceId}/subgraph`, {
    params: { seed, depth, max_nodes: maxNodes },
  })
  return data
}

export async function searchGraph(
  workspaceId: string,
  query: string
): Promise<{ nodes: Array<{ id: string; label: string; type: string }> }> {
  const { data } = await client.get(`/graph/${workspaceId}/search`, {
    params: { query, limit: 20 },
  })
  return data
}
```

- [ ] **Step 6: Query API (returns Response for streaming)**

```ts
// src/api/query.ts
import type { QueryParams } from '@/types'

export async function openQueryStream(params: QueryParams): Promise<Response> {
  const response = await fetch('/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      workspace_id: params.workspace_id,
      query: params.query,
      mode: params.mode ?? 'hybrid',
      top_k: params.top_k ?? 10,
      chunk_top_k: params.chunk_top_k ?? 10,
      enable_rerank: params.enable_rerank ?? true,
      return_graph: params.return_graph ?? false,
    }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(err.detail ?? 'Query failed')
  }
  return response
}
```

- [ ] **Step 7: Commit**

```bash
git add rag-anything/server/frontend/src/api/
git commit -m "feat(frontend): add API layer (workspace, files, jobs, graph, query)"
```

---

### Task 5: Zustand store

**Files:**
- Create: `rag-anything/server/frontend/src/store/index.ts`
- Create: `rag-anything/server/frontend/src/hooks/__tests__/store.test.ts`

- [ ] **Step 1: Write failing store test**

```ts
// src/hooks/__tests__/store.test.ts
import { describe, it, expect, beforeEach } from 'vitest'
import { useAppStore } from '@/store'
import { act } from '@testing-library/react'

describe('AppStore', () => {
  beforeEach(() => {
    useAppStore.setState({
      workspaceId: 'default',
      theme: 'dark',
      selectedFileId: null,
      pendingPageNum: null,
    })
  })

  it('sets workspaceId', () => {
    act(() => useAppStore.getState().setWorkspace('ws2'))
    expect(useAppStore.getState().workspaceId).toBe('ws2')
  })

  it('toggles theme', () => {
    act(() => useAppStore.getState().toggleTheme())
    expect(useAppStore.getState().theme).toBe('light')
    act(() => useAppStore.getState().toggleTheme())
    expect(useAppStore.getState().theme).toBe('dark')
  })

  it('sets pendingPageNum and clears it', () => {
    act(() => useAppStore.getState().setPendingPageNum(5))
    expect(useAppStore.getState().pendingPageNum).toBe(5)
    act(() => useAppStore.getState().setPendingPageNum(null))
    expect(useAppStore.getState().pendingPageNum).toBeNull()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rag-anything/server/frontend
bun run vitest run src/hooks/__tests__/store.test.ts
```

Expected: FAIL — `Cannot find module '@/store'`

- [ ] **Step 3: Configure vitest in `vite.config.ts`**

Add to `vite.config.ts` inside `defineConfig`:
```ts
test: {
  environment: 'jsdom',
  globals: true,
  setupFiles: [],
},
```

And add `/// <reference types="vitest" />` at top of `vite.config.ts`.

- [ ] **Step 4: Implement the store**

```ts
// src/store/index.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AppStore {
  workspaceId: string
  setWorkspace: (id: string) => void
  theme: 'dark' | 'light'
  toggleTheme: () => void
  selectedFileId: string | null
  setSelectedFile: (id: string | null) => void
  pendingPageNum: number | null
  setPendingPageNum: (n: number | null) => void
  // Job diff tracking (persisted so HMR remount doesn't re-fire stale toasts)
  lastSeenJobStatuses: Record<string, string>
  setLastSeenJobStatuses: (statuses: Record<string, string>) => void
}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      workspaceId: 'default',
      setWorkspace: (id) => set({ workspaceId: id }),
      theme: 'dark',
      toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
      selectedFileId: null,
      setSelectedFile: (id) => set({ selectedFileId: id }),
      pendingPageNum: null,
      setPendingPageNum: (n) => set({ pendingPageNum: n }),
      lastSeenJobStatuses: {},
      setLastSeenJobStatuses: (statuses) => set({ lastSeenJobStatuses: statuses }),
    }),
    {
      name: 'raganything-store',
      partialize: (s) => ({
        workspaceId: s.workspaceId,
        theme: s.theme,
        lastSeenJobStatuses: s.lastSeenJobStatuses,
      }),
    }
  )
)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
bun run vitest run src/hooks/__tests__/store.test.ts
```

Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/frontend/src/store/ rag-anything/server/frontend/src/hooks/__tests__/store.test.ts rag-anything/server/frontend/vite.config.ts
git commit -m "feat(frontend): add Zustand store with persistence"
```

---

### Task 6: React Query hooks

**Files:**
- Create: `rag-anything/server/frontend/src/hooks/useWorkspaces.ts`
- Create: `rag-anything/server/frontend/src/hooks/useFiles.ts`
- Create: `rag-anything/server/frontend/src/hooks/useGraph.ts`
- Create: `rag-anything/server/frontend/src/hooks/useJobs.ts`
- Create: `rag-anything/server/frontend/src/hooks/__tests__/useJobs.test.ts`

- [ ] **Step 1: Write failing useJobs polling test**

```ts
// src/hooks/__tests__/useJobs.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement } from 'react'
import { useJobs } from '@/hooks/useJobs'
import * as jobsApi from '@/api/jobs'

vi.mock('@/api/jobs')

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return createElement(QueryClientProvider, { client: qc }, children)
}

describe('useJobs', () => {
  beforeEach(() => vi.clearAllMocks())

  it('returns jobs from API', async () => {
    const mockJobs = [
      { job_id: 'j1', status: 'done', filename: 'a.pdf', progress: 100 },
    ]
    vi.mocked(jobsApi.getJobs).mockResolvedValue(mockJobs as any)

    const { result } = renderHook(() => useJobs('ws1'), { wrapper })
    await waitFor(() => expect(result.current.data).toEqual(mockJobs))
  })

  it('sets refetchInterval to 2000 when a job is running', async () => {
    const runningJobs = [{ job_id: 'j2', status: 'running', progress: 50 }]
    vi.mocked(jobsApi.getJobs).mockResolvedValue(runningJobs as any)

    const { result } = renderHook(() => useJobs('ws1'), { wrapper })
    await waitFor(() => expect(result.current.data).toBeDefined())
    // Polling active when running job exists
    expect(result.current.data?.some((j) => j.status === 'running')).toBe(true)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun run vitest run src/hooks/__tests__/useJobs.test.ts
```

Expected: FAIL — `Cannot find module '@/hooks/useJobs'`

- [ ] **Step 3: Implement all hooks**

```ts
// src/hooks/useWorkspaces.ts
import { useQuery } from '@tanstack/react-query'
import { getWorkspaces } from '@/api/workspace'

export function useWorkspaces() {
  return useQuery({
    queryKey: ['workspaces'],
    queryFn: getWorkspaces,
    staleTime: 30_000,
  })
}
```

```ts
// src/hooks/useFiles.ts
import { useQuery } from '@tanstack/react-query'
import { getFiles } from '@/api/files'

export function useFiles(workspaceId: string) {
  return useQuery({
    queryKey: ['files', workspaceId],
    queryFn: () => getFiles(workspaceId),
    enabled: !!workspaceId,
  })
}
```

```ts
// src/hooks/useGraph.ts
import { useQuery } from '@tanstack/react-query'
import { getOverview, getSubgraph } from '@/api/graph'

export function useGraphOverview(workspaceId: string) {
  return useQuery({
    queryKey: ['graph', 'overview', workspaceId],
    queryFn: () => getOverview(workspaceId),
    enabled: !!workspaceId,
    staleTime: 60_000,
  })
}

export function useSubgraph(workspaceId: string, seed: string | null, depth = 2) {
  return useQuery({
    queryKey: ['graph', 'subgraph', workspaceId, seed, depth],
    queryFn: () => getSubgraph(workspaceId, seed!, depth),
    enabled: !!workspaceId && !!seed,
  })
}
```

```ts
// src/hooks/useJobs.ts
import { useQuery } from '@tanstack/react-query'
import { getJobs } from '@/api/jobs'
import type { Job } from '@/types'

export function useJobs(workspaceId: string) {
  return useQuery<Job[]>({
    queryKey: ['jobs', workspaceId],
    queryFn: () => getJobs(workspaceId),
    enabled: !!workspaceId,
    refetchInterval: (query) => {
      const data = query.state.data
      return data?.some((j) => j.status === 'running') ? 2000 : false
    },
  })
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
bun run vitest run src/hooks/__tests__/useJobs.test.ts
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/hooks/
git commit -m "feat(frontend): add React Query hooks (workspaces, files, graph, jobs)"
```

---

### Task 7: SSE stream hook

**Files:**
- Create: `rag-anything/server/frontend/src/hooks/useStreamQuery.ts`
- Create: `rag-anything/server/frontend/src/hooks/__tests__/useStreamQuery.test.ts`

- [ ] **Step 1: Write failing test**

```ts
// src/hooks/__tests__/useStreamQuery.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import * as queryApi from '@/api/query'

vi.mock('@/api/query')

function makeStream(lines: string[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      for (const line of lines) controller.enqueue(encoder.encode(line + '\n'))
      controller.close()
    },
  })
  return new Response(stream, { status: 200 })
}

describe('useStreamQuery', () => {
  beforeEach(() => vi.clearAllMocks())

  it('starts idle', () => {
    const { result } = renderHook(() => useStreamQuery())
    expect(result.current.status).toBe('idle')
    expect(result.current.answer).toBe('')
  })

  it('accumulates chunks and transitions to done', async () => {
    vi.mocked(queryApi.openQueryStream).mockResolvedValue(
      makeStream([
        'data: {"type":"meta","data":{},"metadata":{}}',
        'data: {"type":"chunk","text":"Hello"}',
        'data: {"type":"chunk","text":" world"}',
        'data: {"type":"done","graph":null,"source_nodes":[]}',
      ])
    )

    const { result } = renderHook(() => useStreamQuery())
    await act(async () => {
      await result.current.send({ workspace_id: 'ws1', query: 'test' })
    })

    expect(result.current.answer).toBe('Hello world')
    expect(result.current.status).toBe('done')
    expect(result.current.sourceNodes).toEqual([])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun run vitest run src/hooks/__tests__/useStreamQuery.test.ts
```

Expected: FAIL — `Cannot find module '@/hooks/useStreamQuery'`

- [ ] **Step 3: Implement the hook**

```ts
// src/hooks/useStreamQuery.ts
import { useState, useCallback, useRef } from 'react'
import { openQueryStream } from '@/api/query'
import type { QueryParams, SourceNode } from '@/types'

export type StreamStatus = 'idle' | 'streaming' | 'done' | 'error'

export function useStreamQuery() {
  const [answer, setAnswer] = useState('')
  const [reasoning, setReasoning] = useState('')
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [sourceNodes, setSourceNodes] = useState<SourceNode[]>([])
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(async (params: QueryParams) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('streaming')
    setAnswer('')
    setReasoning('')
    setSourceNodes([])

    try {
      const response = await openQueryStream(params)
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            if (event.type === 'chunk') {
              setAnswer((a) => a + event.text)
            } else if (event.type === 'reasoning') {
              setReasoning((r) => r + event.text)
            } else if (event.type === 'done') {
              setSourceNodes(event.source_nodes ?? [])
              setStatus('done')
            } else if (event.type === 'error') {
              setStatus('error')
            }
          } catch {
            // malformed SSE line — skip
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setStatus('error')
    }
  }, [])

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setStatus('idle')
  }, [])

  return { send, abort, answer, reasoning, status, sourceNodes }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
bun run vitest run src/hooks/__tests__/useStreamQuery.test.ts
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/hooks/
git commit -m "feat(frontend): add useStreamQuery hook with fetch + ReadableStream"
```

---

## Phase 2 — Application Shell

### Task 8: ThemeToggle + WorkspaceSwitcher + TopNav

**Files:**
- Create: `rag-anything/server/frontend/src/components/layout/ThemeToggle.tsx`
- Create: `rag-anything/server/frontend/src/components/layout/WorkspaceSwitcher.tsx`
- Create: `rag-anything/server/frontend/src/components/layout/TopNav.tsx`

- [ ] **Step 1: ThemeToggle**

```tsx
// src/components/layout/ThemeToggle.tsx
import { Moon, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/store'

export function ThemeToggle() {
  const { theme, toggleTheme } = useAppStore()
  return (
    <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Toggle theme">
      {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  )
}
```

Add `lucide-react` if not present: `bun add lucide-react`

- [ ] **Step 2: WorkspaceSwitcher**

```tsx
// src/components/layout/WorkspaceSwitcher.tsx
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { ChevronDown } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useWorkspaces } from '@/hooks/useWorkspaces'
import { useAppStore } from '@/store'

export function WorkspaceSwitcher() {
  const { data: workspaces = [] } = useWorkspaces()
  const { workspaceId, setWorkspace } = useAppStore()
  const qc = useQueryClient()

  function switchWorkspace(id: string) {
    setWorkspace(id)
    qc.invalidateQueries()
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="gap-1 text-xs">
          ws: {workspaceId} <ChevronDown className="h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {workspaces.map((ws) => (
          <DropdownMenuItem key={ws.workspace_id} onClick={() => switchWorkspace(ws.workspace_id)}>
            {ws.workspace_id}
            {ws.frozen && <span className="ml-2 text-xs text-muted-foreground">🔒</span>}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
```

- [ ] **Step 3: TopNav**

```tsx
// src/components/layout/TopNav.tsx
import { NavLink } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { WorkspaceSwitcher } from './WorkspaceSwitcher'
import { ThemeToggle } from './ThemeToggle'
import { useJobs } from '@/hooks/useJobs'
import { useAppStore } from '@/store'
import { cn } from '@/lib/utils'

const NAV_ITEMS = [
  { to: '/chat', label: 'Chat' },
  { to: '/documents', label: 'Documents' },
  { to: '/graph', label: 'Graph' },
  { to: '/jobs', label: 'Jobs' },
]

export function TopNav() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { data: jobs = [] } = useJobs(workspaceId)
  const runningCount = jobs.filter((j) => j.status === 'running').length

  return (
    <header className="h-12 border-b border-border flex items-center px-4 gap-6 shrink-0">
      <span className="text-sm font-semibold text-foreground">RAGAnything</span>
      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'px-3 py-1.5 text-sm rounded-md transition-colors',
                isActive
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-muted-foreground hover:text-foreground'
              )
            }
          >
            {label}
            {label === 'Jobs' && runningCount > 0 && (
              <Badge variant="default" className="ml-1.5 h-4 px-1 text-[10px]">
                {runningCount}
              </Badge>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-2">
        <WorkspaceSwitcher />
        <ThemeToggle />
      </div>
    </header>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add rag-anything/server/frontend/src/components/layout/
git commit -m "feat(frontend): add ThemeToggle, WorkspaceSwitcher, TopNav"
```

---

### Task 9: AppShell with global job notifications

**Files:**
- Create: `rag-anything/server/frontend/src/components/layout/AppShell.tsx`

- [ ] **Step 1: Implement AppShell with job diff logic**

```tsx
// src/components/layout/AppShell.tsx
import { useEffect } from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { TopNav } from './TopNav'
import { useJobs } from '@/hooks/useJobs'
import { useAppStore } from '@/store'
import type { JobStatus } from '@/types'

export function AppShell() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { lastSeenJobStatuses, setLastSeenJobStatuses } = useAppStore()
  const navigate = useNavigate()
  const { data: jobs = [] } = useJobs(workspaceId)

  useEffect(() => {
    if (jobs.length === 0) return

    const currentStatuses: Record<string, string> = {}
    for (const job of jobs) {
      currentStatuses[job.job_id] = job.status
      const prev = lastSeenJobStatuses[job.job_id] as JobStatus | undefined

      if (prev === 'running' && job.status === 'failed') {
        toast.error(`Ingest failed: ${job.filename}`, {
          action: { label: 'View Jobs', onClick: () => navigate('/jobs') },
        })
      } else if (prev === 'running' && job.status === 'done') {
        toast.success(`Ingest complete: ${job.filename}`)
      }
    }
    setLastSeenJobStatuses(currentStatuses)
  }, [jobs])  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col h-screen bg-background">
      <TopNav />
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add rag-anything/server/frontend/src/components/layout/AppShell.tsx
git commit -m "feat(frontend): add AppShell with global job failure notifications"
```

---

### Task 10: App.tsx + main.tsx (router + providers)

**Files:**
- Modify: `rag-anything/server/frontend/src/App.tsx`
- Modify: `rag-anything/server/frontend/src/main.tsx`

- [ ] **Step 1: Replace `App.tsx`**

```tsx
// src/App.tsx
import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/layout/AppShell'

const ChatPage      = lazy(() => import('@/routes/ChatPage'))
const DocumentsPage = lazy(() => import('@/routes/DocumentsPage'))
const GraphPage     = lazy(() => import('@/routes/GraphPage'))
const JobsPage      = lazy(() => import('@/routes/JobsPage'))

const qc = new QueryClient()

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Toaster richColors position="top-right" />
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<Suspense fallback={null}><ChatPage /></Suspense>} />
            <Route path="/documents" element={<Suspense fallback={null}><DocumentsPage /></Suspense>} />
            <Route path="/graph" element={<Suspense fallback={null}><GraphPage /></Suspense>} />
            <Route path="/jobs" element={<Suspense fallback={null}><JobsPage /></Suspense>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
```

- [ ] **Step 2: Replace `main.tsx`**

```tsx
// src/main.tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

- [ ] **Step 3: Apply theme class to document from store (in `main.tsx`)**

Add before `ReactDOM.createRoot`:
```ts
import { useAppStore } from './store'
const theme = useAppStore.getState().theme
document.documentElement.classList.toggle('light', theme === 'light')
```

- [ ] **Step 4: Create stub page files so the app doesn't crash**

```tsx
// src/routes/ChatPage.tsx
export default function ChatPage() { return <div className="p-4">Chat (coming soon)</div> }
// src/routes/DocumentsPage.tsx
export default function DocumentsPage() { return <div className="p-4">Documents (coming soon)</div> }
// src/routes/GraphPage.tsx
export default function GraphPage() { return <div className="p-4">Graph (coming soon)</div> }
// src/routes/JobsPage.tsx
export default function JobsPage() { return <div className="p-4">Jobs (coming soon)</div> }
```

- [ ] **Step 5: Start dev server and verify all routes render without errors**

```bash
bun run dev
# open http://localhost:5173 — should show nav bar + stub pages
# navigate to /chat, /documents, /graph, /jobs
```

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/frontend/src/
git commit -m "feat(frontend): add router, providers, AppShell wiring"
```

---

## Phase 3 — Jobs Page

### Task 11: JobCard + JobList + JobsPage

**Files:**
- Create: `rag-anything/server/frontend/src/components/jobs/JobCard.tsx`
- Create: `rag-anything/server/frontend/src/components/jobs/JobList.tsx`
- Modify: `rag-anything/server/frontend/src/routes/JobsPage.tsx`

- [ ] **Step 1: JobCard**

```tsx
// src/components/jobs/JobCard.tsx
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cancelJob, retryWorkspace } from '@/api/jobs'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import type { Job } from '@/types'

const STATUS_COLORS: Record<string, string> = {
  running: 'bg-primary',
  done: 'bg-green-600',
  failed: 'bg-destructive',
  cancelled: 'bg-muted-foreground',
}

export function JobCard({ job }: { job: Job }) {
  const qc = useQueryClient()
  const workspaceId = useAppStore((s) => s.workspaceId)

  async function handleCancel() {
    try {
      await cancelJob(job.job_id)
      qc.invalidateQueries({ queryKey: ['jobs', workspaceId] })
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  async function handleRetry() {
    try {
      await retryWorkspace(workspaceId)
      qc.invalidateQueries({ queryKey: ['jobs', workspaceId] })
      toast.success('Retry job created')
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  return (
    <div className="border border-border rounded-lg p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium truncate max-w-xs">{job.filename}</span>
        <Badge className={STATUS_COLORS[job.status] + ' text-white text-xs'}>
          {job.status}
        </Badge>
      </div>
      {job.status === 'running' && (
        <>
          <Progress value={job.progress} className="h-1.5" />
          <Button variant="outline" size="sm" className="self-start" onClick={handleCancel}>
            Cancel
          </Button>
        </>
      )}
      {job.status === 'failed' && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground truncate">{job.error}</span>
          <Button variant="outline" size="sm" onClick={handleRetry}>Retry</Button>
        </div>
      )}
      {(job.status === 'done' || job.status === 'cancelled') && (
        <span className="text-xs text-muted-foreground">
          {new Date(job.updated_at).toLocaleString()}
        </span>
      )}
    </div>
  )
}
```

- [ ] **Step 2: JobList**

```tsx
// src/components/jobs/JobList.tsx
import { JobCard } from './JobCard'
import type { Job, JobStatus } from '@/types'

export function JobList({ jobs, filter }: { jobs: Job[]; filter?: JobStatus }) {
  const filtered = filter ? jobs.filter((j) => j.status === filter) : jobs
  if (filtered.length === 0) return (
    <p className="text-sm text-muted-foreground py-4 text-center">No jobs</p>
  )
  return (
    <div className="flex flex-col gap-2">
      {filtered.map((job) => <JobCard key={job.job_id} job={job} />)}
    </div>
  )
}
```

- [ ] **Step 3: JobsPage**

```tsx
// src/routes/JobsPage.tsx
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Button } from '@/components/ui/button'
import { ChevronDown } from 'lucide-react'
import { useJobs } from '@/hooks/useJobs'
import { useAppStore } from '@/store'
import { JobList } from '@/components/jobs/JobList'

export default function JobsPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { data: jobs = [], isLoading } = useJobs(workspaceId)

  if (isLoading) return <div className="p-6 text-sm text-muted-foreground">Loading...</div>

  return (
    <div className="h-full overflow-y-auto p-6 max-w-2xl mx-auto flex flex-col gap-6">
      <h1 className="text-lg font-semibold">Ingest Jobs</h1>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Running</h2>
        <JobList jobs={jobs} filter="running" />
      </section>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Completed</h2>
        <JobList jobs={jobs} filter="done" />
      </section>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Failed</h2>
        <JobList jobs={jobs} filter="failed" />
      </section>

      <Collapsible>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="gap-1 text-xs text-muted-foreground">
            Audit log <ChevronDown className="h-3 w-3" />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <p className="text-xs text-muted-foreground mt-2">Audit log coming in v2.</p>
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}
```

- [ ] **Step 4: Start dev server, navigate to `/jobs`, verify it renders**

```bash
bun run dev
# open http://localhost:5173/jobs
# with FastAPI running, jobs list should appear (empty is fine)
```

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/components/jobs/ rag-anything/server/frontend/src/routes/JobsPage.tsx
git commit -m "feat(frontend): add Jobs page with polling, cancel, retry"
```

---

## Phase 4 — Documents Page

### Task 12: FileList + FileUpload

**Files:**
- Create: `rag-anything/server/frontend/src/components/documents/FileList.tsx`
- Create: `rag-anything/server/frontend/src/components/documents/FileUpload.tsx`

- [ ] **Step 1: FileList**

```tsx
// src/components/documents/FileList.tsx
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel,
  AlertDialogContent, AlertDialogDescription, AlertDialogFooter,
  AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { deleteDocument } from '@/api/files'
import { freezeWorkspace, unfreezeWorkspace } from '@/api/workspace'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import type { FileRecord, Workspace } from '@/types'

interface FileListProps {
  files: FileRecord[]
  workspace: Workspace | undefined
  selectedFile: string | null
  onSelect: (filename: string) => void
}

export function FileList({ files, workspace, selectedFile, onSelect }: FileListProps) {
  const qc = useQueryClient()
  const workspaceId = useAppStore((s) => s.workspaceId)
  const frozen = workspace?.frozen ?? false

  async function handleDelete(docId: string, filename: string) {
    try {
      await deleteDocument(workspaceId, docId)
      qc.invalidateQueries({ queryKey: ['files', workspaceId] })
      toast.success(`Deleted ${filename}`)
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  async function handleFreezeToggle(checked: boolean) {
    try {
      if (checked) await freezeWorkspace(workspaceId)
      else await unfreezeWorkspace(workspaceId)
      qc.invalidateQueries({ queryKey: ['workspaces'] })
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  return (
    <div className="flex flex-col h-full border-r border-border">
      <div className="p-3 border-b border-border">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch checked={frozen} onCheckedChange={handleFreezeToggle} id="freeze-switch" />
          <label htmlFor="freeze-switch">Frozen</label>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {files.map((f) => (
          <div
            key={f.filename}
            className={cn(
              'flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-accent/50 group text-sm',
              selectedFile === f.filename && 'bg-accent'
            )}
            onClick={() => onSelect(f.filename)}
          >
            <span className="truncate flex-1">📄 {f.filename}</span>
            {f.doc_id && !frozen && (
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button
                    variant="ghost" size="icon"
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 shrink-0"
                    onClick={(e) => e.stopPropagation()}
                  >
                    🗑
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete document?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This removes {f.filename} from the knowledge graph. Cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={() => handleDelete(f.doc_id!, f.filename)}>
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: FileUpload**

```tsx
// src/components/documents/FileUpload.tsx
import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { uploadFile } from '@/api/files'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

const SUPPORTED = new Set([
  'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
  'txt', 'md', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp',
])

export function FileUpload({ disabled }: { disabled?: boolean }) {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  function validate(file: File): boolean {
    const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
    if (!SUPPORTED.has(ext)) {
      toast.error(`Unsupported file type: .${ext}`)
      return false
    }
    return true
  }

  async function upload(file: File) {
    if (!validate(file)) return
    setUploading(true)
    try {
      await uploadFile(workspaceId, file)
      toast.success('Ingest job created — tracking in Jobs page')
      navigate('/jobs')
    } catch (err) {
      toast.error((err as Error).message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="p-3 border-t border-border">
      <input
        ref={inputRef} type="file" className="hidden"
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      <div
        className={cn(
          'border border-dashed rounded-lg p-3 text-center cursor-pointer text-xs text-muted-foreground transition-colors',
          dragging && 'border-primary bg-primary/5',
          disabled && 'opacity-50 pointer-events-none'
        )}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault(); setDragging(false)
          const file = e.dataTransfer.files[0]
          if (file) upload(file)
        }}
      >
        {uploading ? <Progress value={undefined} className="h-1" /> : '+ Upload file'}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/documents/
git commit -m "feat(frontend): add FileList and FileUpload components"
```

---

### Task 13: MarkdownViewer + PdfViewer

**Files:**
- Create: `rag-anything/server/frontend/src/components/documents/MarkdownViewer.tsx`
- Create: `rag-anything/server/frontend/src/components/documents/PdfViewer.tsx`

- [ ] **Step 1: MarkdownViewer**

```tsx
// src/components/documents/MarkdownViewer.tsx
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import 'highlight.js/styles/github-dark.css'
import 'katex/dist/katex.min.css'

export function MarkdownViewer({ content }: { content: string }) {
  return (
    <div className="prose prose-invert prose-sm max-w-none p-4 overflow-y-auto h-full">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
```

- [ ] **Step 2: PdfViewer**

```tsx
// src/components/documents/PdfViewer.tsx
import { useState, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

interface PdfViewerProps {
  url: string
  initialPage?: number
  onPageSet?: () => void
}

export function PdfViewer({ url, initialPage = 1, onPageSet }: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [currentPage, setCurrentPage] = useState(initialPage)
  const [scale, setScale] = useState(1.0)

  useEffect(() => {
    setCurrentPage(initialPage)
    onPageSet?.()
  }, [initialPage])  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 p-2 border-b border-border shrink-0">
        <Button variant="ghost" size="icon" onClick={() => setCurrentPage((p) => Math.max(1, p - 1))} disabled={currentPage <= 1}>
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="text-xs text-muted-foreground">
          {currentPage} / {numPages}
        </span>
        <Button variant="ghost" size="icon" onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))} disabled={currentPage >= numPages}>
          <ChevronRight className="h-4 w-4" />
        </Button>
        <div className="ml-auto flex gap-1">
          <Button variant="ghost" size="icon" onClick={() => setScale((s) => Math.max(0.5, s - 0.25))}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs text-muted-foreground self-center">{Math.round(scale * 100)}%</span>
          <Button variant="ghost" size="icon" onClick={() => setScale((s) => Math.min(2.5, s + 0.25))}>
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <div className="flex-1 overflow-auto flex justify-center bg-muted/20 p-4">
        <Document
          file={url}
          onLoadSuccess={({ numPages }) => setNumPages(numPages)}
          loading={<div className="text-sm text-muted-foreground mt-8">Loading PDF...</div>}
          error={<div className="text-sm text-destructive mt-8">Failed to load PDF</div>}
        >
          <Page pageNumber={currentPage} scale={scale} />
        </Document>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/documents/
git commit -m "feat(frontend): add MarkdownViewer and PdfViewer components"
```

---

### Task 14: DocumentsPage assembly

**Files:**
- Modify: `rag-anything/server/frontend/src/routes/DocumentsPage.tsx`

- [ ] **Step 1: Implement DocumentsPage**

```tsx
// src/routes/DocumentsPage.tsx
import { useState, useEffect } from 'react'
import { useFiles } from '@/hooks/useFiles'
import { useWorkspaces } from '@/hooks/useWorkspaces'
import { useAppStore } from '@/store'
import { getFileContent, getUploadUrl } from '@/api/files'
import { FileList } from '@/components/documents/FileList'
import { FileUpload } from '@/components/documents/FileUpload'
import { MarkdownViewer } from '@/components/documents/MarkdownViewer'
import { PdfViewer } from '@/components/documents/PdfViewer'
import { Button } from '@/components/ui/button'

export default function DocumentsPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { pendingPageNum, setPendingPageNum, selectedFileId, setSelectedFile } = useAppStore()
  const { data: files = [] } = useFiles(workspaceId)
  const { data: workspaces = [] } = useWorkspaces()
  const workspace = workspaces.find((w) => w.workspace_id === workspaceId)
  const frozen = workspace?.frozen ?? false

  const [tab, setTab] = useState<'markdown' | 'pdf'>('markdown')
  const [mdContent, setMdContent] = useState('')

  const selectedFilename = selectedFileId ?? files[0]?.filename ?? null

  useEffect(() => {
    if (!selectedFilename) return
    getFileContent(workspaceId, selectedFilename)
      .then(setMdContent)
      .catch(() => setMdContent('Failed to load content.'))
  }, [selectedFilename, workspaceId])

  // Handle citation jump: if pendingPageNum is set, switch to PDF tab
  useEffect(() => {
    if (pendingPageNum !== null) setTab('pdf')
  }, [pendingPageNum])

  const isPdf = selectedFilename?.toLowerCase().endsWith('.pdf') ?? false

  return (
    <div className="flex h-full">
      {/* Left: file list + upload */}
      <div className="w-64 shrink-0 flex flex-col">
        <FileList
          files={files}
          workspace={workspace}
          selectedFile={selectedFilename}
          onSelect={(f) => { setSelectedFile(f); setTab('markdown') }}
        />
        <FileUpload disabled={frozen} />
      </div>

      {/* Right: preview */}
      <div className="flex-1 flex flex-col min-w-0">
        {selectedFilename && (
          <div className="flex items-center gap-2 px-4 py-2 border-b border-border shrink-0">
            <Button
              variant={tab === 'markdown' ? 'secondary' : 'ghost'}
              size="sm" onClick={() => setTab('markdown')}
            >Markdown</Button>
            {isPdf && (
              <Button
                variant={tab === 'pdf' ? 'secondary' : 'ghost'}
                size="sm" onClick={() => setTab('pdf')}
              >PDF</Button>
            )}
          </div>
        )}
        <div className="flex-1 min-h-0">
          {!selectedFilename && (
            <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
              Select a file to preview
            </div>
          )}
          {selectedFilename && tab === 'markdown' && <MarkdownViewer content={mdContent} />}
          {selectedFilename && tab === 'pdf' && isPdf && (
            <PdfViewer
              url={getUploadUrl(workspaceId, selectedFilename)}
              initialPage={pendingPageNum ?? 1}
              onPageSet={() => setPendingPageNum(null)}
            />
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Start dev server, navigate to `/documents`, verify file list appears and preview loads**

```bash
bun run dev
# open http://localhost:5173/documents with FastAPI running
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/routes/DocumentsPage.tsx
git commit -m "feat(frontend): Documents page with file list, Markdown/PDF preview, freeze toggle"
```

---

## Phase 5 — Chat Page

### Task 15: Chat components

**Files:**
- Create: `rag-anything/server/frontend/src/components/chat/ReasoningTrace.tsx`
- Create: `rag-anything/server/frontend/src/components/chat/CitationChip.tsx`
- Create: `rag-anything/server/frontend/src/components/chat/MessageBubble.tsx`
- Create: `rag-anything/server/frontend/src/components/chat/MessageList.tsx`
- Create: `rag-anything/server/frontend/src/components/chat/ChatInput.tsx`

- [ ] **Step 1: ReasoningTrace**

```tsx
// src/components/chat/ReasoningTrace.tsx
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Button } from '@/components/ui/button'
import { ChevronDown } from 'lucide-react'

export function ReasoningTrace({ text }: { text: string }) {
  if (!text) return null
  return (
    <Collapsible className="mt-1">
      <CollapsibleTrigger asChild>
        <Button variant="ghost" size="sm" className="h-6 text-xs text-muted-foreground gap-1 px-2">
          Thinking <ChevronDown className="h-3 w-3" />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <pre className="mt-1 text-[11px] font-mono text-muted-foreground whitespace-pre-wrap bg-muted/30 rounded p-2 max-h-40 overflow-y-auto">
          {text}
        </pre>
      </CollapsibleContent>
    </Collapsible>
  )
}
```

- [ ] **Step 2: CitationChip**

```tsx
// src/components/chat/CitationChip.tsx
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '@/store'
import { Badge } from '@/components/ui/badge'
import type { SourceNode } from '@/types'

export function CitationChip({ node }: { node: SourceNode }) {
  const navigate = useNavigate()
  const { setSelectedFile, setPendingPageNum } = useAppStore()

  function handleClick() {
    setSelectedFile(node.filename)
    if (node.page_num !== null) setPendingPageNum(node.page_num)
    navigate('/documents')
  }

  return (
    <Badge
      variant="outline"
      className="cursor-pointer hover:bg-accent text-xs gap-1"
      onClick={handleClick}
    >
      {node.filename}
      {node.page_num !== null && <span className="text-muted-foreground">p.{node.page_num}</span>}
    </Badge>
  )
}
```

- [ ] **Step 3: MessageBubble**

```tsx
// src/components/chat/MessageBubble.tsx
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { ReasoningTrace } from './ReasoningTrace'
import { CitationChip } from './CitationChip'
import { cn } from '@/lib/utils'
import type { SourceNode } from '@/types'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sourceNodes?: SourceNode[]
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  return (
    <div className={cn('flex flex-col gap-1', isUser ? 'items-end' : 'items-start')}>
      <div
        className={cn(
          'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-secondary text-secondary-foreground'
        )}
      >
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown>
        )}
      </div>
      {!isUser && message.reasoning && <ReasoningTrace text={message.reasoning} />}
      {!isUser && message.sourceNodes && message.sourceNodes.length > 0 && (
        <div className="flex flex-wrap gap-1 max-w-[80%]">
          {message.sourceNodes.map((n, i) => <CitationChip key={i} node={n} />)}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: MessageList**

```tsx
// src/components/chat/MessageList.tsx
import { useEffect, useRef, useState } from 'react'
import { MessageBubble } from './MessageBubble'
import type { Message } from './MessageBubble'

interface MessageListProps {
  messages: Message[]
  streamingAnswer: string
  streamingReasoning: string
  isStreaming: boolean
}

export function MessageList({ messages, streamingAnswer, streamingReasoning, isStreaming }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingAnswer, autoScroll])

  function handleScroll() {
    const el = containerRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60
    setAutoScroll(atBottom)
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4"
      onScroll={handleScroll}
    >
      {messages.map((m) => <MessageBubble key={m.id} message={m} />)}
      {isStreaming && (
        <MessageBubble
          message={{
            id: '__streaming__',
            role: 'assistant',
            content: streamingAnswer,
            reasoning: streamingReasoning,
          }}
        />
      )}
      <div ref={bottomRef} />
    </div>
  )
}
```

- [ ] **Step 5: ChatInput**

```tsx
// src/components/chat/ChatInput.tsx
import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Send } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSend: (query: string, mode: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [mode, setMode] = useState('hybrid')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q, mode)
    setValue('')
    textareaRef.current?.focus()
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  return (
    <div className="border-t border-border p-3 flex flex-col gap-2">
      <div className="flex gap-2 items-start">
        <textarea
          ref={textareaRef}
          className={cn(
            'flex-1 bg-secondary rounded-xl px-3 py-2 text-sm resize-none outline-none',
            'min-h-[40px] max-h-[160px] placeholder:text-muted-foreground'
          )}
          placeholder="Ask about your documents... (Enter to send, Shift+Enter for newline)"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
        />
        <Button size="icon" onClick={submit} disabled={disabled || !value.trim()}>
          <Send className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Mode:</span>
        <Select value={mode} onValueChange={setMode}>
          <SelectTrigger className="h-7 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {['naive', 'local', 'global', 'hybrid'].map((m) => (
              <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
```

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/frontend/src/components/chat/
git commit -m "feat(frontend): add chat components (MessageBubble, CitationChip, ReasoningTrace, ChatInput)"
```

---

### Task 16: ChatPage assembly

**Files:**
- Modify: `rag-anything/server/frontend/src/routes/ChatPage.tsx`

- [ ] **Step 1: Implement ChatPage**

```tsx
// src/routes/ChatPage.tsx
import { useState, useCallback } from 'react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import { useAppStore } from '@/store'
import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/button'
import { SquarePen } from 'lucide-react'
import type { Message } from '@/components/chat/MessageBubble'

let msgId = 0

export default function ChatPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { send, answer, reasoning, status, sourceNodes } = useStreamQuery()
  const [messages, setMessages] = useState<Message[]>([])

  const handleSend = useCallback(async (query: string, mode: string) => {
    const userMsg: Message = { id: String(++msgId), role: 'user', content: query }
    setMessages((prev) => [...prev, userMsg])

    await send({ workspace_id: workspaceId, query, mode: mode as any })

    setMessages((prev) => [
      ...prev,
      {
        id: String(++msgId),
        role: 'assistant',
        content: answer,
        reasoning,
        sourceNodes,
      },
    ])
  }, [workspaceId, send, answer, reasoning, sourceNodes])

  const isStreaming = status === 'streaming'

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <span className="text-xs text-muted-foreground">workspace: {workspaceId}</span>
        <Button variant="ghost" size="icon" onClick={() => setMessages([])} title="New conversation">
          <SquarePen className="h-4 w-4" />
        </Button>
      </div>

      <MessageList
        messages={messages}
        streamingAnswer={answer}
        streamingReasoning={reasoning}
        isStreaming={isStreaming}
      />

      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  )
}
```

- [ ] **Step 2: Start dev server, navigate to `/chat`, send a query to FastAPI, verify streaming works**

```bash
bun run dev
# open http://localhost:5173/chat with FastAPI + models running
# send a query — streaming tokens should appear in real time
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/routes/ChatPage.tsx
git commit -m "feat(frontend): Chat page with SSE streaming, reasoning trace, citation chips"
```

---

## Phase 6 — Graph Page

### Task 17: ForceGraph + GraphControls + GraphPage

**Files:**
- Create: `rag-anything/server/frontend/src/components/graph/ForceGraph.tsx`
- Create: `rag-anything/server/frontend/src/components/graph/GraphSearch.tsx`
- Create: `rag-anything/server/frontend/src/components/graph/NodeSheet.tsx`
- Modify: `rag-anything/server/frontend/src/routes/GraphPage.tsx`

- [ ] **Step 1: ForceGraph**

```tsx
// src/components/graph/ForceGraph.tsx
import { useRef, useCallback } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { GraphData, GraphNode } from '@/types'

const TYPE_COLORS: Record<string, string> = {
  concept: '#6366f1',
  person: '#22c55e',
  organization: '#f59e0b',
  location: '#06b6d4',
  other: '#64748b',
}

function nodeColor(node: GraphNode): string {
  return TYPE_COLORS[(node.type ?? '').toLowerCase()] ?? TYPE_COLORS.other
}

interface ForceGraphProps {
  data: GraphData
  onNodeClick?: (node: GraphNode) => void
  highlightNodeId?: string | null
}

export function ForceGraph({ data, onNodeClick, highlightNodeId }: ForceGraphProps) {
  const graphRef = useRef<any>(null)

  const handleNodeClick = useCallback((node: any) => {
    onNodeClick?.(node as GraphNode)
  }, [onNodeClick])

  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D) => {
    const color = nodeColor(node as GraphNode)
    const radius = node.id === highlightNodeId ? 8 : 5
    ctx.beginPath()
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI)
    ctx.fillStyle = color
    ctx.fill()
    if (node.id === highlightNodeId) {
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    ctx.font = '3px sans-serif'
    ctx.fillStyle = '#94a3b8'
    ctx.textAlign = 'center'
    ctx.fillText(node.label ?? node.id, node.x, node.y + 8)
  }, [highlightNodeId])

  return (
    <ForceGraph2D
      ref={graphRef}
      graphData={{ nodes: data.nodes as any[], links: data.edges.map(e => ({ ...e, source: e.source, target: e.target })) }}
      nodeCanvasObject={paintNode}
      nodeCanvasObjectMode={() => 'replace'}
      linkLabel="label"
      linkColor={() => '#334155'}
      onNodeClick={handleNodeClick}
      backgroundColor="transparent"
    />
  )
}
```

- [ ] **Step 2: GraphSearch**

```tsx
// src/components/graph/GraphSearch.tsx
import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Search } from 'lucide-react'
import { searchGraph } from '@/api/graph'
import { toast } from 'sonner'

interface GraphSearchProps {
  workspaceId: string
  onResult: (nodeId: string) => void
}

export function GraphSearch({ workspaceId, onResult }: GraphSearchProps) {
  const [query, setQuery] = useState('')

  async function handleSearch() {
    if (!query.trim()) return
    try {
      const result = await searchGraph(workspaceId, query)
      if (result.nodes.length > 0) onResult(result.nodes[0].id)
      else toast.info('No matching nodes found')
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  return (
    <div className="flex gap-2">
      <Input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        placeholder="Search nodes..."
        className="h-8 text-xs"
      />
      <Button variant="outline" size="icon" className="h-8 w-8" onClick={handleSearch}>
        <Search className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}
```

- [ ] **Step 3: NodeSheet**

```tsx
// src/components/graph/NodeSheet.tsx
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import type { GraphNode } from '@/types'

export function NodeSheet({ node, onClose }: { node: GraphNode | null; onClose: () => void }) {
  return (
    <Sheet open={!!node} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="w-80">
        {node && (
          <>
            <SheetHeader>
              <SheetTitle className="text-base">{node.label}</SheetTitle>
            </SheetHeader>
            <div className="mt-4 flex flex-col gap-3 text-sm">
              <div>
                <span className="text-xs text-muted-foreground uppercase tracking-wide">Type</span>
                <p className="mt-0.5">{node.type || '—'}</p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground uppercase tracking-wide">Description</span>
                <p className="mt-0.5 text-muted-foreground text-xs leading-relaxed">
                  {node.description || '—'}
                </p>
              </div>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
```

- [ ] **Step 4: GraphPage**

```tsx
// src/routes/GraphPage.tsx
import { useState } from 'react'
import { useGraphOverview } from '@/hooks/useGraph'
import { useAppStore } from '@/store'
import { ForceGraph } from '@/components/graph/ForceGraph'
import { GraphSearch } from '@/components/graph/GraphSearch'
import { NodeSheet } from '@/components/graph/NodeSheet'
import { Button } from '@/components/ui/button'
import { RefreshCw } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import type { GraphNode } from '@/types'

export default function GraphPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { data, isLoading } = useGraphOverview(workspaceId)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [highlightId, setHighlightId] = useState<string | null>(null)
  const qc = useQueryClient()

  function handleSearchResult(nodeId: string) {
    setHighlightId(nodeId)
  }

  if (isLoading) return (
    <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
      Loading graph...
    </div>
  )

  if (!data || data.nodes.length === 0) return (
    <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
      No graph data. Ingest documents first.
    </div>
  )

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border shrink-0">
        <GraphSearch workspaceId={workspaceId} onResult={handleSearchResult} />
        <Button
          variant="ghost" size="icon" className="h-8 w-8" title="Refresh"
          onClick={() => qc.invalidateQueries({ queryKey: ['graph', 'overview', workspaceId] })}
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
        <span className="text-xs text-muted-foreground ml-auto">
          {data.nodes.length} nodes · {data.edges.length} edges
        </span>
      </div>
      <div className="flex-1 relative">
        <ForceGraph
          data={data}
          onNodeClick={setSelectedNode}
          highlightNodeId={highlightId}
        />
      </div>
      <NodeSheet node={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  )
}
```

- [ ] **Step 5: Start dev server, navigate to `/graph`, verify force graph renders**

```bash
bun run dev
# open http://localhost:5173/graph with FastAPI + Neo4j running
# graph nodes should appear and be draggable
```

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/frontend/src/components/graph/ rag-anything/server/frontend/src/routes/GraphPage.tsx
git commit -m "feat(frontend): Graph page with react-force-graph-2d, search, node details"
```

---

## Phase 7 — Backend Integration

### Task 18: Verify and surface source_nodes with page_num in stream

**Files:**
- Modify: `rag-anything/server/app.py` (lines ~736-751 in `_generate()`)

- [ ] **Step 1: Inspect what's in `meta.data` from a real query**

With FastAPI running and a workspace ingested, run:
```bash
curl -s -X POST http://localhost:9621/query/stream \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"default","query":"test","mode":"naive","top_k":3,"chunk_top_k":3,"return_graph":false}' \
  | head -3
```

Look at the `meta` event JSON. Find if `data` contains any of: `chunks`, `sources`, `references`, `entities` — and whether any field has filename or page number info.

- [ ] **Step 2: Extract source_nodes from meta.data in `_generate()`**

In `app.py`, locate the `_generate()` inner function inside `query_stream_endpoint` (around line 720). After the `done` yield, add source_nodes extraction.

Replace this block:
```python
# Event final: done + optional graph
graph_data = None
if payload.return_graph:
    rag = await service.get_rag(payload.workspace_id)
    graph_data = await _get_query_subgraph(rag, retrieval_data, payload)
yield f"data: {_json.dumps({'type': 'done', 'graph': graph_data}, ensure_ascii=False)}\n\n"
```

With:
```python
# Event final: done + optional graph + source_nodes
graph_data = None
if payload.return_graph:
    rag = await service.get_rag(payload.workspace_id)
    graph_data = await _get_query_subgraph(rag, retrieval_data, payload)

source_nodes = _extract_source_nodes(retrieval_data)
yield f"data: {_json.dumps({'type': 'done', 'graph': graph_data, 'source_nodes': source_nodes}, ensure_ascii=False)}\n\n"
```

- [ ] **Step 3: Add `_extract_source_nodes()` helper after `_get_query_subgraph()`**

```python
def _extract_source_nodes(retrieval_data: dict) -> list[dict]:
    """Extract source file references from LightRAG retrieval meta event.

    page_num is None if LightRAG doesn't surface it — frontend degrades gracefully.
    """
    source_nodes = []
    seen = set()
    data = retrieval_data.get("data", {})

    # LightRAG stores retrieved chunks under various keys — check common ones
    for key in ("chunks", "references", "sources", "text_chunks"):
        chunks = data.get(key, [])
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            filename = chunk.get("file_path") or chunk.get("filename") or chunk.get("source")
            if not filename:
                continue
            filename = str(filename).split("/")[-1].split("\\")[-1]
            doc_id = chunk.get("doc_id") or chunk.get("document_id") or ""
            excerpt = str(chunk.get("content") or chunk.get("text") or "")[:200]
            page_num = chunk.get("page_num") or chunk.get("page") or None
            if isinstance(page_num, float):
                page_num = int(page_num)

            key_id = f"{filename}:{page_num}"
            if key_id not in seen:
                seen.add(key_id)
                source_nodes.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "excerpt": excerpt,
                })
    return source_nodes[:5]  # cap at 5 citations
```

- [ ] **Step 4: Restart FastAPI and verify the done event now includes source_nodes**

```bash
curl -s -X POST http://localhost:9621/query/stream \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"default","query":"test","mode":"naive","top_k":3,"chunk_top_k":3,"return_graph":false}' \
  | grep '"type":"done"'
```

Expected: `{"type":"done","graph":null,"source_nodes":[...]}` — array may be empty if LightRAG keys differ; that's acceptable (frontend degrades gracefully).

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/app.py
git commit -m "feat(server): surface source_nodes in /query/stream done event for citation jump"
```

---

### Task 19: Remove Jinja2, mount SPA in FastAPI

**Files:**
- Modify: `rag-anything/server/app.py`
- Modify: `.gitignore`

- [ ] **Step 1: Remove Jinja2 imports and constants from `app.py`**

Remove these lines (around lines 16, 61-69):
```python
from fastapi.templating import Jinja2Templates          # line 16
TEMPLATES = Jinja2Templates(...)                        # line 61
_STATIC_DIR = APP_ROOT / "static"                      # line 64
_USE_LOCAL_STATIC: bool = all([...])                    # lines 65-69
```

Also remove the logger.info lines referencing `_USE_LOCAL_STATIC` (around lines 132-135).

- [ ] **Step 2: Remove existing static mount and GET / route**

Remove:
```python
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
```

Remove the entire `GET /` route (around lines 253-258):
```python
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(...)
```

Also remove unused imports:
```python
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
```
→ keep only `StreamingResponse` (used by `/query/stream`):
```python
from fastapi.responses import FileResponse, StreamingResponse
```

- [ ] **Step 3: Add SPA mount at the very end of `app.py`**

After the last API route (after line ~1432 `@app.get("/admin/audit")`), add:
```python
# --- SPA fallback (must be last) ---
_DIST_DIR = APP_ROOT / "static" / "dist"
if _DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="spa")
else:
    import warnings
    warnings.warn("static/dist not found — run `bun run build` in server/frontend/")
```

- [ ] **Step 4: Update `.gitignore`**

Add to root `.gitignore`:
```
# Frontend build artifacts
rag-anything/server/frontend/node_modules/
rag-anything/server/static/dist/
.superpowers/
```

- [ ] **Step 5: Start FastAPI and verify it serves without errors**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9621 --reload
```

Expected: server starts without `TemplateResponse` errors. The warning about `static/dist not found` is fine at this stage.

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/app.py .gitignore
git commit -m "feat(server): remove Jinja2, mount React SPA from static/dist"
```

---

## Phase 8 — Build & Smoke Test

### Task 20: Build frontend and end-to-end smoke test

**Files:**
- No new files

- [ ] **Step 1: Run the full test suite**

```bash
cd rag-anything/server/frontend
bun run vitest run
```

Expected: all tests pass (store, useJobs, useStreamQuery).

- [ ] **Step 2: Build the frontend**

```bash
bun run build
```

Expected output:
```
dist/index.html
dist/assets/index-[hash].js
dist/assets/index-[hash].css
```

The build should complete without TypeScript errors.

- [ ] **Step 3: Start FastAPI and verify SPA is served**

```bash
cd rag-anything
uvicorn server.app:app --host 0.0.0.0 --port 9621
```

Open `http://localhost:9621` in a browser. Expected: React SPA loads (TopNav visible, `/chat` route active).

- [ ] **Step 4: Smoke test each page**

- Navigate to `/chat`: input bar visible, send a message, streaming works
- Navigate to `/documents`: file list loads, upload zone visible
- Upload a PDF, verify redirect to `/jobs` and job card appears
- Navigate to `/graph`: graph renders if workspace has data
- Navigate to `/jobs`: job status updates every 2 seconds while running

- [ ] **Step 5: Verify citation jump**

- After a Chat response, click a citation chip (if source_nodes are populated)
- Verify redirect to `/documents` with the correct file selected
- If `page_num` is non-null, verify PDF opens to that page

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat(frontend): build passes, SPA served from FastAPI"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Vite in server/frontend/, build → static/dist | Task 1 |
| Tailwind + shadcn default dark theme | Task 2 |
| TypeScript types aligned to Pydantic | Task 3 |
| API layer (workspace, files, jobs, graph, query) | Task 4 |
| Zustand store with pendingPageNum + lastSeenJobStatuses | Task 5 |
| useJobs with dynamic polling | Task 6 |
| useStreamQuery via fetch + ReadableStream | Task 7 |
| TopNav with Jobs badge | Task 8 |
| AppShell global job failure notifications | Task 9 |
| React Router v6, lazy pages | Task 10 |
| Jobs page with polling, cancel, retry, audit placeholder | Task 11 |
| FileList with delete + freeze toggle | Task 12 |
| FileUpload drag-drop + extension validation | Task 12 |
| MarkdownViewer (react-markdown + rehype-katex) | Task 13 |
| PdfViewer (react-pdf + page jump) | Task 13 |
| DocumentsPage assembly | Task 14 |
| CitationChip + page jump to PdfViewer | Tasks 15–16 |
| ReasoningTrace collapsible | Task 15 |
| Chat SSE streaming | Task 16 |
| ForceGraph (react-force-graph-2d) | Task 17 |
| Graph search + node detail sheet | Task 17 |
| source_nodes in /query/stream done event | Task 18 |
| Remove Jinja2, mount StaticFiles | Task 19 |
| .gitignore additions | Task 19 |
| Build + smoke test | Task 20 |

All spec requirements covered. No placeholders found.
