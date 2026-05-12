# Agentic Mode + Trace Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ppr`/`auto`/`agentic` modes to the chat UI and show a structured trace panel below assistant messages.

**Architecture:** Backend `stream_query` gets an agentic branch that calls `query_with_trace` (non-streaming) and emits trace data in the SSE meta event. Frontend captures the meta event's `metadata` field, determines trace type from query mode, and renders a collapsible `AgenticTrace` panel under each assistant message.

**Tech Stack:** FastAPI (backend SSE), React + TypeScript + Tailwind CSS (frontend), vitest + @testing-library/react (frontend tests), pytest + unittest.mock (backend tests)

---

## File Map

| File | Change |
|---|---|
| `server/app.py` | Add `profile` to `QueryRequest`; pass to `stream_query` |
| `raganything/services/local_rag.py` | Add `profile` param to `stream_query`; agentic/auto+profile branch |
| `tests/test_stream_query_agentic.py` | New: backend unit tests |
| `src/types/index.ts` | Extend mode union, add `profile?` to `QueryParams`, extend `Message` |
| `src/api/query.ts` | Pass `profile` conditionally |
| `src/hooks/useStreamQuery.ts` | Capture + expose `metadata` from meta event |
| `src/hooks/__tests__/useStreamQuery.test.ts` | Add metadata test |
| `src/components/chat/ChatInput.tsx` | 7 modes + profile selector |
| `src/components/chat/AgenticTrace.tsx` | New component |
| `src/components/chat/MessageBubble.tsx` | Add `traceType`/`traceMetadata` props |
| `src/routes/ChatPage.tsx` | Wire profile, traceType, metadata |

---

## Task 1: Backend — QueryRequest.profile + stream_query signature

**Files:**
- Modify: `rag-anything/server/app.py` (QueryRequest model, ~line 214)
- Modify: `rag-anything/raganything/services/local_rag.py` (stream_query signature, ~line 2356)

- [ ] **Step 1: Add `profile` to `QueryRequest`**

In `server/app.py`, find `class QueryRequest(BaseModel):` and add after the last existing field:

```python
profile: Optional[str] = None  # auto mode only; None = LLM classifier decides
```

- [ ] **Step 2: Pass profile to stream_query in the endpoint**

In `query_stream_endpoint` (~line 701), find the `service.stream_query(...)` call and add `profile=payload.profile` to the kwargs:

```python
async for event in service.stream_query(
    payload.workspace_id, payload.query,
    mode=payload.mode, top_k=top_k,
    chunk_top_k=chunk_top_k, enable_rerank=payload.enable_rerank,
    multi_hop_depth=payload.multi_hop_depth,
    ppr_damping=payload.ppr_damping,
    ppr_top_k=payload.ppr_top_k,
    passage_node_weight=payload.passage_node_weight,
    recognition_top_k=payload.recognition_top_k,
    ppr_synonym_weight_mode=payload.ppr_synonym_weight_mode,
    qdrant_retrieval_mode=payload.qdrant_retrieval_mode,
    profile=payload.profile,
):
```

- [ ] **Step 3: Add `profile` to `stream_query` signature**

In `raganything/services/local_rag.py`, find `async def stream_query(` (~line 2356) and add `profile: str | None = None` as the last parameter:

```python
async def stream_query(
    self,
    workspace_id: str,
    query: str,
    mode: str = DEFAULT_QUERY_MODE,
    top_k: int = DEFAULT_TOP_K,
    chunk_top_k: int = DEFAULT_CHUNK_TOP_K,
    enable_rerank: bool = DEFAULT_ENABLE_RERANK,
    multi_hop_depth: int | None = None,
    ppr_damping: float | None = None,
    ppr_top_k: int | None = None,
    passage_node_weight: float | None = None,
    recognition_top_k: int | None = None,
    ppr_synonym_weight_mode: str | None = None,
    exclude_synonym_edges: bool | None = None,
    qdrant_retrieval_mode: str | None = None,
    profile: str | None = None,
):
```

- [ ] **Step 4: Verify no import errors**

```bash
cd rag-anything
python -c "from server.app import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/app.py rag-anything/raganything/services/local_rag.py
git commit -m "feat(backend): add profile param to QueryRequest and stream_query"
```

---

## Task 2: Backend — stream_query agentic/auto+profile branch

**Files:**
- Modify: `rag-anything/raganything/services/local_rag.py` (inside stream_query body, after the signature)
- Create: `rag-anything/tests/test_stream_query_agentic.py`

- [ ] **Step 1: Write failing tests**

Create `rag-anything/tests/test_stream_query_agentic.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


def _make_service():
    from raganything.services.local_rag import LocalRagService, LocalRagSettings
    settings = MagicMock(spec=LocalRagSettings)
    settings.working_dir = "/tmp/test"
    settings.output_dir = "/tmp/out"
    service = LocalRagService.__new__(LocalRagService)
    service.settings = settings
    service.logger = MagicMock()
    return service


@pytest.mark.asyncio
async def test_stream_query_agentic_yields_meta_with_trace():
    service = _make_service()
    fake_result = {
        "answer": "72.3% top-1 accuracy",
        "confidence": 0.91,
        "grounded": True,
        "trace": {
            "profile": "precise",
            "router_cache_hit": False,
            "retrieve_cycles_used": 2,
            "check_cycles_used": 1,
            "rewrite_history": [],
            "sub_questions": None,
        },
    }
    service.query_with_trace = AsyncMock(return_value=fake_result)

    events = []
    async for event in service.stream_query("ws1", "test query", mode="agentic"):
        events.append(event)

    assert events[0]["type"] == "meta"
    trace = events[0]["metadata"]["agentic_trace"]
    assert trace["confidence"] == 0.91
    assert trace["grounded"] is True
    assert trace["profile"] == "precise"
    assert trace["retrieve_cycles_used"] == 2

    assert events[1]["type"] == "chunk"
    assert events[1]["text"] == "72.3% top-1 accuracy"

    assert len(events) == 2


@pytest.mark.asyncio
async def test_stream_query_auto_with_profile_uses_query_with_trace():
    service = _make_service()
    fake_result = {
        "answer": "answer text",
        "confidence": 0.85,
        "grounded": True,
        "trace": {
            "routing": {
                "profile": "multihop",
                "confidence": 0.85,
                "paths_activated": ["ppr", "naive"],
                "chunks_after_rrf": 20,
                "chunks_after_rerank": 10,
                "chunks_after_threshold": 8,
                "latency_per_path": {"ppr": 0.4, "naive": 0.1},
            }
        },
    }
    service.query_with_trace = AsyncMock(return_value=fake_result)

    events = []
    async for event in service.stream_query("ws1", "test", mode="auto", profile="multihop"):
        events.append(event)

    service.query_with_trace.assert_awaited_once()
    call_kwargs = service.query_with_trace.call_args
    assert call_kwargs.kwargs.get("profile") == "multihop" or "multihop" in str(call_kwargs)

    assert events[0]["type"] == "meta"
    assert "routing_trace" in events[0]["metadata"]
    assert events[1]["type"] == "chunk"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd rag-anything
python -m pytest tests/test_stream_query_agentic.py -v 2>&1 | head -30
```

Expected: FAILED (agentic branch doesn't exist yet)

- [ ] **Step 3: Implement agentic/auto+profile branch in stream_query**

In `raganything/services/local_rag.py`, inside `stream_query`, add this block at the very start of the `try:` block (before the `from lightrag import QueryParam` line):

```python
        # ── Non-streaming branch: agentic mode or auto+profile override ──
        if mode == "agentic" or (mode == "auto" and profile):
            try:
                extra: dict = {}
                if profile:
                    extra["profile"] = profile
                result = await self.query_with_trace(
                    workspace_id, query,
                    mode=mode,
                    return_trace=True,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    enable_rerank=enable_rerank,
                    **extra,
                )
                trace = result.get("trace", {})
                if mode == "agentic":
                    meta_payload = {
                        "agentic_trace": {
                            "confidence": result.get("confidence"),
                            "grounded": result.get("grounded"),
                            "profile": trace.get("profile"),
                            "router_cache_hit": trace.get("router_cache_hit", False),
                            "retrieve_cycles_used": trace.get("retrieve_cycles_used", 0),
                            "check_cycles_used": trace.get("check_cycles_used", 0),
                            "rewrite_history": trace.get("rewrite_history", []),
                            "sub_questions": trace.get("sub_questions"),
                        }
                    }
                else:
                    # auto + profile: routing trace
                    meta_payload = {
                        "routing_trace": trace.get("routing", trace),
                    }
                yield {"type": "meta", "data": {}, "metadata": meta_payload}
                yield {"type": "chunk", "text": result.get("answer", "")}
            except Exception as exc:
                self.logger.error("stream_query (agentic branch) error: %s", exc)
                yield {"type": "error", "text": str(exc)}
            return
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd rag-anything
python -m pytest tests/test_stream_query_agentic.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add rag-anything/raganything/services/local_rag.py rag-anything/tests/test_stream_query_agentic.py
git commit -m "feat(backend): stream_query agentic/auto+profile non-streaming branch with trace"
```

---

## Task 3: Frontend types & API

**Files:**
- Modify: `rag-anything/server/frontend/src/types/index.ts`
- Modify: `rag-anything/server/frontend/src/api/query.ts`

- [ ] **Step 1: Extend types**

Replace the contents of `src/types/index.ts`. Key changes: extended mode union, added `profile?` to `QueryParams`, added `traceType`/`traceMetadata` to `Message` (the Message interface lives in `MessageBubble.tsx` — update it there in Task 6). Add `TraceType` export here:

In `src/types/index.ts`, replace the `QueryParams` interface:

```typescript
export type TraceType = 'agentic' | 'auto' | 'ppr' | null

export interface QueryParams {
  workspace_id: string
  query: string
  mode?: 'naive' | 'local' | 'global' | 'hybrid' | 'ppr' | 'auto' | 'agentic'
  profile?: string
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
  return_graph?: boolean
}
```

Keep all other interfaces unchanged.

- [ ] **Step 2: Update API client to send profile**

Replace `src/api/query.ts`:

```typescript
import type { QueryParams } from '@/types'

export async function openQueryStream(params: QueryParams): Promise<Response> {
  const body: Record<string, unknown> = {
    workspace_id: params.workspace_id,
    query: params.query,
    mode: params.mode ?? 'hybrid',
    top_k: params.top_k ?? 10,
    chunk_top_k: params.chunk_top_k ?? 10,
    enable_rerank: params.enable_rerank ?? true,
    return_graph: params.return_graph ?? false,
  }
  if (params.mode === 'auto' && params.profile) {
    body.profile = params.profile
  }

  const response = await fetch('/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error((err as { detail?: string }).detail ?? 'Query failed')
  }
  return response
}
```

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/types/index.ts rag-anything/server/frontend/src/api/query.ts
git commit -m "feat(frontend): extend QueryParams with profile and new modes"
```

---

## Task 4: useStreamQuery — expose metadata

**Files:**
- Modify: `rag-anything/server/frontend/src/hooks/useStreamQuery.ts`
- Modify: `rag-anything/server/frontend/src/hooks/__tests__/useStreamQuery.test.ts`

- [ ] **Step 1: Add failing test for metadata capture**

In `src/hooks/__tests__/useStreamQuery.test.ts`, add a new test after the existing ones:

```typescript
  it('captures agentic_trace from meta event metadata', async () => {
    const fakeTrace = { confidence: 0.91, grounded: true, profile: 'precise',
                        retrieve_cycles_used: 2, check_cycles_used: 1 }
    vi.mocked(queryApi.openQueryStream).mockResolvedValue(
      makeStream([
        `data: {"type":"meta","data":{},"metadata":{"agentic_trace":${JSON.stringify(fakeTrace)}}}`,
        'data: {"type":"chunk","text":"72.3% accuracy"}',
        'data: {"type":"done","graph":null,"source_nodes":[]}',
      ])
    )

    const { result } = renderHook(() => useStreamQuery())
    await act(async () => {
      await result.current.send({ workspace_id: 'ws1', query: 'test', mode: 'agentic' })
    })

    expect(result.current.answer).toBe('72.3% accuracy')
    expect(result.current.metadata).toEqual({ agentic_trace: fakeTrace })
    expect(result.current.status).toBe('done')
  })
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd rag-anything/server/frontend
npx vitest run src/hooks/__tests__/useStreamQuery.test.ts 2>&1 | tail -20
```

Expected: FAIL — `result.current.metadata` is undefined

- [ ] **Step 3: Implement metadata in useStreamQuery**

Replace `src/hooks/useStreamQuery.ts`:

```typescript
import { useState, useCallback, useRef } from 'react'
import { openQueryStream } from '@/api/query'
import type { QueryParams, SourceNode } from '@/types'

export type StreamStatus = 'idle' | 'streaming' | 'done' | 'error'

export function useStreamQuery() {
  const [answer, setAnswer] = useState('')
  const [reasoning, setReasoning] = useState('')
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [sourceNodes, setSourceNodes] = useState<SourceNode[]>([])
  const [metadata, setMetadata] = useState<Record<string, unknown>>({})
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(async (params: QueryParams) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('streaming')
    setAnswer('')
    setReasoning('')
    setSourceNodes([])
    setMetadata({})

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
            if (event.type === 'meta') {
              setMetadata((event.metadata as Record<string, unknown>) ?? {})
            } else if (event.type === 'chunk') {
              setAnswer((a) => a + (event.text as string))
            } else if (event.type === 'reasoning') {
              setReasoning((r) => r + (event.text as string))
            } else if (event.type === 'done') {
              setSourceNodes((event.source_nodes as SourceNode[]) ?? [])
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

  return { send, abort, answer, reasoning, status, sourceNodes, metadata }
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd rag-anything/server/frontend
npx vitest run src/hooks/__tests__/useStreamQuery.test.ts
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/hooks/useStreamQuery.ts rag-anything/server/frontend/src/hooks/__tests__/useStreamQuery.test.ts
git commit -m "feat(frontend): expose metadata from SSE meta event in useStreamQuery"
```

---

## Task 5: ChatInput — 7 modes + profile selector

**Files:**
- Modify: `rag-anything/server/frontend/src/components/chat/ChatInput.tsx`

- [ ] **Step 1: Replace ChatInput**

```typescript
import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Send } from 'lucide-react'
import { cn } from '@/lib/utils'

const MODES = ['naive', 'local', 'global', 'hybrid', 'ppr', 'auto', 'agentic'] as const
const PROFILES = ['precise', 'local', 'multihop', 'descriptive', 'full'] as const

interface ChatInputProps {
  onSend: (query: string, mode: string, profile: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [profile, setProfile] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q, mode, profile)
    setValue('')
    textareaRef.current?.focus()
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const isAuto = mode === 'auto'

  return (
    <div className="border-t border-border p-3 flex flex-col gap-2 shrink-0">
      <div className="flex gap-2 items-start">
        <textarea
          ref={textareaRef}
          className={cn(
            'flex-1 bg-secondary rounded-xl px-3 py-2 text-sm resize-none outline-none',
            'min-h-[40px] max-h-[160px] placeholder:text-muted-foreground text-foreground'
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
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-muted-foreground">Mode:</span>
        <Select value={mode} onValueChange={(v) => { setMode(v); if (v !== 'auto') setProfile('') }}>
          <SelectTrigger className="h-7 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODES.map((m) => (
              <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        {isAuto && (
          <>
            <span className="text-xs text-muted-foreground">Profile:</span>
            <Select value={profile} onValueChange={setProfile}>
              <SelectTrigger className="h-7 w-32 text-xs">
                <SelectValue placeholder="— auto detect —" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="" className="text-xs text-muted-foreground">— auto detect —</SelectItem>
                {PROFILES.map((p) => (
                  <SelectItem key={p} value={p} className="text-xs">{p}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1 | head -20
```

Expected: no errors (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/chat/ChatInput.tsx
git commit -m "feat(frontend): ChatInput 7-mode selector + conditional auto profile picker"
```

---

## Task 6: AgenticTrace component

**Files:**
- Create: `rag-anything/server/frontend/src/components/chat/AgenticTrace.tsx`

- [ ] **Step 1: Create the component**

```typescript
import { useState } from 'react'
import type { ReactNode } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { TraceType } from '@/types'

interface AgenticTraceProps {
  traceType: TraceType
  metadata: Record<string, unknown>
}

interface KV { label: string; value: unknown; green?: boolean; red?: boolean }

function TraceGrid({ items }: { items: KV[] }) {
  return (
    <div className="grid grid-cols-3 gap-x-4 gap-y-1 px-3 pb-2 pt-1 text-[11px]">
      {items.map(({ label, value, green, red }) => (
        <div key={label} className="flex gap-1">
          <span className="text-muted-foreground">{label}</span>
          <span className={cn(
            'font-medium',
            green && 'text-green-500',
            red && 'text-red-500',
            !green && !red && 'text-foreground'
          )}>
            {String(value ?? '—')}
          </span>
        </div>
      ))}
    </div>
  )
}

function AgenticPanel({ metadata }: { metadata: Record<string, unknown> }) {
  const t = (metadata.agentic_trace ?? {}) as Record<string, unknown>
  if (!('confidence' in t) && !('profile' in t)) return null
  const grounded = Boolean(t.grounded)
  return (
    <>
      <PillRow>
        <Pill className={grounded ? 'text-green-500 border-green-800' : 'text-red-500 border-red-800'}>
          {grounded ? '✓ grounded' : '✗ not grounded'}
        </Pill>
        <Pill>conf {String(t.confidence ?? '?')}</Pill>
        <Pill>{String(t.profile ?? '?')}</Pill>
      </PillRow>
      <TraceGrid items={[
        { label: 'profile', value: t.profile },
        { label: 'retrieve ×', value: t.retrieve_cycles_used },
        { label: 'check ×', value: t.check_cycles_used },
        { label: 'cache hit', value: String(t.router_cache_hit ?? false) },
        { label: 'confidence', value: t.confidence },
        { label: 'grounded', value: String(grounded), green: grounded, red: !grounded },
      ]} />
    </>
  )
}

function AutoPanel({ metadata }: { metadata: Record<string, unknown> }) {
  const rt = (metadata.routing_trace ?? {}) as Record<string, unknown>
  if (!('profile' in rt) && !('paths_activated' in rt)) return null
  const paths = Array.isArray(rt.paths_activated) ? rt.paths_activated.join(', ') : String(rt.paths_activated ?? '—')
  return (
    <>
      <PillRow>
        <Pill>{String(rt.profile ?? '?')}</Pill>
        <Pill>conf {String(rt.confidence ?? '?')}</Pill>
        <Pill>paths: {paths}</Pill>
      </PillRow>
      <TraceGrid items={[
        { label: 'profile', value: rt.profile },
        { label: 'paths', value: paths },
        { label: 'after rrf', value: rt.chunks_after_rrf },
        { label: 'after rerank', value: rt.chunks_after_rerank },
        { label: 'final chunks', value: rt.chunks_after_threshold },
      ]} />
    </>
  )
}

function PprPanel({ metadata }: { metadata: Record<string, unknown> }) {
  const d = ((metadata.data ?? metadata) as Record<string, unknown>)
  const chunks = Array.isArray(d.chunks) ? d.chunks.length : 0
  const entities = Array.isArray(d.entities) ? d.entities.length : 0
  const relations = Array.isArray(d.relations) ? d.relations.length : 0
  if (chunks === 0 && entities === 0) return null
  return (
    <>
      <PillRow>
        <Pill>chunks {chunks}</Pill>
        <Pill>entities {entities}</Pill>
        <Pill>relations {relations}</Pill>
      </PillRow>
      <TraceGrid items={[
        { label: 'chunks', value: chunks },
        { label: 'entities', value: entities },
        { label: 'relations', value: relations },
      ]} />
    </>
  )
}

function Pill({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn(
      'rounded-full border px-2 py-0.5 text-[10px] text-muted-foreground border-border',
      className
    )}>
      {children}
    </span>
  )
}

function PillRow({ children }: { children: ReactNode }) {
  return <div className="flex items-center gap-1.5 px-3 pt-2 pb-1 flex-wrap">{children}</div>
}

const LABELS: Record<NonNullable<TraceType>, string> = {
  agentic: 'Agentic trace',
  auto: 'Routing trace',
  ppr: 'PPR trace',
}

export function AgenticTrace({ traceType, metadata }: AgenticTraceProps) {
  const [open, setOpen] = useState(true)

  if (!traceType) return null

  const label = LABELS[traceType]

  return (
    <div className="mt-1 rounded-lg border border-border bg-secondary/50 text-xs overflow-hidden max-w-[80%]">
      <button
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span>{label}</span>
      </button>
      {open && (
        <>
          {traceType === 'agentic' && <AgenticPanel metadata={metadata} />}
          {traceType === 'auto' && <AutoPanel metadata={metadata} />}
          {traceType === 'ppr' && <PprPanel metadata={metadata} />}
        </>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1 | head -20
```

Expected: no new errors

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/chat/AgenticTrace.tsx
git commit -m "feat(frontend): AgenticTrace component for agentic/auto/ppr trace display"
```

---

## Task 7: Wire MessageBubble + ChatPage

**Files:**
- Modify: `rag-anything/server/frontend/src/components/chat/MessageBubble.tsx`
- Modify: `rag-anything/server/frontend/src/routes/ChatPage.tsx`

- [ ] **Step 1: Update MessageBubble to accept and render AgenticTrace**

Replace `src/components/chat/MessageBubble.tsx`:

```typescript
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { AgenticTrace } from './AgenticTrace'
import { CitationChip } from './CitationChip'
import { cn } from '@/lib/utils'
import type { SourceNode, TraceType } from '@/types'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sourceNodes?: SourceNode[]
  traceType?: TraceType
  traceMetadata?: Record<string, unknown>
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
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown>
        )}
      </div>
      {!isUser && message.traceType && (
        <AgenticTrace
          traceType={message.traceType}
          metadata={message.traceMetadata ?? {}}
        />
      )}
      {!isUser && message.sourceNodes && message.sourceNodes.length > 0 && (
        <div className="flex flex-wrap gap-1 max-w-[80%]">
          {message.sourceNodes.map((n, i) => <CitationChip key={i} node={n} />)}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Update ChatPage to wire profile, traceType, metadata**

Replace `src/routes/ChatPage.tsx`:

```typescript
import { useState, useCallback, useRef } from 'react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import { useAppStore } from '@/store'
import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/button'
import { SquarePen } from 'lucide-react'
import type { Message } from '@/components/chat/MessageBubble'
import type { SourceNode, TraceType } from '@/types'

let msgId = 0

function modeToTraceType(mode: string): TraceType {
  if (mode === 'agentic') return 'agentic'
  if (mode === 'ppr') return 'ppr'
  if (mode === 'auto') return 'auto'
  return null
}

export default function ChatPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { send, answer, reasoning, status, sourceNodes, metadata } = useStreamQuery()
  const [messages, setMessages] = useState<Message[]>([])
  const latestRef = useRef<{
    answer: string
    reasoning: string
    sourceNodes: SourceNode[]
    metadata: Record<string, unknown>
  }>({ answer: '', reasoning: '', sourceNodes: [], metadata: {} })

  latestRef.current = { answer, reasoning, sourceNodes, metadata }

  const handleSend = useCallback(async (query: string, mode: string, profile: string) => {
    const userMsg: Message = { id: String(++msgId), role: 'user', content: query }
    setMessages((prev) => [...prev, userMsg])

    await send({
      workspace_id: workspaceId,
      query,
      mode: mode as QueryParams['mode'],
      profile: mode === 'auto' && profile ? profile : undefined,
    })

    const { answer: a, reasoning: r, sourceNodes: sn, metadata: m } = latestRef.current
    const traceType = modeToTraceType(mode)
    setMessages((prev) => [
      ...prev,
      {
        id: String(++msgId),
        role: 'assistant',
        content: a,
        reasoning: r,
        sourceNodes: sn,
        traceType,
        traceMetadata: m,
      },
    ])
  }, [workspaceId, send])

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

Note: add `import type { QueryParams } from '@/types'` at the top alongside the other imports.

- [ ] **Step 3: Verify TypeScript compiles cleanly**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1
```

Expected: no errors

- [ ] **Step 4: Run all frontend tests**

```bash
cd rag-anything/server/frontend
npx vitest run
```

Expected: all pass (7 tests)

- [ ] **Step 5: Run backend tests**

```bash
cd rag-anything
python -m pytest tests/test_stream_query_agentic.py -v
```

Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/frontend/src/components/chat/MessageBubble.tsx rag-anything/server/frontend/src/routes/ChatPage.tsx
git commit -m "feat(frontend): wire AgenticTrace into MessageBubble and ChatPage — P0 complete"
```

---

## Done

All 7 tasks complete. The chat UI now has `ppr`/`auto`/`agentic` modes, a conditional profile selector for `auto`, and a collapsible trace panel that shows structured agentic/routing/PPR diagnostics below each assistant reply.
