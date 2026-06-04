# P1: Conversation History + VLM Enhanced Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire multi-turn conversation history and VLM-enhanced query mode end-to-end from frontend to backend, reusing existing raganything/LightRAG infrastructure.

**Architecture:** History flows as `[{role, content}]` from `ChatPage` → `QueryRequest.conversation_history` → `QueryParam.conversation_history` (LightRAG native field) → `llm_model_func(history_messages=...)`. VLM adds a non-streaming branch in `stream_query` that calls `query_with_trace(vlm_enhanced=True)`, parallel to the existing agentic branch. Both features touch the same 6 files independently.

**Tech Stack:** FastAPI + Pydantic v2 (backend), React + TypeScript + Tailwind (frontend), pytest + vitest (tests)

---

## File Map

| File | Change |
|---|---|
| `rag-anything/server/app.py` | Add `conversation_history` to `QueryRequest`; pass to `stream_query` |
| `rag-anything/raganything/services/local_rag.py` | Add `conversation_history` + `vlm_enhanced` to `stream_query`; update `QueryParam`; add VLM branch; update agentic branch to pass history |
| `rag-anything/tests/test_stream_query_p1.py` | New: backend unit tests for history + VLM branches |
| `rag-anything/server/frontend/src/types/index.ts` | Add `conversation_history?` + `vlm_enhanced?` to `QueryParams` |
| `rag-anything/server/frontend/src/api/query.ts` | Send both new fields conditionally |
| `rag-anything/server/frontend/src/components/chat/ChatInput.tsx` | Add VLM toggle chip |
| `rag-anything/server/frontend/src/routes/ChatPage.tsx` | Build history slice and pass `vlm_enhanced` to `send()` |

---

## Task 1: Backend — conversation_history wiring

**Files:**
- Modify: `rag-anything/server/app.py` (~line 231)
- Modify: `rag-anything/raganything/services/local_rag.py` (stream_query signature ~line 2372; QueryParam ~line 2423; agentic branch ~line 2387)

- [ ] **Step 1: Add `conversation_history` to `QueryRequest`**

In `rag-anything/server/app.py`, after the `profile` field (last field in `QueryRequest`):

```python
    profile: Optional[str] = None  # auto mode only; None = LLM classifier decides
    conversation_history: list[dict] = []
```

- [ ] **Step 2: Pass `conversation_history` to `stream_query` in the endpoint**

In `rag-anything/server/app.py`, in `query_stream_endpoint`, update the `service.stream_query(...)` call to add:

```python
                profile=payload.profile,
                conversation_history=payload.conversation_history,
```

- [ ] **Step 3: Add `conversation_history` to `stream_query` signature**

In `rag-anything/raganything/services/local_rag.py`, update the `stream_query` signature — add after `profile: str | None = None`:

```python
        profile: str | None = None,
        conversation_history: list[dict] | None = None,
```

- [ ] **Step 4: Pass `conversation_history` to `QueryParam`**

In the `QueryParam(...)` constructor block (~line 2423), add after the last existing field:

```python
                conversation_history=conversation_history or [],
```

- [ ] **Step 5: Pass `conversation_history` in the agentic/auto+profile branch**

In the agentic branch `query_with_trace(...)` call (~line 2387), add:

```python
                result = await self.query_with_trace(
                    workspace_id, query,
                    mode=mode,
                    return_trace=True,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    enable_rerank=enable_rerank,
                    conversation_history=conversation_history or [],
                    **extra,
                )
```

- [ ] **Step 6: Syntax check**

```bash
cd rag-anything
python -m py_compile server/app.py && echo "app.py OK"
python -m py_compile raganything/services/local_rag.py && echo "local_rag.py OK"
```

Expected: both print OK

- [ ] **Step 7: Commit**

```bash
git add rag-anything/server/app.py rag-anything/raganything/services/local_rag.py
git commit -m "feat(backend): wire conversation_history through QueryRequest → QueryParam"
```

---

## Task 2: Backend — VLM branch in stream_query

**Files:**
- Modify: `rag-anything/raganything/services/local_rag.py` (stream_query signature + new VLM branch)
- Create: `rag-anything/tests/test_stream_query_p1.py`

- [ ] **Step 1: Write failing tests**

Create `rag-anything/tests/test_stream_query_p1.py`:

```python
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

for _mod in [
    "sentence_transformers",
    "sentence_transformers.cross_encoder",
    "torch",
    "raganything.processor",
    "raganything.batch",
    "raganything.batch_parser",
    "raganything.raganything",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import raganything as _ra
_ra.RAGAnything = MagicMock()
_ra.RAGAnythingConfig = MagicMock()


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
async def test_stream_query_agentic_passes_conversation_history():
    """conversation_history must be forwarded to query_with_trace in agentic branch."""
    service = _make_service()
    service.query_with_trace = AsyncMock(return_value={
        "answer": "answer",
        "confidence": 0.9,
        "grounded": True,
        "trace": {"profile": "precise", "retrieve_cycles_used": 1, "check_cycles_used": 0},
    })
    history = [{"role": "user", "content": "prev question"},
               {"role": "assistant", "content": "prev answer"}]

    events = []
    async for event in service.stream_query(
        "ws1", "follow-up", mode="agentic", conversation_history=history
    ):
        events.append(event)

    call_kwargs = service.query_with_trace.call_args.kwargs
    assert call_kwargs.get("conversation_history") == history


@pytest.mark.asyncio
async def test_stream_query_vlm_calls_query_with_trace():
    """vlm_enhanced=True must use query_with_trace(vlm_enhanced=True) and yield meta+chunk."""
    service = _make_service()
    service.query_with_trace = AsyncMock(return_value={
        "answer": "VLM answer about the image",
    })

    events = []
    async for event in service.stream_query(
        "ws1", "what is in the image?", mode="hybrid", vlm_enhanced=True
    ):
        events.append(event)

    service.query_with_trace.assert_awaited_once()
    call_kwargs = service.query_with_trace.call_args.kwargs
    assert call_kwargs.get("vlm_enhanced") is True

    assert events[0]["type"] == "meta"
    assert events[1]["type"] == "chunk"
    assert events[1]["text"] == "VLM answer about the image"
    assert len(events) == 2


@pytest.mark.asyncio
async def test_stream_query_vlm_passes_conversation_history():
    """VLM branch must also forward conversation_history."""
    service = _make_service()
    service.query_with_trace = AsyncMock(return_value={"answer": "ok"})
    history = [{"role": "user", "content": "earlier"}]

    async for _ in service.stream_query(
        "ws1", "q", mode="hybrid", vlm_enhanced=True, conversation_history=history
    ):
        pass

    call_kwargs = service.query_with_trace.call_args.kwargs
    assert call_kwargs.get("conversation_history") == history
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd rag-anything
python -m pytest tests/test_stream_query_p1.py -v 2>&1 | tail -15
```

Expected: 3 FAILED

- [ ] **Step 3: Add `vlm_enhanced` to `stream_query` signature**

In `rag-anything/raganything/services/local_rag.py`, update `stream_query` — add after `conversation_history`:

```python
        conversation_history: list[dict] | None = None,
        vlm_enhanced: bool = False,
```

- [ ] **Step 4: Add VLM branch after the agentic branch `return`**

Insert this block immediately after `            return  # end agentic branch` and before `        try:`:

```python
        # ── Non-streaming branch: VLM enhanced query ──
        if vlm_enhanced and mode not in ("agentic",):
            try:
                result = await self.query_with_trace(
                    workspace_id, query,
                    mode=mode,
                    vlm_enhanced=True,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    enable_rerank=enable_rerank,
                    conversation_history=conversation_history or [],
                    return_trace=False,
                )
                answer = result.get("answer", result) if isinstance(result, dict) else str(result)
                yield {"type": "meta", "data": {}, "metadata": {}}
                yield {"type": "chunk", "text": answer}
            except Exception as exc:
                self.logger.error("stream_query (vlm branch) error: %s", exc)
                yield {"type": "error", "text": str(exc)}
            return
```

- [ ] **Step 5: Pass `vlm_enhanced` from app.py endpoint to stream_query**

In `rag-anything/server/app.py`, in the `service.stream_query(...)` call, add after `conversation_history`:

```python
                conversation_history=payload.conversation_history,
                vlm_enhanced=payload.vlm_enhanced,
```

- [ ] **Step 6: Run tests — expect 3 passed**

```bash
cd rag-anything
python -m pytest tests/test_stream_query_p1.py -v 2>&1 | tail -10
```

Expected: 3 passed

- [ ] **Step 7: Run existing agentic tests to verify no regression**

```bash
python -m pytest tests/test_stream_query_agentic.py tests/test_stream_query_p1.py -v 2>&1 | tail -10
```

Expected: 5 passed

- [ ] **Step 8: Commit**

```bash
git add rag-anything/raganything/services/local_rag.py rag-anything/server/app.py rag-anything/tests/test_stream_query_p1.py
git commit -m "feat(backend): vlm_enhanced branch in stream_query + conversation_history forwarding"
```

---

## Task 3: Frontend types + API

**Files:**
- Modify: `rag-anything/server/frontend/src/types/index.ts`
- Modify: `rag-anything/server/frontend/src/api/query.ts`

- [ ] **Step 1: Extend `QueryParams` in types**

In `src/types/index.ts`, update `QueryParams` to add the two new fields after `return_graph?`:

```typescript
export interface QueryParams {
  workspace_id: string
  query: string
  mode?: 'naive' | 'local' | 'global' | 'hybrid' | 'ppr' | 'auto' | 'agentic'
  profile?: string
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
  return_graph?: boolean
  conversation_history?: { role: string; content: string }[]
  vlm_enhanced?: boolean
}
```

- [ ] **Step 2: Send new fields in API client**

Replace `rag-anything/server/frontend/src/api/query.ts`:

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
  if (params.conversation_history && params.conversation_history.length > 0) {
    body.conversation_history = params.conversation_history
  }
  if (params.vlm_enhanced) {
    body.vlm_enhanced = true
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

- [ ] **Step 3: TypeScript check**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1 | head -10
```

Expected: no output (no errors)

- [ ] **Step 4: Commit**

```bash
git add rag-anything/server/frontend/src/types/index.ts rag-anything/server/frontend/src/api/query.ts
git commit -m "feat(frontend): add conversation_history + vlm_enhanced to QueryParams and API client"
```

---

## Task 4: ChatInput — VLM toggle

**Files:**
- Modify: `rag-anything/server/frontend/src/components/chat/ChatInput.tsx`

The toggle renders as a small chip alongside the mode selector. When active it shows a distinct accent color. The `onSend` signature gains a fourth argument `vlmEnabled: boolean`.

- [ ] **Step 1: Replace ChatInput**

```typescript
import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Send, Eye } from 'lucide-react'
import { cn } from '@/lib/utils'

const MODES = ['naive', 'local', 'global', 'hybrid', 'ppr', 'auto', 'agentic'] as const
const PROFILES = ['precise', 'local', 'multihop', 'descriptive', 'full'] as const

interface ChatInputProps {
  onSend: (query: string, mode: string, profile: string, vlmEnabled: boolean) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [profile, setProfile] = useState('')
  const [vlmEnabled, setVlmEnabled] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q, mode, profile, vlmEnabled)
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
        <Select value={mode} onValueChange={(v) => { if (v != null) { setMode(v); if (v !== 'auto') setProfile('') } }}>
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
            <Select value={profile} onValueChange={(v) => { if (v != null) setProfile(v) }}>
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

        <button
          type="button"
          onClick={() => setVlmEnabled((v) => !v)}
          className={cn(
            'inline-flex items-center gap-1 h-7 px-2 rounded-md border text-xs transition-colors',
            vlmEnabled
              ? 'bg-primary/10 border-primary text-primary'
              : 'border-border text-muted-foreground hover:text-foreground'
          )}
          title="VLM enhanced: use vision model to reason over images in retrieved documents"
        >
          <Eye className="h-3 w-3" />
          VLM
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: TypeScript check**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1 | head -20
```

Expected: errors only for `ChatPage` (it calls `onSend` with 3 args, now expects 4) — that's fine, fixed in Task 5.

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/chat/ChatInput.tsx
git commit -m "feat(frontend): ChatInput VLM toggle chip"
```

---

## Task 5: ChatPage — history slice + vlm_enabled wiring

**Files:**
- Modify: `rag-anything/server/frontend/src/routes/ChatPage.tsx`

- [ ] **Step 1: Replace ChatPage**

```typescript
import { useState, useCallback, useRef } from 'react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import { useAppStore } from '@/store'
import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/button'
import { SquarePen } from 'lucide-react'
import type { Message } from '@/components/chat/MessageBubble'
import type { SourceNode, TraceType, QueryParams } from '@/types'

let msgId = 0

const MAX_HISTORY_TURNS = 5  // send at most 5 complete turns (10 messages)

function modeToTraceType(mode: string): TraceType {
  if (mode === 'agentic') return 'agentic'
  if (mode === 'ppr') return 'ppr'
  if (mode === 'auto') return 'auto'
  return null
}

function buildHistory(messages: Message[]): { role: string; content: string }[] {
  return messages
    .slice(-MAX_HISTORY_TURNS * 2)
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .map((m) => ({ role: m.role, content: m.content }))
}

export default function ChatPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { send, answer, reasoning, status, sourceNodes, metadata } = useStreamQuery()
  const [messages, setMessages] = useState<Message[]>([])

  // Keep a ref so handleSend always reads current messages without needing
  // messages in its dependency array (avoids re-creating the callback on every turn)
  const messagesRef = useRef<Message[]>([])
  messagesRef.current = messages

  const latestRef = useRef<{
    answer: string
    reasoning: string
    sourceNodes: SourceNode[]
    metadata: Record<string, unknown>
  }>({ answer: '', reasoning: '', sourceNodes: [], metadata: {} })

  latestRef.current = { answer, reasoning, sourceNodes, metadata }

  const handleSend = useCallback(async (
    query: string,
    mode: string,
    profile: string,
    vlmEnabled: boolean,
  ) => {
    const userMsg: Message = { id: String(++msgId), role: 'user', content: query }
    setMessages((prev) => [...prev, userMsg])

    const history = buildHistory(messagesRef.current)  // messages before this turn

    await send({
      workspace_id: workspaceId,
      query,
      mode: mode as QueryParams['mode'],
      profile: mode === 'auto' && profile ? profile : undefined,
      vlm_enhanced: vlmEnabled || undefined,
      conversation_history: history.length > 0 ? history : undefined,
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

- [ ] **Step 2: TypeScript check — expect clean**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1
```

Expected: no output

- [ ] **Step 3: Run all frontend tests**

```bash
npx vitest run 2>&1 | tail -8
```

Expected: 3 test files, 8 tests, all passed

- [ ] **Step 4: Run all backend tests**

```bash
cd ../../..
python -m pytest tests/test_stream_query_agentic.py tests/test_stream_query_p1.py -v 2>&1 | tail -12
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag-anything/server/frontend/src/routes/ChatPage.tsx
git commit -m "feat(frontend): conversation history (last 5 turns) + vlm_enhanced wiring in ChatPage — P1 complete"
```

---

## Done

All 5 tasks complete. The chat now:
- Sends the last 5 conversation turns as context on every query
- Has a VLM toggle that routes queries through `aquery_vlm_enhanced` (vision-capable LLM reasons over images in retrieved document chunks)
