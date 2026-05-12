# P2: Phoenix Monitoring + On-Demand Eval Badge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the already-implemented Phoenix OTEL observability and LLM answer evaluator into the product — Phoenix auto-starts with the server via env var, and every assistant message gets an on-demand Evaluate button.

**Architecture:** Phoenix is initialized in the FastAPI lifespan when `ENABLE_PHOENIX=true`; `AnswerEvaluator` (already in `raganything/retrieval/evaluator.py`) is exposed via a new `/evaluate` POST endpoint. Frontend gains a TopNav "Traces" link and an `EvalBadge` component rendered under each assistant message bubble.

**Tech Stack:** FastAPI + Pydantic v2, React + TypeScript + Tailwind CSS, lucide-react icons, vitest

---

## File Map

| File | Change |
|---|---|
| `rag-anything/server/app.py` | Phoenix in lifespan + `EvaluateRequest` model + `/evaluate` endpoint |
| `rag-anything/raganything/services/local_rag.py` | New `evaluate_answer()` method |
| `rag-anything/tests/test_evaluate_endpoint.py` | New: backend unit test |
| `rag-anything/env.example` | Document `ENABLE_PHOENIX` |
| `rag-anything/server/frontend/src/api/evaluate.ts` | New: POST /evaluate client |
| `rag-anything/server/frontend/src/components/chat/EvalBadge.tsx` | New: score badge component |
| `rag-anything/server/frontend/src/components/chat/MessageBubble.tsx` | Add `query?` + `workspaceId` prop + render EvalBadge |
| `rag-anything/server/frontend/src/components/chat/MessageList.tsx` | Pass `workspaceId` to MessageBubble |
| `rag-anything/server/frontend/src/components/layout/TopNav.tsx` | Add Traces link |
| `rag-anything/server/frontend/src/routes/ChatPage.tsx` | Pass `workspaceId` to MessageList + set `query` on assistant messages |

---

## Task 1: Backend — Phoenix lifespan + evaluate endpoint

**Files:**
- Modify: `rag-anything/server/app.py` (lifespan ~line 109; add model + endpoint after QueryRequest block)
- Modify: `rag-anything/raganything/services/local_rag.py` (add method before `if __name__ == "__main__"`)
- Modify: `rag-anything/env.example` (append ENABLE_PHOENIX)

- [ ] **Step 1: Add Phoenix startup to lifespan**

In `rag-anything/server/app.py`, find the line `logger.info("lifespan: startup complete")` (~line 109) and insert **before** it:

```python
    phoenix_enabled = os.getenv("ENABLE_PHOENIX", "").lower() in ("1", "true", "yes")
    if phoenix_enabled:
        from raganything.observability import setup_phoenix
        setup_phoenix()
        logger.info("lifespan: Phoenix tracing enabled at http://localhost:6006")
```

- [ ] **Step 2: Add `EvaluateRequest` model and `/evaluate` endpoint**

In `rag-anything/server/app.py`, after the `QueryRequest` class definition (around line 234), add:

```python
class EvaluateRequest(BaseModel):
    workspace_id: str
    query: str
    answer: str
```

Then add the endpoint after the existing `/query/stream` route:

```python
@app.post("/evaluate")
async def evaluate_endpoint(
    payload: EvaluateRequest,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    """Run LLM-based answer quality evaluation. Returns {score: float, gap: str}."""
    _validate_workspace_id(payload.workspace_id)
    return await service.evaluate_answer(payload.workspace_id, payload.query, payload.answer)
```

- [ ] **Step 3: Add `evaluate_answer` to `LocalRagService`**

In `rag-anything/raganything/services/local_rag.py`, insert this method just before the `if __name__ == "__main__":` block at the end of the file:

```python
    async def evaluate_answer(self, workspace_id: str, query: str, answer: str) -> dict:
        """Run AnswerEvaluator on a query+answer pair.

        Returns: {"score": float (0-1), "gap": str}
        """
        from raganything.retrieval.evaluator import AnswerEvaluator
        rag = await self.get_rag(workspace_id)
        evaluator = AnswerEvaluator(rag.lightrag.llm_model_func)
        return await evaluator.evaluate(query, answer)
```

- [ ] **Step 4: Document ENABLE_PHOENIX in env.example**

Append to `rag-anything/env.example`:

```
# Arize Phoenix OTEL tracing (requires: pip install -e ".[agentic]")
ENABLE_PHOENIX=false
```

- [ ] **Step 5: Syntax check**

```bash
cd rag-anything
python -m py_compile server/app.py && echo "app.py OK"
python -m py_compile raganything/services/local_rag.py && echo "local_rag.py OK"
```

Expected: both print OK

- [ ] **Step 6: Commit**

```bash
git add rag-anything/server/app.py rag-anything/raganything/services/local_rag.py rag-anything/env.example
git commit -m "feat(backend): Phoenix lifespan init + /evaluate endpoint using AnswerEvaluator"
```

---

## Task 2: Backend — evaluate endpoint test

**Files:**
- Create: `rag-anything/tests/test_evaluate_endpoint.py`

- [ ] **Step 1: Write the test**

Create `rag-anything/tests/test_evaluate_endpoint.py`:

```python
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
    service = LocalRagService.__new__(LocalRagService)
    service.settings = settings
    service.logger = MagicMock()
    return service


@pytest.mark.asyncio
async def test_evaluate_answer_returns_score_and_gap():
    """evaluate_answer must call AnswerEvaluator.evaluate and return its result."""
    service = _make_service()

    fake_rag = MagicMock()
    fake_rag.lightrag.llm_model_func = AsyncMock()
    service.get_rag = AsyncMock(return_value=fake_rag)

    with patch("raganything.retrieval.evaluator.AnswerEvaluator") as MockEval:
        instance = MockEval.return_value
        instance.evaluate = AsyncMock(return_value={"score": 0.87, "gap": ""})

        result = await service.evaluate_answer("ws1", "What is X?", "X is Y.")

    MockEval.assert_called_once_with(fake_rag.lightrag.llm_model_func)
    instance.evaluate.assert_awaited_once_with("What is X?", "X is Y.")
    assert result == {"score": 0.87, "gap": ""}


@pytest.mark.asyncio
async def test_evaluate_answer_with_gap():
    """Gap string is forwarded from the evaluator."""
    service = _make_service()

    fake_rag = MagicMock()
    fake_rag.lightrag.llm_model_func = AsyncMock()
    service.get_rag = AsyncMock(return_value=fake_rag)

    with patch("raganything.retrieval.evaluator.AnswerEvaluator") as MockEval:
        instance = MockEval.return_value
        instance.evaluate = AsyncMock(return_value={"score": 0.45, "gap": "Missing accuracy numbers"})

        result = await service.evaluate_answer("ws1", "What is the accuracy?", "It is good.")

    assert result["score"] == 0.45
    assert "accuracy" in result["gap"]
```

- [ ] **Step 2: Run to confirm tests pass**

```bash
cd rag-anything
python -m pytest tests/test_evaluate_endpoint.py -v 2>&1 | tail -10
```

Expected: 2 passed

- [ ] **Step 3: Run all backend tests — no regression**

```bash
python -m pytest tests/test_stream_query_agentic.py tests/test_stream_query_p1.py tests/test_evaluate_endpoint.py -v 2>&1 | tail -12
```

Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add rag-anything/tests/test_evaluate_endpoint.py
git commit -m "test(backend): evaluate_answer unit tests"
```

---

## Task 3: TopNav — Traces link

**Files:**
- Modify: `rag-anything/server/frontend/src/components/layout/TopNav.tsx`

- [ ] **Step 1: Add Traces link**

Replace `rag-anything/server/frontend/src/components/layout/TopNav.tsx`:

```typescript
import { NavLink } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { WorkspaceSwitcher } from './WorkspaceSwitcher'
import { ThemeToggle } from './ThemeToggle'
import { useJobs } from '@/hooks/useJobs'
import { useAppStore } from '@/store'
import { cn } from '@/lib/utils'
import { BarChart2 } from 'lucide-react'

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
    <header className="h-12 border-b border-border flex items-center px-4 gap-6 shrink-0 bg-background">
      <span className="text-sm font-semibold text-foreground">RAGAnything</span>
      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-1 px-3 py-1.5 text-sm rounded-md transition-colors',
                isActive
                  ? 'text-primary border-b-2 border-primary font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              )
            }
          >
            {label}
            {label === 'Jobs' && runningCount > 0 && (
              <Badge variant="default" className="h-4 min-w-4 px-1 text-[10px]">
                {runningCount}
              </Badge>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-2">
        <a
          href="http://localhost:6006"
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            'flex items-center gap-1 h-7 px-2 rounded-md border border-border',
            'text-xs text-muted-foreground hover:text-foreground transition-colors'
          )}
          title="Open Arize Phoenix trace dashboard (requires ENABLE_PHOENIX=true)"
        >
          <BarChart2 className="h-3 w-3" />
          Traces
        </a>
        <WorkspaceSwitcher />
        <ThemeToggle />
      </div>
    </header>
  )
}
```

- [ ] **Step 2: TypeScript check**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1 | head -5
```

Expected: no output

- [ ] **Step 3: Commit**

```bash
git add rag-anything/server/frontend/src/components/layout/TopNav.tsx
git commit -m "feat(frontend): add Traces link to TopNav pointing to Arize Phoenix"
```

---

## Task 4: Frontend — EvalBadge component + API client

**Files:**
- Create: `rag-anything/server/frontend/src/api/evaluate.ts`
- Create: `rag-anything/server/frontend/src/components/chat/EvalBadge.tsx`

- [ ] **Step 1: Create `src/api/evaluate.ts`**

```typescript
export interface EvalResult {
  score: number
  gap: string
}

export async function evaluateAnswer(
  workspaceId: string,
  query: string,
  answer: string,
): Promise<EvalResult> {
  const res = await fetch('/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workspace_id: workspaceId, query, answer }),
  })
  if (!res.ok) throw new Error('Evaluation failed')
  return res.json() as Promise<EvalResult>
}
```

- [ ] **Step 2: Create `src/components/chat/EvalBadge.tsx`**

```typescript
import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { evaluateAnswer } from '@/api/evaluate'
import type { EvalResult } from '@/api/evaluate'

interface EvalBadgeProps {
  workspaceId: string
  query: string
  answer: string
}

function scoreColor(score: number) {
  if (score >= 0.85) return 'text-green-600 border-green-300 bg-green-50'
  if (score >= 0.65) return 'text-amber-600 border-amber-300 bg-amber-50'
  return 'text-red-600 border-red-300 bg-red-50'
}

export function EvalBadge({ workspaceId, query, answer }: EvalBadgeProps) {
  const [state, setState] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const [result, setResult] = useState<EvalResult | null>(null)
  const [open, setOpen] = useState(true)

  async function runEval() {
    setState('loading')
    try {
      const r = await evaluateAnswer(workspaceId, query, answer)
      setResult(r)
      setState('done')
    } catch {
      setState('error')
    }
  }

  if (state === 'idle') {
    return (
      <button
        onClick={runEval}
        className="mt-1 inline-flex items-center h-6 px-2 rounded border border-border
                   text-[10px] text-muted-foreground hover:text-foreground transition-colors"
      >
        Evaluate
      </button>
    )
  }

  if (state === 'loading') {
    return (
      <span className="mt-1 inline-flex items-center gap-1 h-6 px-2 rounded border border-border text-[10px] text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        Evaluating…
      </span>
    )
  }

  if (state === 'error') {
    return (
      <span className="mt-1 inline-flex items-center h-6 px-2 rounded border border-red-200
                       text-[10px] text-red-500">
        Eval failed
      </span>
    )
  }

  // done
  const score = result!.score
  const gap = result!.gap

  return (
    <div className="mt-1 flex flex-col gap-0.5">
      <button
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'inline-flex items-center h-6 px-2 rounded border text-[10px] font-medium w-fit transition-colors',
          scoreColor(score)
        )}
      >
        {score >= 0.85 ? '●' : score >= 0.65 ? '◐' : '○'} {score.toFixed(2)}
      </button>
      {open && gap && (
        <p className="text-[10px] text-muted-foreground pl-1">gap: {gap}</p>
      )}
      {open && !gap && (
        <p className="text-[10px] text-muted-foreground pl-1">gap: —</p>
      )}
    </div>
  )
}
```

- [ ] **Step 3: TypeScript check**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1 | head -5
```

Expected: no output

- [ ] **Step 4: Commit**

```bash
git add rag-anything/server/frontend/src/api/evaluate.ts rag-anything/server/frontend/src/components/chat/EvalBadge.tsx
git commit -m "feat(frontend): EvalBadge component + /evaluate API client"
```

---

## Task 5: Wire MessageBubble + MessageList + ChatPage

**Files:**
- Modify: `rag-anything/server/frontend/src/components/chat/MessageBubble.tsx`
- Modify: `rag-anything/server/frontend/src/components/chat/MessageList.tsx`
- Modify: `rag-anything/server/frontend/src/routes/ChatPage.tsx`

- [ ] **Step 1: Update `MessageBubble`**

Replace `rag-anything/server/frontend/src/components/chat/MessageBubble.tsx`:

```typescript
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { AgenticTrace } from './AgenticTrace'
import { EvalBadge } from './EvalBadge'
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
  query?: string   // original user question (set on assistant messages for EvalBadge)
}

interface MessageBubbleProps {
  message: Message
  workspaceId?: string
}

export function MessageBubble({ message, workspaceId }: MessageBubbleProps) {
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
      {!isUser && message.query && message.content && workspaceId && (
        <EvalBadge
          workspaceId={workspaceId}
          query={message.query}
          answer={message.content}
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

- [ ] **Step 2: Update `MessageList` to accept and pass `workspaceId`**

Replace `rag-anything/server/frontend/src/components/chat/MessageList.tsx`:

```typescript
import { useEffect, useRef, useState } from 'react'
import { MessageBubble } from './MessageBubble'
import type { Message } from './MessageBubble'

interface MessageListProps {
  messages: Message[]
  streamingAnswer: string
  streamingReasoning: string
  isStreaming: boolean
  workspaceId?: string
}

export function MessageList({
  messages,
  streamingAnswer,
  streamingReasoning,
  isStreaming,
  workspaceId,
}: MessageListProps) {
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
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} workspaceId={workspaceId} />
      ))}
      {isStreaming && streamingAnswer && (
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

- [ ] **Step 3: Update `ChatPage` — pass workspaceId to MessageList + set query on assistant messages**

Replace `rag-anything/server/frontend/src/routes/ChatPage.tsx`:

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

const MAX_HISTORY_TURNS = 5

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

    const history = buildHistory(messagesRef.current)

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
        query,           // store original query for EvalBadge
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
        workspaceId={workspaceId}
      />
      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  )
}
```

- [ ] **Step 4: TypeScript check — expect clean**

```bash
cd rag-anything/server/frontend
npx tsc --noEmit 2>&1
```

Expected: no output

- [ ] **Step 5: Run all frontend tests**

```bash
npx vitest run 2>&1 | tail -8
```

Expected: 3 test files, 8 tests, all passed

- [ ] **Step 6: Run all backend tests**

```bash
cd ../../..
python -m pytest tests/test_stream_query_agentic.py tests/test_stream_query_p1.py tests/test_evaluate_endpoint.py -v 2>&1 | tail -12
```

Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add rag-anything/server/frontend/src/components/chat/MessageBubble.tsx \
        rag-anything/server/frontend/src/components/chat/MessageList.tsx \
        rag-anything/server/frontend/src/routes/ChatPage.tsx
git commit -m "feat(frontend): wire EvalBadge into chat — P2 complete"
```

---

## Done

All 5 tasks complete. The app now:
- Auto-starts Arize Phoenix OTEL tracing when `ENABLE_PHOENIX=true` (all LLM calls instrumented)
- Shows a "📊 Traces" link in TopNav opening `localhost:6006`
- Renders an "Evaluate" button under every assistant message; clicking it calls `/evaluate` and shows a color-coded score (green ≥ 0.85 / amber ≥ 0.65 / red < 0.65) with gap details
