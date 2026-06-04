# P2: Phoenix Monitoring + On-Demand Eval Badge — Design Spec

Date: 2026-05-13
Branch: frontend-modernization

## Goal

Wire the already-implemented Phoenix OTEL observability and LLM-based answer evaluator into the product — Phoenix auto-starts with the server, and every assistant message gets an on-demand "Evaluate" button.

---

## Scope

**In scope:**
- `server/app.py` lifespan: call `setup_phoenix()` when `ENABLE_PHOENIX=true`
- New `/evaluate` POST endpoint using existing `AnswerEvaluator`
- New `evaluate_answer()` method on `LocalRagService`
- TopNav: "📊 Traces" chip linking to `http://localhost:6006`
- New `EvalBadge` component: Evaluate button → score + gap expandable card
- `MessageBubble`: integrate `EvalBadge`, store `query` on assistant messages

**Out of scope:**
- `HallucinationChecker` in the eval endpoint (requires chunk context not available in frontend)
- RAGAS library (not needed — custom evaluators already exist)
- Configurable Phoenix URL (hardcode `localhost:6006` for now)
- Agentic mode: `grounded` already shown in AgenticTrace, eval badge still shows answer score

---

## Sub-system A: Phoenix Server Integration

### Backend

**`server/app.py` lifespan** — on startup, after existing init, add:

```python
# In the lifespan startup block
phoenix_enabled = os.getenv("ENABLE_PHOENIX", "").lower() in ("1", "true", "yes")
if phoenix_enabled:
    from raganything.observability import setup_phoenix
    setup_phoenix()
```

`setup_phoenix()` already silently no-ops if `arize-phoenix` is not installed, so no guard needed.

### Frontend

**`src/components/layout/TopNav.tsx`** — add a "📊 Traces" anchor to the right side (before `ThemeToggle`):

```tsx
<a
  href="http://localhost:6006"
  target="_blank"
  rel="noopener noreferrer"
  className="flex items-center gap-1 h-7 px-2 rounded-md border border-border
             text-xs text-muted-foreground hover:text-foreground transition-colors"
  title="Open Arize Phoenix trace dashboard"
>
  <BarChart2 className="h-3 w-3" />
  Traces
</a>
```

Import `BarChart2` from `lucide-react`.

---

## Sub-system B: On-Demand Eval Badge

### Backend

**`raganything/services/local_rag.py`** — new method:

```python
async def evaluate_answer(self, workspace_id: str, query: str, answer: str) -> dict:
    """Run AnswerEvaluator on a query+answer pair.
    Returns: {score: float, gap: str}
    """
    from raganything.retrieval.evaluator import AnswerEvaluator
    rag = await self.get_rag(workspace_id)
    evaluator = AnswerEvaluator(rag.lightrag.llm_model_func)
    return await evaluator.evaluate(query, answer)
```

**`server/app.py`** — new Pydantic model + endpoint:

```python
class EvaluateRequest(BaseModel):
    workspace_id: str
    query: str
    answer: str

@app.post("/evaluate")
async def evaluate_endpoint(
    payload: EvaluateRequest,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(payload.workspace_id)
    return await service.evaluate_answer(payload.workspace_id, payload.query, payload.answer)
```

Response shape: `{"score": 0.87, "gap": ""}` — score is 0.0–1.0, gap is empty string when answer is complete.

### Frontend — API

**New `src/api/evaluate.ts`**:

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
  return res.json()
}
```

### Frontend — Types

**`src/types/index.ts`** — add `EvalResult` re-export and extend `Message` (in `MessageBubble.tsx`):

The `Message` interface in `MessageBubble.tsx` gains `query?: string` so `EvalBadge` can call the API without needing props from a parent.

### Frontend — EvalBadge Component

**New `src/components/chat/EvalBadge.tsx`**:

Props: `{ workspaceId: string; query: string; answer: string }`

States:
- **idle** — shows "Evaluate" chip button
- **loading** — shows spinner inside the chip
- **done** — chip becomes score badge (color-coded) + optional gap text below

Color coding for score:
- ≥ 0.85 → green (`text-green-600`)
- 0.65–0.84 → amber (`text-amber-600`)
- < 0.65 → red (`text-red-600`)

The entire badge is collapsible: clicking the score badge toggles gap text visibility.

Layout (done state):
```
[● 0.87]  ← score chip (green/amber/red)
gap: —    ← shown when open, hidden when collapsed
```

### Frontend — MessageBubble

**`src/components/chat/MessageBubble.tsx`** — extend `Message` and render `EvalBadge`:

```typescript
export interface Message {
  // ...existing fields...
  query?: string  // original user query (set for assistant messages)
}
```

Render after `AgenticTrace`:

```tsx
{!isUser && message.query && message.content && (
  <EvalBadge
    workspaceId={workspaceId}     // passed as prop to MessageBubble
    query={message.query}
    answer={message.content}
  />
)}
```

`MessageBubble` receives `workspaceId` as a new prop (passed from `MessageList` → `ChatPage`).

### Frontend — ChatPage

Set `query` on assistant messages when they are created:

```typescript
setMessages((prev) => [...prev, {
  id: String(++msgId),
  role: 'assistant',
  content: a,
  query: query,        // ← the user's question for this turn
  // ...other fields
}])
```

Pass `workspaceId` down through `MessageList` → `MessageBubble`.

---

## File Map

| File | Sub-system |
|---|---|
| `server/app.py` | A (Phoenix startup) + B (evaluate endpoint) |
| `raganything/services/local_rag.py` | B (evaluate_answer method) |
| `src/components/layout/TopNav.tsx` | A (Traces link) |
| `src/api/evaluate.ts` | B (new file) |
| `src/components/chat/EvalBadge.tsx` | B (new file) |
| `src/components/chat/MessageBubble.tsx` | B (query prop + EvalBadge) |
| `src/components/chat/MessageList.tsx` | B (pass workspaceId down) |
| `src/routes/ChatPage.tsx` | B (set query on messages, pass workspaceId) |

8 files. Phoenix and Eval are independent within each file.

---

## Testing

**Backend:**
- `test_evaluate_endpoint.py`: mock `AnswerEvaluator.evaluate`, verify `/evaluate` returns `{score, gap}`

**Frontend:**
- TypeScript compile check
- Vitest: existing 8 tests remain passing
- No new hook tests needed (EvalBadge uses local component state + fetch)

---

## Environment

Add to `.env.example`:
```
ENABLE_PHOENIX=false   # set to true to start Arize Phoenix OTEL tracing on server startup
```

Install Phoenix (optional): `pip install -e ".[agentic]"`
