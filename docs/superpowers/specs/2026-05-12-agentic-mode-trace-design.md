# Agentic Mode + Trace Panel — Design Spec

Date: 2026-05-12  
Branch: frontend-modernization

## Goal

Add `ppr`, `auto`, `agentic` to the chat mode selector and show a structured trace panel below assistant messages, matching the trace output from `query_ppr.py`.

---

## Scope

**In scope:**
- ChatInput: 7-mode dropdown + conditional profile selector (auto mode only)
- New `AgenticTrace` component replacing `ReasoningTrace`
- Backend: `stream_query` agentic branch using `query_with_trace`
- `QueryRequest` / `QueryParams` type extension

**Out of scope:** RAGAS evaluation, Phoenix UI integration, VLM image upload, conversation history

---

## 1. ChatInput Changes

**Mode selector** — extend from 4 to 7 options:
`naive | local | global | hybrid | ppr | auto | agentic`

**Profile selector** — new `<Select>` appearing in the same controls row:
- Visible at all times, disabled (grayed out) unless mode = `auto`
- Options: `— auto detect —` (value: `""`) / `precise` / `local` / `multihop` / `descriptive` / `full`
- Default: `""` (empty = let LLM classifier decide)
- `onSend` signature extends to `(query: string, mode: string, profile: string) => void`

---

## 2. AgenticTrace Component

Replaces `ReasoningTrace`. File: `src/components/chat/AgenticTrace.tsx`

**Props:**
```ts
interface AgenticTraceProps {
  traceType: 'agentic' | 'auto' | 'ppr' | null
  metadata: Record<string, unknown>
}
```

**Behavior:**
- Returns `null` if `traceType` is null or no relevant keys present
- Default state: **expanded** (for demo screenshots)
- Click header row to toggle collapse

**Rendering by trace type:**

| traceType | Header pills | Grid fields |
|---|---|---|
| `agentic` | grounded (green/red) · conf · profile | profile, cache_hit, retrieve_cycles, check_cycles |
| `auto` | profile · conf · paths | paths_activated, chunks_after_rrf, chunks_after_rerank, latency summary |
| `ppr` | chunks · entities · relations | top_chunk_score |

**Styling:** uses `bg-secondary`, `border-border`, `text-muted-foreground` — adapts to both light and dark themes. Grounded = green (`text-green-500`), not grounded = red (`text-red-500`).

---

## 3. Types & API

**`src/types/index.ts`:**
```ts
// Extend QueryParams
export interface QueryParams {
  // ...existing...
  mode?: 'naive' | 'local' | 'global' | 'hybrid' | 'ppr' | 'auto' | 'agentic'
  profile?: string  // auto mode only; empty string = omit from request
}

// Extend StreamMetaEvent metadata shape (informational — no runtime change needed)
// metadata.agentic_trace?: { confidence, grounded, profile, router_cache_hit,
//                             retrieve_cycles_used, check_cycles_used,
//                             rewrite_history, sub_questions }
// metadata.routing?:       { profile, confidence, paths_activated, ... }
```

**`src/api/query.ts`:**
- Pass `profile` in body only when mode = `auto` and profile is non-empty

---

## 4. ChatPage Changes

- `handleSend` receives `(query, mode, profile)` from `ChatInput`
- After streaming completes, read `metadata` from the final `meta` event (already stored in `useStreamQuery`)
- Determine `traceType`:
  - `mode === 'agentic'` → `'agentic'`
  - `mode === 'auto'` → `'auto'`
  - `mode === 'ppr'` → `'ppr'`
  - else → `null`
- Pass `traceType` + `metadata` to `AgenticTrace` in each assistant `MessageBubble`

**Message type extension:**
```ts
// in MessageBubble types
traceType?: 'agentic' | 'auto' | 'ppr' | null
traceMetadata?: Record<string, unknown>
```

---

## 5. Backend Changes

### `server/app.py` — QueryRequest

Add field:
```python
profile: Optional[str] = None  # auto mode only
```

Pass to `stream_query`:
```python
profile=payload.profile if payload.mode == "auto" else None,
```

### `raganything/services/local_rag.py` — stream_query

Add `profile: str | None = None` parameter.

Add agentic branch **before** the existing `aquery_llm` path:

```python
if mode == "agentic":
    # Non-streaming: call query_with_trace, emit as single chunk
    result = await self.query_with_trace(
        workspace_id, query,
        mode="agentic", return_trace=True, ...
    )
    trace = result.get("trace", {})
    yield {
        "type": "meta",
        "data": {},
        "metadata": {
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
        },
    }
    yield {"type": "chunk", "text": result.get("answer", "")}
    return
```

For `auto` mode with `profile`: pass `profile=profile` into existing `QueryParam` construction.

---

## 6. useStreamQuery Hook

Currently stores `answer`, `reasoning`, `status`, `sourceNodes`. Need to also expose `metadata`:

```ts
// Add to hook return:
metadata: Record<string, unknown>
```

The `meta` event already arrives first — just store `event.metadata` in a ref.

---

## File Checklist

| File | Change |
|---|---|
| `src/types/index.ts` | Extend `QueryParams.mode`, add `profile?`; extend `Message` type |
| `src/api/query.ts` | Pass `profile` conditionally |
| `src/hooks/useStreamQuery.ts` | Expose `metadata` from meta event |
| `src/components/chat/ChatInput.tsx` | 7 modes + profile selector |
| `src/components/chat/AgenticTrace.tsx` | New component (replaces ReasoningTrace) |
| `src/components/chat/MessageBubble.tsx` | Accept + render `AgenticTrace` |
| `src/routes/ChatPage.tsx` | Wire profile, traceType, metadata |
| `server/app.py` | `QueryRequest.profile`, pass to stream_query |
| `raganything/services/local_rag.py` | `stream_query` agentic branch |
