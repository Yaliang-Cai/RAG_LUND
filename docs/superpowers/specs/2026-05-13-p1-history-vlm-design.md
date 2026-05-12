# P1: Multi-turn History + VLM Enhanced Query — Design Spec

Date: 2026-05-13  
Branch: frontend-modernization

## Goal

Two independent features on top of the P0 chat:
1. **Multi-turn conversation history** — pass recent chat turns to the LLM for context-aware follow-up questions
2. **VLM enhanced query** — toggle that routes queries through `aquery_vlm_enhanced`, letting the VLM reason over images already indexed in the knowledge base

---

## Scope

**In scope:**
- Conversation history: last N turns (default 5) sent as `QueryParam.conversation_history`
- VLM enhanced: `vlm_enhanced` toggle in ChatInput; stream_query non-streaming branch reusing `query_with_trace`
- `QueryRequest` additions for both features
- Frontend-only state management for history (no server-side session store)

**Out of scope:**
- `history_summary` compression (QueryParam doesn't support it in lightrag 1.4.15)
- User-uploaded images in chat (P2)
- Server-side session persistence

---

## Sub-system 1: Conversation History

### How it works

LightRAG's `QueryParam.conversation_history: list[dict[str, str]]` already exists and is forwarded to `llm_model_func` as `history_messages`. We just need to wire it from the frontend request through to `QueryParam`.

### Frontend

**History format:** `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`

**Truncation rule:** Send the last `MAX_HISTORY_TURNS = 5` complete turns (10 messages) before the current query. Applied client-side in `ChatPage.handleSend`. No UI setting needed.

**State:** `messages` array already in `ChatPage` — slice the last 10 items before the current user message to build history.

### Backend

**`QueryRequest`** — add:
```python
conversation_history: list[dict[str, str]] = Field(default_factory=list)
```

**`stream_query`** — add `conversation_history` param and pass to `QueryParam`:
```python
conversation_history: list[dict[str, str]] | None = None,
# ...
param = QueryParam(
    ...
    conversation_history=conversation_history or [],
)
```

**agentic/auto+profile branch** — pass `conversation_history` to `query_with_trace`:
```python
result = await self.query_with_trace(
    workspace_id, query,
    ...
    conversation_history=conversation_history or [],
)
```

### File changes

| File | Change |
|---|---|
| `server/app.py` | `QueryRequest.conversation_history` field |
| `raganything/services/local_rag.py` | `stream_query` param + QueryParam construction + agentic branch |
| `src/api/query.ts` | Send `conversation_history` in body |
| `src/routes/ChatPage.tsx` | Build history slice before calling `send()` |
| `src/types/index.ts` | `QueryParams.conversation_history?` |

---

## Sub-system 2: VLM Enhanced Query

### How it works

When `vlm_enhanced=True`:
1. `aquery_vlm_enhanced()` runs standard RAG retrieval (any mode)
2. Finds `Image Path:` markers in retrieved chunks
3. Encodes those images to base64
4. Sends text + images to VLM for a multimodal answer

`vlm_enhanced` field already exists in `QueryRequest` and the non-streaming query endpoint already passes it. The gap is that `stream_query` ignores it.

### Frontend

**ChatInput** — add a `VLM` toggle chip next to the mode selector. Small, unobtrusive:
- Default: off
- When on: sends `vlm_enhanced: true` in request body
- Visual: a small camera icon badge or labelled toggle button

### Backend

**`stream_query`** — add VLM branch alongside the existing agentic branch:

```python
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
        yield {"type": "meta", "data": {}, "metadata": {}}
        yield {"type": "chunk", "text": result.get("answer", result) if isinstance(result, dict) else str(result)}
    except Exception as exc:
        self.logger.error("stream_query (vlm branch) error: %s", exc)
        yield {"type": "error", "text": str(exc)}
    return
```

**`stream_query` signature** — add `vlm_enhanced: bool = False`.

**`QueryRequest` endpoint** — `vlm_enhanced` already present and passed; just ensure `stream_query` receives it.

### File changes

| File | Change |
|---|---|
| `raganything/services/local_rag.py` | `stream_query` signature + VLM branch |
| `src/components/chat/ChatInput.tsx` | VLM toggle chip |
| `src/api/query.ts` | Send `vlm_enhanced` when true |
| `src/types/index.ts` | `QueryParams.vlm_enhanced?: boolean` |

---

## Combined File Map

| File | Sub-system |
|---|---|
| `server/app.py` | History |
| `raganything/services/local_rag.py` | History + VLM |
| `src/types/index.ts` | History + VLM |
| `src/api/query.ts` | History + VLM |
| `src/routes/ChatPage.tsx` | History |
| `src/components/chat/ChatInput.tsx` | VLM toggle |

6 files total. History and VLM touch the same files but are independent changes within each file — can be implemented in sequence without conflicts.

---

## Testing

**Backend (pytest):**
- `test_stream_query_history.py`: verify `conversation_history` is passed to `QueryParam` in the streaming path
- `test_stream_query_vlm.py`: verify VLM branch calls `query_with_trace(vlm_enhanced=True)` and yields meta+chunk

**Frontend (vitest):**
- `useStreamQuery`: existing tests remain passing
- No new hook tests needed (history logic is in ChatPage, VLM is a boolean toggle)
- TypeScript compile check covers interface correctness

---

## Non-goals / Deferred

- `history_summary` compression: deferred until lightrag upstream adds `QueryParam.history_summary`
- User image upload in chat: P2
- Session persistence across page reload: P2
