# Chunk-Level References in Chat — Design

**Date**: 2026-05-20
**Branch context**: `frontend-modernization`
**Status**: spec approved, awaiting plan

## Problem

The chat answer currently shows references as a flat strip of **file-level** `CitationChip`s under the bubble (`MessageBubble.tsx:159-163`), populated from `_extract_source_nodes` in `server/app.py:879`. That helper dedupes by `filename:page` and caps at 5, collapsing chunk-level granularity. Users cannot:

1. See which specific chunks fed the answer (only which files).
2. Click through to the exact location in the source PDF.

The chunk array is in fact already plumbed end-to-end: the SSE `meta` event carries `data.chunks`, `ChatPage.tsx:21-25` extracts it into `message.chunks`, and `ChunkRef` already exposes `id` / `file_path` / `page_idx` / `content` (`types/index.ts:62-72`). Today the only consumer is `buildChunkMap` for inline `[DC1]` tooltips. The reference strip ignores it.

The Documents page already supports external citation jumps via `setPendingPageNum` + `setSelectedFilename` (DocumentsPage.tsx:32-34). The piece missing is the UI that surfaces chunk-level rows and the jump trigger.

## Scope

**In**: chat page responses returned via `/query/stream` — covers both non-agentic modes (`naive`, `local`, `global`, `hybrid`) **and** agentic mode (same SSE path, same `extractChunks` pipeline in ChatPage:133).

**Out (v1)**:
- Multimodal non-streaming path (`postMultimodalQuery`) — does not populate `message.chunks`; the list naturally won't render. Acceptable degradation.
- InlineCitation → ReferenceList row scroll-into-view linkage (future).
- In-PDF text highlighting (requires bbox coordinates the parser doesn't surface).

## Design

### Backend

**No changes required.** Investigation of the ingestion path showed:

- LightRAG already plumbs `page_idx` end-to-end via `convert_to_user_format` (`lightrag/utils.py:3611`).
- **Multimodal chunks** (image/table/equation descriptions) write `page_idx` to both the chunk KV store and the chunks_vdb on insertion (`raganything/modalprocessors.py:608, 622`) → `data.chunks[i].page_idx` is populated naturally.
- **Plain-text chunks** lose `page_idx`: `separate_content` (`raganything/utils.py:14`) joins all text items with `\n\n` before calling `insert_text_content → lightrag.ainsert(joined_text)`, and LightRAG's chunker stores no page metadata for the resulting chunks → `data.chunks[i].page_idx` is `None`.

Rather than patching the ingestion path (would require re-indexing every existing corpus), the click-through has two branches based on `page_idx`:

- Has `page_idx` (multimodal chunks): jump to the **PDF** at that page using the existing `setPendingPageNum` channel.
- No `page_idx` (text chunks): jump to the **markdown preview** and scroll to the first substring match of the chunk's content, flash-highlight briefly.

This means zero indexing changes and old documents work immediately.

`_extract_source_nodes` becomes unused by the frontend. Keep it for now (the `done` event's `source_nodes` field stays in the wire protocol to avoid version churn); remove in a follow-up after frontend stops reading it.

### Frontend

**New component** `server/frontend/src/components/chat/ReferenceList.tsx`:

- Props: `{ chunks: ChunkRef[] }`.
- Returns `null` if `chunks.length === 0`.
- Local state `open: boolean` (default `false`, not persisted — fresh-fold per reply).
- Header button: `▾ 引用 (N)` / `▸ 引用 (N)`, toggles `open`.
- Body (when open): one row per chunk, in retrieval order.
  - Left: `[<chunk.id ?? index+1>]`.
  - Middle: `<basename(file_path)> · 第 <page_idx> 页` (omit the `· 第 N 页` segment if `page_idx == null`).
  - Below: one-line excerpt from `content` (or `excerpt`/`text` fallback), trimmed to ≤120 chars + `…`.
  - Whole row clickable.
- Click handler:
  1. `useNavigate()` → `/documents`.
  2. `setSelectedFile(basename(file_path))` (Zustand store action).
  3. If `page_idx != null` → `setPendingPageNum(page_idx)` (DocumentsPage switches to PDF tab, PdfViewer jumps).
  4. Else → `setPendingChunkText(chunk.content)` (new store action; MarkdownViewer finds the first substring match and scrolls to it).

### Store + MarkdownViewer changes

- Add `pendingChunkText: string | null` to the Zustand store with `setPendingChunkText`, mirroring the existing `pendingPageNum` shape.
- DocumentsPage forces `tab='markdown'` (not `'pdf'`) when `pendingChunkText` is set, and passes the text to `MarkdownViewer` as a `scrollToText?: string` prop along with an `onScrollComplete?: () => void` callback to clear the pending state once consumed.
- `MarkdownViewer` accepts `scrollToText`; on prop change, after the next render flush it searches the rendered DOM for the first occurrence of `scrollToText.slice(0, 80)` (trimmed of leading/trailing whitespace and punctuation) using `TreeWalker`, calls `range.scrollIntoView`, and flashes a `bg-yellow-300/30` highlight for ~2 seconds. Match failure: silent no-op, file is still opened.

Styling: match the existing collapsed-panel look of `AgenticTrace.tsx` (border, bg-secondary/50, text-xs, max-w-[80%]) so the two panels stack consistently.

**`MessageBubble.tsx` changes**:

1. **Remove** the file-level `CitationChip` strip (lines 159-163, plus the `CitationChip` import on line 5).
2. Insert `<ReferenceList chunks={message.chunks ?? []} />` between the `ReasoningTrace` block (line 152) and the `AgenticTrace` block (line 153), so the final stacking is:

```
bubble
  └─ reasoning trace (if any)
  └─ ReferenceList    ← new
  └─ AgenticTrace     ← existing, agentic-only
```

`InlineCitation` behavior (the `[DC1]` tooltips inside the answer text) is unchanged.

Agentic mode requires **zero additional code**: it uses the same `useStreamQuery` → `extractChunks` → `message.chunks` pipeline, so simply rendering `ReferenceList` unconditionally gives it the same list.

## Data flow

```
LightRAG retrieval
   └─ stream meta event {data: {chunks: [{id, file_path, content, page_idx?…}, …]}}
         │ page_idx populated only for multimodal chunks (image/table/equation)
         ▼
   useStreamQuery → finalMetadata.data.chunks
         │
   ChatPage.extractChunks(snap.metadata) → message.chunks: ChunkRef[]
         │
   MessageBubble → <ReferenceList chunks={message.chunks} />
         │ user clicks row
         ▼
   navigate('/documents') + store.setSelectedFile + (
       page_idx != null  → store.setPendingPageNum   → PdfViewer jumps to page
       page_idx == null  → store.setPendingChunkText → MarkdownViewer scrolls to match
   )
```

## Testing

**Backend**: no changes, no new tests.

**Frontend** (`server/frontend/src/components/chat/__tests__/`):
- `ReferenceList.test.tsx`:
  - Renders nothing when `chunks` is empty.
  - Default state is collapsed (only header visible).
  - Clicking header toggles open/closed.
  - Renders one row per chunk with id / filename / page / excerpt.
  - Excerpt truncated to 120 chars with ellipsis.
  - Row click invokes `setSelectedFile` + navigate; when `page_idx != null` also `setPendingPageNum`; when `page_idx == null` also `setPendingChunkText`.
  - Page suffix `· 第 N 页` hidden when null.

- `MarkdownViewer.test.tsx`:
  - When `scrollToText` prop is set, the matching node is highlighted and `onScrollComplete` is called.
  - When no match found, `onScrollComplete` still called (cleanup), no error.

**Manual acceptance**:
1. Upload a PDF, ask a question in `hybrid` mode → expand list. Click a multimodal chunk row (has 第 N 页) → Documents page opens the PDF at the correct page.
2. Click a plain-text chunk row (no page suffix) → Documents page opens the markdown tab, scrolls to a matching passage, briefly highlights it.
3. Repeat in `agentic` mode → same behavior; Agentic trace still appears below the list.
4. Question in `naive` mode that returns no chunks → no list shown, no error.

## Risks & mitigations

- **Markdown text-match miss** when the chunk straddles token boundaries or contains markdown syntax not present verbatim in the rendered DOM → degrade silently: file is still opened on the markdown tab, just no scroll/highlight.
- **Empty `data.chunks`** in some retrieval modes → `ReferenceList` returns `null`, safe.
- **Wire protocol**: `source_nodes` field in the `done` event stays for now (deprecated, unused) to avoid coupling this change to a protocol break.

## Out-of-scope / future

- Remove `_extract_source_nodes` and the `done.source_nodes` field once nothing reads them.
- Click on `InlineCitation [DC1]` → scroll the matching `ReferenceList` row into view & flash highlight.
- In-PDF text highlighting via bbox (requires parser changes upstream).
