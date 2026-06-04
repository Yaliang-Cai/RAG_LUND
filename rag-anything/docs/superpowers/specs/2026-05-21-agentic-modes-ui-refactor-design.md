# Agentic Modes & UI Refactor — Design Spec

**Date:** 2026-05-21
**Branch:** frontend-modernization
**Status:** draft, awaiting user review

## Goal

Refactor the chat experience around four user-facing query modes (Naive,
LightRAG, Multi-hop, Agentic) backed by a slimmed-down profile registry
(`semantic`, `multihop`, `full`), and expose retrieval-tuning parameters in a
LightRAG-webui–style settings UI: a persistent control strip above the chat
input plus a collapsible "Advanced Options" panel, with controls scoped per
mode.

Frontend strings are all in English.

## Non-Goals

- No changes to ingestion / parsing / VLM pipelines.
- No backend storage of per-user UI settings; all settings live in the
  request body and a frontend-only Zustand store.
- No new evaluation harness; existing `evaluate_local/` scripts that
  reference removed profile names will be updated or have their cases dropped.

## 1. Profile Registry Changes — `raganything/retrieval/profiles.py`

**Delete:** `precise`, `local`, `gfm_multihop`, the existing `semantic`,
the existing `multihop`.

**Add `mix` to `KNOWN_PATHS`.**

**New / kept profiles:**

| profile    | paths                  | rrf_weights                       | enable_rerank | notes        |
|------------|------------------------|-----------------------------------|---------------|--------------|
| `semantic` | `mix`, `qdrant_hybrid` | `mix: 1.0, qdrant_hybrid: 0.8`    | `True`        | LightRAG mix + vector hybrid |
| `multihop` | `ppr`, `qdrant_hybrid` | `ppr: 1.0, qdrant_hybrid: 0.8`    | `False`       | PPR + vector hybrid          |
| `full`     | unchanged              | unchanged                         | `True`        | LLM-classifier fallback      |

Removing profiles is a breaking change for `evaluate_local/` configs and any
cached router results. The router cache schema does not need to change, but
stale cache entries naming dropped profiles will be ignored on read.

## 2. Router Support for `mix` Path

`raganything/retrieval/router.py` currently dispatches to known path executors.
Verify whether a `mix` executor exists; if not, add one that calls
`LightRAG.aquery(mode="mix", ...)` and adapts the result to the router's
`PathResult` shape.

**Implementation gate:** this is the riskiest unknown. The first implementation
task is to confirm/extend router path support before touching profiles or tests.

## 3. Naive Mode Backend Path

Naive mode does **not** go through the router/profile system. It uses the
existing `mode='naive'` branch in `QueryMixin.aquery`. Two enhancements:

- Pass `qdrant_retrieval_mode='hybrid'` and `enable_rerank=True` through.
- Apply a **rerank candidate cap = 4 × chunk_top_k**: only the top
  `4 × chunk_top_k` chunks by vector similarity are sent to the reranker.

**Cap implementation:** check whether `QueryParam` in LightRAG accepts
`rerank_candidate_cap`. If yes, pass it through. If not, implement the cap
inside `QueryMixin.aquery`'s naive branch: truncate the retrieved chunk list
to `4 × chunk_top_k` before invoking the reranker.

Only applied when `mode == 'naive'` and `enable_rerank` is true.

## 4. Backend API Surface — `server/app.py`

### 4.1 `QueryRequest` new fields

All Optional; `None` means "use backend default". Added alongside existing
`mode` / `profile`:

```python
top_k: Optional[int] = None
chunk_top_k: Optional[int] = None
enable_rerank: Optional[bool] = None
qdrant_retrieval_mode: Optional[str] = None        # 'hybrid' | 'sparse' | 'dense'
rerank_candidate_cap: Optional[int] = None         # naive mode only
ppr_damping: Optional[float] = None                # multihop only
ppr_top_k: Optional[int] = None                    # multihop only
ppr_synonym_weight_mode: Optional[str] = None      # multihop only
recognition_top_k: Optional[int] = None            # multihop only
```

### 4.2 Request forwarding

`/query/stream` and `/query` build the kwargs dict, drop `None` values, and
pass to `aquery(**kwargs)`. The kwargs already pass through to LightRAG's
`QueryParam` via `query.py` (confirmed: all the PPR/recognition kwargs are in
the relevant_kwargs list at `raganything/query.py:180-218`).

### 4.3 Mode dispatch

| Frontend `mode` value | Backend `mode` | Backend `profile`           |
|-----------------------|----------------|-----------------------------|
| `naive`               | `naive`        | n/a                         |
| `lightrag`            | `auto`         | `semantic`                  |
| `multihop`            | `auto`         | `multihop`                  |
| `agentic`             | `auto`         | from UI (or `None` = Auto)  |

`mode='auto'` with `profile=<name>` short-circuits the LLM classifier and
locks the chosen profile (existing router behavior).

## 5. Mode → Preset Mapping (Single Source of Truth)

Lives in `server/frontend/src/config/modePresets.ts`. Values must mirror the
backend constants in `raganything/constants.py`.

| field                     | Naive  | LightRAG | Multi-hop | Agentic    |
|---------------------------|--------|----------|-----------|------------|
| `mode` (sent to backend)  | naive  | auto     | auto      | auto       |
| `profile`                 | —      | semantic | multihop  | undefined  |
| `top_k`                   | hidden | 10       | 10        | 10         |
| `chunk_top_k`             | 5      | 5        | 5         | 5          |
| `enable_rerank`           | true   | true     | **false** | true       |
| `qdrant_retrieval_mode`   | hybrid | hybrid   | hybrid    | hybrid     |
| `rerank_candidate_cap`    | 20     | —        | —         | —          |
| `ppr_damping`             | —      | —        | 0.5       | —          |
| `ppr_top_k`               | —      | —        | (const)   | —          |
| `ppr_synonym_weight_mode` | —      | —        | (const)   | —          |
| `recognition_top_k`       | —      | —        | (const)   | —          |

`(const)` = pull from `constants.py` defaults (`DEFAULT_PPR_TOP_K`,
`DEFAULT_PPR_SYNONYM_WEIGHT_MODE`, `DEFAULT_RECOGNITION_TOP_K`). These values
are duplicated into `modePresets.ts` so the UI can render and reset them
without an API round-trip; spec notes this mirroring requirement.

## 6. Frontend UI

### 6.1 Layout

Inspired by `lightrag/lightrag_webui/src/components/retrieval/QuerySettings.tsx`,
adapted to the chat flow:

```
┌────────────────────────────────────────────────────────────┐
│  Chat history                                              │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  [Mode▾] [top_k] [chunk_top_k] [☐ Rerank]        [⚙ Advanced ▾] │   ← persistent strip
├────────────────────────────────────────────────────────────┤
│  ▼ Advanced Options                                        │   ← collapsible
│    Qdrant retrieval mode: [hybrid ▾]                       │
│    (mode-conditional fields below)                         │
├────────────────────────────────────────────────────────────┤
│  [ chat input box                                        ] │
│  [ Send ]                                                  │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Persistent control strip

Each control: label (with tooltip), input/select, small `RotateCcw` reset
button that resets to the **current mode's preset value**.

- `Mode` select: Naive / LightRAG / Multi-hop / Agentic (default Agentic).
- `top_k`: number input; hidden when mode = Naive.
- `chunk_top_k`: number input.
- `Rerank`: checkbox.

### 6.3 Advanced Options panel (collapsible, default collapsed)

Universal:
- `Qdrant retrieval mode`: select (hybrid / sparse / dense).

Mode-conditional:
- **Agentic only:** `Profile` select — Auto (LLM) / semantic / multihop / full.
- **Multi-hop only:** `PPR damping`, `PPR top_k`, `PPR synonym weight mode`,
  `Recognition top_k`.
- **Naive:** no extra fields; panel shows "No advanced options for this mode."
- **LightRAG:** no extra fields beyond universal.

### 6.4 Mode-switch behavior

Switching `Mode` **resets all settings to the new mode's preset.** Any
overrides the user had made in the previous mode are discarded. This is
explicit per user requirement; the reset is silent (no confirmation prompt).

### 6.5 State management

New Zustand slice `frontend/src/store/querySettings.ts`:

```ts
interface QuerySettingsState {
  mode: ModeKey
  // Effective current values, populated from preset on mode change
  // and overwritten by user edits within a session.
  top_k?: number
  chunk_top_k: number
  enable_rerank: boolean
  qdrant_retrieval_mode: 'hybrid' | 'sparse' | 'dense'
  rerank_candidate_cap?: number
  profile?: AgenticProfile        // agentic mode only
  ppr_damping?: number             // multihop only
  ppr_top_k?: number               // multihop only
  ppr_synonym_weight_mode?: string // multihop only
  recognition_top_k?: number       // multihop only

  setMode(m: ModeKey): void        // applies preset, discards prior overrides
  set<K extends keyof State>(k: K, v: State[K]): void
  resetField(k: keyof State): void // back to current-mode preset
}
```

### 6.6 Localization

All UI strings in English; no i18n keys for this iteration.

## 7. Files Touched

### Backend
- `raganything/retrieval/profiles.py` — rewrite registry
- `raganything/retrieval/router.py` — add `mix` path executor if missing
- `raganything/retrieval/classifier.py` — update LLM prompt profile list
- `raganything/query.py` — naive branch: candidate cap implementation if
  `QueryParam` doesn't support `rerank_candidate_cap`
- `server/app.py` — `QueryRequest` field additions, kwargs forwarding

### Frontend
- `server/frontend/src/config/modePresets.ts` (new)
- `server/frontend/src/store/querySettings.ts` (new)
- `server/frontend/src/components/chat/QuerySettings.tsx` (new — persistent strip)
- `server/frontend/src/components/chat/AdvancedOptions.tsx` (new — collapsible)
- `server/frontend/src/components/chat/ChatInput.tsx` — remove internal
  mode/profile state, read from store
- `server/frontend/src/routes/ChatPage.tsx` — `handleSend` reads full
  param set from store
- `server/frontend/src/types/index.ts` — extend `QueryParams`
- `server/frontend/src/api/query.ts` — forward new fields

### Tests
- `tests/retrieval/test_profiles.py` — drop precise/local/gfm_multihop;
  add new semantic/multihop assertions
- `tests/retrieval/test_classifier.py` — update prompt fixture
- `tests/retrieval/test_router*.py` — add `mix` path test
- `tests/retrieval/test_profiles_gfm.py` — drop or rescope (gfm path retained
  in `KNOWN_PATHS` but no longer in any profile)
- `evaluate_local/run_ablation_evals.py`, `run_retrieval_ablation.py`,
  `evaluate_surge_fast.py` — update references to dropped profile names

## 8. Risks and Open Questions

1. **`mix` as a router path** — biggest unknown. If `router.py` cannot easily
   dispatch to LightRAG's `mix` query mode, the `semantic` profile semantics
   collapse to just `qdrant_hybrid`, which is the old `semantic` definition,
   defeating the change. **Mitigation:** first implementation task is to
   verify and extend the router; if blocked, return to spec.
2. **`rerank_candidate_cap` support in LightRAG `QueryParam`** — if not
   present, implement the truncation in `QueryMixin.aquery`'s naive branch.
   Either way the user-facing behavior is identical.
3. **Evaluation regression** — `evaluate_local/ablation_runs/graph20260421/`
   has saved configs referencing old profile names. We do **not** retroactively
   rewrite those JSON artifacts; new runs use new names.
4. **Cache invalidation** — `router_cache` entries keyed on dropped profile
   names will simply miss. No active eviction needed.

## 9. Acceptance Criteria

- The four UI modes are selectable and each loads its preset values on switch.
- Switching modes discards prior overrides.
- Agentic mode's Advanced panel exposes a `profile` select; Multi-hop exposes
  the four PPR/recognition fields; Naive hides advanced fields entirely.
- Backend `/query/stream` accepts all new fields and forwards them to
  `aquery`.
- Naive mode applies a rerank candidate cap of `4 × chunk_top_k`.
- All existing tests pass after registry rewrite; new profile-shape tests
  pass.
- No frontend strings in non-English.
