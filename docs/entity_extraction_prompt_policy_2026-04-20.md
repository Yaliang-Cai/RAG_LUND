# Entity Extraction Prompt Policy Update (2026-04-20)

## Scope

- File changed: `lightrag/lightrag/prompt.py`.
- This update changes only prompt text and examples.
- No extraction, merge, normalization, endpoint-check, graph, or vector-store code path is changed.

## Entity Policy

- Entity nodes should represent specific, stable, source-grounded referents.
- Allowed referents include people, organizations, locations, artifacts, works, natural entities, named events, named methods/processes, and stable domain concepts.
- Generic filler phrases are not entity nodes, even when they are useful for explaining a relation.
- Examples of skipped generic fillers: `User Query`, `Retrieved Documents`, `External Web Source`, `Generator`, `Values`, `Layer`, `Representation Subspaces`.
- Generic relationship semantics should be expressed in `relationship_keywords` or `relationship_description`.
- Do not create a generic placeholder entity just to make a relation endpoint valid.

## Relation Policy

- Relations should be as complete as the source text supports.
- Every relation endpoint must match an extracted entity in the same extraction state.
- Do not create shortcut edges when the real middle entity/event is unnamed or intentionally skipped.
- Metrics, role titles, latency, accuracy, power, and similar attributes belong in descriptions, not entity nodes.
- Negated entities can be extracted, but negated facts must not become positive edges.

## Normalization Policy

- The prompt now says to preserve source lexical content.
- Normalized mode may canonicalize casing and separator artifacts only.
- Normalized mode must not rewrite aliases, abbreviations, or lexical meaning.
- Example: `RAG` must not become `Retrieval-Augmented Generation` unless that full lexical form appears in the source text.

## Continuation Prompt

- The continuation prompt documents the actual chat-history shape:
  - previous user message contains the original extraction task and source text;
  - previous assistant message contains the first extraction;
  - continuation output should contain only missed or corrected entries.
- Endpoint closure is evaluated against previous extraction plus the continuation output.
- If a new relation uses one already-extracted endpoint and one missing endpoint, only the missing endpoint entity and the new relation should be output.
- Casing/separator corrections may output the corrected normalized form for the same lexical content.
- The prompt does not promise lexical supersede/delete behavior, because the current code does not implement delete or replacement records for different lexical referents.

## Example Set

- The default extraction examples were replaced with fewer, denser examples.
- The RAG example uses named modules and benchmarks and avoids generic nodes such as `User Query` and `Generator`.
- The metrics/roles example uses a named model (`AtlasLM`) so `AtlasLM -> GLUE` and `AtlasLM -> H100` are valid without a `H100 -> GLUE` shortcut.
- The concept/process example keeps stable methods such as `Attention`, `Self-Attention`, `Multi-Head Attention`, `Backpropagation`, and `Chain Rule`, while removing filler nodes.
- The enterprise example uses source-grounded names such as `Q3 2024 Root Cause Analysis` and `Apollo Remediation Plan`.
- The negation example keeps `GPT-3`, `GPT-4`, and `RLHF`, but does not create a positive `GPT-3 -> RLHF` edge.
- The normalization examples preserve referents, including `OpenAI API Documentation` rather than rewriting it to `OpenAI API`.

## Validation Checklist

- `python -m py_compile lightrag/lightrag/prompt.py`
- Parse extraction and normalization examples and assert every relation endpoint appears in that example's entity list.
- Assert removed generic filler names do not appear as example entity names.
- Assert no `H100 -> GLUE` shortcut edge exists.
- Assert the continuation example does not repeat already-correct entities when adding a missed relation.
