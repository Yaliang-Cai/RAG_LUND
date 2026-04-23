# Retrieval Ablation Plan for `graphbm25_20260421`

## Fixed Execution Flow
- Run `rag-anything/scripts/debug_retrieval_ablation.py` first on one DocBench query and one SurGE query.
- In debug mode, use the same named groups as the formal runner by setting `RUN_GROUP_MATRIX = True`.
- Only run the formal runner after debug traces confirm the expected counts, prompt mode, chunk rerank stages, and PPR behavior.
- Keep query cache bypassed (`bypass_query_cache=True`) and keyword cache enabled (`bypass_keywords_cache=False`).

## Error-Prevention Rules
- Do not compare SurGE answer-context groups; SurGE is retrieval-only.
- Do not add `kg_chunk_selection_source` to PPR groups; PPR-ranked chunks replace entity/relation-derived chunk selection.
- Do not enable synonym-edge filtering as a non-PPR experiment axis; non-PPR defaults to excluding synonym edges, PPR uses synonym edges.
- Treat `retrieval_mode=hybrid` as one unified setting for entity/relation VDB retrieval and chunk VDB/candidate-pool retrieval.
- Record `after_chunk_top_k` and final chunk counts for DocBench before interpreting answer quality.

## DocBench Groups
Fixed settings:
- `top_k=40`
- `chunk_top_k=20`
- `max_total_tokens=45000`
- Non-PPR: `exclude_synonym_edges=True`

Groups:
- `baseline_kg`: `hybrid`, `joined`, `dense`, `truncated`, `kg_prompt`
- `per_keyword_kg`: baseline + `keyword_fanout_mode=per_keyword_rrf`
- `retrieval_hybrid_kg`: baseline + `retrieval_mode=hybrid`
- `untruncated_kg`: baseline + `kg_chunk_selection_source=untruncated`
- `baseline_chunk_only`: baseline + `answer_context_mode=chunk_only_prompt`
- `untruncated_chunk_only`: baseline + `kg_chunk_selection_source=untruncated`, `answer_context_mode=chunk_only_prompt`
- `ppr_dense`: `ppr`, `joined`, `dense`, synonym edges enabled, `chunk_only_prompt`
- `ppr_hybrid_per_keyword`: `ppr`, `per_keyword_rrf`, `hybrid`, synonym edges enabled, `chunk_only_prompt`

## SurGE Groups
Fixed settings:
- `top_k=40`
- `chunk_top_k=0`
- `max_total_tokens=45000`
- `kg_chunk_selection_source=untruncated`
- Non-PPR: `exclude_synonym_edges=True`

Groups:
- `baseline`: `hybrid`, `joined`, `dense`
- `per_keyword`: baseline + `keyword_fanout_mode=per_keyword_rrf`
- `retrieval_hybrid`: baseline + `retrieval_mode=hybrid`
- `ppr_dense`: `ppr`, `joined`, `dense`
- `ppr_hybrid_per_keyword`: `ppr`, `per_keyword_rrf`, `hybrid`

## Interpretation Notes
- `chunk_only_prompt` keeps KG retrieval and chunk expansion, but removes entity/relation context from the final answer prompt.
- `untruncated` changes only the KG source set used for chunk candidate selection; it does not place untruncated entity/relation records in the answer prompt.
- `max_total_tokens=45000` is intended to make DocBench `chunk_top_k=20` more stable, but final chunk count can still be lower when prompt overhead or image tokens consume budget.
