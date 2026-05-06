# Retrieval Ablation Results for `graphbm25_20260421`

This file summarizes the reduced retrieval ablation results produced from:

```bash
python run_retrieval_ablation.py \
  --tasks both \
  --matrix-mode reduced \
  --run-id retrieval_0424_graph0421 \
  --allow-legacy-index-profile-adoption
```

All metric values below are transcribed exactly from the provided result outputs.

## Fixed Settings

### DocBench

- Dataset/task: DocBench shared retrieval + answer generation + judge evaluation
- `top_k=40`
- `chunk_top_k=20`
- `max_total_tokens=45000`
- `multimodal_top_k=3`
- `keyword_entity_rrf_k=10`
- `keyword_relation_rrf_k=20`
- Non-PPR groups: `exclude_synonym_edges=True`
- PPR groups: `exclude_synonym_edges=False`
- PPR groups: `ppr_top_k=50`, `ppr_qa_top_k=20`
- `entity_retrieval_mode` and `chunk_retrieval_mode` follow `retrieval_mode`

### SurGE

- Dataset/task: SurGE retrieval-only evaluation
- `top_k=40`
- `chunk_top_k=0`
- `max_total_tokens=45000`
- `keyword_entity_rrf_k=10`
- `keyword_relation_rrf_k=20`
- Non-PPR groups: `exclude_synonym_edges=True`
- PPR groups: `exclude_synonym_edges=False`
- Non-PPR groups: `kg_chunk_selection_source=untruncated`
- PPR groups: `ppr_top_k=50`, `ppr_qa_top_k=50`
- `entity_retrieval_mode` and `chunk_retrieval_mode` follow `retrieval_mode`

## Switch Glossary

This section explains each retrieval switch in plain language, without assuming the reader has seen the code.

### `query_mode`

This is the highest-level retrieval family.

- `hybrid`
  - This is the standard non-PPR graph retrieval path.
  - It first extracts keywords from the query.
  - It retrieves entities and relations from vector/BM25/hybrid stores using those keywords.
  - It merges local and global KG evidence.
  - It then collects candidate chunks from the selected entities and relations.
  - After that, chunk rerank and final prompt construction happen downstream.

- `ppr`
  - This is the graph-propagation path.
  - It still starts from query-derived seeds, but then uses Personalized PageRank on the KG.
  - It produces a ranked graph-based chunk list first.
  - The final QA context only keeps up to `ppr_qa_top_k` PPR chunks.
  - In the current implementation, PPR skips KG entity/relation rerank; only chunk rerank may still happen.

### `keyword_fanout_mode`

This controls how extracted keywords are sent into retrieval.

- `joined`
  - Keywords at the same level are concatenated into one retrieval query.
  - This is the simpler and more conservative setting.

- `per_keyword_rrf`
  - Each keyword is queried independently.
  - The per-keyword result lists are then fused with Reciprocal Rank Fusion (RRF).
  - This is intended to reduce the chance that one keyword is drowned out by other terms in a single joined query.

### `retrieval_mode`

This controls which retrieval backend is used for entity VDB lookup and chunk candidate-pool lookup.

- `dense`
  - Embedding/vector retrieval only.

- `bm25`
  - Lexical sparse retrieval only.

- `hybrid`
  - Dense and BM25 are fused inside the retrieval backend.

In this experiment suite, `retrieval_mode` is unified:

- `entity_retrieval_mode = retrieval_mode`
- `chunk_retrieval_mode = retrieval_mode`

So when a group is marked `retrieval_mode=hybrid`, both entity retrieval and chunk candidate-pool retrieval use hybrid retrieval, not just one of them.

### `exclude_synonym_edges`

This controls whether SYNONYM edges in the KG are filtered during query-time graph expansion.

- `True`
  - SYNONYM relations are removed from the query-time graph evidence.
  - This usually reduces trivial synonym spreading and graph noise.

- `False`
  - SYNONYM relations are kept.
  - In this experiment suite, PPR groups keep synonym edges enabled.

Important:

- This is a query-time graph filtering switch.
- It is different from build-time synonym linking.

### `kg_chunk_selection_source`

This only matters for non-PPR KG retrieval.

It controls which entity/relation set is allowed to contribute chunks to the downstream chunk pool.

- `truncated`
  - Chunks are collected only from the post-truncation entity/relation set.
  - This is stricter and usually lower-noise.

- `untruncated`
  - Chunks may be collected from the larger pre-truncation KG evidence set.
  - This is designed to expand chunk recall even if the final entity/relation prompt context was already trimmed.

This switch does **not** mean the answer prompt directly contains untruncated entities/relations.
It only changes the source set used for chunk candidate collection.

### `answer_context_mode`

This only matters for DocBench, because DocBench is answer-generation based.

- `kg_prompt`
  - The final answer prompt contains KG context plus chunk context.
  - In practice this means entity/relation context consumes part of the total token budget before chunks are packed into the prompt.

- `chunk_only_prompt`
  - The final answer prompt contains chunk evidence only.
  - KG retrieval still happens upstream.
  - The difference is only in the final answer prompt context, not in whether the KG is used during retrieval.

### `enable_rerank`

This controls downstream chunk reranking.

- `True`
  - Candidate chunks are reranked against the query before final chunk selection.

- `False`
  - Chunk rerank is skipped.

In PPR groups, this is the main rerank comparison axis.

### `enable_kg_rerank`

This controls reranking of KG entities and KG relations.

- `True`
  - After local/global KG evidence is merged, entities and relations are reranked by a reranker model.

- `False`
  - The merged entity/relation order is kept as-is.

Important:

- `enable_kg_rerank` is independent from `enable_rerank`.
- In other words:
  - `enable_kg_rerank` controls entity/relation rerank.
  - `enable_rerank` controls chunk rerank.

In the current implementation, PPR groups do not use KG entity/relation rerank, so `enable_kg_rerank` is effectively irrelevant there.

### `top_k`

This is the main retrieval breadth for entity/relation retrieval.

- Higher `top_k` means more graph candidates are admitted into the retrieval pipeline.
- Lower `top_k` means stricter early filtering.

In this experiment:

- DocBench: `top_k=40`
- SurGE: `top_k=40`

### `chunk_top_k`

This is the post-rerank chunk window before final token-budget packing.

- If `chunk_top_k=20`, then after chunk rerank, at most 20 chunks are kept before the final prompt budget is applied.
- If `chunk_top_k=0`, then this hard top-k truncation is disabled at that stage.

Important:

- `chunk_top_k` does **not** guarantee that the final prompt will still contain that many chunks.
- The final chunk count can still be smaller because of token-budget packing.

In this experiment:

- DocBench: `chunk_top_k=20`
- SurGE: `chunk_top_k=0`

### `max_total_tokens`

This is the final context budget used when packing the answer/retrieval context.

- It is not “all available for chunks”.
- Fixed prompt overhead is removed first.
- Then the remaining budget is used to pack chunk context.

For `kg_prompt`, entity and relation context also consume budget before chunk packing is finalized.

In this experiment:

- DocBench: `max_total_tokens=45000`
- SurGE: `max_total_tokens=45000`

### `multimodal_top_k`

This only matters for DocBench VLM-enhanced answering.

- It caps how many image references can be promoted into actual multimodal VLM inputs.
- It does **not** mean “retrieve only 3 multimodal chunks”.
- It is a cap on how many image paths are used downstream after chunk selection.

In this experiment:

- DocBench: `multimodal_top_k=3`

### `keyword_entity_rrf_k` and `keyword_relation_rrf_k`

These two are the RRF smoothing constants used only when `keyword_fanout_mode=per_keyword_rrf`.

- `keyword_entity_rrf_k`
  - Used when fusing per-keyword entity retrieval lists.

- `keyword_relation_rrf_k`
  - Used when fusing per-keyword relation retrieval lists.

Larger values flatten the fusion more; smaller values make top-ranked items matter more.

In this experiment:

- `keyword_entity_rrf_k=10`
- `keyword_relation_rrf_k=20`

### `ppr_top_k`

This is the PPR retrieval breadth.

- It controls how many graph/PPR candidates are retained at the retrieval stage.
- Larger values make PPR broader.

In this experiment:

- DocBench PPR: `ppr_top_k=50`
- SurGE PPR: `ppr_top_k=50`

### `ppr_qa_top_k`

This is the PPR QA-context cap.

- PPR retrieval may first rank a broader set up to `ppr_top_k`.
- But only the top `ppr_qa_top_k` PPR chunks are allowed into the downstream QA/final-chunk pipeline.

So:

- `ppr_top_k` controls retrieval breadth.
- `ppr_qa_top_k` controls how many PPR chunks survive into the answer/retrieval context stage.

In this experiment:

- DocBench PPR: `ppr_qa_top_k=20`
- SurGE PPR: `ppr_qa_top_k=50`

## Query Flow Summary

This is the simplest way to understand what each family is doing end-to-end.

### Non-PPR (`query_mode=hybrid`)

1. Extract low-level and high-level keywords from the query.
2. Retrieve entity candidates and relation candidates.
3. Merge local and global KG evidence.
4. Optionally KG-rerank entities and relations if `enable_kg_rerank=True`.
5. Collect chunk candidates from the selected KG evidence.
6. Optionally rerank chunks if `enable_rerank=True`.
7. Apply `chunk_top_k`.
8. Apply final token-budget packing.
9. Build either `kg_prompt` or `chunk_only_prompt` for DocBench, or export retrieval results for SurGE.

### PPR (`query_mode=ppr`)

1. Extract keywords and obtain PPR seed evidence.
2. Run Personalized PageRank on the KG.
3. Produce ranked PPR chunks.
4. Keep only the top `ppr_qa_top_k` PPR chunks for downstream use.
5. Optionally rerank chunks if `enable_rerank=True`.
6. Build `chunk_only_prompt` for DocBench, or export retrieval results for SurGE.

In the current implementation, PPR does not use KG entity/relation rerank as a main stage.

## Experiment Settings

### DocBench Groups

| Group | query_mode | keyword_fanout_mode | retrieval_mode | kg_chunk_selection_source | answer_context_mode | chunk rerank | KG rerank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `docbench_baseline_chunk_only` | `hybrid` | `joined` | `dense` | `truncated` | `chunk_only_prompt` | `True` | `True` |
| `docbench_baseline_kg` | `hybrid` | `joined` | `dense` | `truncated` | `kg_prompt` | `True` | `True` |
| `docbench_per_keyword_kg` | `hybrid` | `per_keyword_rrf` | `dense` | `truncated` | `kg_prompt` | `True` | `True` |
| `docbench_per_keyword_no_kg_rerank_kg` | `hybrid` | `per_keyword_rrf` | `dense` | `truncated` | `kg_prompt` | `True` | `False` |
| `docbench_retrieval_hybrid_kg` | `hybrid` | `joined` | `hybrid` | `truncated` | `kg_prompt` | `True` | `True` |
| `docbench_untruncated_chunk_only` | `hybrid` | `joined` | `dense` | `untruncated` | `chunk_only_prompt` | `True` | `True` |
| `docbench_untruncated_kg` | `hybrid` | `joined` | `dense` | `untruncated` | `kg_prompt` | `True` | `True` |
| `docbench_ppr_dense_no_rerank` | `ppr` | `joined` | `dense` | `N/A` | `chunk_only_prompt` | `False` | `False` |
| `docbench_ppr_dense_rerank` | `ppr` | `joined` | `dense` | `N/A` | `chunk_only_prompt` | `True` | `False` |
| `docbench_ppr_hybrid_per_keyword` | `ppr` | `per_keyword_rrf` | `hybrid` | `N/A` | `chunk_only_prompt` | `True` | `False` |

### SurGE Groups

| Group | query_mode | keyword_fanout_mode | retrieval_mode | kg_chunk_selection_source | chunk rerank | KG rerank |
| --- | --- | --- | --- | --- | --- | --- |
| `surge_baseline` | `hybrid` | `joined` | `dense` | `untruncated` | `True` | `True` |
| `surge_per_keyword` | `hybrid` | `per_keyword_rrf` | `dense` | `untruncated` | `True` | `True` |
| `surge_per_keyword_no_kg_rerank` | `hybrid` | `per_keyword_rrf` | `dense` | `untruncated` | `True` | `False` |
| `surge_retrieval_hybrid` | `hybrid` | `joined` | `hybrid` | `untruncated` | `True` | `True` |
| `surge_ppr_dense_no_rerank` | `ppr` | `joined` | `dense` | `N/A` | `False` | `False` |
| `surge_ppr_dense_rerank` | `ppr` | `joined` | `dense` | `N/A` | `True` | `False` |
| `surge_ppr_hybrid_per_keyword` | `ppr` | `per_keyword_rrf` | `hybrid` | `N/A` | `True` | `False` |

## DocBench Results

### Overall Accuracy

| Group | overall accuracy | correct | total |
| --- | ---: | ---: | ---: |
| `docbench_baseline_chunk_only` | `58.08580858085809` | `176` | `303` |
| `docbench_baseline_kg` | `55.44554455445545` | `168` | `303` |
| `docbench_per_keyword_kg` | `56.76567656765676` | `172` | `303` |
| `docbench_per_keyword_no_kg_rerank_kg` | `55.44554455445545` | `168` | `303` |
| `docbench_ppr_dense_no_rerank` | `56.76567656765676` | `172` | `303` |
| `docbench_ppr_dense_rerank` | `59.4059405940594` | `180` | `303` |
| `docbench_ppr_hybrid_per_keyword` | `59.07590759075908` | `179` | `303` |
| `docbench_retrieval_hybrid_kg` | `54.78547854785478` | `166` | `303` |
| `docbench_untruncated_chunk_only` | `58.745874587458744` | `178` | `303` |
| `docbench_untruncated_kg` | `55.44554455445545` | `168` | `303` |

### By-Type Accuracy

| Group | meta-data | text-only | multimodal-t | multimodal-f | unanswerable | una-web |
| --- | --- | --- | --- | --- | --- | --- |
| `docbench_baseline_chunk_only` | `8.333333333333332 (4/48)` | `73.77049180327869 (45/61)` | `75.65217391304347 (87/115)` | `73.80952380952381 (31/42)` | `30.0 (9/30)` | `0.0 (0/7)` |
| `docbench_baseline_kg` | `4.166666666666666 (2/48)` | `70.49180327868852 (43/61)` | `74.78260869565217 (86/115)` | `76.19047619047619 (32/42)` | `16.666666666666664 (5/30)` | `0.0 (0/7)` |
| `docbench_per_keyword_kg` | `6.25 (3/48)` | `72.1311475409836 (44/61)` | `74.78260869565217 (86/115)` | `76.19047619047619 (32/42)` | `23.333333333333332 (7/30)` | `0.0 (0/7)` |
| `docbench_per_keyword_no_kg_rerank_kg` | `6.25 (3/48)` | `75.40983606557377 (46/61)` | `73.04347826086956 (84/115)` | `71.42857142857143 (30/42)` | `16.666666666666664 (5/30)` | `0.0 (0/7)` |
| `docbench_ppr_dense_no_rerank` | `4.166666666666666 (2/48)` | `72.1311475409836 (44/61)` | `73.91304347826086 (85/115)` | `78.57142857142857 (33/42)` | `23.333333333333332 (7/30)` | `14.285714285714285 (1/7)` |
| `docbench_ppr_dense_rerank` | `4.166666666666666 (2/48)` | `75.40983606557377 (46/61)` | `77.39130434782608 (89/115)` | `78.57142857142857 (33/42)` | `33.33333333333333 (10/30)` | `0.0 (0/7)` |
| `docbench_ppr_hybrid_per_keyword` | `6.25 (3/48)` | `77.04918032786885 (47/61)` | `76.52173913043478 (88/115)` | `76.19047619047619 (32/42)` | `30.0 (9/30)` | `0.0 (0/7)` |
| `docbench_retrieval_hybrid_kg` | `4.166666666666666 (2/48)` | `73.77049180327869 (45/61)` | `73.04347826086956 (84/115)` | `73.80952380952381 (31/42)` | `13.333333333333334 (4/30)` | `0.0 (0/7)` |
| `docbench_untruncated_chunk_only` | `8.333333333333332 (4/48)` | `77.04918032786885 (47/61)` | `75.65217391304347 (87/115)` | `73.80952380952381 (31/42)` | `30.0 (9/30)` | `0.0 (0/7)` |
| `docbench_untruncated_kg` | `6.25 (3/48)` | `72.1311475409836 (44/61)` | `74.78260869565217 (86/115)` | `73.80952380952381 (31/42)` | `13.333333333333334 (4/30)` | `0.0 (0/7)` |

## SurGE Results

### Average Recall@K

| Group | R@5 | R@10 | R@20 | R@30 | R@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `surge_baseline` | `0.07048` | `0.131215` | `0.20201` | `0.257833` | `0.367536` |
| `surge_per_keyword` | `0.07387` | `0.128955` | `0.194209` | `0.247349` | `0.340135` |
| `surge_per_keyword_no_kg_rerank` | `0.07387` | `0.128955` | `0.194209` | `0.247349` | `0.340135` |
| `surge_ppr_dense_no_rerank` | `0.083681` | `0.110093` | `0.168144` | `0.23103` | `0.362471` |
| `surge_ppr_dense_rerank` | `0.078249` | `0.139124` | `0.227063` | `0.29549` | `0.355974` |
| `surge_ppr_hybrid_per_keyword` | `0.074717` | `0.133475` | `0.247208` | `0.325945` | `0.393103` |
| `surge_retrieval_hybrid` | `0.07613` | `0.125` | `0.210081` | `0.27226` | `0.390579` |

## Conclusions

- **DocBench best overall**: `docbench_ppr_dense_rerank` with `59.4059405940594` overall accuracy.
- **DocBench second-best overall**: `docbench_ppr_hybrid_per_keyword` with `59.07590759075908`.
- **Best non-PPR DocBench group**: `docbench_untruncated_chunk_only` with `58.745874587458744`.
- **SurGE best broad-recall group**: `surge_ppr_hybrid_per_keyword`, because it is highest at `R@20`, `R@30`, and `R@50`.
- **SurGE best very-early recall group**: `surge_ppr_dense_no_rerank`, because it is highest at `R@5`.

## Short Analysis

- On DocBench, `chunk_only_prompt` is better than `kg_prompt` under the same baseline retrieval setup: `58.08580858085809` vs `55.44554455445545`.
- On DocBench, `untruncated` helps the `chunk_only_prompt` branch (`58.745874587458744` vs `58.08580858085809`), but it does not improve overall accuracy for the `kg_prompt` branch (`55.44554455445545` vs `55.44554455445545`).
- On DocBench, `per_keyword_rrf` helps the KG branch when KG rerank is kept on (`56.76567656765676` vs `55.44554455445545`), but that gain disappears when KG rerank is turned off (`55.44554455445545`).
- On DocBench, `retrieval_mode=hybrid` is worse than the dense baseline in this run (`54.78547854785478` vs `55.44554455445545`).
- On DocBench, PPR is the strongest family in this result set: both `docbench_ppr_dense_rerank` and `docbench_ppr_hybrid_per_keyword` are above every non-PPR group.
- On SurGE, `surge_per_keyword` and `surge_per_keyword_no_kg_rerank` are exactly identical on all reported recall@k values, so KG rerank produced no measurable difference on this metric for that setting.
- On SurGE, `surge_ppr_hybrid_per_keyword` is the strongest late-recall setting, while `surge_ppr_dense_no_rerank` is only strongest at the smallest cutoff `R@5`.

## Recommendation

If one configuration must be chosen as the main retrieval winner, the safest choice from this result set is:

- **`ppr_dense_rerank`**

Reason:

- It is the **best answer-based result** on DocBench, which is the cleaner end-to-end metric in this report.
- Your SurGE results do not point to the same single winner at every cutoff.
- `surge_ppr_hybrid_per_keyword` is stronger for broader recall (`R@20/30/50`), but `surge_ppr_dense_no_rerank` is better at `R@5`.

So the practical reading is:

- If you prioritize **final answer quality**, choose **`ppr_dense_rerank`**.
- If you prioritize **broad retrieval recall on SurGE**, choose **`ppr_hybrid_per_keyword`** as the recall-oriented alternative.
