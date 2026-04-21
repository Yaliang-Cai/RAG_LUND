# Multi-Hop QA Benchmark Experiment Design

**Date:** 2026-04-21  
**Goal:** Compare PPR query mode against LightRAG's hybrid/mix modes on open-domain multi-hop QA benchmarks used by HippoRAG2.

---

## Scope

- **Task:** Query-mode comparison only — no re-indexing. One shared index per dataset, multiple query modes run against it.
- **Datasets:** HotpotQA, MuSIQue, 2WikiMultiHopQA, SimpleQA
- **Sample size:** Same as HippoRAG2 paper (default 500 per dataset, configurable via `--n-samples`)
- **Query modes:** `ppr`, `hybrid`, `mix` by default; configurable via CLI (valid values: `ppr ppr_local global local hybrid mix naive rrf bypass`)
- **Metrics:** EM + F1 for all datasets; Recall@K (K=5,10,20) for HotpotQA / MuSIQue / 2Wiki (SimpleQA has no supporting facts → N/A)
- **Environment:** Same server as DocBench evaluation (`/data/y50056788/...`)

---

## Directory Structure

```
rag-anything/evaluate_local/
└── MultiHopQA/
    ├── evaluate_multihop.py       # Main eval script
    ├── dataset_adapters.py        # Per-dataset load + score functions
    └── download_datasets.py       # One-time HuggingFace download

rag-anything/
└── run_multihop_evals.py          # Orchestrator (same level as run_ablation_evals.py)
```

---

## File Responsibilities

### `dataset_adapters.py`

**Load functions** — return `List[{"id", "question", "answer", "supporting_facts"}]`:
- `load_hotpotqa(n, seed)` — HuggingFace `hotpot_qa` distractor split, dev set
- `load_musique(n, seed)` — HuggingFace `musique`, dev set
- `load_2wiki(n, seed)` — HuggingFace `wiki_hop` or `2WikiMultiHopQA`, dev set
- `load_simpleqa(n, seed)` — SimpleQA test set; `supporting_facts=None`

**Score functions:**
- `normalize_answer(s)` — lowercase, strip punctuation/articles (standard HotpotQA normalization)
- `score_em(pred, gold) -> float` — exact match after normalization
- `score_f1(pred, gold) -> float` — token-level F1 after normalization
- `score_recall_at_k(chunks, supporting_facts, k) -> float` — fraction of supporting facts covered by at least one top-K chunk (substring match)

**Dataset-specific prompt config:**
- `get_eval_query_overrides(dataset) -> dict` — returns `{"response_type": ..., "user_prompt": ...}` for each dataset

The `PROMPTS["rag_response"]` template in LightRAG produces verbose Markdown + References output by default, which breaks EM/F1 evaluation (gold answers are short phrases like `"yes"` or `"Berlin"`). We override two `QueryParam` fields at eval time:

| Dataset | `response_type` | `user_prompt` instruction |
|---|---|---|
| HotpotQA | `"Short Answer"` | Answer with a short phrase or entity. For yes/no questions, reply only 'yes' or 'no'. No markdown, no citations. |
| MuSIQue | `"Short Answer"` | Same as HotpotQA |
| 2WikiMultiHopQA | `"Short Answer"` | Same as HotpotQA |
| SimpleQA | `"Short Answer"` | Answer with a concise factual phrase. No markdown, no citations. |

`evaluate_multihop.py` also post-processes LLM output to strip residual `### References` sections in case the model ignores the instruction.

### `evaluate_multihop.py`

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | `hotpotqa\|musique\|2wiki\|simpleqa` |
| `--workspace` | required | Workspace ID (pre-built index) |
| `--working-dir` | required | Path to workspace storage |
| `--modes` | `ppr hybrid mix` | Space-separated query modes to compare |
| `--n-samples` | `500` | Number of questions to evaluate |
| `--recall-k` | `5 10 20` | K values for Recall@K |
| `--output-dir` | required | Directory for JSONL + summary output |
| `--resume` | flag | Skip questions already in JSONL (crash recovery) |
| `--seed` | `42` | Random seed for sampling |

**Execution flow:**
1. Load dataset (sample `n` questions with `seed`)
2. For each mode (sequential):
   a. For each question: call `LocalRagService.query_with_trace()` with the given mode
   b. Extract `answer` and `trace["data"]["chunks"]` from response
   c. Compute EM, F1, Recall@K
   d. Append result to `{dataset}_{mode}_results.jsonl` immediately (crash-safe)
3. After all modes complete: aggregate per-mode metrics → `{dataset}_summary.json` → print comparison table

**Output files:**
```
output-dir/
├── {dataset}_{mode}_results.jsonl   # Per-question results (streamed)
└── {dataset}_summary.json           # Aggregated EM / F1 / Recall@K per mode
```

### `download_datasets.py`

One-time script: downloads all four datasets from HuggingFace to a local cache directory (configurable via `--data-dir`). Run once before evaluation.

### `run_multihop_evals.py`

Orchestrator following the same pattern as `run_ablation_evals.py`:
- Accepts `--datasets` (default: all four) and `--modes` (default: `ppr hybrid mix`)
- Calls `evaluate_multihop.py` via subprocess for each dataset sequentially
- Passes through `--workspace`, `--working-dir`, `--output-dir`, `--n-samples`, `--seed`

---

## Metrics Definition

### EM (Exact Match)
After `normalize_answer()`: 1 if prediction equals gold answer exactly, else 0.

### F1 (Token-level)
Precision = `|pred_tokens ∩ gold_tokens| / |pred_tokens|`  
Recall = `|pred_tokens ∩ gold_tokens| / |gold_tokens|`  
F1 = harmonic mean. Take max F1 over multiple gold answers if present.

### Recall@K
For each supporting fact `f` in `supporting_facts`:  
- Check if any chunk in top-K contains `f` as substring (after normalization)  
- `Recall@K = |covered facts| / |total facts|`  
- Macro-average over all questions.  
- SimpleQA: skip (no supporting facts).

---

## Constraints & Assumptions

- The shared index for each dataset must be built before running this script (not in scope here).
- `query_with_trace()` must return `trace["data"]["chunks"]` as a list with rank order preserved.
- PPR mode parameters (`recognition_top_k`, `linking_top_k`, `ppr_qa_top_k`) use defaults from `raganything/constants.py` unless overridden.
- `--resume` matches by question `id` field in existing JSONL.
