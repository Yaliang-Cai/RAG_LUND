# Multi-Hop QA Evaluation

Multi-hop QA evaluation measures how well RAG-Anything retrieves and reasons across multiple evidence paragraphs to answer questions that cannot be answered from any single source.  The evaluation lives in `evaluate_local/MultiHopQA/`.

---

## Overview

The evaluation follows a two-step pipeline:

```
Step 1  build_index.py       — extract corpus subset → index into workspace
Step 2  evaluate_multihop.py — run query modes → compute EM / F1 / Recall@K
```

Only the paragraphs that are bundled with the sampled questions are indexed (closed-book distractor setting).  For 500 questions this is typically 2 000–5 000 unique Wikipedia paragraphs — fast to index and fast to query.

---

## Datasets

| Key | Full name | HuggingFace identifier | Split | Size |
|-----|-----------|------------------------|-------|------|
| `hotpotqa` | HotpotQA distractor | `hotpot_qa / distractor` | validation | ~7 400 |
| `musique` | MuSiQue (answerable) | `dgslibisey/MuSiQue` | validation | ~1 200 |
| `2wiki` | 2WikiMultiHopQA | `framolfese/2WikiMultihopQA` | validation | ~12 600 |
| `simpleqa` | SimpleQA | `basicv8vc/SimpleQA` | test | ~4 300 |

SimpleQA is single-hop; it serves as a single-step reasoning baseline alongside the three multi-hop datasets.

### Corpus format (distractor setting)

HotpotQA, MuSiQue, and 2WikiMultiHopQA bundle their own candidate paragraphs with each question (10–20 paragraphs per question, 2 gold + the rest distractors).  The evaluation indexes **only the paragraphs for the sampled questions**, not a full Wikipedia dump.

Each indexed document is formatted as:

```
{title}
{paragraph text joined from sentences}
```

This puts the article title inside the chunk so the LLM can cite it naturally.

### Corpus scale at n=500

| Dataset | Unique paragraphs | Tokens / paragraph | Chunks at 1 200 tok/chunk |
|---------|-------------------|--------------------|--------------------------|
| HotpotQA | 3 000–4 500 | ~150–200 | 1 per paragraph |
| MuSiQue | 5 000–8 000 | ~100–200 | 1 per paragraph |
| 2WikiMultiHopQA | 3 000–5 000 | ~100–200 | 1 per paragraph |

Paragraphs are well below the 1 200-token chunk limit, so each paragraph becomes exactly one chunk.

---

## Step 1 — Build the index

Download datasets to HuggingFace cache (one-time):

```bash
python evaluate_local/MultiHopQA/download_datasets.py
```

Index the corpus for a dataset:

```bash
python evaluate_local/MultiHopQA/build_index.py \
    --dataset hotpotqa \
    --n-samples 500 \
    --workspace hotpotqa_500_seed42 \
    --working-dir /data/rag_workspaces/hotpotqa_500_seed42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `hotpotqa` / `musique` / `2wiki` |
| `--workspace` | required | Workspace ID (also the directory name) |
| `--working-dir` | required | Absolute path where LightRAG stores its index |
| `--n-samples` | 500 | Questions to sample; determines corpus size |
| `--seed` | 42 | Must match `--seed` in Step 2 |
| `--batch-size` | 50 | Paragraphs per `ainsert` call |

**Important:** use the same `--n-samples` and `--seed` in both steps.  The corpus is built from exactly the paragraphs bundled with the sampled questions.

---

## Step 2 — Run evaluation

```bash
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset hotpotqa \
    --workspace hotpotqa_500_seed42 \
    --working-dir /data/rag_workspaces/hotpotqa_500_seed42 \
    --output-dir ./multihop_results \
    --modes naive hybrid ppr auto \
    --n-samples 500 \
    --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `hotpotqa` / `musique` / `2wiki` / `simpleqa` |
| `--workspace` | required | Workspace ID from Step 1 |
| `--working-dir` | required | Same path as Step 1 |
| `--output-dir` | required | Where to write result JSONL and summary JSON |
| `--modes` | `naive hybrid ppr auto` | One or more query modes (space-separated) |
| `--n-samples` | 500 | Must match Step 1 |
| `--seed` | 42 | Must match Step 1 |
| `--recall-k` | `5 10 20` | K values for Recall@K |
| `--resume` | false | Skip already-answered questions (append to existing JSONL) |

The evaluator streams results to `{output_dir}/{dataset}_{mode}_results.jsonl` and writes a summary to `{output_dir}/{dataset}_summary.json`.

### Smoke test (verify pipeline end-to-end with 5 questions)

The smoke test runs the full two-step pipeline internally — it builds its own
small workspace (n=5) and then evaluates against it.  No pre-built workspace needed.

```bash
# Working dir for the temporary smoke workspace defaults to /tmp.
# Override with SMOKE_WS_ROOT if /tmp is too small or on a slow filesystem.
bash evaluate_local/MultiHopQA/smoke_test.sh

# Or point the workspace at a faster disk:
SMOKE_WS_ROOT=/data/rag_workspaces bash evaluate_local/MultiHopQA/smoke_test.sh
```

The script creates a uniquely-named workspace under `SMOKE_WS_ROOT` and prints
the full `build_index.py` + `evaluate_multihop.py` commands to run the 500-question
version when the smoke test passes.

---

## Query modes under comparison

| Mode | Description |
|------|-------------|
| `naive` | Flat dense vector search over all chunks; no graph traversal. Pure retrieval baseline. |
| `hybrid` | LightRAG hybrid: combines `local` (entity-anchored dense retrieval) and `global` (community/theme-level dense retrieval). All retrieval is pure dense — no BM25 or keyword search. Second baseline. |
| `ppr` | Personalized PageRank over the knowledge graph. Multi-hop graph traversal on top of dense retrieval. |
| `auto` | RetrievalRouter selects the best mode per query via LLM classification. |

`naive` is the minimal baseline (dense-only, no graph).  `hybrid` is the standard LightRAG baseline — it uses more retrieval signals than `naive` (both local entity context and global thematic context) but still relies purely on dense vectors.  `ppr` adds graph-based multi-hop traversal on top; the gap between `hybrid` and `ppr` measures the value of graph reasoning.  `auto` tests whether the router correctly dispatches hard multi-hop queries to `ppr` and simpler queries to cheaper modes.

---

## Metrics

### Exact Match (EM)

Binary score: 1 if the predicted answer matches any gold answer after normalization, 0 otherwise.

Normalization: lowercase, strip punctuation, remove articles (a / an / the), collapse whitespace.  Standard SQuAD / HotpotQA normalization.

$$\text{EM} = \mathbf{1}[\text{normalize}(\hat{a}) = \text{normalize}(a^*)]$$

### F1

Token-level overlap between predicted and gold answer after normalization.

$$F1 = \frac{2 \cdot P \cdot R}{P + R}, \quad P = \frac{|\hat{T} \cap T^*|}{|\hat{T}|}, \quad R = \frac{|\hat{T} \cap T^*|}{|T^*|}$$

When there are multiple gold answers, the maximum F1 over all gold strings is used.

### Recall@K (retrieval metric)

Fraction of gold supporting paragraphs that appear in the top-K retrieved chunks.

$$\text{Recall@K} = \frac{|\{p \in \text{gold}\;:\;\exists\,c \in \text{top-}K,\;p \sqsubseteq c\}|}{|\text{gold}|}$$

where $p \sqsubseteq c$ means the first 10 words of paragraph $p$ appear in chunk $c$ (case-insensitive).  Using the first-10-words fingerprint instead of full-text normalization avoids false misses from punctuation stripping on long texts.

K defaults to 5, 10, 20 (configurable via `--recall-k`).

SimpleQA has no supporting paragraph labels; its Recall@K is reported as `N/A`.

---

## Output files

```
multihop_results/
├── hotpotqa_naive_results.jsonl    # one JSON record per question
├── hotpotqa_hybrid_results.jsonl
├── hotpotqa_ppr_results.jsonl
├── hotpotqa_auto_results.jsonl
└── hotpotqa_summary.json           # aggregated metrics across all modes
```

### Per-question record (JSONL)

```json
{
  "id": "5a8b57f25542995d1e6f1371",
  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "gold": "Arthur's Magazine",
  "pred": "Arthur's Magazine",
  "em": 1.0,
  "f1": 1.0,
  "recall@5":  1.0,
  "recall@10": 1.0,
  "recall@20": 1.0
}
```

### Summary JSON

```json
{
  "dataset": "hotpotqa",
  "n_samples": 500,
  "seed": 42,
  "recall_k": [5, 10, 20],
  "timestamp": "2026-04-27T10:00:00+00:00",
  "results": {
    "naive":  {"em": 0.312, "f1": 0.421, "recall@5": 0.51, "recall@10": 0.64, "recall@20": 0.73, "n": 500},
    "hybrid": {"em": 0.338, "f1": 0.447, "recall@5": 0.58, "recall@10": 0.71, "recall@20": 0.80, "n": 500},
    "ppr":    {"em": 0.361, "f1": 0.472, "recall@5": 0.63, "recall@10": 0.76, "recall@20": 0.84, "n": 500},
    "auto":   {"em": 0.355, "f1": 0.468, "recall@5": 0.62, "recall@10": 0.75, "recall@20": 0.83, "n": 500}
  }
}
```

(Numbers above are illustrative; run the evaluation to get real numbers.)

---

## Implementation details

### Corpus extraction

`dataset_adapters.py` exposes paired functions for each multi-hop dataset:

| Load function | Corpus function | What it returns |
|---------------|-----------------|-----------------|
| `load_hotpotqa(n, seed)` | `extract_corpus_hotpotqa(n, seed)` | Q&A items / unique paragraphs |
| `load_musique(n, seed)` | `extract_corpus_musique(n, seed)` | Q&A items / unique paragraphs |
| `load_2wiki(n, seed)` | `extract_corpus_2wiki(n, seed)` | Q&A items / unique paragraphs |
| `load_simpleqa(n, seed)` | — | Q&A items (no corpus; SimpleQA is open-book) |

Both functions accept the same `n` and `seed`, so the corpus exactly covers the evaluation questions.

### Supporting facts granularity

`supporting_facts` in each item is a list of **full paragraph texts** (not individual sentences).  HotpotQA and 2WikiMultiHopQA provide sentence-level annotations (`sent_id`); the adapter resolves these to the full paragraph by joining all sentences for the gold article titles.  This is consistent with MuSiQue (which provides full paragraph texts natively) and with HippoRAG2's document-level Recall@K methodology.

### Recall matching

Because `normalize_answer` (which strips punctuation) is destructive on long paragraph texts, Recall@K uses a fingerprint approach:

```python
key = " ".join(fact.lower().split()[:10])   # first 10 words
covered = any(key in chunk_content.lower() for chunk in top_k_chunks)
```

Ten words is distinctive enough to avoid false positives while being robust to minor formatting differences between the original dataset text and the indexed chunk text.

### Resume / incremental evaluation

Pass `--resume` to skip questions whose IDs are already in the output JSONL.  This allows a run to be interrupted and restarted without duplicates:

```bash
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset hotpotqa --workspace ... --working-dir ... \
    --output-dir ./multihop_results --resume
```

### Query prompt overrides

Each dataset injects a short-answer constraint into the query:

- HotpotQA / MuSiQue / 2WikiMultiHopQA: "Answer with a short phrase or entity name only. For yes/no questions, reply only 'yes' or 'no'."
- SimpleQA: "Answer with a concise factual phrase only."

This suppresses chain-of-thought reasoning and citation sections from the LLM so that EM / F1 scoring against short gold strings works correctly.

---

## Complete example (HotpotQA, 500 questions)

```bash
# 0. One-time dataset download
python evaluate_local/MultiHopQA/download_datasets.py

# 1. Build index (~10–30 min depending on LLM/embedding speed)
python evaluate_local/MultiHopQA/build_index.py \
    --dataset hotpotqa \
    --n-samples 500 \
    --workspace hotpotqa_500_seed42 \
    --working-dir /data/rag_workspaces/hotpotqa_500_seed42

# 2. Smoke test — builds its own n=5 workspace and runs end-to-end (~2–5 min)
SMOKE_WS_ROOT=/data/rag_workspaces bash evaluate_local/MultiHopQA/smoke_test.sh

# 3. Full evaluation across all four modes (~2–4 h for 500 questions × 4 modes)
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset hotpotqa \
    --workspace hotpotqa_500_seed42 \
    --working-dir /data/rag_workspaces/hotpotqa_500_seed42 \
    --output-dir ./multihop_results \
    --modes naive hybrid ppr auto \
    --n-samples 500 \
    --seed 42 \
    --recall-k 5 10 20
```

The final console output is a comparison table:

```
============================================================
Dataset: hotpotqa  n=500
Mode              EM       F1    R@5    R@10   R@20
------------------------------------------------------------
naive          0.XXXX   0.XXXX  0.XXXX  0.XXXX  0.XXXX
hybrid         0.XXXX   0.XXXX  0.XXXX  0.XXXX  0.XXXX
ppr            0.XXXX   0.XXXX  0.XXXX  0.XXXX  0.XXXX
auto           0.XXXX   0.XXXX  0.XXXX  0.XXXX  0.XXXX
```
