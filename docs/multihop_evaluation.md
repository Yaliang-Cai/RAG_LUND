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

## Evaluation design and comparison context

### What this evaluation measures

This pipeline uses the **closed-corpus distractor setting** native to each dataset.  Every question already bundles its own candidate paragraphs (2 gold + 8–18 distractors).  At `n=500`, the index contains roughly 3 000–8 000 unique paragraphs — all gold paragraphs are guaranteed to be present.  The retrieval task is therefore: find the correct 2 paragraphs among a pool of ~5 000, where the distractors come from the other 499 questions in the sample.

This is a valid and internally consistent benchmark for **comparing query modes against each other**.  The same index, same LLM, same embedding model, and same prompt are used for every mode, so the only variable is the retrieval strategy.  Differences in EM / F1 / Recall@K across modes directly measure the value of graph traversal (ppr), routing (auto), or flat vector search (naive).

The one thing this setting does **not** support is a direct apples-to-apples comparison with systems evaluated on a full open-domain corpus.

### How HippoRAG2's evaluation differs

HippoRAG2 (Gutierrez et al., 2024; [arXiv:2405.14831](https://arxiv.org/abs/2405.14831)) targets **open-domain multi-hop retrieval** and is evaluated under a fundamentally different protocol:

| Dimension | This evaluation | HippoRAG2 |
|-----------|-----------------|------------|
| Corpus | Closed — only the paragraphs bundled with the sampled questions | Open — full Wikipedia dump (HotpotQA fullwiki: ~5 M paragraphs) or full dataset corpus |
| Pool size at n=500 | 3 000–8 000 paragraphs | Millions of paragraphs |
| Gold docs in index | Guaranteed by construction | Must be retrieved from the full corpus |
| Retrieval difficulty | Moderate (needle in ~5 000) | Hard (needle in millions) |
| Primary metric | EM / F1 / Recall@K (end-to-end) | Recall@2 / F1 (retrieval-focused) |
| Index construction | LightRAG KG + dense vectors (LLM entity extraction per paragraph) | Lightweight graph over entity co-occurrence |
| LLM / embedding | Configurable (local vLLM, bge-m3) | Fixed in paper (OpenAI embeddings) |

Because the corpus size, retrieval difficulty, LLM, and embedding model all differ, **numbers from this evaluation cannot be placed in the same table as HippoRAG2's published results** without a protocol-alignment note.

### System positioning

RAG-Anything is a **general-purpose multimodal document RAG system**.  Its design goal is ingesting heterogeneous documents (PDF, Office, images, tables, equations) and answering questions over them — not winning retrieval benchmarks on Wikipedia text.

HippoRAG2 is a **specialist multi-hop retrieval system** built specifically to maximise recall on Wikipedia-scale corpora.  It does not handle multimodal documents, structured tables, or equation parsing.

The appropriate use of this evaluation pipeline is therefore:

- **Use it** to compare ppr vs. hybrid vs. naive vs. auto on the same index and decide which mode to default to.
- **Use it** to track regressions as the retrieval code evolves.
- **Do not use it** to claim that RAG-Anything outperforms or underperforms HippoRAG2 — the systems solve different problems under different constraints.

If a paper section needs to position RAG-Anything relative to retrieval-specialist systems, the recommended framing is:

> "We evaluate query-mode performance using the closed-corpus distractor setting of HotpotQA / MuSiQue / 2WikiMultiHopQA (n=500, seed=42).  This differs from the open-domain fullwiki protocol used by HippoRAG2 [cite]; our corpus contains only the paragraphs bundled with the sampled questions (~3 000–8 000 paragraphs), making absolute numbers not directly comparable.  The evaluation is designed to measure the incremental value of graph-based multi-hop traversal (ppr) and adaptive routing (auto) over flat dense retrieval (naive) within a unified general-purpose multimodal RAG system."

---

## Step 1 — Build the index

Download datasets to HuggingFace cache (one-time).  Prefer absolute paths and
set `HF_HOME` so HuggingFace's `hub/`, `datasets/`, and `xet/` directories stay
under the evaluation data root:

```bash
mkdir -p /data/y50056788/Yaliang/datasets_for_eval/hf_cache/datasets

HF_HOME=/data/y50056788/Yaliang/datasets_for_eval/hf_cache \
HF_DATASETS_CACHE=/data/y50056788/Yaliang/datasets_for_eval/hf_cache/datasets \
python evaluate_local/MultiHopQA/download_datasets.py \
    --data-dir /data/y50056788/Yaliang/datasets_for_eval/hf_cache/datasets
```

Index the corpus for a dataset:

```bash
python evaluate_local/MultiHopQA/build_index.py \
    --dataset hotpotqa \
    --n-samples 500 \
    --workspace hotpotqa_500_seed42 \
    --working-dir /data/rag_workspaces/hotpotqa_500_seed42 \
    --ingest-batch-size 256 \
    --batch-doc-concurrency 2 \
    --llm-model-max-async 48 \
    --resume
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `hotpotqa` / `musique` / `2wiki` |
| `--workspace` | required | Workspace ID (also the directory name) |
| `--working-dir` | required | Absolute path where LightRAG stores its index |
| `--n-samples` | 500 | Questions to sample; determines corpus size |
| `--seed` | 42 | Must match `--seed` in Step 2 |
| `--ingest-batch-size` / `--batch-size` | 256 | Source paragraphs packed into one virtual batch document |
| `--batch-doc-concurrency` | 2 | Concurrent virtual batch document inserts |
| `--llm-model-max-async` | 48 | LightRAG LLM extraction worker concurrency during ingest |
| `--max-retries` | 0 | Retries per failed virtual batch insert |
| `--resume` | false | Skip virtual batches already marked `ok` in the progress JSONL |

**Important:** use the same `--n-samples` and `--seed` in both steps.  The corpus is built from exactly the paragraphs bundled with the sampled questions.

`build_index.py` uses a SurGE-style fast ingest path.  It packs many source
paragraphs into virtual batch documents, splits each virtual document back into
one paragraph per LightRAG chunk with `split_by_character_only=True`, and runs
multiple virtual batch inserts concurrently.  Each workspace directory also gets
source-map artifacts:

```
multihopqa_source_records.jsonl       # one original source paragraph per row
multihopqa_chunk_source_map.json      # LightRAG chunk id -> source paragraph
multihopqa_ingest_progress.jsonl      # successful virtual batches for resume
multihopqa_ingest_failures.jsonl      # failed virtual batches, if any
multihopqa_ingest_manifest.json       # build parameters and counts
```

The source map is what lets evaluation and debugging resolve retrieved chunks
back to their original dataset paragraph.

---

## Step 2 — Run evaluation

```bash
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset hotpotqa \
    --workspace hotpotqa_500_seed42 \
    --working-dir /data/rag_workspaces/hotpotqa_500_seed42 \
    --output-dir ./multihop_results \
    --modes naive hybrid ppr auto full \
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
| `auto` | RetrievalRouter classifies each query via LLM and dispatches it to the best-matching retrieval profile (e.g. `multihop` → ppr+hybrid, `local` → hybrid+naive, `descriptive` → mix+qdrant_hybrid). Results from the selected profile's paths are fused with weighted RRF and optionally reranked. |
| `full` | Forces the router's `full` profile for every query — all six retrieval paths (naive, hybrid, mix, ppr, qdrant_hybrid, qdrant_sparse) run in parallel and are fused via weighted RRF. No LLM classification overhead; the widest possible evidence net per query. |

`naive` is the minimal baseline (dense-only, no graph).  `hybrid` is the standard LightRAG baseline — it uses more retrieval signals than `naive` (both local entity context and global thematic context) but still relies purely on dense vectors.  `ppr` adds graph-based multi-hop traversal on top; the gap between `hybrid` and `ppr` measures the value of graph reasoning.

`auto` and `full` both go through the `RetrievalRouter`; the difference is how the retrieval profile is chosen:

- **`auto`**: LLM classifies the query type per question (multi-hop, local fact, descriptive, …) and activates only the paths relevant to that type.  It is the smarter but more expensive option — it pays one extra LLM call per question in exchange for precision.  The gap between `ppr` and `auto` measures whether the router's profile selection adds value over always using PPR.
- **`full`**: Pins the `full` profile for every query — all six paths fire unconditionally, results are merged with weighted RRF, then reranked.  This is the maximum-recall upper bound for the router-based architecture.  The gap between `auto` and `full` measures whether LLM-based routing beats brute-force path fusion, and at what latency cost.

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
├── hotpotqa_full_results.jsonl
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
    "auto":   {"em": 0.355, "f1": 0.468, "recall@5": 0.62, "recall@10": 0.75, "recall@20": 0.83, "n": 500},
    "full":   {"em": 0.368, "f1": 0.479, "recall@5": 0.65, "recall@10": 0.78, "recall@20": 0.86, "n": 500}
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
    --modes naive hybrid ppr auto full \
    --n-samples 500 \
    --seed 42 \
    --recall-k 5 10 20
```

For current 500-question runs on `/data/y50056788`, prefer this explicit
cache + concurrent-ingest form:

```bash
mkdir -p /data/y50056788/Yaliang/datasets_for_eval/hf_cache/datasets
HF_HOME=/data/y50056788/Yaliang/datasets_for_eval/hf_cache \
HF_DATASETS_CACHE=/data/y50056788/Yaliang/datasets_for_eval/hf_cache/datasets \
python evaluate_local/MultiHopQA/download_datasets.py \
    --data-dir /data/y50056788/Yaliang/datasets_for_eval/hf_cache/datasets

python evaluate_local/MultiHopQA/build_index.py \
    --dataset hotpotqa \
    --n-samples 500 \
    --seed 42 \
    --workspace hotpotqa_500_seed42 \
    --working-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/MultiHopQA/rag_workspaces/hotpotqa_500_seed42 \
    --ingest-batch-size 256 \
    --batch-doc-concurrency 2 \
    --llm-model-max-async 48 \
    --max-retries 1 \
    --resume
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
full           0.XXXX   0.XXXX  0.XXXX  0.XXXX  0.XXXX
```
