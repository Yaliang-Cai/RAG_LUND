# Adaptive Agentic Retrieval-Augmented Generation

**Date:** 2026-04-29  
**Branch:** `feature/agentic-rag` → merged into `main`  
**Status:** Implemented, 93 tests passing  
**Query mode:** `mode="agentic"`

---

## 1. Motivation

Conventional retrieval-augmented generation (RAG) systems execute a fixed, single-pass pipeline: a query is mapped to a retrieval strategy, relevant passages are fetched once, and a language model generates a response from the retrieved context. This design is computationally efficient for simple, single-hop factual queries, but it is structurally inadequate for at least three classes of questions that regularly arise in knowledge-intensive tasks:

1. **Multi-entity reasoning queries**, which require chaining evidence across multiple documents and cannot be resolved from a single retrieval pass.
2. **Informationally incomplete responses**, where an initial retrieval succeeds in finding partially relevant passages but the generated answer exhibits identifiable knowledge gaps.
3. **Queries of variable complexity**, for which a uniform retrieval strategy imposes either insufficient computation (missed evidence) or unnecessary overhead (wasted inference budget).

A central tension in prior work is the trade-off between retrieval depth and latency. Methods such as Iterative Retrieval Augmented Generation (IRAG) and Self-Ask with Search address this by chaining retrieval calls, but typically do so unconditionally, incurring full multi-hop overhead on all queries regardless of their actual complexity. Conversely, standard RAG pipelines treat all queries identically, ignoring the heterogeneity of information needs.

The present system addresses this tension through **adaptive agentic retrieval**: a query-complexity-aware orchestration mechanism that selects and executes an appropriate retrieval and self-refinement strategy for each individual query, optimising the allocation of inference budget across the complexity spectrum.

---

## 2. System Overview

The agentic query mode (`mode="agentic"`) is implemented as a directed acyclic state machine over a finite set of retrieval and generation states. The execution policy is determined at runtime by a language model classifier that estimates the structural complexity of the incoming query and assigns it to one of three complexity tiers. Each tier activates a distinct execution track with a corresponding ceiling on the number of retrieval iterations.

The system is designed around three core components:

- **Complexity Classifier**: a lightweight LLM-based routing module that maps a query to a complexity level.
- **Adaptive Execution Graph**: a LangGraph-based state machine implementing the three execution tracks and their conditional transition logic.
- **Evaluator–Optimizer**: a self-reflection module that assesses the quality of each generated response and triggers supplementary retrieval when a quality threshold is not met.

All three tracks reuse the existing retrieval infrastructure — `RetrievalRouter`, `run_path`, PPR, and Qdrant-based hybrid retrieval — without modification. The agentic layer is purely orchestration; retrieval semantics are unchanged.

---

## 3. Complexity Classification

### 3.1 Taxonomy

Queries are classified into three complexity tiers based on their structural properties:

**Simple** queries are single-hop factual questions for which a single retrieval pass over the knowledge base is sufficient to produce a complete and accurate response. These queries exhibit no multi-entity dependencies and require no iterative evidence chaining.

**Medium** queries involve moderate informational depth — typically a single focal entity with multiple attributes, or a process explanation requiring several related passages. A single retrieval pass may be sufficient, but there is a non-negligible probability of an incomplete response that would benefit from one supplementary retrieval step.

**Complex** queries involve multiple distinct entities or require cross-document reasoning — comparative analysis, causal chain tracing, or synthesis of evidence distributed across heterogeneous sources. These queries cannot be adequately resolved without decomposition into independent sub-queries and concurrent retrieval across each sub-query's retrieval neighbourhood.

### 3.2 Classification Mechanism

The classifier is instantiated as a prompted language model that receives the user query and outputs a structured JSON object containing a complexity label and a confidence score. The classification prompt includes disambiguation rules that encode a conservative prior: ambiguous cases are resolved in favour of the lower complexity tier rather than the higher one. This design choice reflects the asymmetry in error costs — the overhead of an unnecessary retrieval iteration is smaller than the quality degradation of an insufficiently retrieved response.

A confidence threshold of 0.6 is applied: classifications below this threshold are overridden to `medium`, the intermediate tier. This prevents high-complexity assignments on queries where the classifier itself expresses uncertainty, avoiding expensive decomposition for queries that may in fact be tractable in a single pass.

The classification decision is not exposed to the user and is treated as an internal routing signal. Callers may bypass classification entirely by specifying a retrieval profile directly via the `profile` parameter, which is consistent with the design of the existing `mode="auto"` router.

---

## 4. Execution Tracks

The three execution tracks correspond directly to the three complexity tiers. Each track is a distinct path through the state machine, with a hard upper bound on the number of retrieval iterations.

### 4.1 Simple Track

The simple track consists of three sequential states: query classification, single-pass retrieval via the `RetrievalRouter`, and language model generation. No quality evaluation is performed. The total inference budget is bounded by one LLM classification call and one LLM generation call. This track is structurally equivalent to the existing `mode="auto"` pipeline and imposes no additional overhead relative to it.

### 4.2 Medium Track

The medium track augments the simple track with a post-generation quality evaluation step. After an initial response is generated, the Evaluator–Optimizer module (described in §5) assigns a quality score to the response. If the score meets or exceeds the quality threshold (0.7), execution terminates. If the score falls below the threshold, a single targeted retrieval step is executed, the response is regenerated from the augmented context, and execution terminates unconditionally regardless of the second evaluation score. The maximum number of supplementary retrieval steps is therefore bounded at one.

### 4.3 Complex Track

The complex track introduces two structural differences relative to the medium track. First, before any retrieval is performed, an LLM-based decomposition step breaks the original query into two to four independent sub-queries. Each sub-query is formulated to be self-contained — it contains no cross-references to other sub-queries and can be answered independently by the retrieval system. Second, the sub-queries are executed concurrently by dispatching each to the `RetrievalRouter` under a concurrency semaphore, which bounds the peak number of simultaneous retrieval path executions. The retrieved passages from all sub-queries are pooled and deduplicated before being passed to the generation stage.

The quality evaluation and supplementary retrieval mechanism is identical to the medium track, but the maximum number of supplementary retrieval steps is raised to two. On each retry, the decomposition step is not re-executed; only a single targeted supplementary retrieval is performed, guided by the gap description produced by the evaluator (§5).

---

## 5. Evaluator–Optimizer

The Evaluator–Optimizer is a self-reflection module that implements the quality gate governing whether execution terminates or continues with supplementary retrieval. It operates after each generation step on all tracks except the simple track.

### 5.1 Quality Assessment

The evaluator receives the original query and the current generated response, and produces two outputs: a scalar quality score in the interval [0, 1], and a natural language description of any identifiable knowledge gaps in the response. The scoring rubric is defined over four qualitative bands:

- [0.9, 1.0]: response is complete, accurate, and well-supported by the retrieved context.
- [0.7, 0.9]: response is mostly complete with minor gaps.
- [0.5, 0.7]: response is partial with notable informational gaps.
- [0.0, 0.5]: response is incomplete or substantially off-topic.

The quality threshold is set at 0.7, corresponding to the boundary between the "mostly complete" and "partial" bands. Responses that meet or exceed this threshold are accepted; responses below the threshold trigger a supplementary retrieval step, provided the iteration budget has not been exhausted.

### 5.2 Targeted Supplementary Retrieval

When a supplementary retrieval step is triggered, the retrieval query is constructed by concatenating the original user query with the evaluator's gap description. This targeted formulation increases the probability that the supplementary retrieval surfaces passages specifically relevant to the identified knowledge gap, rather than retrieving the same passages already present in the context. The original query is preserved unchanged throughout all iterations and is always used as the primary generation context; only the retrieval query is modified.

### 5.3 Context Accumulation

Retrieved passages are accumulated across all retrieval iterations using an append-mode state reducer rather than a replacement operation. Passages retrieved in supplementary steps are merged with the initial retrieval set and deduplicated prior to each generation step. Deduplication retains the occurrence of each passage with the highest retrieval relevance score (as assigned by the Reciprocal Rank Fusion scorer), discarding lower-scored duplicates. This design ensures that the language model generates from the broadest possible context at each generation step, incorporating evidence from both the initial and supplementary retrievals.

---

## 6. Iteration Budget and Termination Conditions

Each execution track enforces a hard upper bound on the number of supplementary retrieval iterations: zero for the simple track, one for the medium track, and two for the complex track. Execution terminates when either of the following conditions is met:

1. The evaluator assigns a quality score of 0.7 or above.
2. The iteration counter reaches the track-specific upper bound.

The second condition guarantees termination regardless of evaluator output, preventing unbounded iteration on queries for which the knowledge base does not contain sufficient evidence to produce a high-quality response.

---

## 7. Relationship to Existing Query Modes

The agentic mode is implemented as an independent execution path within the query dispatch layer and does not share state with or modify any existing query mode. The retrieval infrastructure (`RetrievalRouter`, `run_path`, PPR, Qdrant hybrid retrieval, reranker) is invoked through the same interfaces used by `mode="auto"` and other existing modes. Existing query modes are unaffected by the introduction of `mode="agentic"`.

The agentic mode subsumes the capabilities of `mode="auto"` for the simple track — both invoke `RetrievalRouter` with the same retrieval infrastructure — but adds evaluation and iterative retrieval for medium and complex queries. The two modes are not interchangeable: `mode="auto"` performs profile-based retrieval routing with no post-generation evaluation, while `mode="agentic"` performs complexity-based track selection with optional iterative self-refinement.

---

## 8. Observability

The agentic pipeline is instrumented via OpenTelemetry using Arize Phoenix in local deployment mode. Every state transition in the execution graph, every LLM call (classification, decomposition, generation, evaluation), and every retrieval invocation is recorded as a span within a distributed trace. This instrumentation enables post-hoc analysis of:

- **Complexity distribution**: the fraction of queries routed to each of the three tracks.
- **Evaluator score distribution**: the empirical distribution of quality scores across queries and tracks, which can be used to calibrate the quality threshold.
- **Retrieval path contribution**: the relative relevance scores of passages retrieved by each path within the `RetrievalRouter`, as a proxy for path-level recall.
- **Iteration frequency**: the fraction of medium and complex queries that trigger one or more supplementary retrieval steps, and the corresponding latency overhead.

These metrics are available for arbitrary production queries without requiring a labelled evaluation set. Formal retrieval metrics (Recall@k, MRR, NDCG) require a ground-truth corpus and are computed separately using the existing offline evaluation scripts.

---

## 9. Usage

```
aquery(query, mode="agentic")                        # adaptive track selection
aquery(query, mode="agentic", return_trace=True)     # also return execution trace
```

The `return_trace=True` parameter causes the method to return a dictionary containing the generated answer and a structured trace of the execution, including the assigned complexity level, the final evaluator score, the number of supplementary retrieval iterations performed, and the routing metadata from each retrieval step.

Observability must be initialised once at service startup:

```
from raganything.observability import setup_phoenix
setup_phoenix()   # Phoenix UI at http://localhost:6006
```

Dependencies for the agentic mode are declared as an optional extras group and are not required for other query modes:

```
 pip install "langgraph>=0.2" "arize-phoenix[otel]>=4.0" opentelemetry-sdk opentelemetry-exporter-otlp
```
启动 Phoenix 服务端：
你需要打开另一个终端窗口，在同一个虚拟环境下运行：

Bash
```
python -m phoenix.server.main serve
# 或者如果安装了 CLI 工具：
phoenix serve
```
---

## 10. Complexity and Latency Characteristics

The inference budget of each track is summarised below. LLM call counts are approximate and exclude caching effects.

| Track | LLM calls (approx.) | Retrieval passes | Max iterations |
|-------|-------------------|-----------------|---------------|
| Simple | 2 (classify + generate) | 1 | 0 |
| Medium | 2–4 | 1–2 | 1 |
| Complex | 4–7 (+ decompose) | 2–6 (parallel) | 2 |

The dominant latency contributor for the simple track is the retrieval pass. For the complex track, the parallel sub-query retrieval reduces wall-clock latency relative to serial execution, but the decomposition and evaluation calls introduce additional serialised LLM latency. In a local vLLM deployment, the expected total latency for a complex query with one retry is approximately 3–5× that of a simple query, bounded by the LLM inference throughput of the local server.

---

## References

- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Trivedi, H., et al. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023*.
- Gutierrez, B.J., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.
