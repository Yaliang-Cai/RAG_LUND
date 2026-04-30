# raganything/retrieval/agent_graph.py
from __future__ import annotations

import asyncio
import json
import logging
import operator
from typing import Annotated, Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from opentelemetry import trace as otel_trace

from lightrag import QueryParam
from .complexity import ComplexityClassifier
from .evaluator import AnswerEvaluator
from .router import RetrievalRouter

logger = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer("raganything.agentic")

MAX_CONCURRENT_SUBQUESTIONS = 3
MAX_ITER_BY_COMPLEXITY: dict[str, int] = {"simple": 0, "medium": 1, "complex": 2}
EVAL_PASS_THRESHOLD = 0.7

_DECOMPOSE_PROMPT = """\
Break this complex question into 2-4 independent sub-questions that can each be \
answered by searching a knowledge base independently.

Complex question: {query}

Rules:
- Each sub-question must be self-contained (no references to other sub-questions).
- Sub-questions must not overlap in scope.
- Prefer 2 sub-questions over 4 unless truly needed.

Output JSON: {{"sub_questions": ["...", "..."]}}
"""


class AgentState(TypedDict):
    query: str
    complexity: str
    sub_questions: list[str]
    retrieved_chunks: Annotated[list[dict], operator.add]
    answer: str
    eval_score: float
    eval_gap: str
    current_search_query: str
    iteration: int
    routing_trace: dict


class AdaptiveAgentGraph:
    def __init__(
        self,
        lightrag: Any,
        llm_func: Any = None,
        *,
        _complexity_clf: ComplexityClassifier | None = None,
        _evaluator: AnswerEvaluator | None = None,
        _router: RetrievalRouter | None = None,
    ) -> None:
        self._lightrag = lightrag
        self._llm = llm_func or lightrag.llm_model_func
        self._complexity_clf = _complexity_clf or ComplexityClassifier(self._llm)
        self._evaluator = _evaluator or AnswerEvaluator(self._llm)
        self._router = _router or RetrievalRouter(lightrag, self._llm)
        self._graph = self._build_graph()

    # ── Nodes ───────────────────────────────────────────────────────────────

    async def _node_classify(self, state: AgentState) -> dict:
        with _tracer.start_as_current_span("agentic.classify_complexity") as span:
            complexity, meta = await self._complexity_clf.classify(state["query"])
            span.set_attribute("complexity", complexity)
            span.set_attribute("confidence", meta.get("confidence", 0.0))
            span.set_attribute("latency_s", meta.get("latency", 0.0))
            span.set_attribute("reasoning", meta.get("reasoning", ""))
        return {
            "complexity": complexity,
            "current_search_query": state["query"],
            "routing_trace": {"complexity": meta},
            "iteration": 0,
        }

    async def _node_retrieve(self, state: AgentState) -> dict:
        with _tracer.start_as_current_span("agentic.retrieve") as span:
            param = QueryParam(mode="hybrid")
            chunks, trace = await self._router.route(state["current_search_query"], param)
            span.set_attribute("chunks_retrieved", len(chunks))
            span.set_attribute("profile", trace.get("profile", ""))
        return {
            "retrieved_chunks": chunks,
            "routing_trace": {**state.get("routing_trace", {}), "retrieve": trace},
        }

    async def _node_decompose(self, state: AgentState) -> dict:
        with _tracer.start_as_current_span("agentic.decompose") as span:
            sub_questions = await self._decompose_query(state["query"])
            span.set_attribute("sub_questions_count", len(sub_questions))
            span.set_attribute("sub_questions", str(sub_questions))
        return {"sub_questions": sub_questions}

    async def _node_parallel_retrieve(self, state: AgentState) -> dict:
        with _tracer.start_as_current_span("agentic.parallel_retrieve") as span:
            span.set_attribute("sub_questions_count", len(state["sub_questions"]))
            sem = asyncio.Semaphore(MAX_CONCURRENT_SUBQUESTIONS)
            param = QueryParam(mode="hybrid")

            async def _one(sub_q: str) -> list[dict]:
                async with sem:
                    chunks, _ = await self._router.route(sub_q, param)
                    return chunks

            results = await asyncio.gather(*[_one(q) for q in state["sub_questions"]])
            all_chunks = [c for batch in results for c in batch]
            span.set_attribute("total_chunks_retrieved", len(all_chunks))
        return {"retrieved_chunks": all_chunks}

    async def _node_generate(self, state: AgentState) -> dict:
        with _tracer.start_as_current_span("agentic.generate") as span:
            deduped = _dedup_chunks(state["retrieved_chunks"])
            span.set_attribute("chunks_after_dedup", len(deduped))
            span.set_attribute("iteration", state["iteration"])
            answer = await self._generate_answer(state["query"], deduped)
        return {"answer": answer}

    async def _node_evaluate(self, state: AgentState) -> dict:
        with _tracer.start_as_current_span("agentic.evaluate") as span:
            result = await self._evaluator.evaluate(state["query"], state["answer"])
            span.set_attribute("eval_score", result["score"])
            span.set_attribute("eval_gap", result["gap"])
            span.set_attribute("iteration", state["iteration"])
            span.set_attribute("will_retry", result["score"] < EVAL_PASS_THRESHOLD
                               and state["iteration"] < MAX_ITER_BY_COMPLEXITY.get(state["complexity"], 1))
        return {"eval_score": result["score"], "eval_gap": result["gap"]}

    async def _node_targeted_retrieve(self, state: AgentState) -> dict:
        with _tracer.start_as_current_span("agentic.targeted_retrieve") as span:
            new_query = f"{state['query']} — supplementary retrieval: {state['eval_gap']}"
            span.set_attribute("iteration", state["iteration"])
            span.set_attribute("eval_gap", state["eval_gap"])
            param = QueryParam(mode="hybrid")
            chunks, trace = await self._router.route(new_query, param)
            span.set_attribute("chunks_retrieved", len(chunks))
        iter_key = f"targeted_retrieve_{state['iteration']}"
        return {
            "retrieved_chunks": chunks,
            "current_search_query": new_query,
            "iteration": state["iteration"] + 1,
            "routing_trace": {**state.get("routing_trace", {}), iter_key: trace},
        }

    # ── Conditional edges ────────────────────────────────────────────────────

    def _should_retry(self, state: AgentState) -> str:
        max_iter = MAX_ITER_BY_COMPLEXITY.get(state["complexity"], 1)
        if state["eval_score"] >= EVAL_PASS_THRESHOLD or state["iteration"] >= max_iter:
            return "end"
        return "retry"

    # ── Graph ────────────────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("classify", self._node_classify)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("decompose", self._node_decompose)
        builder.add_node("parallel_retrieve", self._node_parallel_retrieve)
        builder.add_node("generate", self._node_generate)
        builder.add_node("evaluate", self._node_evaluate)
        builder.add_node("targeted_retrieve", self._node_targeted_retrieve)

        builder.set_entry_point("classify")
        builder.add_conditional_edges(
            "classify",
            lambda s: "decompose" if s["complexity"] == "complex" else "retrieve",
            {"decompose": "decompose", "retrieve": "retrieve"},
        )
        builder.add_edge("retrieve", "generate")
        builder.add_edge("decompose", "parallel_retrieve")
        builder.add_edge("parallel_retrieve", "generate")
        builder.add_conditional_edges(
            "generate",
            lambda s: "end" if s["complexity"] == "simple" else "evaluate",
            {"end": END, "evaluate": "evaluate"},
        )
        builder.add_conditional_edges(
            "evaluate",
            self._should_retry,
            {"end": END, "retry": "targeted_retrieve"},
        )
        builder.add_edge("targeted_retrieve", "generate")

        return builder.compile()

    # ── Public API ───────────────────────────────────────────────────────────

    async def run(self, query: str, return_trace: bool = False, **kwargs: Any) -> str | dict:
        with _tracer.start_as_current_span("agentic.run") as span:
            span.set_attribute("query", query[:200])
        initial: AgentState = {
            "query": query,
            "complexity": "medium",
            "sub_questions": [],
            "retrieved_chunks": [],
            "answer": "",
            "eval_score": 0.0,
            "eval_gap": "",
            "current_search_query": query,
            "iteration": 0,  # overwritten to 0 by _node_classify; kept for LangGraph state completeness
            "routing_trace": {},
        }
        final = await self._graph.ainvoke(initial)
        if return_trace:
            return {
                "answer": final["answer"],
                "trace": {
                    "complexity": final["complexity"],
                    "eval_score": final["eval_score"],
                    "iteration": final["iteration"],
                    "routing": final["routing_trace"],
                },
            }
        return final["answer"]

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _generate_answer(self, query: str, chunks: list[dict]) -> str:
        if not chunks:
            context = "No relevant information found."
        else:
            parts = [
                f"[{i + 1}] Source: {c.get('file_path', 'unknown')}\n{c.get('content', '')}"
                for i, c in enumerate(chunks)
            ]
            context = "\n\n---\n\n".join(parts)
        prompt = (
            f"Answer the following question based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a comprehensive response."
        )
        answer = await self._llm(prompt)
        return answer if isinstance(answer, str) else str(answer)

    async def _decompose_query(self, query: str) -> list[str]:
        try:
            raw = await self._llm(
                _DECOMPOSE_PROMPT.format(query=query),
                response_format={"type": "json_object"},
            )
            result = json.loads(raw)
            sub_qs = result.get("sub_questions", [])
            if not sub_qs or not isinstance(sub_qs, list):
                return [query]
            return [str(q) for q in sub_qs[:4]]
        except Exception:
            logger.warning("Decompose failed, using original query", exc_info=True)
            return [query]


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate by chunk_id, keeping the entry with the highest rrf_score."""
    seen: dict[str, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id", "")
        if not cid:
            logger.debug("_dedup_chunks: dropping chunk with no chunk_id or id: %r", c)
            continue
        if cid not in seen or c.get("rrf_score", 0.0) > seen[cid].get("rrf_score", 0.0):
            seen[cid] = c
    return list(seen.values())
