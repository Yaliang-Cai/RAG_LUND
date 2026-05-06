from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from lightrag import QueryParam
from raganything.constants import (
    DEFAULT_AGENTIC_MAX_RETRIEVE_CYCLES,
    DEFAULT_AGENTIC_MAX_CHECK_CYCLES,
    DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS,
    DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY,
)
from .classifier import QueryClassifier
from .grader import Grader, build_shared_prefix
from .rewriter import Rewriter
from .hallucination_checker import HallucinationChecker
from .router import RetrievalRouter
from .router_cache import RouterCache

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """\
Break this question into {max_sub} or fewer independent sub-questions, \
each answerable by searching a knowledge base independently.

Question: {query}

Rules:
- Each sub-question must be self-contained (no references to other sub-questions).
- Sub-questions must not overlap in scope.
- Prefer 2 sub-questions over 4 unless truly needed.

Output JSON: {{"sub_questions": ["...", "..."]}}
"""

_GENERATOR_SUFFIX = """\
Question: {query}

Answer the question based ONLY on the context above.
If the context lacks the information needed to answer accurately, \
say so explicitly rather than speculating.

Provide a comprehensive response.
"""


class AgentState(TypedDict):
    query: str
    current_query: str
    profile: str
    chunks: list[dict]
    grader_sufficient: bool
    grader_reason: str
    answer: str
    grounded: bool
    ungrounded_claims: list[str]
    retrieve_cycle: int
    check_cycle: int
    routing_trace: dict


class AdaptiveAgentGraph:
    def __init__(
        self,
        lightrag: Any,
        llm_func: Any = None,
        *,
        _classifier: QueryClassifier | None = None,
        _grader: Grader | None = None,
        _rewriter: Rewriter | None = None,
        _checker: HallucinationChecker | None = None,
        _router: RetrievalRouter | None = None,
        _cache: RouterCache | None = None,
        max_retrieve_cycles: int = DEFAULT_AGENTIC_MAX_RETRIEVE_CYCLES,
        max_check_cycles: int = DEFAULT_AGENTIC_MAX_CHECK_CYCLES,
    ) -> None:
        self._lightrag = lightrag
        self._llm = llm_func or lightrag.llm_model_func
        self._clf = _classifier or QueryClassifier(self._llm)
        self._grader = _grader or Grader(self._llm)
        self._rewriter = _rewriter or Rewriter(self._llm)
        self._checker = _checker or HallucinationChecker(self._llm)
        self._router = _router or RetrievalRouter(lightrag, self._llm)
        self._cache = _cache or RouterCache()
        self._max_retrieve_cycles = max_retrieve_cycles
        self._max_check_cycles = max_check_cycles
        self._graph = self._build_graph()

    # ── Nodes ──────────────────────────────────────────────────────────────

    async def _node_router(self, state: AgentState) -> dict:
        query = state["query"]
        cached = self._cache.get(query)
        if cached and cached["outcome"] != "failed":
            profile = cached["profile"]
            cache_hit = True
        else:
            avoid = self._cache.get_avoid_profiles(query)
            profile, meta = await self._clf.classify(query, avoid=avoid)
            self._cache.put(query, profile)
            cache_hit = False
            logger.debug("Router LLM: %r → %s (conf=%.2f)", query[:60], profile, meta["confidence"])
        return {
            "current_query": query,
            "profile": profile,
            "retrieve_cycle": 0,
            "check_cycle": 0,
            "routing_trace": {
                "profile": profile,
                "router_cache_hit": cache_hit,
                "rewrite_history": [query],
                "sub_questions": None,
                "chunks_per_path": {},
            },
        }

    async def _node_retriever(self, state: AgentState) -> dict:
        param = QueryParam(mode="hybrid")
        routing_trace = dict(state.get("routing_trace", {}))
        routing_trace.setdefault("chunks_per_path", {})
        try:
            chunks, trace = await self._router.route(
                state["current_query"], param, profile_name=state["profile"]
            )
            routing_trace["chunks_per_path"].update(trace.get("chunks_per_path", {}))
        except Exception:
            logger.warning("Retriever failed (all paths failed), returning empty chunks", exc_info=True)
            chunks = []
        return {"chunks": chunks, "routing_trace": routing_trace}

    async def _node_grader(self, state: AgentState) -> dict:
        result = await self._grader.grade(state["current_query"], state["chunks"])
        return {"grader_sufficient": result["sufficient"], "grader_reason": result["reason"]}

    async def _node_rewriter(self, state: AgentState) -> dict:
        new_q = await self._rewriter.rewrite(state["current_query"], state["grader_reason"])
        history = list(state["routing_trace"].get("rewrite_history", []))
        history.append(new_q)
        return {
            "current_query": new_q,
            "retrieve_cycle": state["retrieve_cycle"] + 1,
            "routing_trace": {**state["routing_trace"], "rewrite_history": history},
        }

    async def _node_decomposer(self, state: AgentState) -> dict:
        try:
            raw = await self._llm(
                _DECOMPOSE_PROMPT.format(
                    query=state["query"],
                    max_sub=DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS,
                ),
                response_format={"type": "json_object"},
            )
            sub_qs = json.loads(raw).get("sub_questions", [])
            if not sub_qs:
                sub_qs = [state["query"]]
        except Exception:
            logger.warning("Decomposer failed, using original query", exc_info=True)
            sub_qs = [state["query"]]
        sub_qs = [str(q) for q in sub_qs[:DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS]]
        return {
            "retrieve_cycle": state["retrieve_cycle"] + 1,
            "routing_trace": {**state["routing_trace"], "sub_questions": sub_qs},
        }

    async def _node_parallel_retriever(self, state: AgentState) -> dict:
        sub_qs = state["routing_trace"].get("sub_questions") or [state["query"]]
        sem = asyncio.Semaphore(DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY)
        param = QueryParam(mode="hybrid")

        async def _one(q: str) -> list[dict]:
            async with sem:
                chunks, _ = await self._router.route(q, param, profile_name="full")
                return chunks

        results = await asyncio.gather(*[_one(q) for q in sub_qs], return_exceptions=True)
        all_chunks: list[dict] = []
        for r in results:
            if not isinstance(r, BaseException):
                all_chunks.extend(r)
        return {"chunks": _dedup_chunks(all_chunks)[:30]}

    async def _node_generator(self, state: AgentState) -> dict:
        prefix = build_shared_prefix(state["chunks"])
        prompt = prefix + _GENERATOR_SUFFIX.format(query=state["query"])
        try:
            raw = await self._llm(prompt)
            answer = raw if isinstance(raw, str) else str(raw)
        except Exception:
            logger.warning("Generator failed", exc_info=True)
            answer = ""
        return {"answer": answer}

    async def _node_hallucination_check(self, state: AgentState) -> dict:
        result = await self._checker.verify(state["query"], state["answer"], state["chunks"])
        return {
            "grounded": result["grounded"],
            "ungrounded_claims": result.get("ungrounded_claims", []),
        }

    async def _node_targeted_retriever(self, state: AgentState) -> dict:
        new_q = " ".join(state["ungrounded_claims"]) or state["query"]
        param = QueryParam(mode="hybrid")
        new_chunks, _ = await self._router.route(new_q, param, profile_name=state["profile"])
        combined = _dedup_chunks(state["chunks"] + new_chunks)[:30]
        return {
            "chunks": combined,
            "check_cycle": state["check_cycle"] + 1,
        }

    async def _node_end_grounded(self, state: AgentState) -> dict:
        self._cache.mark_success(state["query"])
        return {}

    async def _node_end_insufficient(self, state: AgentState) -> dict:
        self._cache.mark_failed(state["query"])
        return {}

    # ── Conditional edges ──────────────────────────────────────────────────

    def _after_grade(self, state: AgentState) -> str:
        if state["grader_sufficient"]:
            return "generate"
        if state["retrieve_cycle"] < 1:
            return "rewrite"
        if state["retrieve_cycle"] < 2:
            return "decompose"
        return "end_insufficient"

    def _after_check(self, state: AgentState) -> str:
        if state["grounded"]:
            return "end_grounded"
        if state["check_cycle"] < self._max_check_cycles:
            return "targeted"
        return "end_insufficient"

    # ── Graph ──────────────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("router", self._node_router)
        builder.add_node("retriever", self._node_retriever)
        builder.add_node("grader", self._node_grader)
        builder.add_node("rewriter", self._node_rewriter)
        builder.add_node("decomposer", self._node_decomposer)
        builder.add_node("parallel_retriever", self._node_parallel_retriever)
        builder.add_node("generator", self._node_generator)
        builder.add_node("hallucination_check", self._node_hallucination_check)
        builder.add_node("targeted_retriever", self._node_targeted_retriever)
        builder.add_node("end_grounded", self._node_end_grounded)
        builder.add_node("end_insufficient", self._node_end_insufficient)

        builder.set_entry_point("router")
        builder.add_edge("router", "retriever")
        builder.add_edge("retriever", "grader")
        builder.add_conditional_edges("grader", self._after_grade, {
            "generate": "generator",
            "rewrite": "rewriter",
            "decompose": "decomposer",
            "end_insufficient": "end_insufficient",
        })
        builder.add_edge("rewriter", "retriever")
        builder.add_edge("decomposer", "parallel_retriever")
        builder.add_edge("parallel_retriever", "grader")
        builder.add_edge("generator", "hallucination_check")
        builder.add_conditional_edges("hallucination_check", self._after_check, {
            "end_grounded": "end_grounded",
            "targeted": "targeted_retriever",
            "end_insufficient": "end_insufficient",
        })
        builder.add_edge("targeted_retriever", "generator")
        builder.add_edge("end_grounded", END)
        builder.add_edge("end_insufficient", END)

        return builder.compile()

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(self, query: str, return_trace: bool = False, **kwargs: Any) -> str | dict:
        initial: AgentState = {
            "query": query,
            "current_query": query,
            "profile": "semantic",
            "chunks": [],
            "grader_sufficient": False,
            "grader_reason": "",
            "answer": "",
            "grounded": False,
            "ungrounded_claims": [],
            "retrieve_cycle": 0,
            "check_cycle": 0,
            "routing_trace": {},
        }
        final = await self._graph.ainvoke(initial)
        answer: str | None = final.get("answer") or None
        grounded = final.get("grounded", False)
        if grounded:
            confidence = "high"
        else:
            confidence = "none"
            answer = None

        if return_trace:
            return {
                "answer": answer,
                "confidence": confidence,
                "grounded": grounded,
                "ungrounded_claims": final.get("ungrounded_claims", []),
                "trace": {
                    **final.get("routing_trace", {}),
                    "retrieve_cycles_used": final.get("retrieve_cycle", 0),
                    "check_cycles_used": final.get("check_cycle", 0),
                },
            }
        return answer if answer is not None else ""


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id", "")
        if not cid:
            continue
        if cid not in seen or c.get("rrf_score", 0.0) > seen[cid].get("rrf_score", 0.0):
            seen[cid] = c
    return list(seen.values())
