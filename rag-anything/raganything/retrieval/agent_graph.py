from __future__ import annotations

import asyncio
import dataclasses
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
from .json_utils import load_json_object
from .rewriter import Rewriter
from .hallucination_checker import HallucinationChecker
from .router import RetrievalRouter
from .router_cache import RouterCache

logger = logging.getLogger(__name__)
_QUERY_PARAM_FIELDS = {f.name for f in dataclasses.fields(QueryParam)}

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
    grader_unanswerable: bool
    grader_reason: str
    answer: str
    grounded: bool
    ungrounded_claims: list[str]
    retrieve_cycle: int
    check_cycle: int
    routing_trace: dict
    query_param_kwargs: dict[str, Any]


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
            meta = {
                "confidence": 1.0,
                "reasoning": "router cache hit",
                "latency": 0.0,
                "candidate_profile": profile,
                "selected_profile": profile,
                "fallback_used": False,
                "fallback_reason": "",
            }
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
                "classifier": {
                    **meta,
                    "selected_profile": meta.get("selected_profile", profile),
                },
                "rewrite_history": [query],
                "sub_questions": None,
                "chunks_per_path": {},
                "retrieval_steps": [],
                "grader_events": [],
                "hallucination_events": [],
            },
        }

    async def _node_retriever(self, state: AgentState) -> dict:
        param = _query_param_from_state(state)
        routing_trace = dict(state.get("routing_trace", {}))
        routing_trace.setdefault("chunks_per_path", {})
        retrieval_steps = list(routing_trace.get("retrieval_steps", []))
        try:
            chunks, trace = await self._router.route(
                state["current_query"], param, profile_name=state["profile"]
            )
            chunks = chunks[:_chunk_limit(param)]
            routing_trace["profile"] = trace.get("profile", state["profile"])
            routing_trace["paths_activated"] = trace.get("paths_activated", [])
            routing_trace["paths_failed"] = trace.get("paths_failed", [])
            routing_trace["chunks_per_path"].update(trace.get("chunks_per_path", {}))
            retrieval_steps.append(_retrieval_step(
                step_type="initial" if state["retrieve_cycle"] == 0 else "rewrite",
                query=state["current_query"],
                profile=state["profile"],
                trace=trace,
                chunks=chunks,
                cycle=state["retrieve_cycle"],
            ))
        except Exception:
            logger.warning("Retriever failed (all paths failed), returning empty chunks", exc_info=True)
            chunks = []
            retrieval_steps.append({
                "type": "initial" if state["retrieve_cycle"] == 0 else "rewrite",
                "query": state["current_query"],
                "profile": state["profile"],
                "cycle": state["retrieve_cycle"],
                "chunks": 0,
                "paths_activated": [],
                "paths_failed": ["all"],
                "chunks_per_path": {},
                "chunks_after_rrf": 0,
                "chunks_after_rerank": 0,
                "chunks_after_threshold": 0,
            })
        routing_trace["retrieval_steps"] = retrieval_steps
        return {"chunks": chunks, "routing_trace": routing_trace}

    async def _node_grader(self, state: AgentState) -> dict:
        result = await self._grader.grade(state["current_query"], state["chunks"])
        routing_trace = dict(state.get("routing_trace", {}))
        events = list(routing_trace.get("grader_events", []))
        events.append({
            **result,
            "cycle": state["retrieve_cycle"],
            "query": state["current_query"],
            "chunks": len(state["chunks"]),
        })
        routing_trace["grader_events"] = events
        return {
            "grader_sufficient": result["sufficient"],
            "grader_unanswerable": result.get("unanswerable", False),
            "grader_reason": result["reason"],
            "routing_trace": routing_trace,
        }

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
            sub_qs = load_json_object(raw).get("sub_questions", [])
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
        param = _query_param_from_state(state)

        async def _one(q: str) -> tuple[str, list[dict], dict]:
            async with sem:
                chunks, trace = await self._router.route(q, param, profile_name="full")
                return q, chunks, trace

        results = await asyncio.gather(*[_one(q) for q in sub_qs], return_exceptions=True)
        all_chunks: list[dict] = []
        routing_trace = dict(state.get("routing_trace", {}))
        routing_trace.setdefault("chunks_per_path", {})
        retrieval_steps = list(routing_trace.get("retrieval_steps", []))
        for r in results:
            if not isinstance(r, BaseException):
                sub_q, chunks, trace = r
                all_chunks.extend(chunks)
                routing_trace["chunks_per_path"].update(trace.get("chunks_per_path", {}))
                retrieval_steps.append(_retrieval_step(
                    step_type="decompose",
                    query=sub_q,
                    profile="full",
                    trace=trace,
                    chunks=chunks,
                    cycle=state["retrieve_cycle"],
                ))
        deduped = _dedup_chunks(all_chunks)[:_chunk_limit(param)]
        routing_trace["retrieval_steps"] = retrieval_steps
        return {"chunks": deduped, "routing_trace": routing_trace}

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
        routing_trace = dict(state.get("routing_trace", {}))
        events = list(routing_trace.get("hallucination_events", []))
        events.append({
            **result,
            "cycle": state["check_cycle"],
            "chunks": len(state["chunks"]),
        })
        routing_trace["hallucination_events"] = events
        return {
            "grounded": result["grounded"],
            "ungrounded_claims": result.get("ungrounded_claims", []),
            "routing_trace": routing_trace,
        }

    async def _node_targeted_retriever(self, state: AgentState) -> dict:
        new_q = " ".join(state["ungrounded_claims"]) or state["query"]
        param = _query_param_from_state(state)
        routing_trace = dict(state.get("routing_trace", {}))
        routing_trace.setdefault("chunks_per_path", {})
        retrieval_steps = list(routing_trace.get("retrieval_steps", []))
        try:
            new_chunks, trace = await self._router.route(
                new_q, param, profile_name=state["profile"]
            )
            routing_trace["chunks_per_path"].update(trace.get("chunks_per_path", {}))
        except Exception:
            logger.warning("Targeted retriever failed, keeping existing chunks", exc_info=True)
            new_chunks = []
            trace = {
                "paths_activated": [],
                "paths_failed": ["all"],
                "chunks_per_path": {},
            }
        combined = _dedup_chunks(state["chunks"] + new_chunks)[:_chunk_limit(param)]
        retrieval_steps.append(_retrieval_step(
            step_type="targeted",
            query=new_q,
            profile=state["profile"],
            trace=trace,
            chunks=new_chunks,
            cycle=state["check_cycle"],
        ))
        routing_trace["retrieval_steps"] = retrieval_steps
        return {
            "chunks": combined,
            "check_cycle": state["check_cycle"] + 1,
            "routing_trace": routing_trace,
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
        if state.get("grader_unanswerable"):
            return "end_insufficient"
        if state["retrieve_cycle"] + 1 >= self._max_retrieve_cycles:
            return "end_insufficient"
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
        query_param_kwargs = {k: v for k, v in kwargs.items() if k in _QUERY_PARAM_FIELDS}
        initial: AgentState = {
            "query": query,
            "current_query": query,
            "profile": "semantic",
            "chunks": [],
            "grader_sufficient": False,
            "grader_unanswerable": False,
            "grader_reason": "",
            "answer": "",
            "grounded": False,
            "ungrounded_claims": [],
            "retrieve_cycle": 0,
            "check_cycle": 0,
            "routing_trace": {},
            "query_param_kwargs": query_param_kwargs,
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
            routing_trace = final.get("routing_trace", {})
            chunks = final.get("chunks") or []
            return {
                "answer": answer,
                "confidence": confidence,
                "grounded": grounded,
                "ungrounded_claims": final.get("ungrounded_claims", []),
                "trace": {
                    **routing_trace,
                    "grounded": grounded,
                    "retrieve_cycles_used": final.get("retrieve_cycle", 0),
                    "check_cycles_used": final.get("check_cycle", 0),
                    "data": {"chunks": chunks},
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


def _query_param_from_state(state: AgentState) -> QueryParam:
    return QueryParam(mode="hybrid", **state.get("query_param_kwargs", {}))


def _chunk_limit(param: QueryParam) -> int:
    try:
        return max(1, int(getattr(param, "chunk_top_k", 10) or 10))
    except (TypeError, ValueError):
        return 10


def _retrieval_step(
    *,
    step_type: str,
    query: str,
    profile: str,
    trace: dict,
    chunks: list[dict],
    cycle: int,
) -> dict:
    return {
        "type": step_type,
        "query": query,
        "profile": profile,
        "cycle": cycle,
        "chunks": len(chunks),
        "paths_activated": trace.get("paths_activated", []),
        "paths_failed": trace.get("paths_failed", []),
        "chunks_per_path": trace.get("chunks_per_path", {}),
        "chunks_after_rrf": int(trace.get("chunks_after_rrf", len(chunks)) or 0),
        "chunks_after_rerank": int(trace.get("chunks_after_rerank", len(chunks)) or 0),
        "chunks_after_threshold": int(trace.get("chunks_after_threshold", len(chunks)) or 0),
    }
