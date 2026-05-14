from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
from collections import Counter
from typing import Any

from lightrag import QueryParam

from raganything.constants import (
    DEFAULT_AGENTIC_MAX_CHECK_CYCLES,
    DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS,
    DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY,
)
from .classifier import QueryClassifier
from .grader import build_shared_prefix
from .grader_v2 import GraderV2
from .hallucination_checker import HallucinationChecker
from .json_utils import call_json_object
from .recovery_policy import RecoveryAction, RecoveryPolicy
from .rewriter import Rewriter
from .router import RetrievalRouter, _select_with_path_floors
from .router_cache import RouterCache

logger = logging.getLogger(__name__)

_QUERY_PARAM_FIELDS = {f.name for f in dataclasses.fields(QueryParam)}
_DEFAULT_MAX_RETRIEVE_STEPS = 5

_DECOMPOSE_PROMPT = """\
Break this question into {max_sub} or fewer independent sub-questions,
each answerable by searching a knowledge base independently.

Question: {query}

Rules:
- Each sub-question must be self-contained.
- Sub-questions must not overlap in scope.
- Prefer 2 sub-questions over 4 unless truly needed.

Output JSON only: {{"sub_questions": ["...", "..."]}}
"""

_GENERATOR_SUFFIX = """\
Question: {query}

Answer the question based ONLY on the context above.
If the context lacks the information needed to answer accurately, say so explicitly.

Provide a comprehensive response.
"""


class AdaptiveAgentGraphV2:
    def __init__(
        self,
        lightrag: Any,
        llm_func: Any = None,
        *,
        _classifier: QueryClassifier | None = None,
        _grader: GraderV2 | None = None,
        _rewriter: Rewriter | None = None,
        _checker: HallucinationChecker | None = None,
        _router: RetrievalRouter | None = None,
        _cache: RouterCache | None = None,
        _policy: RecoveryPolicy | None = None,
        max_retrieve_steps: int = _DEFAULT_MAX_RETRIEVE_STEPS,
        max_check_cycles: int = DEFAULT_AGENTIC_MAX_CHECK_CYCLES,
    ) -> None:
        self._lightrag = lightrag
        self._llm = llm_func or lightrag.llm_model_func
        self._clf = _classifier or QueryClassifier(self._llm)
        self._grader = _grader or GraderV2(self._llm)
        self._rewriter = _rewriter or Rewriter(self._llm)
        self._checker = _checker or HallucinationChecker(self._llm)
        self._router = _router or RetrievalRouter(lightrag, self._llm)
        self._cache = _cache or RouterCache()
        self._policy = _policy or RecoveryPolicy()
        self._max_retrieve_steps = max(1, int(max_retrieve_steps))
        self._max_check_cycles = max(0, int(max_check_cycles))

    async def run(self, query: str, return_trace: bool = False, **kwargs: Any) -> str | dict:
        param = QueryParam(
            mode="hybrid",
            **{k: v for k, v in kwargs.items() if k in _QUERY_PARAM_FIELDS},
        )
        profile, classifier_meta, cache_hit = await self._select_profile(query)
        trace: dict[str, Any] = {
            "profile": profile,
            "router_cache_hit": cache_hit,
            "classifier": {
                **classifier_meta,
                "selected_profile": classifier_meta.get("selected_profile", profile),
            },
            "rewrite_history": [query],
            "sub_questions": None,
            "chunks_per_path": {},
            "retrieval_steps": [],
            "grader_events": [],
            "hallucination_events": [],
            "recovery_actions": [],
            "failure_type_counts": {},
        }

        tried_signatures: set[tuple[str, str, str]] = {("initial", profile, "original")}
        tried_profiles: set[str] = {profile}
        failure_counts: Counter[str] = Counter()
        best: dict[str, Any] | None = None
        last_chunks: list[dict] = []
        action = RecoveryAction("initial", profile, "original")
        current_query = query
        terminal_reason = "insufficient"
        grounded = False
        answer: str | None = None
        ungrounded_claims: list[str] = []
        retrieve_cycle = 0
        check_cycle = 0

        for retrieve_cycle in range(self._max_retrieve_steps):
            chunks, step_trace, step_type, step_query, step_profile = await self._execute_retrieval(
                action=action,
                query=query,
                current_query=current_query,
                param=param,
            )
            last_chunks = chunks
            _merge_path_trace(trace, step_trace)
            trace["retrieval_steps"].append(_retrieval_step(
                step_type=step_type,
                query=step_query,
                profile=step_profile,
                trace=step_trace,
                chunks=chunks,
                cycle=retrieve_cycle,
            ))

            grade = await self._grader.grade(query, chunks)
            failure_counts[grade["failure_type"]] += 1
            trace["failure_type_counts"] = dict(sorted(failure_counts.items()))
            grade_event = {
                **grade,
                "cycle": retrieve_cycle,
                "query": step_query,
                "profile": step_profile,
                "step_type": step_type,
                "chunks": len(chunks),
            }
            trace["grader_events"].append(grade_event)
            candidate = {
                "chunks": chunks,
                "grade": grade,
                "step": {
                    "cycle": retrieve_cycle,
                    "profile": step_profile,
                    "step_type": step_type,
                    "failure_type": grade["failure_type"],
                    "coverage_score": grade["coverage_score"],
                    "sufficient": grade["sufficient"],
                    "chunks": len(chunks),
                },
            }
            if _is_better_candidate(candidate, best):
                best = candidate
                trace["best_step"] = candidate["step"]

            if grade["sufficient"]:
                (
                    answer,
                    grounded,
                    ungrounded_claims,
                    check_cycle,
                    terminal_reason,
                    best,
                    last_chunks,
                ) = (
                    await self._generate_check_and_target(
                        query=query,
                        best=best,
                        trace=trace,
                        param=param,
                        check_cycle=0,
                    )
                )
                break

            if retrieve_cycle + 1 >= self._max_retrieve_steps:
                terminal_reason = (
                    "unanswerable" if grade.get("unanswerable") else "insufficient"
                )
                break

            action = self._policy.choose(
                failure_type=grade["failure_type"],
                original_profile=profile,
                original_query=query,
                tried_profiles=tried_profiles,
                tried_signatures=tried_signatures,
            )
            if action is None:
                terminal_reason = (
                    "unanswerable" if grade.get("unanswerable") else "insufficient"
                )
                break

            tried_signatures.add(action.signature)
            tried_profiles.add(action.profile)
            trace["recovery_actions"].append({
                "cycle": retrieve_cycle,
                "action_type": action.action_type,
                "profile": action.profile,
                "query_kind": action.query_kind,
                "failure_type": grade["failure_type"],
            })
            if action.action_type == "rewrite":
                current_query = await self._rewrite_query(query, grade)
                trace["rewrite_history"].append(current_query)
            else:
                current_query = query

        else:
            terminal_reason = "insufficient"

        best_chunks = _candidate_chunks(best)
        if terminal_reason == "grounded":
            self._cache.mark_success(query)
        else:
            self._cache.mark_failed(query)
            if not grounded:
                answer = None

        if return_trace:
            return {
                "answer": answer,
                "confidence": "high" if grounded else "none",
                "grounded": grounded,
                "ungrounded_claims": ungrounded_claims,
                "trace": {
                    **trace,
                    "grounded": grounded,
                    "terminal_reason": terminal_reason,
                    "retrieve_cycles_used": retrieve_cycle,
                    "check_cycles_used": check_cycle,
                    "data": {
                        "chunks": best_chunks,
                        "last_chunks": last_chunks,
                    },
                },
            }
        return answer if answer is not None and grounded else ""

    async def _select_profile(self, query: str) -> tuple[str, dict[str, Any], bool]:
        cached = self._cache.get(query)
        if cached and cached["outcome"] != "failed":
            profile = cached["profile"]
            return profile, {
                "confidence": 1.0,
                "reasoning": "router cache hit",
                "latency": 0.0,
                "candidate_profile": profile,
                "selected_profile": profile,
                "fallback_used": False,
                "fallback_reason": "",
            }, True
        avoid = self._cache.get_avoid_profiles(query)
        profile, meta = await self._clf.classify(query, avoid=avoid)
        self._cache.put(query, profile)
        return profile, meta, False

    async def _execute_retrieval(
        self,
        *,
        action: RecoveryAction,
        query: str,
        current_query: str,
        param: QueryParam,
    ) -> tuple[list[dict], dict, str, str, str]:
        if action.action_type == "decompose":
            chunks, trace = await self._retrieve_decomposed(query, param)
            return chunks, trace, "decompose", query, action.profile

        step_query = current_query if action.query_kind == "rewrite" else query
        step_type = "initial" if action.action_type == "initial" else action.action_type
        try:
            chunks, trace = await self._router.route(
                step_query,
                param,
                profile_name=action.profile,
            )
            return chunks[:_chunk_limit(param)], trace, step_type, step_query, action.profile
        except Exception:
            logger.warning("Agentic v2 retrieval failed", exc_info=True)
            return [], _failed_trace(action.profile), step_type, step_query, action.profile

    async def _retrieve_decomposed(self, query: str, param: QueryParam) -> tuple[list[dict], dict]:
        sub_questions = await self._decompose(query)
        sub_questions = sub_questions or [query]
        sem = asyncio.Semaphore(DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY)

        async def _one(sub_query: str) -> tuple[str, list[dict], dict]:
            async with sem:
                chunks, trace = await self._router.route(
                    sub_query,
                    param,
                    profile_name="full_v2",
                )
                return sub_query, chunks, trace

        results = await asyncio.gather(
            *[_one(q) for q in sub_questions],
            return_exceptions=True,
        )
        all_chunks: list[dict] = []
        sub_results: list[tuple[str, list[dict]]] = []
        paths_activated: list[str] = []
        paths_failed: list[str] = []
        chunks_per_path: dict[str, int] = {}
        ppr_floor_count = 0
        path_floor_applied: dict[str, dict[str, int]] = {}
        sub_traces = []
        for result in results:
            if isinstance(result, BaseException):
                paths_failed.append("full_v2")
                continue
            sub_query, chunks, trace = result
            all_chunks.extend(chunks)
            sub_results.append((sub_query, chunks))
            sub_traces.append({"query": sub_query, **trace})
            for path in trace.get("paths_activated", []):
                if path not in paths_activated:
                    paths_activated.append(path)
            paths_failed.extend(trace.get("paths_failed", []))
            for path, count in trace.get("chunks_per_path", {}).items():
                chunks_per_path[path] = chunks_per_path.get(path, 0) + int(count)
            ppr_floor_count += int(trace.get("ppr_floor_count", 0) or 0)
            for path, floor_data in trace.get("path_floor_applied", {}).items():
                aggregate = path_floor_applied.setdefault(
                    path,
                    {"requested": 0, "available": 0, "selected": 0},
                )
                aggregate["requested"] += int(floor_data.get("requested", 0) or 0)
                aggregate["available"] += int(floor_data.get("available", 0) or 0)
                aggregate["selected"] += int(floor_data.get("selected", 0) or 0)
        merged_chunks, merge_trace = _merge_decomposed_chunks(
            sub_results,
            chunk_top_k=_chunk_limit(param),
            path_floors={"ppr": 3},
        )
        return merged_chunks, {
            "profile": "full_v2",
            "sub_questions": sub_questions,
            "paths_activated": paths_activated,
            "paths_failed": paths_failed,
            "chunks_per_path": chunks_per_path,
            "chunks_after_rrf": len(all_chunks),
            "chunks_after_rerank": len(all_chunks),
            "chunks_after_threshold": len(merged_chunks),
            "sub_traces": sub_traces,
            "path_floor_applied": path_floor_applied,
            "ppr_floor_count": ppr_floor_count,
            **merge_trace,
        }

    async def _decompose(self, query: str) -> list[str]:
        try:
            parsed = await call_json_object(
                self._llm,
                _DECOMPOSE_PROMPT.format(
                    query=query,
                    max_sub=DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS,
                ),
                max_tokens=512,
            )
            questions = parsed.get("sub_questions", [])
            if not questions:
                questions = [query]
        except Exception:
            logger.warning("Agentic v2 decomposer failed", exc_info=True)
            questions = [query]
        return [str(q) for q in questions[:DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS]]

    async def _rewrite_query(self, query: str, grade: dict) -> str:
        if hasattr(self._rewriter, "rewrite_with_feedback"):
            return await self._rewriter.rewrite_with_feedback(
                query,
                failure_type=grade["failure_type"],
                reason=grade.get("reason", ""),
                found_facts=grade.get("found_facts", []),
                missing_facts=grade.get("missing_facts", []),
            )
        return await self._rewriter.rewrite(query, grade.get("reason", ""))

    async def _generate_check_and_target(
        self,
        *,
        query: str,
        best: dict[str, Any] | None,
        trace: dict[str, Any],
        param: QueryParam,
        check_cycle: int,
    ) -> tuple[str | None, bool, list[str], int, str, dict[str, Any] | None, list[dict]]:
        best_chunks = _candidate_chunks(best)
        last_chunks = best_chunks
        answer = await self._generate(query, best_chunks)
        grounded, claims = await self._check(query, answer, best_chunks, trace, check_cycle)
        if grounded:
            return answer, True, claims, check_cycle, "grounded", best, last_chunks

        for targeted_idx in range(self._max_check_cycles):
            profile = _candidate_profile(best) if targeted_idx == 0 else "full_v2"
            target_query = " ".join(claims) or query
            try:
                new_chunks, target_trace = await self._router.route(
                    target_query,
                    param,
                    profile_name=profile,
                )
            except Exception:
                logger.warning("Agentic v2 targeted retrieval failed", exc_info=True)
                new_chunks = []
                target_trace = _failed_trace(profile)
            last_chunks = new_chunks
            combined = _dedup_chunks(best_chunks + new_chunks)[:_chunk_limit(param)]
            trace["retrieval_steps"].append(_retrieval_step(
                step_type="targeted",
                query=target_query,
                profile=profile,
                trace=target_trace,
                chunks=new_chunks,
                cycle=targeted_idx,
            ))
            _merge_path_trace(trace, target_trace)
            grade = await self._grader.grade(query, combined)
            trace["grader_events"].append({
                **grade,
                "cycle": targeted_idx,
                "query": target_query,
                "profile": profile,
                "step_type": "targeted",
                "chunks": len(combined),
            })
            candidate = {
                "chunks": combined,
                "grade": grade,
                "step": {
                    "cycle": targeted_idx,
                    "profile": profile,
                    "step_type": "targeted",
                    "failure_type": grade["failure_type"],
                    "coverage_score": grade["coverage_score"],
                    "sufficient": grade["sufficient"],
                    "chunks": len(combined),
                },
            }
            if _is_better_candidate(candidate, best):
                best = candidate
                best_chunks = combined
                trace["best_step"] = candidate["step"]
            answer = await self._generate(query, best_chunks)
            check_cycle = targeted_idx + 1
            grounded, claims = await self._check(
                query, answer, best_chunks, trace, check_cycle
            )
            if grounded:
                return answer, True, claims, check_cycle, "grounded", best, last_chunks
        return None, False, claims, check_cycle, "ungrounded", best, last_chunks

    async def _generate(self, query: str, chunks: list[dict]) -> str:
        prompt = build_shared_prefix(chunks) + _GENERATOR_SUFFIX.format(query=query)
        try:
            raw = await self._llm(prompt)
            return raw if isinstance(raw, str) else str(raw)
        except Exception:
            logger.warning("Agentic v2 generator failed", exc_info=True)
            return ""

    async def _check(
        self,
        query: str,
        answer: str,
        chunks: list[dict],
        trace: dict[str, Any],
        cycle: int,
    ) -> tuple[bool, list[str]]:
        if not str(answer).strip():
            result = {
                "grounded": False,
                "ungrounded_claims": [query],
                "check_status": "empty_answer",
            }
            trace["hallucination_events"].append({
                **result,
                "cycle": cycle,
                "chunks": len(chunks),
            })
            return False, [query]
        result = await self._checker.verify(query, answer, chunks)
        trace["hallucination_events"].append({
            **result,
            "cycle": cycle,
            "chunks": len(chunks),
        })
        return bool(result.get("grounded", False)), [
            str(claim) for claim in result.get("ungrounded_claims", [])
        ]


def _retrieval_step(
    *,
    step_type: str,
    query: str,
    profile: str,
    trace: dict,
    chunks: list[dict],
    cycle: int,
) -> dict:
    step = {
        "type": step_type,
        "query": query,
        "profile": profile,
        "cycle": cycle,
        "chunks": len(chunks),
        "paths_activated": trace.get("paths_activated", []),
        "paths_failed": trace.get("paths_failed", []),
        "chunks_per_path": trace.get("chunks_per_path", {}),
        "chunks_after_rrf": trace.get("chunks_after_rrf", len(chunks)),
        "chunks_after_rerank": trace.get("chunks_after_rerank", len(chunks)),
        "chunks_after_threshold": trace.get("chunks_after_threshold", len(chunks)),
    }
    if trace.get("sub_questions"):
        step["sub_questions"] = trace["sub_questions"]
    for key in (
        "path_floor_applied",
        "ppr_floor_count",
        "decompose_merge_policy",
        "per_subquestion_cap",
        "sub_question_chunk_counts",
    ):
        if key in trace:
            step[key] = trace[key]
    return step


def _merge_path_trace(trace: dict[str, Any], step_trace: dict[str, Any]) -> None:
    trace["paths_activated"] = step_trace.get("paths_activated", [])
    trace["paths_failed"] = step_trace.get("paths_failed", [])
    trace.setdefault("chunks_per_path", {}).update(step_trace.get("chunks_per_path", {}))
    if step_trace.get("sub_questions"):
        trace["sub_questions"] = step_trace["sub_questions"]


def _failed_trace(profile: str) -> dict:
    return {
        "profile": profile,
        "paths_activated": [],
        "paths_failed": ["all"],
        "chunks_per_path": {},
        "chunks_after_rrf": 0,
        "chunks_after_rerank": 0,
        "chunks_after_threshold": 0,
    }


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for index, chunk in enumerate(chunks):
        cid = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or _synthetic_chunk_key(chunk, index)
        )
        if not cid:
            continue
        if cid not in seen or chunk.get("rrf_score", 0.0) > seen[cid].get("rrf_score", 0.0):
            seen[cid] = chunk
    return list(seen.values())


def _merge_decomposed_chunks(
    sub_results: list[tuple[str, list[dict]]],
    *,
    chunk_top_k: int,
    path_floors: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    limit = max(1, int(chunk_top_k or 10))
    if not sub_results:
        return [], {
            "decompose_merge_policy": "balanced_subquestion_then_global_score",
            "per_subquestion_cap": 0,
            "sub_question_chunk_counts": {},
        }

    per_subquestion_cap = min(4, max(2, math.ceil(limit / len(sub_results))))
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_chunk(chunk: dict) -> bool:
        chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or _synthetic_chunk_key(chunk, len(selected))
        )
        if not chunk_id or chunk_id in selected_ids:
            return False
        selected.append(chunk)
        selected_ids.add(str(chunk_id))
        return True

    for _, chunks in sub_results:
        taken = 0
        for chunk in chunks:
            if len(selected) >= limit or taken >= per_subquestion_cap:
                break
            if add_chunk(chunk):
                taken += 1

    all_unique = _dedup_chunks([chunk for _, chunks in sub_results for chunk in chunks])
    all_unique.sort(key=_chunk_score, reverse=True)
    for chunk in all_unique:
        add_chunk(chunk)

    final_chunks, floor_trace = _select_with_path_floors(
        selected,
        chunk_top_k=limit,
        path_floors=path_floors or {},
    )
    return final_chunks, {
        "decompose_merge_policy": "balanced_subquestion_then_global_score",
        "per_subquestion_cap": per_subquestion_cap,
        "sub_question_chunk_counts": {
            sub_query: len(chunks)
            for sub_query, chunks in sub_results
        },
        "path_floor_applied": floor_trace,
        "ppr_floor_count": floor_trace.get("ppr", {}).get("selected", 0),
    }


def _chunk_score(chunk: dict) -> float:
    for key in ("rerank_score", "rrf_score", "score"):
        try:
            return float(chunk.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _synthetic_chunk_key(chunk: dict, index: int) -> str:
    content = str(chunk.get("content") or "").strip()
    source = str(chunk.get("file_path") or chunk.get("source") or "").strip()
    return f"synthetic:{source}:{content[:200]}:{index}"


def _chunk_limit(param: QueryParam) -> int:
    try:
        return max(1, int(getattr(param, "chunk_top_k", 10) or 10))
    except (TypeError, ValueError):
        return 10


def _candidate_chunks(candidate: dict[str, Any] | None) -> list[dict]:
    if not candidate:
        return []
    return list(candidate.get("chunks") or [])


def _candidate_profile(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "full_v2"
    return str(candidate.get("step", {}).get("profile") or "full_v2")


def _is_better_candidate(new: dict[str, Any], old: dict[str, Any] | None) -> bool:
    if old is None:
        return True
    new_key = _candidate_key(new)
    old_key = _candidate_key(old)
    return new_key > old_key


def _candidate_key(candidate: dict[str, Any]) -> tuple:
    grade = candidate.get("grade", {})
    step = candidate.get("step", {})
    failure_type = str(grade.get("failure_type") or "")
    return (
        1 if grade.get("sufficient") else 0,
        float(grade.get("coverage_score") or 0.0),
        0 if failure_type in {"empty", "off_topic"} else 1,
        _stage_rank(str(step.get("profile") or ""), str(step.get("step_type") or "")),
        len(candidate.get("chunks") or []),
    )


def _stage_rank(profile: str, step_type: str) -> int:
    if step_type == "decompose":
        return 6
    if profile == "full_v2":
        return 5
    if profile == "multihop":
        return 4
    if profile == "semantic":
        return 3
    if profile in {"global", "local"}:
        return 2
    if profile == "precise":
        return 1
    return 0
