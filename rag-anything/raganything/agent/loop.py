# raganything/agent/loop.py
"""Agent 主循环（spec §4/§6.5/§8.3/§9.3）。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import asyncio

from raganything.retrieval.json_utils import call_json_object
from raganything.retrieval.recovery_policy import RecoveryPolicy

from raganything.agent.budget import Budget
from raganything.agent.citations import verify_citations
from raganything.agent.decision import Decision, decision_signature, normalize_decision
from raganything.agent.evidence import EvidencePool, FactLedger
from raganything.agent.generate import generate_answer
from raganything.agent.grading import LedgerGrader, should_final_review
from raganything.agent.models import ModelPool
from raganything.agent.planner import PlanResult, make_plan
from raganything.agent.session import SessionMemory
from raganything.agent.tools import ToolRegistry
from raganything.agent.trace import TraceBuilder

logger = logging.getLogger(__name__)

MAX_STEPS = 8  # 硬上限 §4.3
# RecoveryPolicy profile → 工具名映射（降级大脑 §4.4）
_FALLBACK_TOOL = {"precise": "search_sparse", "semantic": "search_dense",
                  "multihop": "search_ppr", "full_v2": "search_hybrid",
                  "global": "search_graph", "local": "search_graph"}

_DECIDE_PROMPT = """\
You decide the next action for an evidence-gathering agent. Tools:
{cards}

Rules: start cheap, escalate only on evidence; prefer enlarging top_k on the same
tool before switching; targeted retrieval for missing facts beats query expansion;
when budget is low, prefer answering with current best evidence.
Archetype is a prior — override it when evidence disagrees (set "reclassify").

decide the next action. Output JSON only:
{{"thought": "<one sentence>", "action": "<tool name>", "params": {{...}},
  "stop": false, "reclassify": null}}

Archetype: {archetype}
Query: {query}
Evidence pool: {pool_summary}
Fact ledger: {ledger}
Action history:
{history}
Tool status: {tool_status}
Budget: {budget}
"""

RetrieveFn = Callable[[str, dict], Awaitable[tuple[list[dict], dict]]]
RerankFn = Callable[[str, list[str]], Awaitable[list[float]]]


@dataclass
class AgentResult:
    answer: str | None
    grounded: bool
    refusal: dict | None
    ledger: dict
    trace: dict
    cancelled: bool = False


@dataclass
class AgentLoop:
    model_pool: ModelPool
    registry: ToolRegistry
    retrieve_fn: RetrieveFn
    rerank_fn: RerankFn | None = None
    vision_fn: Callable | None = None
    recovery: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    max_context_tokens: int = 12_000
    _grade_override: dict | None = None

    async def run(self, query: str, session: SessionMemory, *,
                  budget: Budget | None = None, **qp_kwargs: Any) -> AgentResult:
        plan = await make_plan(self.model_pool, query, session)
        budget = budget or Budget.for_archetype(plan.archetype)
        pool, ledger = EvidencePool(), FactLedger()
        grader = LedgerGrader(self.model_pool)
        tb = TraceBuilder(profile=f"agent:{plan.archetype}", query=query)
        tb.add_rewrite(plan.standalone_query)
        tried: set[tuple] = set()
        dup_rates: list[float] = []
        ledger_steps = 0
        cq = plan.standalone_query

        if plan.fast_path:
            new = await self._execute_search(plan.preset["tool"],
                                             {"query": cq, "top_k": plan.preset["top_k"]},
                                             pool, session, cq, tb, step=0, budget=budget)
            if new is None:
                return self._cancelled_result(ledger, tb, session, query)
            grade = await self._grade(grader, cq, ledger, pool,
                                      list(pool.entries.values()))
            tb.add_grader_event(grade, cycle=0)
            if grade["sufficient"]:
                return await self._finish(cq, plan, pool, ledger, tb, budget, session,
                                          generation_mode=plan.preset["generation_mode"])

        for step in range(MAX_STEPS):
            if session.cancel_event.is_set():
                return self._cancelled_result(ledger, tb, session, query)
            reason = budget.exhausted()
            if reason:
                return await self._exhausted(cq, plan, pool, ledger, tb, budget,
                                             session, reason)
            decision = await self._decide(plan, cq, pool, ledger, tb, budget,
                                          tried, step)
            if decision is None:
                return await self._exhausted(cq, plan, pool, ledger, tb, budget,
                                             session, "no_action")
            if decision.reclassify and decision.reclassify != plan.archetype:
                if budget.upgrade(decision.reclassify):
                    tb.add_reclassify(plan.archetype, decision.reclassify, cycle=step)
                    plan.archetype = decision.reclassify
            if decision.action == "answer":
                # 过早作答守卫 §8.3：证据池为空、预算未低/未耗尽时，不允许凭零证据收尾，
                # 降级为一次确定性 preset 检索，确保账本被填充后再走耗尽/拒答路径。
                if not pool.entries and not budget.low() and not budget.exhausted():
                    decision = Decision(
                        thought="premature answer rejected: gather evidence first",
                        action=plan.preset["tool"],
                        params=self.registry.get(plan.preset["tool"]).clamp(
                            {"query": cq, "top_k": plan.preset["top_k"]}),
                        fallback=True)
                    tb.add_decision(thought=decision.thought, action=decision.action,
                                    params=decision.params,
                                    budget_snapshot=budget.snapshot(), fallback=True)
                else:
                    return await self._finish(
                        cq, plan, pool, ledger, tb, budget, session,
                        generation_mode=str(decision.params.get("generation_mode", "direct")))
            sig = decision_signature(decision)
            if sig in tried:
                continue
            tried.add(sig)
            if decision.action == "rewrite_query":
                cq = str(decision.params.get("query") or cq)
                tb.add_rewrite(cq)
                budget.charge(points=self.registry.get(decision.action).cost)
                continue
            new_entries = await self._execute_search(
                decision.action, decision.params, pool, session, cq, tb,
                step=step, budget=budget)
            if new_entries is None:
                return self._cancelled_result(ledger, tb, session, query)
            dup_rates.append(pool.last_dup_rate)
            for fact in ledger.missing():
                ledger.record_attempt(fact["id"], decision.action)
            grade = await self._grade(grader, cq, ledger, pool, new_entries)
            ledger_steps += 1
            tb.add_grader_event(grade, cycle=step)
            if grade["sufficient"]:
                if should_final_review(ledger_steps=ledger_steps, ledger=ledger,
                                       pool=pool, recent_dup_rates=dup_rates):
                    review = await grader.final_review(cq, pool)
                    tb.add_grader_event({**review, "final_review": True}, cycle=step)
                    if not review["sufficient"]:
                        fresh = FactLedger()
                        fresh.update({"facts": review.get("facts", [])})
                        ledger = fresh
                        continue
                return await self._finish(
                    cq, plan, pool, ledger, tb, budget, session,
                    generation_mode=plan.preset["generation_mode"])
        return await self._exhausted(cq, plan, pool, ledger, tb, budget, session, "max_steps")

    async def _cancellable(self, coro, session: SessionMemory):
        task = asyncio.ensure_future(coro)
        waiter = asyncio.ensure_future(session.cancel_event.wait())
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            waiter.cancel()
            return task.result()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return None

    async def _execute_search(self, tool_name, params, pool, session, cq, tb, *,
                              step, budget):
        spec = self.registry.get(tool_name)
        budget.charge(points=spec.cost)
        result = await self._cancellable(self.retrieve_fn(tool_name, dict(params)), session)
        if result is None:
            return None
        chunks, rtrace = result
        new_entries = pool.add(chunks, step=step, tool=tool_name,
                               sub_query=str(params.get("query", cq)))
        session.cache_chunks(chunks)
        if new_entries:
            if self.rerank_fn is not None:
                scores = await self.rerank_fn(cq, [e.content for e in new_entries])
                pool.set_scores({e.chunk_id: s for e, s in zip(new_entries, scores)})
            else:
                pool.set_scores({e.chunk_id: max((p["rrf_score"] for p in e.provenance),
                                                 default=0.0)
                                 for e in new_entries})
        pool.evict()
        tb.add_retrieval_step(step_type=tool_name, query=str(params.get("query", cq)),
                              tool=tool_name, chunks=len(chunks), trace=rtrace, cycle=step)
        return new_entries

    async def _grade(self, grader, cq, ledger, pool, new_entries) -> dict:
        if self._grade_override is not None:
            ledger.update(self._grade_override)
            return dict(self._grade_override)
        return await grader.grade(cq, ledger, pool, new_entries=new_entries)

    async def _decide(self, plan, cq, pool, ledger, tb, budget, tried, step) -> Decision | None:
        history = "\n".join(
            f"{i + 1}. {d['action']}({d['params']}) " for i, d in
            enumerate(tb._trace["agent_decisions"])) or "(none)"
        tool_status = "search_ppr: ready"
        prompt = _DECIDE_PROMPT.format(
            cards=self.registry.card_text(), archetype=plan.archetype, query=cq,
            pool_summary=pool.summary(), ledger=str(ledger.to_dict())[:1500],
            history=history, tool_status=tool_status, budget=budget.snapshot())
        for attempt in range(2):
            try:
                raw = await call_json_object(
                    lambda p, **kw: self.model_pool.call("planner", p, **kw),
                    prompt, max_tokens=256)
                d = normalize_decision(raw, self.registry, cq)
                tb.add_decision(thought=d.thought, action=d.action, params=d.params,
                                budget_snapshot=budget.snapshot(), fallback=False)
                return d
            except Exception as exc:
                prompt += f"\nPrevious output invalid: {exc}. Output ONLY the JSON object."
        failure = ledger.missing()[0]["text"] if ledger.missing() else "partial_evidence"
        action = self.recovery.choose(
            failure_type="partial_evidence", original_profile="semantic",
            original_query=cq, tried_profiles=set(), tried_signatures=set())
        if action is None:
            return None
        tool = "decompose_search" if action.action_type == "decompose" else \
            _FALLBACK_TOOL.get(action.profile, "search_hybrid")
        d = Decision(thought=f"fallback:{failure[:50]}", action=tool,
                     params=self.registry.get(tool).clamp({"query": cq}), fallback=True)
        tb.add_decision(thought=d.thought, action=d.action, params=d.params,
                        budget_snapshot=budget.snapshot(), fallback=True)
        return d

    async def _finish(self, cq, plan, pool, ledger, tb, budget, session, *,
                      generation_mode) -> AgentResult:
        answer = await self._cancellable(
            generate_answer(self.model_pool, cq, pool, ledger, mode=generation_mode,
                            max_context_tokens=self.max_context_tokens,
                            visual_intent=plan.visual_intent), session)
        if answer is None:
            return self._cancelled_result(ledger, tb, session, cq)
        chunks = [{"chunk_id": e.chunk_id, "content": e.content} for e in pool.top(20)]
        grounded, ungrounded = await verify_citations(self.model_pool, cq, answer, chunks)
        tb.add_hallucination_event({"grounded": grounded,
                                    "ungrounded_claims": ungrounded}, cycle=0)
        if not grounded and generation_mode == "cot_reflect" and not budget.exhausted():
            repair_q = " ".join(ungrounded)[:300]
            await self._execute_search("search_dense", {"query": repair_q, "top_k": 10},
                                       pool, session, cq, tb, step=MAX_STEPS, budget=budget)
            answer = await generate_answer(self.model_pool, cq, pool, ledger,
                                           mode=generation_mode,
                                           max_context_tokens=self.max_context_tokens)
            grounded, ungrounded = await verify_citations(self.model_pool, cq, answer, chunks)
            tb.add_hallucination_event({"grounded": grounded,
                                        "ungrounded_claims": ungrounded}, cycle=1)
        session.add_turn(cq, answer if grounded else "")
        unver = ledger.unverifiable()
        if unver and grounded:
            answer += "\n\n（以下细节在语料中无法证实：" + "；".join(f["text"] for f in unver) + "）"
        return AgentResult(answer=answer if grounded else answer,
                           grounded=grounded, refusal=None,
                           ledger=ledger.to_dict(),
                           trace=tb.build(terminal_reason="grounded" if grounded else "ungrounded",
                                          grounded=grounded))

    async def _exhausted(self, cq, plan, pool, ledger, tb, budget, session,
                         reason) -> AgentResult:
        if ledger.coverage >= 0.5 and pool.entries:
            answer = await generate_answer(
                self.model_pool, cq, pool, ledger, mode="direct",
                max_context_tokens=self.max_context_tokens)
            answer += "\n\n（基于不完整证据作答，未覆盖：" + \
                "；".join(f["text"] for f in ledger.missing()) + "）"
            session.add_turn(cq, answer)
            return AgentResult(answer=answer, grounded=False, refusal=None,
                               ledger=ledger.to_dict(),
                               trace=tb.build(terminal_reason=reason, grounded=False))
        refusal = {"reason": reason,
                   "missing_facts": [f["text"] for f in ledger.missing()],
                   "unverifiable": [f["text"] for f in ledger.unverifiable()],
                   "attempts": [d["action"] for d in tb._trace["agent_decisions"]]}
        session.add_turn(cq, "")
        return AgentResult(answer=None, grounded=False, refusal=refusal,
                           ledger=ledger.to_dict(),
                           trace=tb.build(terminal_reason=reason, grounded=False))

    def _cancelled_result(self, ledger, tb, session, query) -> AgentResult:
        session.add_turn(query, "", cancelled=True)
        session.cancel_event.clear()
        return AgentResult(answer=None, grounded=False,
                           refusal={"reason": "cancelled"},
                           ledger=ledger.to_dict(),
                           trace=tb.build(terminal_reason="cancelled", grounded=False),
                           cancelled=True)
