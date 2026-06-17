# raganything/agent/loop.py
"""Agent 主循环（spec §4/§6.5/§8.3/§9.3）。"""
from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import asyncio

from raganything.retrieval.json_utils import call_json_object
from raganything.retrieval.recovery_policy import RecoveryPolicy

from raganything.agent.budget import Budget
from raganything.agent.citations import split_claims, verify_answer
from raganything.agent.decision import Decision, decision_signature, normalize_decision
from raganything.agent.evidence import EvidencePool, FactLedger
from raganything.agent.expand import hyde_query, mqe_queries
from raganything.agent.generate import generate_answer
from raganything.agent.grading import LedgerGrader, should_final_review
from raganything.agent.models import ModelPool
from raganything.agent.planner import PlanResult, make_plan
from raganything.agent.session import SessionMemory
from raganything.agent.tools import ToolRegistry
from raganything.agent.trace import TraceBuilder

logger = logging.getLogger(__name__)

from contextlib import contextmanager

try:  # OTEL is pulled in by Phoenix; degrade to no-op span if absent.
    from opentelemetry import trace as _otel_trace
    _TRACER = _otel_trace.get_tracer("raganything.agent")
except Exception:  # pragma: no cover - observability optional
    _TRACER = None


@contextmanager
def _span(name: str, kind: str = "CHAIN", **attrs: Any):
    """A child OTEL span tagged with the OpenInference span *kind* so Phoenix
    renders a typed tree (AGENT → CHAIN/RETRIEVER/LLM …) instead of one opaque
    node. Spans nest under whatever span is current (the per-turn agent.turn),
    so child LLM calls auto-instrumented by Phoenix land under the right phase.
    No-op when tracing is unavailable. The yielded span (or None) lets callers
    attach an ``output.value`` after the awaited work completes."""
    if _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as sp:
        try:
            sp.set_attribute("openinference.span.kind", kind)
            for k, v in attrs.items():
                if v is None:
                    continue
                sp.set_attribute(k, v if isinstance(v, (int, float, bool)) else str(v)[:1000])
        except Exception:  # pragma: no cover - attributes best-effort
            pass
        yield sp


def _set_out(sp: Any, value: Any) -> None:
    """Best-effort ``output.value`` on a span returned by :func:`_span`."""
    if sp is None:
        return
    try:
        sp.set_attribute("output.value", str(value)[:1000])
    except Exception:  # pragma: no cover
        pass


# Per-run UI event sink (the SSE endpoint sets it). A ContextVar keeps it isolated
# across concurrent turns that share one AgentLoop instance, unlike an attribute.
_event_sink: contextvars.ContextVar = contextvars.ContextVar("agent_event_sink", default=None)


async def _emit(phase: str, detail: str = "", items: list | None = None) -> None:
    """Push a live ``phase`` event to the current run's sink (no-op when absent),
    so the UI can render the agent's thinking chain in real time. ``items`` carries
    short sub-lines (e.g. the MQE/HyDE variants, the PPR entity-anchor seed)."""
    sink = _event_sink.get()
    if sink is None:
        return
    ev: dict = {"type": "phase", "phase": phase, "detail": str(detail)[:300]}
    if items:
        ev["items"] = [str(x)[:200] for x in items][:6]
    try:
        await sink(ev)
    except Exception:  # pragma: no cover - UI streaming is best-effort
        pass

MAX_STEPS = 5  # hard cap on agent steps (spec §4.3; lowered for latency)
# RecoveryPolicy profile → tool name (deterministic fallback brain, spec §4.4).
# Minimal toolset: anything non-multihop falls back to `search`; bridge/decompose
# profiles fall back to `search_multihop`.
_FALLBACK_TOOL = {"precise": "search", "semantic": "search",
                  "multihop": "search_multihop", "full_v2": "search_multihop",
                  "global": "search", "local": "search"}

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
    chunks: list[dict] = field(default_factory=list)  # references for the UI
    citations: list[dict] = field(default_factory=list)  # verbatim supporting spans
    metrics: dict = field(default_factory=dict)  # faithfulness / coverage panel


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
                  budget: Budget | None = None,
                  event_cb: Callable[[dict], Awaitable[None]] | None = None,
                  **qp_kwargs: Any) -> AgentResult:
        """Run one turn under a single OTEL span so every child LLM/retrieval call
        nests into one chain trace per Q&A in Phoenix (spec §13 / observability).
        ``event_cb`` (when given) receives live phase events for UI streaming."""
        token = _event_sink.set(event_cb)
        try:
            return await self._run_traced(query, session, budget=budget, **qp_kwargs)
        finally:
            _event_sink.reset(token)

    async def _run_traced(self, query: str, session: SessionMemory, *,
                          budget: Budget | None = None, **qp_kwargs: Any) -> AgentResult:
        if _TRACER is None:
            return await self._run_inner(query, session, budget=budget, **qp_kwargs)
        with _TRACER.start_as_current_span("agent.turn") as span:
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("input.value", query[:500])
            span.set_attribute("agent.query", query[:200])
            span.set_attribute("agent.workspace", session.workspace_id)
            span.set_attribute("agent.session", session.session_id)
            result = await self._run_inner(query, session, budget=budget, **qp_kwargs)
            try:
                span.set_attribute("agent.grounded", bool(result.grounded))
                span.set_attribute("agent.cancelled", bool(result.cancelled))
                span.set_attribute(
                    "agent.terminal_reason",
                    str(result.trace.get("terminal_reason", "")))
                span.set_attribute(
                    "agent.steps", len(result.trace.get("agent_decisions", [])))
                span.set_attribute("output.value", str(result.answer or
                                   (result.refusal or {}).get("reason", ""))[:1000])
            except Exception:  # pragma: no cover - attribute best-effort
                pass
            return result

    async def _run_inner(self, query: str, session: SessionMemory, *,
                         budget: Budget | None = None, **qp_kwargs: Any) -> AgentResult:
        with _span("plan", "CHAIN", **{"input.value": query}) as sp:
            plan = await make_plan(self.model_pool, query, session)
            _set_out(sp, f"archetype={plan.archetype} | {plan.standalone_query}")
        await _emit("plan", f"{plan.archetype} · {plan.standalone_query}")
        budget = budget or Budget.for_archetype(plan.archetype)
        tokens_at_start = self.model_pool.approx_tokens_used
        pool, ledger = EvidencePool(), FactLedger()
        grader = LedgerGrader(self.model_pool)
        tb = TraceBuilder(profile=f"agent:{plan.archetype}", query=query)
        # v2 parity: a first-turn question retrieves best on its own surface form.
        # The coreference-resolved rewrite is only needed for follow-ups; applying it
        # to a fresh standalone question tends to narrow or embellish it and poison
        # every downstream retrieval (the rewrite is the ONLY text we search with).
        cq = plan.standalone_query if session.recent_turns else query
        tb.add_rewrite(cq)
        tried: set[tuple] = set()
        dup_rates: list[float] = []
        ledger_steps = 0

        if plan.fast_path:
            fp_params = self._apply_expand_default(
                plan.preset["tool"], {"query": cq, "top_k": plan.preset["top_k"]}, plan)
            new = await self._execute_search(plan.preset["tool"], fp_params,
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
            budget.spent_tokens = self.model_pool.approx_tokens_used - tokens_at_start
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
                    # 守卫替换 _decide 已记录的 answer 决策，避免轨迹出现未执行的幽灵决策
                    if (tb._trace["agent_decisions"]
                            and tb._trace["agent_decisions"][-1]["action"] == "answer"):
                        tb._trace["agent_decisions"].pop()
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
            # PPR seeds best from clean entity anchors, not the context-laden
            # rewrite — swap the query for multihop only (within-repo #5 mitigation).
            # `search` instead gets the planner's expand suggestion (unless the
            # decider explicitly chose one); multihop is never expanded.
            if decision.action == "search_multihop":
                # Seed PPR from the canonical question (its semantics drive the graph
                # walk), augmented with entity anchors — not the decider's free-form
                # query, which can be verbose and dilute the keyword extraction.
                seed = _multihop_seed(plan, cq)
                search_params = {**decision.params, "query": seed}
                await _emit("seed", "PPR seed (question + entity anchors)", items=[seed])
            else:
                search_params = self._apply_expand_default(
                    decision.action, dict(decision.params), plan)
            new_entries = await self._execute_search(
                decision.action, search_params, pool, session, cq, tb,
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

    def _apply_expand_default(self, tool_name, params, plan) -> dict:
        """Resolve the effective ``expand`` strategy for a retrieval tool. An explicit
        non-"none" choice by the decider wins; otherwise the planner's
        ``suggested_expand`` applies. Tools that don't support expansion are untouched."""
        spec = self.registry.get(tool_name)
        if spec.allowed_expand == ("none",):
            return params
        chosen = str(params.get("expand", "none"))
        if chosen == "none" and plan.suggested_expand in spec.allowed_expand:
            chosen = plan.suggested_expand
        return {**params, "expand": chosen}

    async def _expanded_queries(self, base_q: str, expand: str) -> list[str]:
        """Base query plus any expansion probes (MQE paraphrases / HyDE passage),
        traced as an ``expand`` child span so Phoenix shows what was generated."""
        if expand not in ("mqe", "hyde"):
            return [base_q]
        with _span("expand", "CHAIN",
                   **{"input.value": base_q, "expand.strategy": expand}) as esp:
            queries = [base_q]
            if expand == "mqe":
                queries += await mqe_queries(self.model_pool, base_q)
            else:  # hyde
                hypo = await hyde_query(self.model_pool, base_q)
                if hypo:
                    queries.append(hypo)
            _set_out(esp, f"{len(queries)} queries")
            # Surface the actual variants so the demo shows the recall-broadening
            # chain (original → N paraphrases/hypothetical → fused), not just a count.
            await _emit("expand", f"{expand}: original + {len(queries) - 1} variants",
                        items=queries[1:])
            return queries

    async def _execute_search(self, tool_name, params, pool, session, cq, tb, *,
                              step, budget):
        spec = self.registry.get(tool_name)
        expand = str(params.get("expand", "none"))
        if expand not in spec.allowed_expand:
            expand = "none"
        # Expansion adds an LLM call + extra retrievals; charge a flat surcharge so
        # the budget guardrail still bounds the turn.
        budget.charge(points=spec.cost + (1.0 if expand != "none" else 0.0))
        base_q = str(params.get("query", cq))
        queries = await self._expanded_queries(base_q, expand)
        # Strip non-retrieval params before handing to the retrieve_fn.
        retrieve_base = {k: v for k, v in params.items() if k != "expand"}

        with _span("retrieve", "RETRIEVER",
                   **{"input.value": base_q, "tool.name": tool_name,
                      "retrieval.top_k": params.get("top_k"),
                      "retrieval.queries": len(queries)}) as sp:
            results = await self._cancellable(
                asyncio.gather(*(self.retrieve_fn(tool_name, {**retrieve_base, "query": q})
                                 for q in queries)), session)
            if results is None:
                return None
            new_entries: list = []
            total_chunks = 0
            last_trace: dict = {}
            for (chunks, rtrace), q in zip(results, queries):
                total_chunks += len(chunks)
                last_trace = rtrace
                new_entries.extend(pool.add(chunks, step=step, tool=tool_name, sub_query=q))
                session.cache_chunks(chunks)
            _set_out(sp, f"{total_chunks} chunks via "
                         f"{last_trace.get('profile', tool_name)} ({len(queries)} q)")
        if new_entries:
            # PPR (rerank=False) already ranks by multi-hop relevance — never
            # cross-encoder-rerank it (user requirement). Other tools rerank at
            # the pool admission point. Expansion results are reranked together
            # against the canonical query, fusing the variants by relevance.
            if self.rerank_fn is not None and spec.rerank:
                scores = await self.rerank_fn(cq, [e.content for e in new_entries])
                pool.set_scores({e.chunk_id: s for e, s in zip(new_entries, scores)})
            else:
                pool.set_scores({e.chunk_id: max((p["rrf_score"] for p in e.provenance),
                                                 default=0.0)
                                 for e in new_entries})
        pool.evict()
        tb.add_retrieval_step(step_type=tool_name, query=base_q,
                              tool=tool_name, chunks=total_chunks, trace=last_trace, cycle=step)
        await _emit("retrieve", f"{tool_name} → {total_chunks} chunks"
                                + (f" ({len(queries)} queries)" if len(queries) > 1 else ""))
        return new_entries

    async def _grade(self, grader, cq, ledger, pool, new_entries) -> dict:
        if self._grade_override is not None:
            ledger.update(self._grade_override)
            return dict(self._grade_override)
        with _span("grade", "CHAIN", **{"input.value": cq}) as sp:
            grade = await grader.grade(cq, ledger, pool, new_entries=new_entries)
            _set_out(sp, f"sufficient={grade.get('sufficient')} "
                         f"coverage={grade.get('coverage')}")
        found = sum(1 for f in ledger.facts.values() if f["status"] == "found")
        await _emit("grade", f"facts {found}/{len(ledger.facts)} · "
                             f"sufficient={bool(grade.get('sufficient'))}")
        return grade

    async def _decide(self, plan, cq, pool, ledger, tb, budget, tried, step) -> Decision | None:
        history = "\n".join(
            f"{i + 1}. {d['action']}({d['params']}) " for i, d in
            enumerate(tb._trace["agent_decisions"])) or "(none)"
        tool_status = "search, search_multihop: ready"
        prompt = _DECIDE_PROMPT.format(
            cards=self.registry.card_text(), archetype=plan.archetype, query=cq,
            pool_summary=pool.summary(), ledger=str(ledger.to_dict())[:1500],
            history=history, tool_status=tool_status, budget=budget.snapshot())
        with _span("decide", "CHAIN",
                   **{"input.value": cq, "agent.archetype": plan.archetype}) as sp:
            for attempt in range(2):
                try:
                    raw = await call_json_object(
                        lambda p, **kw: self.model_pool.call("planner", p, **kw),
                        prompt, max_tokens=256)
                    d = normalize_decision(raw, self.registry, cq)
                    tb.add_decision(thought=d.thought, action=d.action, params=d.params,
                                    budget_snapshot=budget.snapshot(), fallback=False)
                    _set_out(sp, f"{d.action}({d.params}) :: {d.thought}")
                    await _emit("decide", f"{d.action} — {d.thought}")
                    return d
                except Exception as exc:
                    prompt += f"\nPrevious output invalid: {exc}. Output ONLY the JSON object."
        failure = ledger.missing()[0]["text"] if ledger.missing() else "partial_evidence"
        # 把已尝试动作签名喂给 RecoveryPolicy，使其跨步逐级升级而非每次返回首选
        tried_profiles = {self.registry.get(a).profile for a in
                          (d["action"] for d in tb._trace["agent_decisions"])
                          if a in self.registry.names()}
        action = self.recovery.choose(
            failure_type="partial_evidence", original_profile="semantic",
            original_query=cq, tried_profiles=tried_profiles, tried_signatures=set())
        if action is None:
            return None
        # Map the recovery brain's profile onto the minimal toolset. Decomposition
        # is inherently multi-hop; everything else falls back to plain search.
        tool = "search_multihop" if action.action_type == "decompose" else \
            _FALLBACK_TOOL.get(action.profile, "search")
        d = Decision(thought=f"fallback:{failure[:50]}", action=tool,
                     params=self.registry.get(tool).clamp({"query": cq}), fallback=True)
        tb.add_decision(thought=d.thought, action=d.action, params=d.params,
                        budget_snapshot=budget.snapshot(), fallback=True)
        return d

    async def _finish(self, cq, plan, pool, ledger, tb, budget, session, *,
                      generation_mode) -> AgentResult:
        await _emit("generate", "writing answer")
        with _span("generate", "LLM",
                   **{"input.value": cq, "llm.generation_mode": generation_mode}) as sp:
            answer = await self._cancellable(
                generate_answer(self.model_pool, cq, pool, ledger, mode=generation_mode,
                                max_context_tokens=self.max_context_tokens,
                                visual_intent=plan.visual_intent,
                                vision_fn=self.vision_fn), session)
            _set_out(sp, answer)
        if answer is None:
            return self._cancelled_result(ledger, tb, session, cq)
        chunks = [{"chunk_id": e.chunk_id, "content": e.content} for e in pool.top(20)]
        with _span("verify_citations", "CHAIN", **{"input.value": cq}) as sp:
            vres = await verify_answer(self.model_pool, cq, answer, chunks)
            _set_out(sp, f"grounded={vres.grounded} ungrounded={len(vres.ungrounded)}")
        grounded, ungrounded, citations = vres.grounded, vres.ungrounded, vres.citations
        tb.add_hallucination_event({"grounded": grounded,
                                    "ungrounded_claims": ungrounded}, cycle=0)
        if not grounded and generation_mode == "cot_reflect" and not budget.exhausted():
            repair_q = " ".join(ungrounded)[:300]
            await self._execute_search("search", {"query": repair_q, "top_k": 12},
                                       pool, session, cq, tb, step=MAX_STEPS, budget=budget)
            answer = await generate_answer(self.model_pool, cq, pool, ledger,
                                           mode=generation_mode,
                                           max_context_tokens=self.max_context_tokens,
                                           visual_intent=plan.visual_intent,
                                           vision_fn=self.vision_fn)
            chunks = [{"chunk_id": e.chunk_id, "content": e.content} for e in pool.top(20)]
            vres = await verify_answer(self.model_pool, cq, answer, chunks)
            grounded, ungrounded, citations = vres.grounded, vres.ungrounded, vres.citations
            tb.add_hallucination_event({"grounded": grounded,
                                        "ungrounded_claims": ungrounded}, cycle=1)

        # Grounding is a quality signal, not a gate: one unsupported sentence must
        # not bury an otherwise good answer. Treat the answer as deliverable unless
        # a *majority* of its claims are unsupported (then a brief note is added).
        total_claims = max(1, len(split_claims(answer))) if answer else 1
        unsupported_ratio = len(ungrounded) / total_claims
        deliverable = bool(answer) and unsupported_ratio <= 0.5
        metrics = _answer_metrics(total_claims, ungrounded, citations, ledger, grounded)
        refs = _result_chunks(pool)
        session.add_turn(cq, answer if deliverable else "")
        unver = ledger.unverifiable()
        # Carry what's still unresolved so a "continue / give me the full answer"
        # follow-up can be rewritten against concrete gaps, not the bare instruction.
        session.open_gaps = ([f["text"] for f in unver]
                             + (list(ungrounded) if not deliverable else []))

        if deliverable:
            notes = []
            if unver:
                notes.append("Some details could not be verified against the sources: "
                             + "; ".join(f["text"] for f in unver))
            if ungrounded and unsupported_ratio >= 0.34:
                notes.append("Some statements may need verification against the cited sources.")
            if notes:
                answer += "\n\n_" + " ".join(notes) + "_"
            return AgentResult(answer=answer, grounded=grounded, refusal=None,
                               ledger=ledger.to_dict(), chunks=refs,
                               citations=citations, metrics=metrics,
                               trace=tb.build(
                                   terminal_reason="grounded" if grounded else "partial",
                                   grounded=grounded))
        # Mostly-unsupported: return what we have, flagged, with structured metadata.
        partial = (answer + "\n\n_Note: the retrieved sources do not fully support this "
                   "answer; please verify._") if answer else None
        return AgentResult(answer=partial, grounded=False,
                           refusal={"reason": "ungrounded", "ungrounded_claims": ungrounded},
                           ledger=ledger.to_dict(), chunks=refs,
                           citations=citations, metrics=metrics,
                           trace=tb.build(terminal_reason="ungrounded", grounded=False))

    async def _exhausted(self, cq, plan, pool, ledger, tb, budget, session,
                         reason) -> AgentResult:
        refs = _result_chunks(pool)
        # Unresolved facts carry to the next turn so a follow-up can target them.
        session.open_gaps = ([f["text"] for f in ledger.missing()]
                             + [f["text"] for f in ledger.unverifiable()])
        if ledger.coverage >= 0.5 and pool.entries:
            answer = await generate_answer(
                self.model_pool, cq, pool, ledger, mode="direct",
                max_context_tokens=self.max_context_tokens,
                visual_intent=plan.visual_intent, vision_fn=self.vision_fn)
            missing = ledger.missing()
            if missing:
                answer += ("\n\n_Answered from partial evidence; not covered: "
                           + "; ".join(f["text"] for f in missing) + "_")
            session.add_turn(cq, answer)
            return AgentResult(answer=answer, grounded=False, refusal=None,
                               ledger=ledger.to_dict(), chunks=refs,
                               metrics={"coverage": ledger.to_dict().get("coverage", 0.0),
                                        "grounded": False},
                               trace=tb.build(terminal_reason=reason, grounded=False))
        refusal = {"reason": reason,
                   "missing_facts": [f["text"] for f in ledger.missing()],
                   "unverifiable": [f["text"] for f in ledger.unverifiable()],
                   "attempts": [d["action"] for d in tb._trace["agent_decisions"]]}
        session.add_turn(cq, "")
        return AgentResult(answer=None, grounded=False, refusal=refusal,
                           ledger=ledger.to_dict(), chunks=refs,
                           trace=tb.build(terminal_reason=reason, grounded=False))

    def _cancelled_result(self, ledger, tb, session, query) -> AgentResult:
        session.add_turn(query, "", cancelled=True)
        session.cancel_event.clear()
        return AgentResult(answer=None, grounded=False,
                           refusal={"reason": "cancelled"},
                           ledger=ledger.to_dict(),
                           trace=tb.build(terminal_reason="cancelled", grounded=False),
                           cancelled=True)


def _answer_metrics(total_claims: int, ungrounded: list, citations: list,
                    ledger: Any, grounded: bool) -> dict:
    """Answer-quality numbers for the UI confidence panel. Faithfulness = share of
    sentences with a verified verbatim source; coverage = found/needed facts."""
    total = max(1, total_claims)
    supported = max(0, total - len(ungrounded))
    return {
        "faithfulness": round(supported / total, 3),
        "claims_total": total,
        "claims_supported": supported,
        "citations": len(citations),
        "coverage": ledger.to_dict().get("coverage", 0.0),
        "grounded": bool(grounded),
    }


def _multihop_seed(plan: PlanResult, query: str) -> str:
    """Seed PPR with the full question so its semantics drive the graph walk,
    augmented with the planner's entity anchors (exact_terms) to strengthen the
    pivot nodes. An earlier revision seeded from entity anchors ALONE, which dropped
    the question's meaning and hurt multi-hop recall — LightRAG's low/high keyword
    extraction already distills the pivots from the full question text, so the right
    move is to keep the question and merely reinforce its anchors."""
    terms = [t.strip() for t in plan.exact_terms if t and t.strip()]
    base = (query or "").strip()
    if terms:
        return f"{base} {' '.join(terms)}".strip()
    return base


def _result_chunks(pool: EvidencePool, n: int = 12) -> list[dict]:
    """Top pool entries as UI reference records. Carries page/modal metadata so the
    frontend ReferenceList can jump to the right PDF page — including for multimodal
    chunks, which previously lost their page and fell back to text-highlight only."""
    return [
        {"id": e.chunk_id, "file_path": e.file_path,
         "content": e.content, "score": e.canonical_score,
         "page_idx": e.page_idx, "page_num": e.page_num,
         "modal_type": e.modal_type, "image_paths": e.image_paths}
        for e in pool.top(n)
    ]
