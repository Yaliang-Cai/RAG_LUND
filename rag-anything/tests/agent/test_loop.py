import asyncio
import json
import pytest
from raganything.agent.budget import Budget
from raganything.agent.loop import AgentLoop, AgentResult, _multihop_seed
from raganything.agent.planner import PlanResult
from raganything.agent.models import ModelPool
from raganything.agent.session import SessionMemory
from raganything.agent.tools import build_default_registry


def test_multihop_seed_prefers_exact_terms():
    plan = PlanResult(standalone_query="given the earlier company, what is its 5G revenue",
                      archetype="multihop", confidence=0.9,
                      exact_terms=["Huawei", "Ericsson"])
    # Clean entity anchors replace the context-laden rewrite for PPR seeding.
    assert _multihop_seed(plan, plan.standalone_query) == "Huawei Ericsson"


def test_multihop_seed_falls_back_without_terms():
    plan = PlanResult(standalone_query="how do A and B compare", archetype="multihop",
                      confidence=0.9, exact_terms=[])
    assert _multihop_seed(plan, plan.standalone_query) == "how do A and B compare"


def test_apply_expand_default_planner_suggestion_then_decider_override():
    loop = _loop([], [])
    plan = PlanResult(standalone_query="q", archetype="factoid", confidence=0.9,
                      suggested_expand="mqe")
    # Planner suggestion fills in when the decider didn't choose.
    assert loop._apply_expand_default("search", {"query": "q"}, plan)["expand"] == "mqe"
    # An explicit decider choice wins over the planner suggestion.
    assert loop._apply_expand_default(
        "search", {"query": "q", "expand": "hyde"}, plan)["expand"] == "hyde"
    # search_multihop never expands (allowed_expand == ("none",)).
    assert "expand" not in loop._apply_expand_default("search_multihop", {"query": "q"}, plan)


def make_llm(script):
    """script: list of(matcher_substring, reply_json_or_text)，按序消费匹配项。"""
    async def llm(prompt, **kw):
        for i, (needle, reply) in enumerate(script):
            if needle in prompt:
                script.pop(i)
                return reply if isinstance(reply, str) else json.dumps(reply)
        return json.dumps({"thought": "stop", "action": "answer",
                           "params": {"generation_mode": "direct"}})
    return llm


def make_retrieve(chunks):
    calls = []
    async def retrieve(tool_name, params):
        calls.append((tool_name, dict(params)))
        return list(chunks), {"paths_activated": [tool_name], "chunks_per_path": {tool_name: len(chunks)}}
    retrieve.calls = calls
    return retrieve


PLAN = {"standalone_query": "规范查询", "archetype": "factoid", "confidence": 0.9,
        "exact_terms": [], "suggested_expand": "none", "visual_intent": False,
        "entities_referenced": []}
GRADE_OK = {"sufficient": True, "facts": [
    {"id": "f1", "text": "事实", "status": "found", "chunks": ["c1"]}]}
VERIFY_OK = {"claims": [{"id": 0, "quote": "证据内容证据内容", "supported": True}]}


def _loop(script, chunks):
    llm = make_llm(script)
    return AgentLoop(
        model_pool=ModelPool(main_func=llm),
        registry=build_default_registry(),
        retrieve_fn=make_retrieve(chunks),
        rerank_fn=None,
        vision_fn=None,
    )


@pytest.mark.asyncio
async def test_fast_path_zero_decision_calls():
    chunks = [{"chunk_id": "c1", "content": "证据内容证据内容"}]
    script = [("prepare a user question", PLAN),
              ("fact ledger", GRADE_OK),
              ("Answer based ONLY", "答案：证据内容证据内容。"),
              ("verify answer claims", VERIFY_OK)]
    result = await _loop(script, chunks).run(
        "问题", SessionMemory(session_id="s", workspace_id="w"))
    assert isinstance(result, AgentResult)
    assert result.grounded is True and result.answer
    assert result.trace["agent_decisions"] == []  # 快速通道零决策 §4.5


@pytest.mark.asyncio
async def test_budget_exhaustion_returns_structured_refusal():
    chunks = [{"chunk_id": "c1", "content": "无关内容"}]
    grade_bad = {"sufficient": False, "facts": [
        {"id": "f1", "text": "找不到的事实", "status": "missing", "chunks": []}]}
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9})]
    loop = _loop(script, chunks)
    loop._grade_override = grade_bad  # 测试钩子：grader 恒不充分
    result = await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"),
                            budget=Budget(points=2, max_seconds=None))
    assert result.answer is None and result.refusal  # 结构化拒答 §8.3
    assert "找不到的事实" in json.dumps(result.refusal, ensure_ascii=False)


@pytest.mark.asyncio
async def test_repeat_action_rejected_and_noted():
    chunks = [{"chunk_id": "c1", "content": "x"}]
    decision = {"thought": "重复", "action": "search_dense",
                "params": {"query": "同查询", "top_k": 10}}
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9}),
              ("decide the next action", decision),
              ("decide the next action", decision)]  # 第二次重复
    loop = _loop(script, chunks)
    loop._grade_override = {"sufficient": False, "facts": []}
    result = await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"),
                            budget=Budget(points=3, max_seconds=None))
    assert loop.retrieve_fn.calls.count(("search_dense", {"query": "同查询", "top_k": 10})) <= 1


@pytest.mark.asyncio
async def test_cancellation_mid_tool_call():
    session = SessionMemory(session_id="s", workspace_id="w")
    async def slow_retrieve(tool_name, params):
        await asyncio.sleep(10)
        return [], {}
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9}),
              ("decide the next action",
               {"thought": "", "action": "search_dense", "params": {"query": "q"}})]
    loop = _loop(script, [])
    loop.retrieve_fn = slow_retrieve
    task = asyncio.create_task(loop.run("问题", session, budget=Budget(points=5, max_seconds=None)))
    await asyncio.sleep(0.1)
    session.cancel_event.set()
    result = await asyncio.wait_for(task, timeout=2)  # 秒级中断，不等 10s
    assert result.cancelled is True


@pytest.mark.asyncio
async def test_empty_pool_answer_is_guarded_into_retrieval():
    # LLM 立刻想 answer，但池为空且预算充足 → 守卫降级为一次 preset 检索（fallback=True）
    chunks = [{"chunk_id": "c1", "content": "实际检索到的内容"}]
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9}),
              ("decide the next action",
               {"thought": "想直接答", "action": "answer",
                "params": {"generation_mode": "direct"}}),
              ("fact ledger", GRADE_OK),
              ("Answer based ONLY", "答案：实际检索到的内容。"),
              ("verify answer claims",
               {"claims": [{"id": 0, "quote": "实际检索到的内容", "supported": True}]})]
    loop = _loop(script, chunks)
    result = await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"),
                            budget=Budget(points=8, max_seconds=None))
    decisions = result.trace["agent_decisions"]
    assert decisions and decisions[0]["fallback"] is True
    assert decisions[0]["action"] != "answer"  # 被替换成检索工具
    assert len(loop.retrieve_fn.calls) >= 1


@pytest.mark.asyncio
async def test_token_guardrail_charges_per_run():
    from raganything.agent.models import ModelPool
    chunks = [{"chunk_id": "c1", "content": "x" * 4000}]
    # 长 prompt 让近似 token 迅速累计；max_tokens 很小 → 应触发 token 护栏而非点数
    grade_bad = {"sufficient": False, "facts": [{"id": "f1", "text": "缺", "status": "missing", "chunks": []}]}
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9})]
    loop = _loop(script, chunks)
    loop._grade_override = grade_bad
    result = await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"),
                            budget=Budget(points=100, max_tokens=50, max_seconds=None))
    assert result.trace["terminal_reason"] in ("tokens", "no_action", "max_steps")
    # 关键断言：ModelPool 计数确实累加了
    assert loop.model_pool.approx_tokens_used > 0


@pytest.mark.asyncio
async def test_search_expands_with_mqe_when_planner_suggests():
    chunks = [{"chunk_id": "c1", "content": "证据内容证据内容", "score": 1.0}]
    script = [("prepare a user question", {**PLAN, "suggested_expand": "mqe"}),
              ("alternative search queries", {"queries": ["变体一", "变体二"]}),
              ("fact ledger", GRADE_OK),
              ("Answer based ONLY", "答案：证据内容证据内容。"),
              ("verify answer claims", VERIFY_OK)]
    loop = _loop(script, chunks)
    await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"))
    # base query + 2 MQE variants = 3 retrievals into the shared pool
    assert len(loop.retrieve_fn.calls) == 3
    queried = {c[1]["query"] for c in loop.retrieve_fn.calls}
    assert "变体一" in queried and "变体二" in queried
    # the retrieve_fn never sees the non-retrieval 'expand' param
    assert all("expand" not in c[1] for c in loop.retrieve_fn.calls)


@pytest.mark.asyncio
async def test_open_gaps_carried_after_ungrounded_turn():
    chunks = [{"chunk_id": "c1", "content": "无关内容", "score": 1.0}]
    script = [("prepare a user question", PLAN),
              ("fact ledger", GRADE_OK),
              ("Answer based ONLY", "一个无法被证据支撑的断言。"),
              ("verify answer claims",
               {"claims": [{"id": 0, "quote": "完全不在证据里的引文", "supported": True}]})]
    session = SessionMemory(session_id="s", workspace_id="w")
    result = await _loop(script, chunks).run("问题", session)
    assert result.grounded is False
    assert session.open_gaps  # unresolved items carried to the next turn's planner


@pytest.mark.asyncio
async def test_multihop_never_expands_even_if_suggested():
    chunks = [{"chunk_id": "c1", "content": "证据内容证据内容", "score": 1.0}]
    plan = {**PLAN, "archetype": "multihop", "confidence": 0.9,
            "suggested_expand": "mqe", "exact_terms": ["Huawei"]}
    decision = {"thought": "hop", "action": "search_multihop",
                "params": {"query": "context-laden rewrite", "top_k": 20}}
    script = [("prepare a user question", plan),
              ("decide the next action", decision),
              ("fact ledger", GRADE_OK),
              ("Answer based ONLY", "答案：证据内容证据内容。"),
              ("verify answer claims", VERIFY_OK)]
    loop = _loop(script, chunks)
    await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"),
                   budget=Budget(points=8, max_seconds=None))
    # exactly one retrieval (the seed) — no MQE fan-out on the multihop path
    assert len(loop.retrieve_fn.calls) == 1
    assert loop.retrieve_fn.calls[0][0] == "search_multihop"
    # seeded from the clean entity anchor, not the context-laden rewrite
    assert loop.retrieve_fn.calls[0][1]["query"] == "Huawei"


@pytest.mark.asyncio
async def test_ungrounded_returns_partial_answer_and_refusal_meta():
    chunks = [{"chunk_id": "c1", "content": "无关内容"}]
    script = [("prepare a user question", PLAN),  # factoid 高置信 → 快速通道
              ("fact ledger", GRADE_OK),
              ("Answer based ONLY", "一个无法被证据支撑的断言。"),
              ("verify answer claims",
               {"claims": [{"id": 0, "quote": "完全不在证据里的引文", "supported": True}]})]
    result = await _loop(script, chunks).run(
        "问题", SessionMemory(session_id="s", workspace_id="w"))
    assert result.grounded is False
    # All claims unsupported (ratio 1.0 > 0.5) → partial answer + English verify note.
    assert result.answer is not None and "verify" in result.answer.lower()
    assert result.refusal and result.refusal["reason"] == "ungrounded"
    assert result.refusal["ungrounded_claims"]  # structured gap metadata
