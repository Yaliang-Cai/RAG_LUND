import asyncio
import json
import pytest
from raganything.agent.budget import Budget
from raganything.agent.loop import AgentLoop, AgentResult
from raganything.agent.models import ModelPool
from raganything.agent.session import SessionMemory
from raganything.agent.tools import build_default_registry


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
