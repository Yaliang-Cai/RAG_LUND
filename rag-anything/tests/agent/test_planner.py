import json
import pytest
from raganything.agent.planner import make_plan, ARCHETYPE_PRESETS, PlanResult
from raganything.agent.session import SessionMemory


class FakePool:
    def __init__(self, payload):
        self.payload, self.calls = payload, []
    async def call(self, role, prompt, **kw):
        self.calls.append((role, prompt))
        return json.dumps(self.payload)


PAYLOAD = {
    "standalone_query": "Transformer attention 与 CNN 的区别",
    "archetype": "comparison", "confidence": 0.9,
    "exact_terms": [], "suggested_expand": "none", "visual_intent": False,
    "entities_referenced": [{"name": "CNN", "note": "对比对象", "last_turn": 2}],
}


@pytest.mark.asyncio
async def test_merged_call_rewrites_and_classifies():
    s = SessionMemory(session_id="s", workspace_id="w")
    s.recent_turns.append({"q": "attention 是什么", "a": "是加权求和", "cancelled": False})
    pool = FakePool(PAYLOAD)
    plan = await make_plan(pool, "它和 CNN 有什么区别", s)
    assert plan.standalone_query.startswith("Transformer")
    assert pool.calls[0][0] == "rewriter"
    assert "attention 是什么" in pool.calls[0][1]  # 历史进改写 prompt §6.2
    assert any(e["name"] == "CNN" for e in s.active_entities)  # 实体顺手登记


@pytest.mark.asyncio
async def test_fast_path_only_high_confidence_factoid():
    s = SessionMemory(session_id="s", workspace_id="w")
    p1 = await make_plan(FakePool({**PAYLOAD, "archetype": "factoid", "confidence": 0.9}), "q1", s)
    assert p1.fast_path is True
    p2 = await make_plan(FakePool({**PAYLOAD, "archetype": "factoid", "confidence": 0.5}), "q2", s)
    assert p2.fast_path is False
    p3 = await make_plan(FakePool({**PAYLOAD, "archetype": "summary", "confidence": 0.99}), "q3", s)
    assert p3.fast_path is False  # 仅 factoid §4.5


@pytest.mark.asyncio
async def test_plan_cached_per_session():
    s = SessionMemory(session_id="s", workspace_id="w")
    pool = FakePool(PAYLOAD)
    await make_plan(pool, "同一个问题", s)
    await make_plan(pool, "同一个问题", s)
    assert len(pool.calls) == 1


@pytest.mark.asyncio
async def test_unknown_on_parse_failure():
    class Broken:
        async def call(self, role, prompt, **kw):
            return "not json at all" * 3
    plan = await make_plan(Broken(), "问题", SessionMemory(session_id="s", workspace_id="w"))
    assert plan.archetype == "unknown" and plan.standalone_query == "问题"


def test_presets_match_minimal_toolset():
    # Minimal toolset: search for everything except true multi-hop / comparison
    # (which use PPR via search_multihop).
    assert ARCHETYPE_PRESETS["factoid"]["tool"] == "search"
    assert ARCHETYPE_PRESETS["summary"]["tool"] == "search"
    assert ARCHETYPE_PRESETS["unknown"]["tool"] == "search"
    assert ARCHETYPE_PRESETS["multihop"]["tool"] == "search_multihop"
    assert ARCHETYPE_PRESETS["comparison"]["tool"] == "search_multihop"
    assert ARCHETYPE_PRESETS["multihop"]["generation_mode"] == "cot_reflect"


@pytest.mark.asyncio
async def test_factoid_uses_search():
    s = SessionMemory(session_id="s", workspace_id="w")
    plan = await make_plan(
        FakePool({**PAYLOAD, "archetype": "factoid", "confidence": 0.9, "exact_terms": []}),
        "什么是注意力机制", s)
    assert plan.preset["tool"] == "search"


@pytest.mark.asyncio
async def test_multihop_uses_search_multihop():
    s = SessionMemory(session_id="s", workspace_id="w")
    plan = await make_plan(
        FakePool({**PAYLOAD, "archetype": "multihop", "confidence": 0.9}),
        "A 和 B 通过谁联系起来", s)
    assert plan.preset["tool"] == "search_multihop"


@pytest.mark.asyncio
async def test_moderate_confidence_keeps_archetype_as_prior():
    s = SessionMemory(session_id="s", workspace_id="w")
    # 0.4 used to collapse to "unknown" (killing the preset matrix); now a moderately
    # confident archetype is kept as a prior — only the fast path is withheld.
    plan = await make_plan(
        FakePool({**PAYLOAD, "archetype": "multihop", "confidence": 0.4}), "某问题", s)
    assert plan.archetype == "multihop" and plan.fast_path is False


@pytest.mark.asyncio
async def test_very_low_confidence_still_downgraded_to_unknown():
    s = SessionMemory(session_id="s", workspace_id="w")
    plan = await make_plan(
        FakePool({**PAYLOAD, "archetype": "multihop", "confidence": 0.2}), "某问题", s)
    assert plan.archetype == "unknown"  # genuinely unsure → safe default


@pytest.mark.asyncio
async def test_continuation_query_gets_gaps_and_rule_in_prompt():
    s = SessionMemory(session_id="s", workspace_id="w")
    s.recent_turns.append({"q": "华为2023年5G营收", "a": "", "cancelled": False})
    s.open_gaps = ["华为2023年5G营收具体数字"]
    pool = FakePool({**PAYLOAD, "standalone_query": "华为2023年5G营收具体数字"})
    await make_plan(pool, "继续检索相关事实，给我综合答案", s)
    prompt = pool.calls[0][1]
    assert "华为2023年5G营收具体数字" in prompt   # unresolved gaps surfaced to planner
    assert "continue" in prompt.lower()          # continuation/refine rewrite rule present
