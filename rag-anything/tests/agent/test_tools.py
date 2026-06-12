import pytest
from raganything.agent.tools import (
    ToolRegistry, ToolSpec, ParamSpec, build_default_registry, register_agent_profiles,
)
from raganything.retrieval.profiles import PROFILE_REGISTRY


def test_agent_profiles_registered_without_rerank():
    register_agent_profiles()
    for name in ["agent_sparse", "agent_dense", "agent_hybrid", "agent_graph", "agent_ppr"]:
        assert name in PROFILE_REGISTRY
        assert PROFILE_REGISTRY[name].enable_rerank is False  # rerank 移到池准入口 §5.1


def test_default_registry_costs_match_spec():
    reg = build_default_registry()
    costs = {n: reg.get(n).cost for n in reg.names()}
    assert costs["search_sparse"] == 1 and costs["search_hybrid"] == 2
    assert costs["search_ppr"] == 4 and costs["decompose_search"] == 8
    assert costs["answer"] == 0


def test_param_clamp_via_spec():
    spec = ToolSpec(name="t", cost=1, description="", profile="",
                    params={"top_k": ParamSpec(default=10, min=1, max=50)})
    assert spec.clamp({"top_k": 999}) == {"top_k": 50}
    assert spec.clamp({"top_k": "abc", "bogus": 1}) == {"top_k": 10}  # 非法回默认、未知丢弃


def test_expand_allowed_per_tool():
    reg = build_default_registry()
    assert "hyde" in reg.get("search_dense").allowed_expand   # hyde 仅 dense §7.4
    assert "hyde" not in reg.get("search_sparse").allowed_expand
    assert "mqe" in reg.get("search_hybrid").allowed_expand


def test_card_text_static():
    reg = build_default_registry()
    text = reg.card_text()
    assert "search_ppr" in text and "4" in text  # 成本进卡片 §4.2
