from raganything.agent.tools import (
    ToolRegistry, ToolSpec, ParamSpec, build_default_registry, register_agent_profiles,
)
from raganything.retrieval.profiles import PROFILE_REGISTRY


def test_agent_profiles_registered_without_rerank():
    register_agent_profiles()
    # Minimal toolset: two retrieval profiles, both rerank-free at the router
    # (search reranks at the pool; multihop/PPR is never reranked).
    for name in ["agent_search", "agent_multihop"]:
        assert name in PROFILE_REGISTRY
        assert PROFILE_REGISTRY[name].enable_rerank is False


def test_agent_search_fuses_local_global_and_vector():
    register_agent_profiles()
    paths = set(PROFILE_REGISTRY["agent_search"].paths)
    assert paths == {"local_kg", "global_kg", "qdrant_chunks_hybrid"}
    assert PROFILE_REGISTRY["agent_multihop"].paths == ["ppr"]


def test_default_registry_is_minimal_three_tools():
    reg = build_default_registry()
    assert set(reg.names()) == {"search", "search_multihop", "answer"}
    costs = {n: reg.get(n).cost for n in reg.names()}
    assert costs["search"] == 2 and costs["search_multihop"] == 4 and costs["answer"] == 0


def test_multihop_tool_skips_pool_rerank():
    reg = build_default_registry()
    assert reg.get("search").rerank is True       # pool cross-encoder reranks
    assert reg.get("search_multihop").rerank is False  # PPR scores trusted as-is


def test_param_clamp_via_spec():
    spec = ToolSpec(name="t", cost=1, description="", profile="",
                    params={"top_k": ParamSpec(default=10, min=1, max=50)})
    assert spec.clamp({"top_k": 999}) == {"top_k": 50}
    assert spec.clamp({"top_k": "abc", "bogus": 1}) == {"top_k": 10}  # invalid→default, unknown dropped


def test_card_text_lists_tools_with_costs():
    reg = build_default_registry()
    text = reg.card_text()
    assert "search" in text and "search_multihop" in text
    assert "cost 2" in text and "cost 4" in text
