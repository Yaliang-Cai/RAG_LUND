from raganything.agent.trace import TraceBuilder


def test_v2_compatible_top_level_keys():
    tb = TraceBuilder(profile="agent", query="q")
    tb.add_retrieval_step(step_type="initial", query="q", tool="search_dense",
                          chunks=3, trace={"paths_activated": ["naive"]}, cycle=0)
    tb.add_grader_event({"sufficient": False, "coverage": 0.3}, cycle=0)
    tb.add_decision(thought="先便宜的", action="search_dense", params={"top_k": 10},
                    budget_snapshot={"remaining_points": 5}, fallback=False)
    tb.add_hallucination_event({"grounded": True}, cycle=0)
    tb.add_reclassify("factoid", "multihop", cycle=1)
    out = tb.build(terminal_reason="grounded", grounded=True)
    # v2 同构键（agent_graph_v2.py trace 契约）
    for key in ("retrieval_steps", "grader_events", "hallucination_events",
                "rewrite_history", "profile", "terminal_reason", "grounded"):
        assert key in out
    # v3 新增键
    assert out["agent_decisions"][0]["action"] == "search_dense"
    assert out["reclassify_events"][0]["to"] == "multihop"
    assert out["used_fallback"] is False


def test_used_fallback_flag_for_ab_stratification():
    tb = TraceBuilder(profile="agent", query="q")
    tb.add_decision(thought="", action="search_hybrid", params={},
                    budget_snapshot={}, fallback=True)
    assert tb.build(terminal_reason="insufficient", grounded=False)["used_fallback"] is True
