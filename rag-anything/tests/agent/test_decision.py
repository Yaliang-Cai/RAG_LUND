# tests/agent/test_decision.py
from raganything.agent.decision import Decision, normalize_decision, decision_signature
from raganything.agent.tools import build_default_registry

REG = build_default_registry()


def test_unknown_action_difflib_matched():
    d = normalize_decision({"thought": "t", "action": "serch", "params": {}}, REG, "默认查询")
    assert d.action == "search"  # difflib fuzzy-matches the typo to the real tool
    assert d.params["query"] == "默认查询"  # missing query backfilled §4.3


def test_params_clamped_and_unknown_dropped():
    d = normalize_decision(
        {"thought": "t", "action": "search", "params": {"top_k": 9999, "evil": 1}}, REG, "q")
    assert d.params["top_k"] == 30 and "evil" not in d.params  # search top_k max is 30


def test_answer_overrides_stop_flag():
    d = normalize_decision({"thought": "t", "action": "answer", "stop": False,
                            "params": {"generation_mode": "cot_reflect"}}, REG, "q")
    assert d.stop is True  # stop 与 action 不一致以 action 为准 §4.1


def test_unmatchable_action_raises():
    import pytest
    with pytest.raises(ValueError):
        normalize_decision({"thought": "t", "action": "zzzzzz", "params": {}}, REG, "q")


def test_signature_normalizes_query():
    a = decision_signature(Decision("t", "search_dense", {"query": " 大 模型 ", "top_k": 10}))
    b = decision_signature(Decision("t", "search_dense", {"query": "大模型", "top_k": 10}))
    assert a == b  # 空白差异不绕过重复守卫
