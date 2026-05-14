import json
from unittest.mock import AsyncMock, MagicMock

from raganything.retrieval.agent_graph_v2 import (
    AdaptiveAgentGraphV2,
    _merge_decomposed_chunks,
)
from raganything.retrieval.router_cache import RouterCache


def _lightrag() -> MagicMock:
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value="the answer")
    return lg


def _chunks(prefix: str, n: int = 2, score: float = 0.5) -> list[dict]:
    return [
        {"chunk_id": f"{prefix}{i}", "content": f"{prefix} content {i}", "rrf_score": score}
        for i in range(n)
    ]


def _prefix_count(chunks: list[dict], prefix: str) -> int:
    return sum(1 for chunk in chunks if str(chunk["chunk_id"]).startswith(prefix))


def _trace(profile: str, chunks: list[dict]) -> dict:
    path = "ppr" if profile in {"multihop", "full_v2"} else "qdrant_chunks_hybrid"
    return {
        "profile": profile,
        "paths_activated": [path],
        "paths_failed": [],
        "chunks_per_path": {path: len(chunks)},
        "chunks_after_rrf": len(chunks),
        "chunks_after_rerank": len(chunks),
        "chunks_after_threshold": len(chunks),
    }


def _classifier(profile: str = "semantic") -> MagicMock:
    m = MagicMock()
    m.classify = AsyncMock(return_value=(profile, {
        "confidence": 0.9,
        "reasoning": "test",
        "latency": 0.01,
        "candidate_profile": profile,
        "selected_profile": profile,
        "fallback_used": False,
        "fallback_reason": "",
    }))
    return m


def _checker(grounded: bool = True, claims: list[str] | None = None) -> MagicMock:
    m = MagicMock()
    m.verify = AsyncMock(return_value={
        "grounded": grounded,
        "ungrounded_claims": claims or [],
    })
    return m


async def test_v2_unanswerable_signal_does_not_stop_before_full_recovery():
    router = MagicMock()
    router.route = AsyncMock(side_effect=[
        (_chunks("initial"), _trace("semantic", _chunks("initial"))),
        (_chunks("full"), _trace("full_v2", _chunks("full"))),
    ])
    grader = MagicMock()
    grader.grade = AsyncMock(side_effect=[
        {
            "sufficient": False,
            "unanswerable": True,
            "failure_type": "unanswerable_candidate",
            "coverage_score": 0.1,
            "found_facts": [],
            "missing_facts": ["answer"],
            "reason": "not enough",
        },
        {
            "sufficient": True,
            "unanswerable": False,
            "failure_type": "partial_evidence",
            "coverage_score": 0.9,
            "found_facts": ["answer"],
            "missing_facts": [],
            "reason": "enough",
        },
    ])
    graph = AdaptiveAgentGraphV2(
        _lightrag(),
        _classifier=_classifier("semantic"),
        _grader=grader,
        _router=router,
        _checker=_checker(True),
        _cache=RouterCache(),
    )
    result = await graph.run("question", return_trace=True)
    assert result["grounded"] is True
    assert router.route.call_args_list[1].kwargs["profile_name"] == "full_v2"
    assert result["trace"]["terminal_reason"] == "grounded"


async def test_v2_missing_relation_escalates_to_multihop():
    router = MagicMock()
    router.route = AsyncMock(side_effect=[
        (_chunks("sem"), _trace("semantic", _chunks("sem"))),
        (_chunks("ppr"), _trace("multihop", _chunks("ppr"))),
    ])
    grader = MagicMock()
    grader.grade = AsyncMock(side_effect=[
        {
            "sufficient": False,
            "unanswerable": False,
            "failure_type": "missing_relation",
            "coverage_score": 0.2,
            "found_facts": ["A"],
            "missing_facts": ["A-B relation"],
            "reason": "missing bridge",
        },
        {
            "sufficient": True,
            "unanswerable": False,
            "failure_type": "partial_evidence",
            "coverage_score": 0.8,
            "found_facts": ["bridge"],
            "missing_facts": [],
            "reason": "ok",
        },
    ])
    graph = AdaptiveAgentGraphV2(
        _lightrag(),
        _classifier=_classifier("semantic"),
        _grader=grader,
        _router=router,
        _checker=_checker(True),
        _cache=RouterCache(),
    )
    result = await graph.run("How is A connected to B?", return_trace=True)
    assert router.route.call_args_list[1].kwargs["profile_name"] == "multihop"
    assert result["trace"]["best_step"]["profile"] == "multihop"


async def test_v2_best_chunks_are_returned_instead_of_last_chunks():
    router = MagicMock()
    router.route = AsyncMock(side_effect=[
        (_chunks("good", 2), _trace("semantic", _chunks("good", 2))),
        (_chunks("bad", 1), _trace("full_v2", _chunks("bad", 1))),
    ])
    grader = MagicMock()
    grader.grade = AsyncMock(side_effect=[
        {
            "sufficient": False,
            "unanswerable": False,
            "failure_type": "missing_entity",
            "coverage_score": 0.7,
            "found_facts": ["useful"],
            "missing_facts": ["one fact"],
            "reason": "partial",
        },
        {
            "sufficient": False,
            "unanswerable": False,
            "failure_type": "off_topic",
            "coverage_score": 0.1,
            "found_facts": [],
            "missing_facts": ["all"],
            "reason": "worse",
        },
    ])
    rewriter = MagicMock()
    rewriter.rewrite_with_feedback = AsyncMock(return_value="rewritten")
    graph = AdaptiveAgentGraphV2(
        _lightrag(),
        _classifier=_classifier("semantic"),
        _grader=grader,
        _rewriter=rewriter,
        _router=router,
        _checker=_checker(True),
        _cache=RouterCache(),
        max_retrieve_steps=2,
    )
    result = await graph.run("question", return_trace=True)
    assert [c["chunk_id"] for c in result["trace"]["data"]["chunks"]] == ["good0", "good1"]
    assert [c["chunk_id"] for c in result["trace"]["data"]["last_chunks"]] == ["bad0"]
    assert result["trace"]["terminal_reason"] == "insufficient"


async def test_v2_targeted_retrieval_uses_best_profile_then_full_v2():
    router = MagicMock()
    router.route = AsyncMock(side_effect=[
        (_chunks("initial"), _trace("multihop", _chunks("initial"))),
        (_chunks("target1"), _trace("multihop", _chunks("target1"))),
        (_chunks("target2"), _trace("full_v2", _chunks("target2"))),
    ])
    grader = MagicMock()
    grader.grade = AsyncMock(return_value={
        "sufficient": True,
        "unanswerable": False,
        "failure_type": "partial_evidence",
        "coverage_score": 0.9,
        "found_facts": ["answer"],
        "missing_facts": [],
        "reason": "ok",
    })
    checker = MagicMock()
    checker.verify = AsyncMock(side_effect=[
        {"grounded": False, "ungrounded_claims": ["claim"]},
        {"grounded": False, "ungrounded_claims": ["claim"]},
        {"grounded": True, "ungrounded_claims": []},
    ])
    graph = AdaptiveAgentGraphV2(
        _lightrag(),
        _classifier=_classifier("multihop"),
        _grader=grader,
        _router=router,
        _checker=checker,
        _cache=RouterCache(),
    )
    result = await graph.run("question", return_trace=True)
    assert result["grounded"] is True
    assert router.route.call_args_list[1].kwargs["profile_name"] == "multihop"
    assert router.route.call_args_list[2].kwargs["profile_name"] == "full_v2"
    assert result["trace"]["check_cycles_used"] == 2


def test_decompose_merge_balances_subquestions_for_k5():
    merged, trace = _merge_decomposed_chunks(
        [
            ("q1", _chunks("a", 5, 0.9)),
            ("q2", _chunks("b", 5, 0.8)),
            ("q3", _chunks("c", 5, 0.7)),
        ],
        chunk_top_k=5,
    )
    assert len(merged) == 5
    assert trace["per_subquestion_cap"] == 2
    assert _prefix_count(merged, "a") <= 2
    assert _prefix_count(merged, "b") <= 2
    assert _prefix_count(merged, "c") <= 2


def test_decompose_merge_balances_subquestions_for_k10():
    merged, trace = _merge_decomposed_chunks(
        [
            ("q1", _chunks("a", 5, 0.9)),
            ("q2", _chunks("b", 5, 0.8)),
            ("q3", _chunks("c", 5, 0.7)),
        ],
        chunk_top_k=10,
    )
    assert len(merged) == 10
    assert trace["per_subquestion_cap"] == 4
    assert _prefix_count(merged, "a") <= 4
    assert _prefix_count(merged, "b") <= 4
    assert _prefix_count(merged, "c") <= 4
    assert trace["sub_question_chunk_counts"] == {"q1": 5, "q2": 5, "q3": 5}


def test_decompose_merge_can_apply_final_ppr_floor():
    hybrid_chunks = [
        {"chunk_id": f"h{i}", "rrf_score": 0.9 - i * 0.01, "rrf_source_paths": ["qdrant_chunks_hybrid"]}
        for i in range(5)
    ]
    ppr_chunks = [
        {"chunk_id": f"p{i}", "rrf_score": 0.1 - i * 0.01, "rrf_source_paths": ["ppr"]}
        for i in range(3)
    ]
    merged, trace = _merge_decomposed_chunks(
        [("q1", hybrid_chunks), ("q2", ppr_chunks)],
        chunk_top_k=5,
        path_floors={"ppr": 3},
    )
    ids = {chunk["chunk_id"] for chunk in merged}
    assert {"p0", "p1", "p2"}.issubset(ids)
    assert trace["ppr_floor_count"] == 3
