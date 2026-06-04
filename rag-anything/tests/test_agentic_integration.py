# tests/test_agentic_integration.py
"""
End-to-end smoke tests for AdaptiveAgentGraph.
LLM and retrieval are mocked at their boundaries; graph logic is real.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from raganything.retrieval.agent_graph import AdaptiveAgentGraph


def _chunks(n=3):
    return [{"chunk_id": f"c{i}", "content": f"relevant content {i}", "rrf_score": 0.6 + i * 0.05} for i in range(n)]


def _lightrag():
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value="integrated answer")
    return lg


def _router(chunks=None):
    m = MagicMock()
    route_chunks = chunks or _chunks()
    m.route = AsyncMock(return_value=(
        route_chunks,
        {
            "profile": "semantic",
            "confidence": 0.88,
            "paths_activated": ["qdrant_chunks_hybrid"],
            "paths_failed": [],
            "chunks_per_path": {"qdrant_chunks_hybrid": len(route_chunks)},
            "chunks_after_rrf": len(route_chunks),
            "chunks_after_rerank": len(route_chunks),
            "chunks_after_threshold": len(route_chunks),
        },
    ))
    return m


def _classifier(profile="semantic", confidence=0.9):
    m = MagicMock()
    m.classify = AsyncMock(return_value=(
        profile,
        {
            "confidence": confidence,
            "reasoning": "test",
            "latency": 0.01,
            "candidate_profile": profile,
            "selected_profile": profile,
            "fallback_used": False,
            "fallback_reason": "",
        },
    ))
    return m


def _grader(sufficient=True):
    m = MagicMock()
    m.grade = AsyncMock(return_value={"sufficient": sufficient, "reason": "ok"})
    return m


def _checker(grounded=True):
    m = MagicMock()
    m.verify = AsyncMock(return_value={"grounded": grounded, "ungrounded_claims": []})
    return m


async def test_full_simple_track():
    clf = _classifier("semantic", confidence=0.95)
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=clf,
        _grader=_grader(sufficient=True),
        _checker=_checker(grounded=True),
        _router=_router(),
    )
    result = await graph.run("What does BERT stand for?")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_full_medium_track_with_retry():
    grader = MagicMock()
    grader.grade = AsyncMock(side_effect=[
        {"sufficient": False, "reason": "missing information about layer normalization"},
        {"sufficient": True, "reason": ""},
    ])
    rewriter = MagicMock()
    rewriter.rewrite = AsyncMock(return_value="Explain BERT layer normalization.")
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier("semantic", confidence=0.8),
        _grader=grader,
        _rewriter=rewriter,
        _checker=_checker(grounded=True),
        _router=_router(),
    )
    result = await graph.run("Explain BERT's architecture in detail.")
    assert isinstance(result, str)
    assert grader.grade.call_count == 2
    rewriter.rewrite.assert_called_once()


async def test_full_complex_track():
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value="synthesized comparison answer")

    graph = AdaptiveAgentGraph(
        lg,
        _classifier=_classifier("multihop", confidence=0.87),
        _grader=_grader(sufficient=True),
        _checker=_checker(grounded=True),
        _router=_router(),
    )
    result = await graph.run("Compare BERT and GPT architectures.")
    assert isinstance(result, str)


async def test_return_trace_structure():
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier("semantic", confidence=0.9),
        _grader=_grader(sufficient=True),
        _checker=_checker(grounded=True),
        _router=_router(),
    )
    result = await graph.run("test", return_trace=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "trace" in result
    assert result["trace"]["profile"] == "semantic"
    assert result["trace"]["classifier"]["selected_profile"] == "semantic"
    assert result["trace"]["retrieval_steps"]
    assert result["trace"]["data"]["chunks"]


async def test_empty_retrieval_does_not_crash():
    router = MagicMock()
    router.route = AsyncMock(return_value=([], {}))
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier("semantic", confidence=0.9),
        _grader=_grader(sufficient=True),
        _checker=_checker(grounded=True),
        _router=router,
    )
    result = await graph.run("question with no matching docs")
    assert isinstance(result, str)
