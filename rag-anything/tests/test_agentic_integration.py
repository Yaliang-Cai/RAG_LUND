# tests/test_agentic_integration.py
"""
End-to-end smoke tests for AdaptiveAgentGraph.
LLM and retrieval are mocked at their boundaries; graph logic is real.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from raganything.retrieval.agent_graph import AdaptiveAgentGraph
from raganything.retrieval.complexity import ComplexityClassifier
from raganything.retrieval.evaluator import AnswerEvaluator


def _chunks(n=3):
    return [{"chunk_id": f"c{i}", "content": f"relevant content {i}", "rrf_score": 0.6 + i * 0.05} for i in range(n)]


def _lightrag():
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value="integrated answer")
    return lg


def _router(chunks=None):
    m = MagicMock()
    m.route = AsyncMock(return_value=(chunks or _chunks(), {"profile": "semantic", "confidence": 0.88}))
    return m


async def test_full_simple_track():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("simple", {"confidence": 0.95, "reasoning": "one hop", "latency": 0.01}))
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=_router())
    result = await graph.run("What does BERT stand for?")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_full_medium_track_with_retry():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("medium", {"confidence": 0.8, "reasoning": "moderate", "latency": 0.02}))
    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.55, "gap": "missing information about layer normalization"},
        {"score": 0.88, "gap": ""},
    ])
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=_router(), _evaluator=ev)
    result = await graph.run("Explain BERT's architecture in detail.")
    assert isinstance(result, str)
    assert ev.evaluate.call_count == 2


async def test_full_complex_track():
    llm_responses = [
        json.dumps({"sub_questions": ["What is BERT?", "What is GPT?"]}),
        "synthesized comparison answer",
    ]
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(side_effect=llm_responses)

    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("complex", {"confidence": 0.87, "reasoning": "multi-entity", "latency": 0.03}))
    ev = MagicMock()
    ev.evaluate = AsyncMock(return_value={"score": 0.92, "gap": ""})

    graph = AdaptiveAgentGraph(lg, _complexity_clf=clf, _router=_router(), _evaluator=ev)
    result = await graph.run("Compare BERT and GPT architectures.")
    assert isinstance(result, str)


async def test_return_trace_structure():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("simple", {"confidence": 0.9, "reasoning": "ok", "latency": 0.01}))
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=_router())
    result = await graph.run("test", return_trace=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "trace" in result
    assert result["trace"]["complexity"] == "simple"
    assert "eval_score" in result["trace"]
    assert "iteration" in result["trace"]
    assert "routing" in result["trace"]


async def test_empty_retrieval_does_not_crash():
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=("simple", {"confidence": 0.9, "reasoning": "ok", "latency": 0.01}))
    router = MagicMock()
    router.route = AsyncMock(return_value=([], {}))
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=clf, _router=router)
    result = await graph.run("question with no matching docs")
    assert isinstance(result, str)
