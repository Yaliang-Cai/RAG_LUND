# tests/retrieval/test_agent_graph.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from raganything.retrieval.agent_graph import AdaptiveAgentGraph


def _lightrag(answer: str = "the answer") -> MagicMock:
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value=answer)
    return lg


def _chunks(n: int = 2) -> list[dict]:
    return [{"chunk_id": f"c{i}", "content": f"content {i}", "rrf_score": 0.5 + i * 0.1} for i in range(n)]


def _clf(complexity: str = "simple") -> MagicMock:
    m = MagicMock()
    m.classify = AsyncMock(return_value=(complexity, {"confidence": 0.9, "reasoning": "test", "latency": 0.01}))
    return m


def _router(chunks=None) -> MagicMock:
    m = MagicMock()
    m.route = AsyncMock(return_value=(chunks or _chunks(), {"profile": "semantic"}))
    return m


def _evaluator(score: float = 0.9, gap: str = "") -> MagicMock:
    m = MagicMock()
    m.evaluate = AsyncMock(return_value={"score": score, "gap": gap})
    return m


# ── Simple path ─────────────────────────────────────────────────────────────

async def test_simple_path_returns_answer():
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("simple"), _router=_router())
    result = await graph.run("What is BERT?")
    assert result == "the answer"


async def test_simple_path_does_not_call_evaluator():
    ev = _evaluator()
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("simple"), _router=_router(), _evaluator=ev)
    await graph.run("simple question")
    ev.evaluate.assert_not_called()


async def test_simple_path_calls_router_once():
    router = _router()
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("simple"), _router=router)
    await graph.run("simple question")
    assert router.route.call_count == 1


# ── Medium path ──────────────────────────────────────────────────────────────

async def test_medium_path_no_retry_when_eval_passes():
    router = _router()
    graph = AdaptiveAgentGraph(
        _lightrag(), _complexity_clf=_clf("medium"), _router=router, _evaluator=_evaluator(0.85)
    )
    result = await graph.run("Explain attention.")
    assert isinstance(result, str)
    assert router.route.call_count == 1


async def test_medium_path_retries_once_on_low_score():
    router = _router()
    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.4, "gap": "missing details about multi-head"},
        {"score": 0.82, "gap": ""},
    ])
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("medium"), _router=router, _evaluator=ev)
    result = await graph.run("What is multi-head attention?")
    assert isinstance(result, str)
    assert router.route.call_count == 2  # initial + 1 targeted


async def test_medium_path_stops_at_max_iter_even_if_eval_fails():
    router = _router()
    ev = _evaluator(0.2, "always bad")
    graph = AdaptiveAgentGraph(_lightrag(), _complexity_clf=_clf("medium"), _router=router, _evaluator=ev)
    await graph.run("hard medium question")
    assert router.route.call_count == 2  # initial + max 1 retry for medium


# ── Complex path ─────────────────────────────────────────────────────────────

async def test_complex_path_decomposes_and_parallel_retrieves():
    router = _router()
    llm = AsyncMock(side_effect=[
        '{"sub_questions": ["sub1", "sub2"]}',
        "synthesized answer",
    ])
    lg = MagicMock()
    lg.llm_model_func = llm
    graph = AdaptiveAgentGraph(
        lg,
        _complexity_clf=_clf("complex"),
        _router=router,
        _evaluator=_evaluator(0.9),
    )
    result = await graph.run("Compare LightRAG and HippoRAG indexing.")
    assert isinstance(result, str)
    assert router.route.call_count == 2  # one per sub-question


async def test_complex_path_retries_targeted_not_full_decompose():
    route_calls = []

    async def fake_route(q, param, **kw):
        route_calls.append(q)
        return _chunks(), {}

    router = MagicMock()
    router.route = fake_route

    llm = AsyncMock(side_effect=[
        '{"sub_questions": ["q1", "q2"]}',
        "partial answer",
        "better answer",
    ])
    lg = MagicMock()
    lg.llm_model_func = llm

    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.4, "gap": "missing region B details"},
        {"score": 0.85, "gap": ""},
    ])

    graph = AdaptiveAgentGraph(lg, _complexity_clf=_clf("complex"), _router=router, _evaluator=ev)
    await graph.run("Complex multi-hop question.")
    assert len(route_calls) == 3  # 2 sub-questions + 1 targeted retry
    assert "region B" in route_calls[2]


async def test_return_trace_includes_complexity_and_score():
    graph = AdaptiveAgentGraph(
        _lightrag(), _complexity_clf=_clf("simple"), _router=_router()
    )
    result = await graph.run("test", return_trace=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert result["trace"]["complexity"] == "simple"


async def test_chunks_deduplicated_keeping_highest_rrf():
    chunk_high = {"chunk_id": "c1", "content": "high quality", "rrf_score": 0.9}
    chunk_low = {"chunk_id": "c1", "content": "high quality", "rrf_score": 0.3}

    call_count = [0]
    async def fake_route(q, param, **kw):
        call_count[0] += 1
        return ([chunk_high] if call_count[0] == 1 else [chunk_low]), {}

    router = MagicMock()
    router.route = fake_route

    ev = MagicMock()
    ev.evaluate = AsyncMock(side_effect=[
        {"score": 0.4, "gap": "needs more"},
        {"score": 0.9, "gap": ""},
    ])

    captured = []
    async def capture_llm(prompt, **kw):
        captured.append(prompt)
        return "answer"

    lg = MagicMock()
    lg.llm_model_func = capture_llm

    graph = AdaptiveAgentGraph(lg, _complexity_clf=_clf("medium"), _router=router, _evaluator=ev)
    await graph.run("dedup test")
    # The final generate prompt should list c1 only once
    last_prompt = captured[-1]
    assert last_prompt.count("[1] Source:") == 1
