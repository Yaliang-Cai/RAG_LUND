import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from raganything.retrieval.agent_graph import AdaptiveAgentGraph
from raganything.retrieval.router_cache import RouterCache


def _lightrag() -> MagicMock:
    lg = MagicMock()
    lg.llm_model_func = AsyncMock(return_value="the answer")
    return lg


def _chunks(n: int = 2) -> list[dict]:
    return [{"chunk_id": f"c{i}", "content": f"content {i}", "rrf_score": 0.5} for i in range(n)]


def _router(chunks=None) -> MagicMock:
    m = MagicMock()
    m.route = AsyncMock(return_value=(chunks or _chunks(), {"profile": "semantic", "chunks_per_path": {}}))
    return m


def _classifier(profile: str = "semantic") -> MagicMock:
    m = MagicMock()
    m.classify = AsyncMock(return_value=(profile, {"confidence": 0.9, "reasoning": "test", "latency": 0.01}))
    return m


def _grader(sufficient: bool = True, reason: str = "") -> MagicMock:
    m = MagicMock()
    m.grade = AsyncMock(return_value={"sufficient": sufficient, "reason": reason})
    return m


def _rewriter(new_query: str = "rewritten query") -> MagicMock:
    m = MagicMock()
    m.rewrite = AsyncMock(return_value=new_query)
    return m


def _checker(grounded: bool = True, claims: list | None = None) -> MagicMock:
    m = MagicMock()
    m.verify = AsyncMock(return_value={"grounded": grounded, "ungrounded_claims": claims or []})
    return m


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_happy_path_returns_answer():
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("What is BERT?")
    assert isinstance(result, str)
    assert result != ""


async def test_return_trace_true_includes_metadata():
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("query", return_trace=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert result["confidence"] == "high"
    assert result["grounded"] is True
    assert "trace" in result


# ── Retrieval cycle: rewriter at cycle 0 ─────────────────────────────────────

async def test_cycle0_fail_triggers_rewriter():
    grader_calls = []
    async def grade_side_effect(query, chunks):
        grader_calls.append(query)
        # fail first, succeed second
        return {"sufficient": len(grader_calls) > 1, "reason": "missing Y"}

    grader = MagicMock()
    grader.grade = grade_side_effect
    rewriter = _rewriter("improved query")

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=grader,
        _rewriter=rewriter,
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("original query")
    assert result != ""
    rewriter.rewrite.assert_called_once()


# ── Retrieval cycle: decompose at cycle 1 ────────────────────────────────────

async def test_cycle1_fail_triggers_decompose_with_full_profile():
    grader_calls = []
    async def grade_side_effect(query, chunks):
        grader_calls.append(query)
        # fail twice, succeed third
        return {"sufficient": len(grader_calls) > 2, "reason": "missing"}

    grader = MagicMock()
    grader.grade = grade_side_effect
    router = _router()

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=grader,
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=router,
        _cache=RouterCache(),
    )

    llm = AsyncMock(side_effect=[
        # classifier
        json.dumps({"reasoning": "r", "profile": "semantic", "confidence": 0.9}),
        # rewriter
        "rewritten",
        # decomposer
        json.dumps({"sub_questions": ["sub1", "sub2"]}),
        # generator
        "the answer",
        # hallucination check
        json.dumps({"grounded": True, "ungrounded_claims": []}),
    ])
    graph._llm = llm

    result = await graph.run("complex query")
    assert result != ""
    # router should have been called with "full" profile for parallel retrieve
    route_calls = router.route.call_args_list
    full_calls = [c for c in route_calls if c.kwargs.get("profile_name") == "full" or
                  (len(c.args) > 2 and c.args[2] == "full")]
    assert len(full_calls) > 0


# ── 3 retrieval cycles all fail → END_INSUFFICIENT ───────────────────────────

async def test_three_retrieve_failures_returns_none_without_generating():
    grader = _grader(sufficient=False)

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=grader,
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=RouterCache(),
    )
    # Patch llm so we can detect if generator was called
    call_log = []
    original_llm = graph._llm
    async def logged_llm(prompt, **kw):
        call_log.append(prompt)
        return await original_llm(prompt, **kw)
    graph._llm = logged_llm

    result = await graph.run("unanswerable query", return_trace=True)
    assert result["answer"] is None
    assert result["confidence"] == "none"
    # Generator suffix contains this phrase; it must NOT appear in any call
    for call in call_log:
        assert "Answer the question based ONLY" not in call


# ── Hallucination check: retry via targeted_retriever ────────────────────────

async def test_check_fail_triggers_targeted_retriever():
    check_calls = []
    async def check_side_effect(query, answer, chunks):
        check_calls.append(1)
        grounded = len(check_calls) > 1
        return {"grounded": grounded, "ungrounded_claims": ["claim X"] if not grounded else []}

    checker = MagicMock()
    checker.verify = check_side_effect
    router = _router()

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=checker,
        _router=router,
        _cache=RouterCache(),
    )
    result = await graph.run("query", return_trace=True)
    assert result["confidence"] == "high"
    # targeted_retriever should have fired once (check_cycle went 0→1)
    assert result["trace"]["check_cycles_used"] == 1


# ── Hallucination check: 2 failures → END_INSUFFICIENT ───────────────────────

async def test_two_check_failures_returns_none():
    checker = _checker(grounded=False, claims=["unsupported claim"])
    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=_classifier(),
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=checker,
        _router=_router(),
        _cache=RouterCache(),
    )
    result = await graph.run("query", return_trace=True)
    assert result["answer"] is None
    assert result["confidence"] == "none"


# ── Router cache integration ──────────────────────────────────────────────────

async def test_cache_hit_skips_classifier():
    cache = RouterCache()
    cache.put("cached query", "local")
    classifier = _classifier("semantic")  # would return semantic, but cache wins

    graph = AdaptiveAgentGraph(
        _lightrag(),
        _classifier=classifier,
        _grader=_grader(sufficient=True),
        _rewriter=_rewriter(),
        _checker=_checker(grounded=True),
        _router=_router(),
        _cache=cache,
    )
    result = await graph.run("cached query", return_trace=True)
    assert result["trace"]["router_cache_hit"] is True
    classifier.classify.assert_not_called()
