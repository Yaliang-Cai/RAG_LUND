import pytest
from raganything.agent.models import ModelPool, JUDGE_ROLES


def make_fn(name, fail=False):
    calls = []
    async def fn(prompt, **kw):
        calls.append(prompt)
        if fail:
            raise ConnectionError("down")
        return f"{name}-reply"
    fn.calls = calls
    return fn


@pytest.mark.asyncio
async def test_default_all_roles_to_main():
    main = make_fn("main")
    pool = ModelPool(main_func=main)
    assert await pool.call("grader", "p") == "main-reply"
    assert await pool.call("generator", "p") == "main-reply"


@pytest.mark.asyncio
async def test_judge_roles_routed_to_judge():
    main, judge = make_fn("main"), make_fn("judge")
    pool = ModelPool(main_func=main, judge_func=judge)
    assert await pool.call("checker", "p") == "judge-reply"
    assert await pool.call("planner", "p") == "main-reply"  # planner 留大模型 §12.1


@pytest.mark.asyncio
async def test_per_call_fallback_and_breaker():
    main, judge = make_fn("main"), make_fn("judge", fail=True)
    pool = ModelPool(main_func=main, judge_func=judge, breaker_threshold=2, probe_interval=999)
    assert await pool.call("grader", "p") == "main-reply"  # 单次回退 §12.3
    assert await pool.call("grader", "p") == "main-reply"
    assert pool.breaker_open is True  # 连续 2 次失败熔断
    await pool.call("grader", "p")
    assert len(judge.calls) == 2  # 熔断后不再打 judge
