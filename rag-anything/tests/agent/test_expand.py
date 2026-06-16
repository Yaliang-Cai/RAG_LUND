import pytest
from raganything.agent.expand import hyde_query, mqe_queries


class StubPool:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def call(self, role, prompt, **kw):
        self.calls.append((role, prompt))
        return self.reply


class BoomPool:
    async def call(self, role, prompt, **kw):
        raise RuntimeError("endpoint down")


@pytest.mark.asyncio
async def test_mqe_returns_paraphrases_excluding_original():
    pool = StubPool('{"queries": ["5G revenue Huawei", "Huawei wireless sales", "原始问题"]}')
    out = await mqe_queries(pool, "原始问题", n=3)
    assert "5G revenue Huawei" in out
    assert "原始问题" not in out          # original excluded
    assert pool.calls[0][0] == "rewriter"  # uses the cheap judge role


@pytest.mark.asyncio
async def test_mqe_caps_to_n_and_dedups():
    pool = StubPool('{"queries": ["a", "a", "b", "c", "d"]}')
    assert await mqe_queries(pool, "q", n=2) == ["a", "b"]


@pytest.mark.asyncio
async def test_mqe_best_effort_on_failure():
    assert await mqe_queries(BoomPool(), "q") == []


@pytest.mark.asyncio
async def test_hyde_returns_passage():
    pool = StubPool("Huawei's 5G revenue reached X billion in 2023 across regions.")
    out = await hyde_query(pool, "q")
    assert out.startswith("Huawei")
    assert pool.calls[0][0] == "rewriter"


@pytest.mark.asyncio
async def test_hyde_best_effort_on_failure():
    assert await hyde_query(BoomPool(), "q") == ""
