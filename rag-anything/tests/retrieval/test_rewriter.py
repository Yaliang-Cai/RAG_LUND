import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.rewriter import Rewriter


async def test_rewrite_returns_llm_output():
    llm = AsyncMock(return_value="  rewritten query  ")
    rw = Rewriter(llm)
    result = await rw.rewrite("original query", "missing entity Y")
    assert result == "rewritten query"


async def test_rewrite_prompt_contains_original_and_reason():
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return "new query"
    rw = Rewriter(llm)
    await rw.rewrite("original_xyz", "missing_reason_abc")
    assert "original_xyz" in captured[0]
    assert "missing_reason_abc" in captured[0]


async def test_rewrite_exception_returns_original():
    llm = AsyncMock(side_effect=RuntimeError("LLM error"))
    rw = Rewriter(llm)
    result = await rw.rewrite("original query", "some reason")
    assert result == "original query"


async def test_rewrite_empty_response_returns_original():
    llm = AsyncMock(return_value="   ")
    rw = Rewriter(llm)
    result = await rw.rewrite("original query", "reason")
    assert result == "original query"
