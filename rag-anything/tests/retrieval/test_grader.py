import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.grader import Grader, build_shared_prefix


def _chunks(n: int = 3) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "content": f"Evidence paragraph {i}.", "file_path": f"doc{i}.pdf"}
        for i in range(n)
    ]


async def test_grade_sufficient():
    llm = AsyncMock(return_value=json.dumps({"sufficient": True, "reason": "All facts present."}))
    g = Grader(llm)
    result = await g.grade("What is X?", _chunks())
    assert result["sufficient"] is True
    assert "reason" in result


async def test_grade_insufficient():
    llm = AsyncMock(return_value=json.dumps({"sufficient": False, "reason": "Missing Y."}))
    g = Grader(llm)
    result = await g.grade("What is X?", _chunks())
    assert result["sufficient"] is False
    assert result["reason"] == "Missing Y."


async def test_grade_json_parse_failure_falls_back_sufficient():
    llm = AsyncMock(return_value="not json at all")
    g = Grader(llm, fallback_sufficient=True)
    result = await g.grade("query", _chunks())
    assert result["sufficient"] is True


async def test_grade_json_parse_failure_respects_fallback_false():
    llm = AsyncMock(return_value="broken")
    g = Grader(llm, fallback_sufficient=False)
    result = await g.grade("query", _chunks())
    assert result["sufficient"] is False


async def test_grade_llm_exception_falls_back():
    llm = AsyncMock(side_effect=RuntimeError("LLM down"))
    g = Grader(llm, fallback_sufficient=True)
    result = await g.grade("query", _chunks())
    assert result["sufficient"] is True


async def test_build_shared_prefix_contains_all_chunks():
    chunks = _chunks(3)
    prefix = build_shared_prefix(chunks)
    for c in chunks:
        assert c["content"] in prefix
    assert "[1]" in prefix
    assert "[3]" in prefix


async def test_grade_prompt_contains_query():
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return json.dumps({"sufficient": True, "reason": "ok"})
    g = Grader(llm)
    await g.grade("unique_query_string_xyz", _chunks(1))
    assert "unique_query_string_xyz" in captured[0]
