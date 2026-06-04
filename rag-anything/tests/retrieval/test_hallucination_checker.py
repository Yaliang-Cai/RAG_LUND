import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.hallucination_checker import HallucinationChecker


def _chunks(n: int = 2) -> list[dict]:
    return [{"chunk_id": f"c{i}", "content": f"Fact {i}.", "file_path": "doc.pdf"} for i in range(n)]


async def test_grounded_true():
    llm = AsyncMock(return_value=json.dumps({"grounded": True, "ungrounded_claims": []}))
    hc = HallucinationChecker(llm)
    r = await hc.verify("What is X?", "X is A.", _chunks())
    assert r["grounded"] is True
    assert r["ungrounded_claims"] == []


async def test_grounded_false_with_claims():
    llm = AsyncMock(return_value=json.dumps({
        "grounded": False,
        "ungrounded_claims": ["X is 42", "Y happened in 2020"],
    }))
    hc = HallucinationChecker(llm)
    r = await hc.verify("What is X?", "X is 42 and Y happened in 2020.", _chunks())
    assert r["grounded"] is False
    assert "X is 42" in r["ungrounded_claims"]


async def test_exception_defaults_grounded_false():
    llm = AsyncMock(side_effect=RuntimeError("LLM down"))
    hc = HallucinationChecker(llm)
    r = await hc.verify("q", "answer", _chunks())
    assert r["grounded"] is False
    assert r.get("check_status") == "error"


async def test_json_parse_failure_defaults_grounded_false():
    llm = AsyncMock(return_value="not json")
    hc = HallucinationChecker(llm)
    r = await hc.verify("q", "answer", _chunks())
    assert r["grounded"] is False
    assert r.get("check_status") == "error"


async def test_prompt_contains_answer_and_query():
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return json.dumps({"grounded": True, "ungrounded_claims": []})
    hc = HallucinationChecker(llm)
    await hc.verify("unique_query_abc", "unique_answer_xyz", _chunks(1))
    assert "unique_query_abc" in captured[0]
    assert "unique_answer_xyz" in captured[0]


async def test_prompt_shares_prefix_with_grader():
    from raganything.retrieval.grader import build_shared_prefix
    chunks = _chunks(2)
    shared = build_shared_prefix(chunks)
    captured = []
    async def llm(prompt, **kw):
        captured.append(prompt)
        return json.dumps({"grounded": True, "ungrounded_claims": []})
    hc = HallucinationChecker(llm)
    await hc.verify("q", "a", chunks)
    assert captured[0].startswith(shared)
