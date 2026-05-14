import json
from unittest.mock import AsyncMock

import pytest

from raganything.retrieval.grader_v2 import GraderV2


def _chunks(n: int = 2) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "content": f"Evidence {i}", "file_path": f"doc{i}.txt"}
        for i in range(n)
    ]


async def test_grader_v2_returns_structured_failure_metadata():
    llm = AsyncMock(return_value=json.dumps({
        "sufficient": False,
        "unanswerable": True,
        "failure_type": "missing_relation",
        "coverage_score": 0.42,
        "found_facts": ["entity found"],
        "missing_facts": ["bridge relation"],
        "reason": "Missing the bridge relation.",
    }))
    result = await GraderV2(llm).grade("question", _chunks())
    assert result == {
        "sufficient": False,
        "unanswerable": True,
        "failure_type": "missing_relation",
        "coverage_score": 0.42,
        "found_facts": ["entity found"],
        "missing_facts": ["bridge relation"],
        "reason": "Missing the bridge relation.",
    }


async def test_grader_v2_clamps_coverage_and_normalizes_bad_failure_type():
    llm = AsyncMock(return_value=json.dumps({
        "sufficient": False,
        "unanswerable": False,
        "failure_type": "unknown",
        "coverage_score": 9,
    }))
    result = await GraderV2(llm).grade("question", _chunks())
    assert result["failure_type"] == "partial_evidence"
    assert result["coverage_score"] == 1.0


async def test_grader_v2_fails_closed_without_early_unanswerable():
    llm = AsyncMock(side_effect=RuntimeError("down"))
    result = await GraderV2(llm).grade("question", _chunks())
    assert result["sufficient"] is False
    assert result["unanswerable"] is False
    assert result["failure_type"] == "partial_evidence"
    assert result["coverage_score"] == 0.0


async def test_grader_v2_prompt_defers_unanswerable_until_exhaustion():
    captured = []

    async def llm(prompt, **kwargs):
        captured.append(prompt)
        return json.dumps({
            "sufficient": False,
            "unanswerable": False,
            "failure_type": "empty",
            "coverage_score": 0,
        })

    await GraderV2(llm).grade("unique question", [])
    assert "unanswerable_candidate" in captured[0]
    assert "retrieval needs improvement" in captured[0]


async def test_grader_v2_does_not_limit_fact_count_and_uses_larger_token_budget():
    captured = {}
    found = [f"found fact {i}" for i in range(8)]
    missing = [f"missing fact {i}" for i in range(7)]

    async def llm(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return json.dumps({
            "sufficient": False,
            "unanswerable": False,
            "failure_type": "partial_evidence",
            "coverage_score": 0.5,
            "found_facts": found,
            "missing_facts": missing,
            "reason": "Needs more evidence.",
        })

    result = await GraderV2(llm).grade("question", _chunks())
    assert result["found_facts"] == found
    assert result["missing_facts"] == missing
    assert captured["kwargs"]["max_tokens"] == 1536
    assert "no item-count limit" in captured["prompt"]
