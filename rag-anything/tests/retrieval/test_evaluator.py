# tests/retrieval/test_evaluator.py
import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.evaluator import AnswerEvaluator


def _llm(response: str) -> AsyncMock:
    return AsyncMock(return_value=response)


@pytest.mark.asyncio
async def test_high_score_with_no_gap():
    ev = AnswerEvaluator(_llm(json.dumps({"score": 0.92, "gap": ""})))
    result = await ev.evaluate("What is BERT?", "BERT is a transformer model.")
    assert result["score"] == 0.92
    assert result["gap"] == ""


@pytest.mark.asyncio
async def test_low_score_with_gap():
    ev = AnswerEvaluator(_llm(json.dumps({"score": 0.45, "gap": "Missing info about training data"})))
    result = await ev.evaluate("How was BERT trained?", "BERT uses transformers.")
    assert result["score"] == 0.45
    assert "training" in result["gap"].lower()


@pytest.mark.asyncio
async def test_score_clamped_to_0_1():
    ev = AnswerEvaluator(_llm(json.dumps({"score": 1.5, "gap": ""})))
    result = await ev.evaluate("q", "a")
    assert result["score"] == 1.0

    ev2 = AnswerEvaluator(_llm(json.dumps({"score": -0.3, "gap": "something"})))
    result2 = await ev2.evaluate("q", "a")
    assert result2["score"] == 0.0


@pytest.mark.asyncio
async def test_non_json_returns_default_score():
    ev = AnswerEvaluator(_llm("Sorry, cannot evaluate."))
    result = await ev.evaluate("q", "a")
    assert result["score"] == 0.5
    assert isinstance(result["gap"], str)


@pytest.mark.asyncio
async def test_llm_exception_returns_default():
    ev = AnswerEvaluator(AsyncMock(side_effect=RuntimeError("LLM down")))
    result = await ev.evaluate("q", "a")
    assert result["score"] == 0.5
