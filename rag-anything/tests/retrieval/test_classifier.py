import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.classifier import QueryClassifier


async def _make_llm(response_str: str) -> AsyncMock:
    mock = AsyncMock(return_value=response_str)
    return mock


async def test_valid_classification():
    llm = await _make_llm(json.dumps({
        "reasoning": "clear factual query",
        "profile": "local",
        "confidence": 0.9,
    }))
    clf = QueryClassifier(llm)
    name, meta = await clf.classify("How many parameters does BERT have?")
    assert name == "local"
    assert meta["confidence"] == 0.9
    assert "reasoning" in meta
    assert meta["latency"] >= 0.0


async def test_low_confidence_falls_back_to_full():
    llm = await _make_llm(json.dumps({
        "reasoning": "unsure",
        "profile": "local",
        "confidence": 0.4,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("some ambiguous query")
    assert name == "full"


async def test_unknown_profile_falls_back_to_full():
    llm = await _make_llm(json.dumps({
        "reasoning": "ok",
        "profile": "nonexistent_profile",
        "confidence": 0.95,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "full"


async def test_non_json_output_falls_back_to_full():
    llm = await _make_llm("Sorry, I cannot classify this query.")
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "full"


async def test_missing_profile_key_falls_back_to_full():
    llm = await _make_llm(json.dumps({"reasoning": "ok", "confidence": 0.9}))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    # missing "profile" key → result.get("profile", "full") → "full"
    # "full" IS in registry and confidence is 0.9 → returns "full"
    assert name == "full"


async def test_llm_exception_falls_back_to_full():
    llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "full"
