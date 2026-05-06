import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.classifier import QueryClassifier
from raganything.constants import DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE


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


async def test_low_confidence_falls_back_to_semantic():
    llm = await _make_llm(json.dumps({
        "reasoning": "unsure",
        "profile": "local",
        "confidence": 0.4,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("some ambiguous query")
    assert name == "semantic"


async def test_unknown_profile_falls_back_to_semantic():
    llm = await _make_llm(json.dumps({
        "reasoning": "ok",
        "profile": "nonexistent_profile",
        "confidence": 0.95,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("query")
    assert name == "semantic"


async def test_non_json_output_falls_back_to_semantic():
    llm = await _make_llm("Sorry, I cannot classify this query.")
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "semantic"


async def test_missing_profile_key_falls_back_to_semantic():
    llm = await _make_llm(json.dumps({"reasoning": "ok", "confidence": 0.9}))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "semantic"


async def test_llm_exception_falls_back_to_semantic():
    llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("test query")
    assert name == "semantic"


async def test_full_profile_rejected_from_llm_output():
    # Classifier must not accept "full" from LLM — it is reserved for cycle-3 escalation
    llm = await _make_llm(json.dumps({
        "reasoning": "very ambiguous",
        "profile": "full",
        "confidence": 0.95,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("something ambiguous")
    assert name == "semantic"   # rejected and replaced with fallback


async def test_avoid_excludes_profile():
    llm = await _make_llm(json.dumps({
        "reasoning": "multihop fits",
        "profile": "multihop",
        "confidence": 0.85,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("complex query", avoid=["multihop"])
    assert name == "semantic"   # multihop avoided, fallback applied


async def test_avoid_empty_list_no_effect():
    llm = await _make_llm(json.dumps({
        "reasoning": "multihop",
        "profile": "multihop",
        "confidence": 0.85,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("complex query", avoid=[])
    assert name == "multihop"
