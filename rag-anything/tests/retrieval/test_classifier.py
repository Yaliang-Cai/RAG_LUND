import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.classifier import QueryClassifier, _CLASSIFIER_PROMPT
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
    assert meta["candidate_profile"] == "local"
    assert meta["selected_profile"] == "local"
    assert meta["fallback_used"] is False
    assert meta["fallback_reason"] == ""


async def test_low_confidence_falls_back_to_semantic():
    llm = await _make_llm(json.dumps({
        "reasoning": "unsure",
        "profile": "local",
        "confidence": 0.4,
    }))
    clf = QueryClassifier(llm)
    name, meta = await clf.classify("some ambiguous query")
    assert name == "semantic"
    assert meta["candidate_profile"] == "local"
    assert meta["selected_profile"] == "semantic"
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "low_confidence"


async def test_unknown_profile_falls_back_to_semantic():
    llm = await _make_llm(json.dumps({
        "reasoning": "ok",
        "profile": "nonexistent_profile",
        "confidence": 0.95,
    }))
    clf = QueryClassifier(llm)
    name, meta = await clf.classify("query")
    assert name == "semantic"
    assert meta["candidate_profile"] == "nonexistent_profile"
    assert meta["selected_profile"] == "semantic"
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "invalid_profile"


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


async def test_hybrid_experimental_profile_rejected_from_llm_output():
    llm = await _make_llm(json.dumps({
        "reasoning": "broad kg query",
        "profile": "hybrid_experimental",
        "confidence": 0.95,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("find related entities and events")
    assert name == "semantic"


async def test_global_profile_is_valid():
    llm = await _make_llm(json.dumps({
        "reasoning": "relationship or event driven",
        "profile": "global",
        "confidence": 0.9,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("Which acquisitions involved semiconductor suppliers?")
    assert name == "global"


async def test_person_name_alone_does_not_force_precise():
    llm = await _make_llm(json.dumps({
        "reasoning": "proper noun present",
        "profile": "precise",
        "confidence": 0.92,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("What did Barack Obama do before becoming president?")
    assert name != "precise"


async def test_low_confidence_exact_query_falls_back_to_precise():
    llm = await _make_llm(json.dumps({
        "reasoning": "unsure",
        "profile": "semantic",
        "confidence": 0.4,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("What is the impact scope of CVE-2026-001?")
    assert name == "precise"


async def test_low_confidence_multihop_query_falls_back_to_multihop():
    llm = await _make_llm(json.dumps({
        "reasoning": "unsure",
        "profile": "semantic",
        "confidence": 0.4,
    }))
    clf = QueryClassifier(llm)
    name, _ = await clf.classify("Compare how LightRAG and HippoRAG2 connect entity seeds to passages.")
    assert name == "multihop"


def test_classifier_prompt_documents_profile_boundaries():
    assert "A person, organization, or place name alone is NOT precise" in _CLASSIFIER_PROMPT
    assert "Do not use local/global/multihop unless graph structure is clearly useful" in _CLASSIFIER_PROMPT
    assert "full" not in _CLASSIFIER_PROMPT.split("Available profiles", 1)[1]
    assert "hybrid_experimental" not in _CLASSIFIER_PROMPT


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
