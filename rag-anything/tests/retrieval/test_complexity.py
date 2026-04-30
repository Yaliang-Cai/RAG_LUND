# tests/retrieval/test_complexity.py
import json
import pytest
from unittest.mock import AsyncMock
from raganything.retrieval.complexity import ComplexityClassifier


def _llm(response: str) -> AsyncMock:
    return AsyncMock(return_value=response)


async def test_returns_simple_on_high_confidence():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "one hop", "complexity": "simple", "confidence": 0.9})))
    level, meta = await clf.classify("What does BERT stand for?")
    assert level == "simple"
    assert meta["confidence"] == 0.9


async def test_returns_complex_on_high_confidence():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "multi-entity", "complexity": "complex", "confidence": 0.85})))
    level, _ = await clf.classify("Compare indexing in LightRAG vs HippoRAG.")
    assert level == "complex"


async def test_low_confidence_downgrades_to_medium():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "unsure", "complexity": "complex", "confidence": 0.4})))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_unknown_complexity_falls_back_to_medium():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "ok", "complexity": "impossible_value", "confidence": 0.9})))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_non_json_falls_back_to_medium():
    clf = ComplexityClassifier(_llm("Not JSON at all"))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_llm_exception_falls_back_to_medium():
    clf = ComplexityClassifier(AsyncMock(side_effect=RuntimeError("LLM down")))
    level, _ = await clf.classify("some query")
    assert level == "medium"


async def test_metadata_contains_latency():
    clf = ComplexityClassifier(_llm(json.dumps({"reasoning": "ok", "complexity": "simple", "confidence": 0.8})))
    _, meta = await clf.classify("test")
    assert "latency" in meta
    assert meta["latency"] >= 0.0


async def test_arithmetic_query_returns_medium():
    """Prompt now instructs LLM to classify arithmetic/aggregation as medium."""
    clf = ComplexityClassifier(_llm(json.dumps({
        "reasoning": "requires summing wins + ties from table",
        "complexity": "medium",
        "confidence": 0.9,
    })))
    level, _ = await clf.classify(
        "How many total evaluations were collected for RetrieveNRefine++ vs. Seq2Seq?"
    )
    assert level == "medium"


async def test_score_difference_query_returns_medium():
    clf = ComplexityClassifier(_llm(json.dumps({
        "reasoning": "requires subtracting two scores",
        "complexity": "medium",
        "confidence": 0.88,
    })))
    level, _ = await clf.classify(
        "How much did the Engagingness score improve from Seq2Seq to RetrieveNRefine++?"
    )
    assert level == "medium"


async def test_simple_single_fact_still_returns_simple():
    clf = ComplexityClassifier(_llm(json.dumps({
        "reasoning": "direct lookup, no calculation",
        "complexity": "simple",
        "confidence": 0.92,
    })))
    level, _ = await clf.classify("What does BERT stand for?")
    assert level == "simple"
