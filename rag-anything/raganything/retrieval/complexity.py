# raganything/retrieval/complexity.py
from __future__ import annotations
import json
import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6
_FALLBACK = "medium"

_PROMPT = """\
You are a query complexity classifier for a RAG system.
Classify the query into one complexity level:

- simple: Single-hop factual question answered by directly reading one fact.
  The answer requires NO calculation, NO combining of multiple values, and NO reading
  across multiple table cells or document sections.
  Examples: "What does BERT stand for?", "What is the capital of France?",
            "Which model has the highest accuracy?"

- medium: Requires calculation, aggregation, or synthesis — even if it involves only
  one topic. Assign medium whenever ANY of the following signals appear:
    * Arithmetic over multiple values ("how many total", "sum of", "difference between",
      "how much did X improve", "average of")
    * Reading multiple cells from a table and combining them
    * Comparison of a specific metric across two named items
    * Questions whose answer cannot be read verbatim from a single sentence
  Examples: "How many total evaluations were collected for A vs. B?" (requires summing
            wins + losses + ties from a table),
            "How much did the Engagingness score improve from X to Y?",
            "What are all the config options for the Redis module?",
            "Explain the attention mechanism in detail."

- complex: Multi-entity; requires decomposition, causal chains, or cross-document
  reasoning across multiple distinct entities.
  Examples: "How did the network partition in region A cause failures in region B?",
            "Compare the indexing strategies used by LightRAG and HippoRAG."

Rules:
- When unsure between simple and medium → choose medium.
- When unsure between medium and complex → choose medium.
- ANY question involving summing, counting, or arithmetic → at least medium.
- Only choose complex when multiple distinct entities clearly need cross-document reasoning.

Output JSON: {{"reasoning": "...", "complexity": "<simple|medium|complex>", "confidence": <0.0-1.0>}}

Query: {query}
"""


class ComplexityClassifier:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def classify(self, query: str) -> tuple[str, dict[str, Any]]:
        """Return (complexity, metadata). complexity is 'simple', 'medium', or 'complex'."""
        t0 = time.monotonic()
        complexity = _FALLBACK
        confidence = 0.0
        reasoning = ""
        try:
            raw = await self._llm(_PROMPT.format(query=query), response_format={"type": "json_object"})
            result = json.loads(raw)
            complexity = str(result.get("complexity", _FALLBACK)).strip()
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", ""))
            if complexity not in {"simple", "medium", "complex"}:
                complexity = _FALLBACK
            elif confidence < _CONFIDENCE_THRESHOLD:
                logger.warning("Low confidence %.2f for %r → %r", confidence, complexity, _FALLBACK)
                complexity = _FALLBACK
        except Exception:
            logger.warning("ComplexityClassifier failed, fallback to %r", _FALLBACK, exc_info=True)
            complexity = _FALLBACK
        return complexity, {"confidence": confidence, "reasoning": reasoning, "latency": round(time.monotonic() - t0, 4)}
