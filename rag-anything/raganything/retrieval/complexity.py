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

- simple: Single-hop factual question; one retrieval pass is sufficient.
  Examples: "What does BERT stand for?", "What is the capital of France?"

- medium: Moderate depth; one entity or topic, may need one follow-up retrieval.
  Examples: "What are all the config options for the Redis module?",
            "Explain the attention mechanism in detail."

- complex: Multi-entity; requires decomposition, causal chains, or cross-document
  reasoning across multiple distinct entities.
  Examples: "How did the network partition in region A cause failures in region B?",
            "Compare the indexing strategies used by LightRAG and HippoRAG."

Rules:
- When unsure between simple and medium → choose medium.
- When unsure between medium and complex → choose medium.
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
