# raganything/retrieval/classifier.py
import json
import logging
import time
from typing import Any, Awaitable, Callable

from .profiles import PROFILE_REGISTRY

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6

_CLASSIFIER_PROMPT = """\
You are a retrieval routing classifier. Given a user query, select the most
appropriate retrieval profile from the list below.

Available profiles and typical examples:

- precise: Exact character-level match queries (error codes, IDs, rare proper nouns)
  Examples: "What is the impact scope of CVE-2026-001?"
            "Status of order ID ORD-20260424-8821"

- local: Direct query targeting a specific entity or clear single-hop fact
  Examples: "How many parameters does BERT have?"
            "What are the architectural differences between BERT and GPT?"
            "When should you use RAG vs fine-tuning?"

- multihop: Chain reasoning across multiple entities or documents
  Examples: "What other papers have been published by the authors cited in HippoRAG2?"
            "Which components of LightRAG were influenced by HippoRAG2?"

- descriptive: Open-ended question requiring broad, complete context
  Examples: "Describe the overall architecture of LightRAG."
            "Provide a survey of PPR algorithms used in RAG systems."

- full: Fallback when query type is unclear or ambiguous

First briefly state your reasoning in one sentence, then output JSON.
Output format: {{"reasoning": "...", "profile": "<name>", "confidence": <0.0-1.0>}}

Query: {query}
"""


class QueryClassifier:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]):
        self._llm = llm_func

    async def classify(self, query: str) -> tuple[str, dict[str, Any]]:
        """Classify query into a profile name.

        Returns:
            (profile_name, metadata) where metadata contains confidence,
            reasoning, and latency (seconds).
        """
        t0 = time.monotonic()
        profile = "full"
        confidence = 0.0
        reasoning = ""
        try:
            prompt = _CLASSIFIER_PROMPT.format(query=query)
            raw = await self._llm(
                prompt,
                response_format={"type": "json_object"},
            )
            result = json.loads(raw)
            profile = str(result.get("profile", "full")).strip()
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", ""))
            if confidence < _CONFIDENCE_THRESHOLD or profile not in PROFILE_REGISTRY:
                logger.warning(
                    "Classifier fallback: profile=%r confidence=%.2f → 'full'",
                    profile,
                    confidence,
                )
                profile = "full"
        except Exception:
            logger.warning("Classifier output parse failed, fallback to 'full'", exc_info=True)
            profile = "full"
        latency = time.monotonic() - t0
        return profile, {
            "confidence": confidence,
            "reasoning": reasoning,
            "latency": round(latency, 4),
        }
