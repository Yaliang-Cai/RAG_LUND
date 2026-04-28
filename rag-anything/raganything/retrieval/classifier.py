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

Available profiles (ordered from narrow to broad):

- precise: Query contains hard constraints that require exact lexical matching.
  Signals: specific IDs, error codes, version numbers, rare proper nouns, abbreviations.
  Examples: "What is the impact scope of CVE-2026-001?"
            "Status of order ID ORD-20260424-8821"
            "What does the acronym MTTR stand for in this document?"

- semantic: Default workhorse for everyday knowledge queries. No graph traversal needed.
  Signals: factual questions, process/procedure explanations, concept definitions, summaries.
           Single topic, no multi-entity reasoning, no rare terminology.
  Examples: "What is the company leave policy?"
            "How does the attention mechanism work?"
            "Summarise the key findings of this report."

- local: Query is tightly focused on ONE specific entity and asks about its direct
  properties, relationships, or current state. Does not require cross-document hops.
  Signals: "What are the [attributes/dependencies/upstream-downstream] of X?"
  Examples: "What are the upstream systems of the payment service?"
            "What configuration parameters does the Redis cluster expose?"

- multihop: Query involves MULTIPLE distinct entities and requires chaining logic
  across documents — comparisons, causal chains, impact analysis.
  Signals: two or more named entities, causal language ("led to", "caused", "resulted in"),
           comparative language ("difference between", "how does A affect B").
  Examples: "How did the network partition in region A eventually cause the order service in region B to fail?"
            "Compare the indexing strategies used by LightRAG and HippoRAG2."

- full: Use ONLY when the query is genuinely ambiguous or the intent cannot be
  determined with confidence. Do not use as a safe default.

Key disambiguation rules:
- If the query asks about one entity → prefer local over multihop.
- If the query is a plain question with no entity graph needed → prefer semantic over local.
- If precise terminology appears alongside reasoning → prefer multihop or local, not precise.
- Reserve full for true ambiguity, not for difficult classification.

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
