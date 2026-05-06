import json
import logging
import time
from typing import Any, Awaitable, Callable

from raganything.constants import DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE
from .profiles import PROFILE_REGISTRY

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6
_ROUTER_PROFILES = {"precise", "semantic", "local", "multihop"}  # full excluded

_CLASSIFIER_PROMPT = """\
You are a retrieval routing classifier. Given a user query, select the most
appropriate retrieval profile from the list below.

Available profiles (ordered from narrow to broad):

- precise: Query contains hard constraints that require exact lexical matching.
  Signals: specific IDs, error codes, version numbers, rare proper nouns, abbreviations.
  Examples: "What is the impact scope of CVE-2026-001?"
            "Status of order ID ORD-20260424-8821"

- semantic: Default workhorse for everyday knowledge queries. No graph traversal needed.
  Signals: factual questions, process/procedure explanations, concept definitions, summaries.
           Single topic, no multi-entity reasoning.
  Examples: "What is the company leave policy?"
            "How does the attention mechanism work?"

- local: Query is tightly focused on ONE specific entity and its direct properties or relationships.
  Signals: "What are the [attributes/dependencies] of X?"
  Examples: "What are the upstream systems of the payment service?"

- multihop: Query involves MULTIPLE distinct entities requiring cross-document reasoning.
  Signals: two or more named entities, causal/comparative language.
  Examples: "How did the network partition in region A cause failures in region B?"
            "Compare the indexing strategies used by LightRAG and HippoRAG2."

Key disambiguation rules:
- If the query asks about one entity → prefer local over multihop.
- If no entity graph is needed → prefer semantic over local.
- When genuinely unsure → choose semantic (it is the safe default).{avoid_instruction}

First briefly state your reasoning in one sentence, then output JSON.
Output format: {{"reasoning": "...", "profile": "<name>", "confidence": <0.0-1.0>}}

Query: {query}
"""

_AVOID_INSTRUCTION = """
- Do NOT output any of these profiles (already tried and failed): {avoid_list}
"""


class QueryClassifier:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]):
        self._llm = llm_func

    async def classify(
        self, query: str, avoid: list[str] | None = None
    ) -> tuple[str, dict[str, Any]]:
        t0 = time.monotonic()
        fallback = DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE
        profile = fallback
        confidence = 0.0
        reasoning = ""
        avoid = avoid or []
        avoid_instruction = (
            _AVOID_INSTRUCTION.format(avoid_list=", ".join(avoid)) if avoid else ""
        )
        try:
            prompt = _CLASSIFIER_PROMPT.format(
                query=query, avoid_instruction=avoid_instruction
            )
            raw = await self._llm(prompt, response_format={"type": "json_object"})
            result = json.loads(raw)
            candidate = str(result.get("profile", fallback)).strip()
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", ""))

            valid = (
                candidate in _ROUTER_PROFILES
                and candidate not in avoid
                and confidence >= _CONFIDENCE_THRESHOLD
            )
            profile = candidate if valid else fallback
            if not valid:
                logger.warning(
                    "Classifier fallback: profile=%r conf=%.2f avoid=%r -> %r",
                    candidate, confidence, avoid, fallback,
                )
        except Exception:
            logger.warning("Classifier failed, fallback to %r", fallback, exc_info=True)
            profile = fallback

        latency = time.monotonic() - t0
        return profile, {"confidence": confidence, "reasoning": reasoning, "latency": round(latency, 4)}
