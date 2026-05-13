import logging
import re
import time
from typing import Any, Awaitable, Callable

from raganything.constants import DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE
from .json_utils import call_json_object

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6
_ROUTER_PROFILES = {"precise", "semantic", "local", "global", "multihop"}

_CLASSIFIER_PROMPT = """\
You are a retrieval routing classifier. Select exactly one retrieval profile for
the user query. Output JSON only.

Available profiles:

- precise: Exact lexical matching over chunks.
  Use for IDs, error codes, CVEs, order numbers, version numbers, exact titles,
  code names, table names, quoted strings, or rare strings that must match text.
  A person, organization, or place name alone is NOT precise.

- semantic: Default dense+sparse chunk retrieval for ordinary RAG questions.
  Use when the answer is likely in semantically similar passages and graph
  structure is not clearly required.

- local: KG entity-neighbour retrieval.
  Use only for one focal entity and its direct attributes, direct neighbours, or
  direct relationships.

- global: KG edge/relation retrieval.
  Use for relationship, event, theme, participation, interaction, acquisition,
  affiliation, dependency, or other relation-centric questions where the edge is
  more important than one focal entity.

- multihop: PPR graph retrieval.
  Use for bridge, composition, comparison, causal chains, shared intermediate
  entities, or cross-document reasoning. Multiple names alone are not enough;
  the query must require combining facts.

Disambiguation rules:
- Do not use local/global/multihop unless graph structure is clearly useful.
- Prefer semantic for ordinary factual, definition, summary, and procedure
  questions, even if they mention a named entity.
- Prefer local over multihop for one entity and one direct property/relation.
- Prefer global over local when the query is about a relation/event type rather
  than a single entity.
- Prefer multihop when the answer requires connecting two or more facts.

Output format:
{{"reasoning": "<one short sentence>", "profile": "<name>", "confidence": <0.0-1.0>}}
{avoid_instruction}
Query: {query}
"""

_AVOID_INSTRUCTION = """
Do not output any of these profiles because they were already tried and failed:
{avoid_list}
"""

_EXACT_PATTERNS = [
    re.compile(r"\bCVE-\d{4}-\d+\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,}[-_]\d[\w.-]*\b"),
    re.compile(r"\b\d+(?:\.\d+){1,}\b"),
    re.compile(r"['\"`][^'\"`]{3,}['\"`]"),
    re.compile(r"\b(?:id|code|error|ticket|order|version|commit|sha|hash)\b", re.IGNORECASE),
]

_MULTIHOP_TERMS = re.compile(
    r"\b(compare|comparison|connect|connected|bridge|between|both|shared|common|cause|caused|causal|"
    r"lead to|led to|influence|relationship between|how did|why did)\b",
    re.IGNORECASE,
)

_GLOBAL_TERMS = re.compile(
    r"\b(event|events|relationship|relationships|relation|relations|participated|participation|"
    r"interaction|interactions|acquisition|acquired|affiliation|dependency|dependencies|supplier|"
    r"suppliers|partnership|collaboration)\b",
    re.IGNORECASE,
)


class QueryClassifier:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]):
        self._llm = llm_func

    async def classify(
        self, query: str, avoid: list[str] | None = None
    ) -> tuple[str, dict[str, Any]]:
        t0 = time.monotonic()
        fallback = DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE
        candidate: str | None = None
        profile = fallback
        confidence = 0.0
        reasoning = ""
        fallback_used = False
        fallback_reason = ""
        avoid = avoid or []
        avoid_instruction = (
            _AVOID_INSTRUCTION.format(avoid_list=", ".join(avoid)) if avoid else ""
        )
        try:
            prompt = _CLASSIFIER_PROMPT.format(
                query=query, avoid_instruction=avoid_instruction
            )
            result = await call_json_object(self._llm, prompt, max_tokens=256)
            candidate = str(result.get("profile", fallback)).strip()
            confidence = float(result.get("confidence", 0.0))
            reasoning = str(result.get("reasoning", ""))

            valid = (
                candidate in _ROUTER_PROFILES
                and candidate not in avoid
                and confidence >= _CONFIDENCE_THRESHOLD
                and not (candidate == "precise" and not _has_exact_signal(query))
            )
            if valid:
                profile = candidate
            elif candidate in _ROUTER_PROFILES and confidence < _CONFIDENCE_THRESHOLD:
                profile = _low_confidence_fallback(query, avoid, fallback)
                fallback_reason = "low_confidence"
            elif candidate in avoid:
                profile = _first_allowed(fallback, avoid)
                fallback_reason = "avoided_profile"
            elif candidate == "precise" and not _has_exact_signal(query):
                profile = _first_allowed(fallback, avoid)
                fallback_reason = "weak_exact_signal"
            else:
                profile = _first_allowed(fallback, avoid)
                fallback_reason = "invalid_profile"
            if not valid:
                fallback_used = True
                logger.warning(
                    "Classifier fallback: profile=%r conf=%.2f avoid=%r -> %r",
                    candidate,
                    confidence,
                    avoid,
                    profile,
                )
        except Exception:
            logger.warning("Classifier failed, fallback to %r", fallback, exc_info=True)
            profile = _first_allowed(fallback, avoid)
            fallback_used = True
            fallback_reason = "exception"

        latency = time.monotonic() - t0
        return profile, {
            "confidence": confidence,
            "reasoning": reasoning,
            "latency": round(latency, 4),
            "candidate_profile": candidate,
            "selected_profile": profile,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }


def _has_exact_signal(query: str) -> bool:
    return any(pattern.search(query) for pattern in _EXACT_PATTERNS)


def _low_confidence_fallback(query: str, avoid: list[str], default: str) -> str:
    if _has_exact_signal(query):
        return _first_allowed("precise", avoid)
    if _MULTIHOP_TERMS.search(query):
        return _first_allowed("multihop", avoid)
    if _GLOBAL_TERMS.search(query):
        return _first_allowed("global", avoid)
    return _first_allowed(default, avoid)


def _first_allowed(preferred: str, avoid: list[str]) -> str:
    if preferred in _ROUTER_PROFILES and preferred not in avoid:
        return preferred
    if "semantic" not in avoid:
        return "semantic"
    for profile in sorted(_ROUTER_PROFILES):
        if profile not in avoid:
            return profile
    return DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE
