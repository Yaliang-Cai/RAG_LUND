from __future__ import annotations

import logging
from typing import Awaitable, Callable

from .grader import build_shared_prefix
from .json_utils import call_json_object

logger = logging.getLogger(__name__)

FAILURE_TYPES = {
    "empty",
    "off_topic",
    "missing_entity",
    "missing_relation",
    "needs_decomposition",
    "partial_evidence",
    "ambiguous_query",
    "unanswerable_candidate",
}

_GRADER_V2_SUFFIX = """\
Question: {query}

Task:
Evaluate whether the chunks above contain enough evidence to answer the exact question.

Output one compact JSON object only:
{{
  "sufficient": true|false,
  "unanswerable": true|false,
  "failure_type": "empty|off_topic|missing_entity|missing_relation|needs_decomposition|partial_evidence|ambiguous_query|unanswerable_candidate",
  "coverage_score": 0.0,
  "found_facts": ["<fact>", ...],
  "missing_facts": ["<fact>", ...],
  "reason": "<one short sentence>"
}}

Rules:
- coverage_score is the fraction of the facts needed to answer the question that are supported by the chunks.
- coverage_score is not topical relevance. Entity mentions alone are low coverage when bridge facts are missing.
- For multi-hop, comparison, or causal questions, every bridge fact must be present for sufficient=true.
- If sufficient=true, coverage_score should be high, missing_facts should be empty, and unanswerable must be false.
- Use missing_relation when entities are present but a required relation, bridge, comparison, or causal link is missing.
- Use needs_decomposition when the question should be split into independent sub-questions before retrieval.
- Use empty when there are no useful chunks, and off_topic when chunks are unrelated.
- Use unanswerable_candidate only when the chunks explicitly state or strongly imply the indexed documents do not contain the answer.
- unanswerable is only a candidate signal here, not a stopping decision. Do not use it for merely weak retrieval.
- Empty, off-topic, or weak retrieval means retrieval needs improvement; do not mark that as truly unanswerable.
- When in doubt, set unanswerable=false and choose the closest retrieval failure type.
- found_facts and missing_facts have no item-count limit, but every item must be short, atomic, and necessary for judging answer coverage.
- Do not copy chunk text or write long explanations in facts. Record only the facts needed or missing for answering the question.
- Return valid JSON only. Do not include Markdown, comments, or prose outside the JSON object.
"""


class GraderV2:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def grade(self, query: str, chunks: list[dict]) -> dict:
        prompt = build_shared_prefix(chunks) + _GRADER_V2_SUFFIX.format(query=query)
        try:
            result = await call_json_object(self._llm, prompt, max_tokens=1536)
            return _normalize_result(result, has_chunks=bool(chunks))
        except Exception:
            logger.warning("GraderV2 failed, defaulting insufficient", exc_info=True)
            return _fallback_result(has_chunks=bool(chunks))


def _normalize_result(result: dict, *, has_chunks: bool) -> dict:
    if not has_chunks:
        return {
            "sufficient": False,
            "unanswerable": False,
            "failure_type": "empty",
            "coverage_score": 0.0,
            "found_facts": [],
            "missing_facts": _string_list(result.get("missing_facts", [])),
            "reason": str(result.get("reason", "")).strip() or "no chunks retrieved",
        }

    sufficient = bool(result.get("sufficient", False))
    unanswerable = bool(result.get("unanswerable", False)) and not sufficient
    failure_type = str(result.get("failure_type") or "").strip()
    if failure_type not in FAILURE_TYPES:
        failure_type = "partial_evidence"
    coverage_score = _clamp_float(result.get("coverage_score", 0.0))
    if sufficient:
        coverage_score = max(coverage_score, 0.85)
    return {
        "sufficient": sufficient,
        "unanswerable": unanswerable,
        "failure_type": failure_type,
        "coverage_score": coverage_score,
        "found_facts": _string_list(result.get("found_facts", [])),
        "missing_facts": _string_list(result.get("missing_facts", [])),
        "reason": str(result.get("reason", "")).strip(),
    }


def _fallback_result(*, has_chunks: bool) -> dict:
    return {
        "sufficient": False,
        "unanswerable": False,
        "failure_type": "partial_evidence" if has_chunks else "empty",
        "coverage_score": 0.0,
        "found_facts": [],
        "missing_facts": [],
        "reason": "grader error",
        "grader_error": True,
    }


def _clamp_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
