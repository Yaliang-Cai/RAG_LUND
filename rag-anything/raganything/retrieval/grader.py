from __future__ import annotations
import json
import logging
from typing import Awaitable, Callable

from raganything.constants import DEFAULT_AGENTIC_GRADER_FALLBACK_SUFFICIENT

logger = logging.getLogger(__name__)

_CONTEXT_HEADER = "You are a RAG quality controller.\n\nContext:\n"

_GRADER_SUFFIX = """\
Question: {query}

Task:
1. Are the chunks above sufficient to accurately answer the question?
2. If not sufficient, is this because the information genuinely does not exist in these documents \
(unanswerable), rather than because retrieval was incomplete?

Output JSON: {{"sufficient": true|false, "unanswerable": true|false, "reason": "<one short sentence>"}}

Rules for "unanswerable":
- Set true ONLY when the available chunks explicitly state or clearly imply the requested \
information is not present in the indexed documents.
- Set false when chunks are empty or off-topic — that may simply mean retrieval needs improvement.
- When in doubt, set false (prefer retrying over giving up early).
"""


def build_shared_prefix(chunks: list[dict]) -> str:
    """Build the chunk-text prefix shared by grader, generator, and hallucination_check."""
    parts = [
        f"[{i + 1}] Source: {c.get('file_path', 'unknown')}\n{c.get('content', '')}"
        for i, c in enumerate(chunks)
    ]
    return _CONTEXT_HEADER + "\n\n---\n\n".join(parts) + "\n\n---\n\n"


class Grader:
    def __init__(
        self,
        llm_func: Callable[..., Awaitable[str]],
        fallback_sufficient: bool = DEFAULT_AGENTIC_GRADER_FALLBACK_SUFFICIENT,
    ) -> None:
        self._llm = llm_func
        self._fallback_sufficient = fallback_sufficient

    async def grade(self, query: str, chunks: list[dict]) -> dict:
        prefix = build_shared_prefix(chunks)
        prompt = prefix + _GRADER_SUFFIX.format(query=query)
        try:
            raw = await self._llm(prompt, response_format={"type": "json_object"})
            result = json.loads(raw)
            return {
                "sufficient": bool(result.get("sufficient", self._fallback_sufficient)),
                "unanswerable": bool(result.get("unanswerable", False)),
                "reason": str(result.get("reason", "")).strip(),
            }
        except Exception:
            logger.warning("Grader failed, fallback sufficient=%s", self._fallback_sufficient, exc_info=True)
            return {"sufficient": self._fallback_sufficient, "unanswerable": False, "reason": "grader error"}
