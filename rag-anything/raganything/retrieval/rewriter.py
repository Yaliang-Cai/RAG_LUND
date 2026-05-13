from __future__ import annotations
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

_REWRITER_PROMPT = """\
The following query did not retrieve sufficient evidence.

Original query: {query}
Retrieval feedback: {reason}

Rewrite the query to improve retrieval. Strategies:
- Replace ambiguous terms with synonyms
- Add explicit domain context
- Decompose compound noun phrases

Output the rewritten query only. No explanation, no quotation marks.
"""

_REWRITER_V2_PROMPT = """\
The following query did not retrieve sufficient evidence.

Original query: {query}
Failure type: {failure_type}
Retrieval feedback: {reason}
Found facts: {found_facts}
Missing facts: {missing_facts}

Rewrite the query to improve retrieval for the missing facts.
Rules:
- Preserve all entities and constraints from the original question.
- Add explicit bridge entities, relations, or document context only when implied.
- Do not answer the question.

Output the rewritten query only. No explanation, no quotation marks.
"""


class Rewriter:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def rewrite(self, query: str, reason: str) -> str:
        prompt = _REWRITER_PROMPT.format(query=query, reason=reason)
        try:
            raw = await self._llm(prompt)
            result = (raw if isinstance(raw, str) else str(raw)).strip()
            return result if result else query
        except Exception:
            logger.warning("Rewriter failed, returning original query", exc_info=True)
            return query

    async def rewrite_with_feedback(
        self,
        query: str,
        *,
        failure_type: str,
        reason: str,
        found_facts: list[str] | None = None,
        missing_facts: list[str] | None = None,
    ) -> str:
        prompt = _REWRITER_V2_PROMPT.format(
            query=query,
            failure_type=failure_type,
            reason=reason,
            found_facts="; ".join(found_facts or []) or "none",
            missing_facts="; ".join(missing_facts or []) or "unknown",
        )
        try:
            raw = await self._llm(prompt)
            result = (raw if isinstance(raw, str) else str(raw)).strip()
            return result if result else query
        except Exception:
            logger.warning("Rewriter v2 failed, returning original query", exc_info=True)
            return query
