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
