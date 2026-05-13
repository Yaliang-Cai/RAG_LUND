from __future__ import annotations

import logging
from typing import Awaitable, Callable

from .grader import build_shared_prefix
from .json_utils import load_json_object

logger = logging.getLogger(__name__)

_CHECKER_SUFFIX = """\
Answer: {answer}

Question being answered: {query}

For every factual claim in the Answer, verify it is explicitly supported by the Context above.
Only mark grounded=true if each factual claim is directly supported by the supplied chunks.
Statements such as "I cannot determine X from the context" make no factual claims and are grounded.

Output JSON only:
{{
  "grounded": true|false,
  "ungrounded_claims": ["<claim>", ...]
}}
"""


class HallucinationChecker:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def verify(self, query: str, answer: str, chunks: list[dict]) -> dict:
        prefix = build_shared_prefix(chunks)
        prompt = prefix + _CHECKER_SUFFIX.format(answer=answer, query=query)
        try:
            raw = await self._llm(prompt, response_format={"type": "json_object"})
            result = load_json_object(raw)
            claims = result.get("ungrounded_claims", [])
            if not isinstance(claims, list):
                claims = [claims] if claims else []
            return {
                "grounded": bool(result.get("grounded", False)),
                "ungrounded_claims": [str(c) for c in claims if str(c).strip()],
            }
        except Exception:
            logger.warning(
                "HallucinationChecker failed, defaulting grounded=False",
                exc_info=True,
            )
            return {
                "grounded": False,
                "ungrounded_claims": [],
                "check_status": "error",
            }
