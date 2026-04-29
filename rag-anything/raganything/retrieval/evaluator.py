# raganything/retrieval/evaluator.py
from __future__ import annotations
import json
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_PROMPT = """\
Evaluate whether the generated answer adequately addresses the original question.

Original question: {query}

Generated answer:
{answer}

Score 0.0-1.0:
  0.9-1.0  Complete, accurate, well-supported
  0.7-0.9  Mostly complete, minor gaps
  0.5-0.7  Partial, notable gaps
  0.0-0.5  Incomplete or off-topic

If score < 0.7, describe in one sentence what specific information is missing.

Output JSON: {{"score": <float>, "gap": "<missing info or empty string>"}}
"""


class AnswerEvaluator:
    def __init__(self, llm_func: Callable[..., Awaitable[str]]) -> None:
        self._llm = llm_func

    async def evaluate(self, query: str, answer: str) -> dict[str, Any]:
        """Return dict with keys: score (float 0-1), gap (str)."""
        try:
            raw = await self._llm(
                _PROMPT.format(query=query, answer=answer),
                response_format={"type": "json_object"},
            )
            result = json.loads(raw)
            score = float(result.get("score", 0.5))
            gap = str(result.get("gap", "")).strip()
            return {"score": max(0.0, min(1.0, score)), "gap": gap}
        except Exception:
            logger.warning("AnswerEvaluator failed, returning default score", exc_info=True)
            return {"score": 0.5, "gap": "Evaluation failed; please verify the answer."}
