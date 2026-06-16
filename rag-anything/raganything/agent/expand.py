"""Query-expansion strategies for the ``search`` tool: MQE + HyDE.

Single responsibility: turn one query into auxiliary query strings. These helpers
NEVER retrieve — the loop issues the extra retrievals into the shared evidence pool,
where dedup + cross-encoder rerank fuse the results. Expansion is therefore a *knob*
on ``search`` (a planner-set ``expand`` param), not a separate tool: the agent's
decision stays binary (search vs search_multihop). ``search_multihop`` is never
expanded — it keeps the clean entity-anchor seed (see loop._multihop_seed)."""
from __future__ import annotations

import logging
from typing import Any

from raganything.retrieval.json_utils import call_json_object

logger = logging.getLogger(__name__)

EXPAND_STRATEGIES = ("none", "mqe", "hyde")
DEFAULT_MQE_N = 3

_MQE_PROMPT = """\
Rewrite the user question into {n} alternative search queries that surface the same
answer from different angles (synonyms, sub-aspects, related phrasings). Keep proper
nouns, IDs and model numbers verbatim. Do not answer the question. Output JSON only:
{{"queries": ["...", "..."]}}

Question: {query}
"""

_HYDE_PROMPT = """\
Write a short, factual passage (3-5 sentences) that would directly answer the
question, as if copied from an authoritative source document. Do not hedge or say you
are unsure — write the ideal answer paragraph. This passage is used only as a
retrieval probe, not shown to the user.

Question: {query}
"""


async def mqe_queries(model_pool: Any, query: str, n: int = DEFAULT_MQE_N) -> list[str]:
    """Return up to *n* paraphrase / sub-aspect queries (excluding the original).

    Best-effort: returns ``[]`` on any failure so the caller still has the base
    query to retrieve with."""
    n = max(1, int(n))
    try:
        parsed = await call_json_object(
            lambda p, **kw: model_pool.call("rewriter", p, **kw),
            _MQE_PROMPT.format(n=n, query=query), max_tokens=256)
    except Exception:
        logger.warning("MQE expansion failed; using base query only", exc_info=True)
        return []
    out: list[str] = []
    for q in parsed.get("queries", []):
        s = str(q).strip()
        if s and s != query and s not in out:
            out.append(s)
    return out[:n]


async def hyde_query(model_pool: Any, query: str) -> str:
    """Return a hypothetical answer passage to use as a retrieval probe.

    Best-effort: returns ``""`` on failure so the caller falls back to the base
    query."""
    try:
        passage = await model_pool.call(
            "rewriter", _HYDE_PROMPT.format(query=query), max_tokens=256)
    except Exception:
        logger.warning("HyDE expansion failed; using base query only", exc_info=True)
        return ""
    return str(passage or "").strip()
