"""决策解析、四层归一化、重复守卫（spec §4.1/4.3）。"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from raganything.agent.tools import ToolRegistry

_ACTION_CUTOFF = 0.6


@dataclass
class Decision:
    thought: str
    action: str
    params: dict = field(default_factory=dict)
    stop: bool = False
    reclassify: str | None = None
    fallback: bool = False  # RecoveryPolicy 降级产生的决策标记（进 trace/评测分层）


def normalize_decision(raw: dict, registry: ToolRegistry, default_query: str) -> Decision:
    action = str(raw.get("action", "")).strip()
    if action not in registry.names():
        matches = difflib.get_close_matches(action, registry.names(), n=1, cutoff=_ACTION_CUTOFF)
        if not matches:
            raise ValueError(f"unknown action: {action!r}")
        action = matches[0]
    spec = registry.get(action)
    params = spec.clamp(dict(raw.get("params") or {}))
    if "query" in spec.params and not str(params.get("query", "")).strip():
        params["query"] = default_query
    if "expand" in params and str(params["expand"]) not in spec.allowed_expand:
        params["expand"] = "none"
    reclassify = raw.get("reclassify")
    return Decision(
        thought=str(raw.get("thought", ""))[:300],
        action=action,
        params=params,
        stop=(action == "answer"),
        reclassify=str(reclassify) if reclassify else None,
    )


def decision_signature(d: Decision) -> tuple:
    query = re.sub(r"\s+", "", str(d.params.get("query", "")))
    keys = tuple(sorted(
        (k, str(v)) for k, v in d.params.items() if k != "query"
    ))
    return (d.action, query, keys)
