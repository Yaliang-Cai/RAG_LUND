"""v2 兼容 trace + agent 决策链（spec §13、§14.2 分层标记）。"""
from __future__ import annotations

from typing import Any


class TraceBuilder:
    def __init__(self, *, profile: str, query: str) -> None:
        self._trace: dict[str, Any] = {
            "profile": profile,
            "rewrite_history": [query],
            "retrieval_steps": [],
            "grader_events": [],
            "hallucination_events": [],
            "recovery_actions": [],
            "agent_decisions": [],
            "reclassify_events": [],
            "chunks_per_path": {},
            "paths_activated": [],
            "paths_failed": [],
        }

    def add_rewrite(self, query: str) -> None:
        self._trace["rewrite_history"].append(query)

    def add_retrieval_step(self, *, step_type: str, query: str, tool: str,
                           chunks: int, trace: dict, cycle: int) -> None:
        self._trace["retrieval_steps"].append({
            "type": step_type, "query": query, "profile": tool, "cycle": cycle,
            "chunks": chunks,
            "paths_activated": trace.get("paths_activated", []),
            "paths_failed": trace.get("paths_failed", []),
            "chunks_per_path": trace.get("chunks_per_path", {}),
        })
        self._trace["paths_activated"] = trace.get("paths_activated", [])
        self._trace["chunks_per_path"].update(trace.get("chunks_per_path", {}))

    def add_grader_event(self, payload: dict, *, cycle: int) -> None:
        self._trace["grader_events"].append({**payload, "cycle": cycle})

    def add_hallucination_event(self, payload: dict, *, cycle: int) -> None:
        self._trace["hallucination_events"].append({**payload, "cycle": cycle})

    def add_decision(self, *, thought: str, action: str, params: dict,
                     budget_snapshot: dict, fallback: bool) -> None:
        self._trace["agent_decisions"].append({
            "thought": thought, "action": action, "params": params,
            "budget": budget_snapshot, "fallback": fallback,
        })

    def add_reclassify(self, old: str, new: str, *, cycle: int) -> None:
        self._trace["reclassify_events"].append({"from": old, "to": new, "cycle": cycle})

    def build(self, *, terminal_reason: str, grounded: bool, **extra: Any) -> dict:
        return {
            **self._trace,
            "terminal_reason": terminal_reason,
            "grounded": grounded,
            "used_fallback": any(d["fallback"] for d in self._trace["agent_decisions"]),
            **extra,
        }
