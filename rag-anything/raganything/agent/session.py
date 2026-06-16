"""Session 级工作记忆（spec §6）。纯内存 + TTL；dump/load 预留持久化。"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field

RECENT_TURNS_MAX = 6
ACTIVE_ENTITIES_MAX = 12
CACHE_MAX = 1000
TTL_SECONDS = 7200.0
MAX_SESSIONS = 256


def _now() -> float:
    return time.monotonic()


@dataclass
class SessionMemory:
    session_id: str
    workspace_id: str
    cache_max: int = CACHE_MAX
    active_entities: list[dict] = field(default_factory=list)
    recent_turns: list[dict] = field(default_factory=list)
    open_gaps: list[str] = field(default_factory=list)  # facts left unresolved last turn,
                                                        # so a "continue/refine" follow-up
                                                        # can be rewritten against them
    history_summary: str = ""
    chunk_cache: OrderedDict = field(default_factory=OrderedDict)  # chunk_id -> {content, file_path}
    plan_cache: dict = field(default_factory=dict)                 # 规范化 query -> PlanResult dict
    last_access: float = field(default_factory=_now)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_access = _now()

    def register_entities(self, entities: list[dict]) -> None:
        by_name = {e["name"]: e for e in self.active_entities}
        for e in entities:
            by_name[str(e.get("name", ""))] = {
                "name": str(e.get("name", "")), "note": str(e.get("note", "")),
                "last_turn": int(e.get("last_turn", 0)),
            }
        ranked = sorted(by_name.values(), key=lambda e: e["last_turn"], reverse=True)
        self.active_entities = ranked[:ACTIVE_ENTITIES_MAX]

    def add_turn(self, q: str, a: str, *, cancelled: bool = False) -> None:
        self.recent_turns.append({"q": q, "a": a, "cancelled": cancelled})
        if len(self.recent_turns) > RECENT_TURNS_MAX:
            self.recent_turns = self.recent_turns[-RECENT_TURNS_MAX:]

    def cache_chunks(self, chunks: list[dict]) -> None:
        for c in chunks:
            cid = str(c.get("chunk_id", ""))
            if not cid:
                continue
            self.chunk_cache[cid] = {"content": c.get("content", ""), "file_path": c.get("file_path", "")}
            self.chunk_cache.move_to_end(cid)
        while len(self.chunk_cache) > self.cache_max:
            self.chunk_cache.popitem(last=False)

    def get_cached(self, chunk_ids: list[str]) -> dict[str, dict]:
        out = {}
        for cid in chunk_ids:
            if cid in self.chunk_cache:
                self.chunk_cache.move_to_end(cid)
                out[cid] = self.chunk_cache[cid]
        return out

    def drop_chunks(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self.chunk_cache.pop(cid, None)

    def dump(self) -> dict:
        return {
            "session_id": self.session_id, "workspace_id": self.workspace_id,
            "active_entities": self.active_entities, "recent_turns": self.recent_turns,
            "open_gaps": self.open_gaps,
            "history_summary": self.history_summary, "chunk_cache": dict(self.chunk_cache),
        }

    @classmethod
    def load(cls, data: dict) -> "SessionMemory":
        s = cls(session_id=data["session_id"], workspace_id=data["workspace_id"])
        s.active_entities = list(data.get("active_entities", []))
        s.recent_turns = list(data.get("recent_turns", []))
        s.open_gaps = list(data.get("open_gaps", []))
        s.history_summary = str(data.get("history_summary", ""))
        s.chunk_cache = OrderedDict(data.get("chunk_cache", {}))
        return s


class SessionStore:
    def __init__(self, *, ttl_seconds: float = TTL_SECONDS, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[tuple[str, str], SessionMemory] = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_sessions

    def get(self, workspace_id: str, session_id: str) -> SessionMemory:
        key = (workspace_id, session_id)
        self.sweep()
        if key not in self._sessions:
            self._sessions[key] = SessionMemory(session_id=session_id, workspace_id=workspace_id)
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        self._sessions.move_to_end(key)
        s = self._sessions[key]
        s.touch()
        return s

    def sweep(self) -> None:
        cutoff = _now() - self._ttl
        for key in [k for k, s in self._sessions.items() if s.last_access < cutoff]:
            del self._sessions[key]

    def drop_chunks(self, workspace_id: str, chunk_ids: list[str]) -> None:
        for (ws, _), s in self._sessions.items():
            if ws == workspace_id:
                s.drop_chunks(chunk_ids)

    def invalidate_workspace(self, workspace_id: str) -> None:
        for key in [k for k in self._sessions if k[0] == workspace_id]:
            del self._sessions[key]
