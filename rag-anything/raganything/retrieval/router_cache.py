from __future__ import annotations
import hashlib
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1


class RouterCache:
    """
    Process-local LRU cache for query → profile decisions.

    Tri-state outcome: "unknown" → "success" | "failed"
    Evicts entries that fail >= 3 times.
    """

    def __init__(self, maxsize: int = 2048, prompt_hash: str = "") -> None:
        self._maxsize = maxsize
        self._prompt_hash = prompt_hash
        self._store: OrderedDict[str, dict] = OrderedDict()

    def _key(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        raw = f"v{_CACHE_VERSION}:{self._prompt_hash}:{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[dict]:
        key = self._key(query)
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, query: str, profile: str) -> None:
        key = self._key(query)
        self._store[key] = {"profile": profile, "outcome": "unknown", "fail_count": 0}
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def mark_success(self, query: str) -> None:
        key = self._key(query)
        if key in self._store:
            self._store[key]["outcome"] = "success"

    def mark_failed(self, query: str) -> None:
        key = self._key(query)
        if key not in self._store:
            return
        entry = self._store[key]
        entry["fail_count"] += 1
        if entry["fail_count"] >= 2:
            entry["outcome"] = "failed"
        if entry["fail_count"] >= 3:
            del self._store[key]
            logger.debug("RouterCache: evicted %r after 3 failures", query[:60])

    def get_avoid_profiles(self, query: str) -> list[str]:
        entry = self.get(query)
        if entry and entry.get("outcome") == "failed":
            return [entry["profile"]]
        return []
