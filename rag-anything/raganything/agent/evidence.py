"""证据池 + 事实账本（spec §5）。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# 与 query.py 的图片行格式一致（spec §11.2 入池时解析）
_IMAGE_PATH_RE = re.compile(r"Image Path:\s*([^\r\n]+)")


def _content_id(content: str) -> str:
    return "syn-" + hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class PoolEntry:
    chunk_id: str
    content: str
    file_path: str = ""
    modal_type: str = ""
    image_paths: list[str] = field(default_factory=list)
    canonical_score: float | None = None
    provenance: list[dict] = field(default_factory=list)
    hit_count: int = 1
    supports: set[str] = field(default_factory=set)

    def sort_key(self) -> tuple:
        return (self.canonical_score if self.canonical_score is not None else -1.0, self.hit_count)


class EvidencePool:
    def __init__(self, max_entries: int = 200) -> None:
        self.entries: dict[str, PoolEntry] = {}
        self.max_entries = max_entries
        self.last_dup_rate: float = 0.0

    def add(self, chunks: list[dict], *, step: int, tool: str, sub_query: str) -> list[PoolEntry]:
        """入池去重；返回需要 canonical rerank 的新条目（spec §5.1/5.2）。"""
        new_entries: list[PoolEntry] = []
        dups = 0
        for c in chunks:
            content = str(c.get("content") or "")
            cid = str(c.get("chunk_id") or c.get("id") or _content_id(content))
            prov = {"step": step, "tool": tool, "sub_query": sub_query,
                    "rrf_score": float(c.get("rrf_score") or c.get("score") or 0.0)}
            if cid in self.entries:
                self.entries[cid].provenance.append(prov)
                self.entries[cid].hit_count += 1
                dups += 1
                continue
            entry = PoolEntry(
                chunk_id=cid, content=content,
                file_path=str(c.get("file_path") or c.get("source") or ""),
                modal_type=str(c.get("modal_type") or ""),
                image_paths=_IMAGE_PATH_RE.findall(content),
                provenance=[prov],
            )
            self.entries[cid] = entry
            new_entries.append(entry)
        self.last_dup_rate = dups / len(chunks) if chunks else 0.0
        return new_entries

    def set_scores(self, scores: dict[str, float]) -> None:
        for cid, s in scores.items():
            if cid in self.entries:
                self.entries[cid].canonical_score = float(s)

    def evict(self) -> None:
        overflow = len(self.entries) - self.max_entries
        if overflow <= 0:
            return
        victims = sorted(
            (e for e in self.entries.values() if not e.supports),
            key=lambda e: e.sort_key(),
        )[:overflow]
        for v in victims:
            del self.entries[v.chunk_id]

    def top(self, n: int) -> list[PoolEntry]:
        return sorted(self.entries.values(), key=lambda e: e.sort_key(), reverse=True)[:n]

    def summary(self) -> dict[str, Any]:
        return {
            "chunks": len(self.entries),
            "scored": sum(1 for e in self.entries.values() if e.canonical_score is not None),
            "last_dup_rate": round(self.last_dup_rate, 2),
        }
