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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PoolEntry:
    chunk_id: str
    content: str
    file_path: str = ""
    modal_type: str = ""
    page_idx: int | None = None   # 0-based source page (LightRAG ingestion metadata)
    page_num: int | None = None   # 1-based source page, when present
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
                    "rrf_score": _safe_float(c.get("rrf_score") or c.get("score") or 0.0)}
            if cid in self.entries:
                self.entries[cid].provenance.append(prov)
                self.entries[cid].hit_count += 1
                dups += 1
                continue
            entry = PoolEntry(
                chunk_id=cid, content=content,
                file_path=str(c.get("file_path") or c.get("source") or ""),
                modal_type=str(c.get("modal_type") or ""),
                page_idx=_safe_int(c.get("page_idx")),
                page_num=_safe_int(c.get("page_num")),
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
        # 软上限：支撑 found fact 的条目按 spec §5.5 豁免淘汰。极端情况下（几乎全部
        # 条目被保护）池可暂超 max_entries，代价仅为少量内存，属故意行为。
        # sorted 升序 = 最低 canonical_score 优先被逐出；top() 则用 reverse=True。
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


class FactLedger:
    """事实账本：found/missing/unverifiable + 有效 coverage（spec §5.3）。"""

    GIVE_UP_DISTINCT_TOOLS = 2

    def __init__(self) -> None:
        self.facts: dict[str, dict] = {}

    def update(self, payload: dict) -> None:
        for f in payload.get("facts", []):
            text = str(f.get("text", ""))
            # 无 id 时用文本哈希作回退：避免与 grader 自供的 "fN" 撞键导致静默覆盖，
            # 且同文本事实跨周期得到稳定 id（与 EvidencePool 内容寻址同风格）。
            fid = str(f.get("id") or "") or "fx-" + hashlib.sha1(
                text.encode("utf-8", errors="replace")).hexdigest()[:8]
            existing = self.facts.get(fid)
            status = str(f.get("status", "missing"))
            if existing and existing["status"] == "unverifiable":
                status = "unverifiable"  # 已放弃事实不被 grader 复活
            attempts = existing["attempts"] if existing else set()
            self.facts[fid] = {
                "id": fid, "text": text, "status": status,
                # chunks 整体替换而非合并：grader 每周期重推导完整支撑列表（约定不变量）
                "chunks": [str(c) for c in f.get("chunks", [])], "attempts": attempts,
            }

    def record_attempt(self, fact_id: str, tool: str) -> None:
        f = self.facts.get(fact_id)
        if not f or f["status"] != "missing":
            return
        f["attempts"].add(tool)
        if len(f["attempts"]) >= self.GIVE_UP_DISTINCT_TOOLS:
            f["status"] = "unverifiable"

    @property
    def coverage(self) -> float:
        effective = [f for f in self.facts.values() if f["status"] != "unverifiable"]
        if not effective:
            return 0.0
        return sum(1 for f in effective if f["status"] == "found") / len(effective)

    def missing(self) -> list[dict]:
        return [f for f in self.facts.values() if f["status"] == "missing"]

    def unverifiable(self) -> list[dict]:
        return [f for f in self.facts.values() if f["status"] == "unverifiable"]

    def supported_chunks(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for f in self.facts.values():
            if f["status"] == "found":
                for cid in f["chunks"]:
                    out.setdefault(cid, set()).add(f["id"])
        return out

    def to_dict(self) -> dict:
        return {
            "coverage": round(self.coverage, 3),
            "facts": [{**f, "attempts": sorted(f["attempts"])} for f in self.facts.values()],
        }
