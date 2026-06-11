# Agentic RAG v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LLM 驱动的决策循环（证据池 + 事实账本 + 成本点预算 + session 记忆）替代 v2 固定状态机，新增 `/agent/chat` 端点，v2 保留为降级 fallback。

**Architecture:** 新建 `raganything/agent/` 包：纯函数/数据结构模块（budget/evidence/citations/decision/session/trace）先行，依赖注入的服务模块（models/tools/planner/grading/generate）居中，`loop.py` 最后集成，`server/app.py` 暴露端点。所有检索通过包装现有 `retrieval/router.py` 的工具执行，rerank 从 per-path 移到池准入口。

**Tech Stack:** Python 3.10+ asyncio、pytest + pytest-asyncio、现有 LightRAG/RetrievalRouter/RecoveryPolicy/json_utils、vLLM OpenAI 兼容端点。

**Spec:** `docs/superpowers/specs/2026-06-11-agentic-rag-v3-design.md`（任务中以 §N 引用）

**执行约定**
- 工作目录：`rag-anything/`（仓库根的子目录）。测试命令均在此目录运行。
- 测试装饰器统一用 `@pytest.mark.asyncio`（项目已用 pytest-asyncio）。
- 每个 Task 一次 commit。LLM 调用一律通过注入的 async callable，单测中用 fake 函数替身，不连真端点。
- 范围外（后续独立计划）：A/B 评测脚本接入 `evaluate_local/`、前端 stop 按钮、PPR 预热挂载（§7.3 第 1 点）。本计划交付可运行可测试的 `/agent/chat`。

---

## File Structure

```
raganything/agent/
├── __init__.py      # 空包标记（Task 1 创建）
├── budget.py        # Budget：成本点 + 双护栏 + 改判升档（§8）
├── models.py        # ModelPool：角色→端点 + 回退/熔断（§12）
├── evidence.py      # PoolEntry/EvidencePool/FactLedger（§5）
├── citations.py     # 声明拆分 + 引文代码裁决（§12.2）
├── session.py       # SessionMemory/SessionStore（§6）
├── decision.py      # 决策归一化 + 重复守卫（§4.3）
├── tools.py         # ToolSpec/Registry + agent 检索 profile + MQE/HyDE + inspect_image（§7）
├── planner.py       # 改写+分类合并调用 + 画像预设 + 快速通道判定（§9）
├── grading.py       # LedgerGrader 增量评估 + 条件终审（§5.3/5.4）
├── generate.py      # 装填器 + direct/map_reduce/cot_reflect（§10/11.2）
├── trace.py         # v2 兼容 trace（§13）
└── loop.py          # AgentLoop 主循环 + 取消 + RecoveryPolicy 降级（§4/6.5）
server/app.py        # /agent/chat、/agent/sessions/{id}/cancel、409（§6.4/6.5）
tests/agent/         # 各模块单测，文件名 test_<module>.py
```

---

### Task 1: Budget（成本点 + 双护栏）

**Files:**
- Create: `raganything/agent/__init__.py`（空文件）
- Create: `raganything/agent/budget.py`
- Test: `tests/agent/__init__.py`（空文件）、`tests/agent/test_budget.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_budget.py
import time
from raganything.agent.budget import Budget, ARCHETYPE_POINTS


def test_archetype_points_table():
    assert ARCHETYPE_POINTS == {
        "factoid": 6, "comparison": 10, "summary": 12, "multihop": 16, "unknown": 10,
    }


def test_charge_and_exhaustion():
    b = Budget.for_archetype("factoid", max_tokens=100, max_seconds=None)
    assert b.exhausted() is None
    b.charge(points=6)
    assert b.exhausted() == "points"
    b2 = Budget.for_archetype("factoid", max_tokens=100, max_seconds=None)
    b2.charge(tokens=100)
    assert b2.exhausted() == "tokens"


def test_wall_clock_guardrail_optional():
    b = Budget(points=5, max_seconds=0.01)
    time.sleep(0.02)
    assert b.exhausted() == "wall_clock"
    b2 = Budget(points=5, max_seconds=None)  # 评测模式关护栏 §8.1
    assert b2.exhausted() is None


def test_low_soft_threshold():
    b = Budget(points=10, max_seconds=None)
    b.charge(points=8.5)
    assert b.low() is True  # 剩 1.5/10 ≤ 20%


def test_upgrade_once_only():
    b = Budget.for_archetype("factoid", max_seconds=None)
    assert b.upgrade("multihop") is True
    assert b.points == ARCHETYPE_POINTS["multihop"]
    assert b.upgrade("summary") is False  # 每轮最多升一次 §9.3


def test_snapshot_keys():
    snap = Budget(points=10).snapshot()
    assert set(snap) >= {"remaining_points", "spent_tokens", "elapsed_seconds"}
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_budget.py -v`
Expected: FAIL（ModuleNotFoundError: raganything.agent.budget）

- [ ] **Step 3: 实现**

```python
# raganything/agent/budget.py
"""成本点预算 + 墙钟/token 双护栏（spec §8）。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

ARCHETYPE_POINTS: dict[str, int] = {
    "factoid": 6, "comparison": 10, "summary": 12, "multihop": 16, "unknown": 10,
}
DEFAULT_MAX_TOKENS = 30_000
DEFAULT_MAX_SECONDS = 60.0
SOFT_RATIO = 0.2


@dataclass
class Budget:
    points: float
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_seconds: float | None = DEFAULT_MAX_SECONDS
    spent_points: float = 0.0
    spent_tokens: int = 0
    _upgraded: bool = False
    _start: float = field(default_factory=time.monotonic)

    @classmethod
    def for_archetype(cls, archetype: str, **kwargs) -> "Budget":
        return cls(points=ARCHETYPE_POINTS.get(archetype, ARCHETYPE_POINTS["unknown"]), **kwargs)

    def charge(self, *, points: float = 0.0, tokens: int = 0) -> None:
        self.spent_points += points
        self.spent_tokens += tokens

    @property
    def remaining_points(self) -> float:
        return self.points - self.spent_points

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def exhausted(self) -> str | None:
        if self.remaining_points <= 0:
            return "points"
        if self.spent_tokens >= self.max_tokens:
            return "tokens"
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            return "wall_clock"
        return None

    def low(self) -> bool:
        if self.remaining_points <= self.points * SOFT_RATIO:
            return True
        if self.max_seconds is not None and self.elapsed >= self.max_seconds * (1 - SOFT_RATIO * 1.25):
            return True  # 60s 护栏 → 45s 软阈值
        return False

    def upgrade(self, archetype: str) -> bool:
        """改判升档：补到新画像额度，每轮一次（spec §9.3）。"""
        if self._upgraded:
            return False
        self._upgraded = True
        self.points = max(self.points, ARCHETYPE_POINTS.get(archetype, self.points))
        return True

    def snapshot(self) -> dict:
        return {
            "remaining_points": round(self.remaining_points, 2),
            "spent_points": round(self.spent_points, 2),
            "spent_tokens": self.spent_tokens,
            "elapsed_seconds": round(self.elapsed, 2),
            "low": self.low(),
        }
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_budget.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/__init__.py raganything/agent/budget.py tests/agent/
git commit -m "feat(agent): cost-point budget with wall-clock/token guardrails"
```

---

### Task 2: ModelPool（角色分离 + 回退熔断）

**Files:**
- Create: `raganything/agent/models.py`
- Test: `tests/agent/test_models.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_models.py
import pytest
from raganything.agent.models import ModelPool, JUDGE_ROLES


def make_fn(name, fail=False):
    calls = []
    async def fn(prompt, **kw):
        calls.append(prompt)
        if fail:
            raise ConnectionError("down")
        return f"{name}-reply"
    fn.calls = calls
    return fn


@pytest.mark.asyncio
async def test_default_all_roles_to_main():
    main = make_fn("main")
    pool = ModelPool(main_func=main)
    assert await pool.call("grader", "p") == "main-reply"
    assert await pool.call("generator", "p") == "main-reply"


@pytest.mark.asyncio
async def test_judge_roles_routed_to_judge():
    main, judge = make_fn("main"), make_fn("judge")
    pool = ModelPool(main_func=main, judge_func=judge)
    assert await pool.call("checker", "p") == "judge-reply"
    assert await pool.call("planner", "p") == "main-reply"  # planner 留大模型 §12.1


@pytest.mark.asyncio
async def test_per_call_fallback_and_breaker():
    main, judge = make_fn("main"), make_fn("judge", fail=True)
    pool = ModelPool(main_func=main, judge_func=judge, breaker_threshold=2, probe_interval=999)
    assert await pool.call("grader", "p") == "main-reply"  # 单次回退 §12.3
    assert await pool.call("grader", "p") == "main-reply"
    assert pool.breaker_open is True  # 连续 2 次失败熔断
    await pool.call("grader", "p")
    assert len(judge.calls) == 2  # 熔断后不再打 judge
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_models.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/models.py
"""角色→端点映射 + 回退/熔断（spec §12）。"""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

LLMFunc = Callable[..., Awaitable[str]]
JUDGE_ROLES = frozenset({"grader", "checker", "rewriter", "summarizer"})
ALL_ROLES = JUDGE_ROLES | {"planner", "generator"}


class ModelPool:
    def __init__(
        self,
        main_func: LLMFunc,
        judge_func: LLMFunc | None = None,
        *,
        breaker_threshold: int = 5,
        probe_interval: float = 60.0,
    ) -> None:
        self._main = main_func
        self._judge = judge_func
        self._threshold = breaker_threshold
        self._probe_interval = probe_interval
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def breaker_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._probe_interval:
            return False  # 半开：放一次探测 §12.3
        return True

    async def call(self, role: str, prompt: str, **kwargs: Any) -> str:
        if role in JUDGE_ROLES and self._judge is not None and not self.breaker_open:
            try:
                result = await self._judge(prompt, **kwargs)
                self._consecutive_failures = 0
                self._opened_at = None
                return result
            except Exception:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._threshold:
                    self._opened_at = time.monotonic()
                logger.warning("judge endpoint failed (role=%s), fallback to main", role, exc_info=True)
        return await self._main(prompt, **kwargs)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_models.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/models.py tests/agent/test_models.py
git commit -m "feat(agent): ModelPool with judge routing, fallback and circuit breaker"
```

---

### Task 3: EvidencePool（去重、准入、淘汰、图路径解析）

**Files:**
- Create: `raganything/agent/evidence.py`（本任务只含 PoolEntry/EvidencePool；FactLedger 在 Task 4 追加到同文件）
- Test: `tests/agent/test_evidence.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_evidence.py
from raganything.agent.evidence import EvidencePool, PoolEntry


def chunk(cid, content="text", score=0.5, file_path="a.md"):
    return {"chunk_id": cid, "content": content, "rrf_score": score, "file_path": file_path}


def test_dedup_appends_provenance_and_hit_count():
    pool = EvidencePool()
    new = pool.add([chunk("c1")], step=0, tool="search_dense", sub_query="q")
    assert len(new) == 1
    new2 = pool.add([chunk("c1"), chunk("c2")], step=1, tool="search_hybrid", sub_query="q2")
    assert [e.chunk_id for e in new2] == ["c2"]  # 仅新增需 rerank §5.1
    e1 = pool.entries["c1"]
    assert e1.hit_count == 2 and len(e1.provenance) == 2
    assert pool.last_dup_rate == 0.5  # 2 进 1 重


def test_synthetic_id_is_content_hash():
    pool = EvidencePool()
    a = pool.add([{"content": "same text", "rrf_score": 0.1}], step=0, tool="t", sub_query="q")
    b = pool.add([{"content": "same text", "rrf_score": 0.2}], step=1, tool="t", sub_query="q")
    assert len(a) == 1 and len(b) == 0  # 同内容不同批次判同条 §5.2


def test_image_paths_parsed_on_admission():
    pool = EvidencePool()
    pool.add([chunk("c1", content="说明\nImage Path: img/fig1.jpg\n后文")], step=0, tool="t", sub_query="q")
    assert pool.entries["c1"].image_paths == ["img/fig1.jpg"]


def test_eviction_protects_fact_supporters():
    pool = EvidencePool(max_entries=2)
    pool.add([chunk("c1"), chunk("c2"), chunk("c3")], step=0, tool="t", sub_query="q")
    pool.set_scores({"c1": 0.1, "c2": 0.9, "c3": 0.5})
    pool.entries["c1"].supports.add("f1")  # 低分但支撑事实 → 豁免 §5.5
    pool.evict()
    assert set(pool.entries) == {"c1", "c2"}


def test_top_sorted_by_canonical_then_hits():
    pool = EvidencePool()
    pool.add([chunk("c1"), chunk("c2")], step=0, tool="t", sub_query="q")
    pool.set_scores({"c1": 0.3, "c2": 0.8})
    assert [e.chunk_id for e in pool.top(2)] == ["c2", "c1"]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_evidence.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/evidence.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_evidence.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/evidence.py tests/agent/test_evidence.py
git commit -m "feat(agent): evidence pool with content-hash dedup and fact-supporter eviction"
```

---

### Task 4: FactLedger（事实账本 + unverifiable 放弃阈值）

**Files:**
- Modify: `raganything/agent/evidence.py`（文件末尾追加 FactLedger）
- Test: `tests/agent/test_ledger.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_ledger.py
from raganything.agent.evidence import FactLedger


PAYLOAD = {"facts": [
    {"id": "f1", "text": "A 是 B", "status": "found", "chunks": ["c1"]},
    {"id": "f2", "text": "C 的数值", "status": "missing", "chunks": []},
    {"id": "f3", "text": "D 的来源", "status": "missing", "chunks": []},
]}


def test_update_and_effective_coverage():
    led = FactLedger()
    led.update(PAYLOAD)
    assert led.coverage == 1 / 3
    assert [f["id"] for f in led.missing()] == ["f2", "f3"]


def test_unverifiable_after_two_distinct_tools():
    led = FactLedger()
    led.update(PAYLOAD)
    led.record_attempt("f2", "search_dense")
    assert led.facts["f2"]["status"] == "missing"
    led.record_attempt("f2", "search_dense")  # 同工具重复不计第二次
    assert led.facts["f2"]["status"] == "missing"
    led.record_attempt("f2", "search_hybrid")  # 第二个不同工具 → 放弃 §5.3
    assert led.facts["f2"]["status"] == "unverifiable"
    assert led.coverage == 1 / 2  # 分母剔除 unverifiable


def test_found_marks_supports_back():
    led = FactLedger()
    led.update(PAYLOAD)
    assert led.supported_chunks() == {"c1": {"f1"}}


def test_update_merge_keeps_unverifiable():
    led = FactLedger()
    led.update(PAYLOAD)
    led.record_attempt("f2", "a"); led.record_attempt("f2", "b")
    led.update({"facts": [{"id": "f2", "text": "C 的数值", "status": "missing", "chunks": []}]})
    assert led.facts["f2"]["status"] == "unverifiable"  # grader 不能复活已放弃事实
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_ledger.py -v`
Expected: FAIL（ImportError: FactLedger）

- [ ] **Step 3: 实现（追加到 evidence.py 末尾）**

```python
class FactLedger:
    """事实账本：found/missing/unverifiable + 有效 coverage（spec §5.3）。"""

    GIVE_UP_DISTINCT_TOOLS = 2

    def __init__(self) -> None:
        self.facts: dict[str, dict] = {}

    def update(self, payload: dict) -> None:
        for f in payload.get("facts", []):
            fid = str(f.get("id") or f"f{len(self.facts) + 1}")
            existing = self.facts.get(fid)
            status = str(f.get("status", "missing"))
            if existing and existing["status"] == "unverifiable":
                status = "unverifiable"  # 已放弃事实不被 grader 复活
            attempts = existing["attempts"] if existing else set()
            self.facts[fid] = {
                "id": fid, "text": str(f.get("text", "")), "status": status,
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_ledger.py tests/agent/test_evidence.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/evidence.py tests/agent/test_ledger.py
git commit -m "feat(agent): fact ledger with unverifiable give-up threshold"
```

---

### Task 5: Citations（声明拆分 + 引文代码裁决）

**Files:**
- Create: `raganything/agent/citations.py`
- Test: `tests/agent/test_citations.py`

实现要点（评审提示）：LLM 逐字引用常擅改空白/标点，比对前对 quote 和 chunk 都做 `re.sub(r'\W+', '', text)` 归一化（Python3 Unicode 模式下 CJK 属 `\w`，中文安全）。

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_citations.py
import json
import pytest
from raganything.agent.citations import split_claims, verify_citations


def test_split_claims_cjk_and_ascii():
    claims = split_claims("注意力是加权求和。它源于 2017 年论文！短。OK? Final sentence here.")
    assert "注意力是加权求和" in claims
    assert "Final sentence here" in claims
    assert "短" not in claims  # 过短句过滤


class FakePool:
    def __init__(self, payload):
        self.payload = payload
    async def call(self, role, prompt, **kw):
        assert role == "checker"
        return json.dumps(self.payload)


@pytest.mark.asyncio
async def test_quote_whitespace_tolerated_but_fabrication_rejected():
    chunks = [{"chunk_id": "c1", "content": "Attention 是  加权\n求和 机制"}]
    payload = {"claims": [
        {"id": 0, "quote": "Attention 是加权求和机制", "supported": True},   # 空白差异 → 应判支持
        {"id": 1, "quote": "完全捏造的引文内容", "supported": True},          # 伪造 → 代码裁决推翻
    ]}
    grounded, ungrounded = await verify_citations(
        FakePool(payload), "q", "Attention 是加权求和机制。模型于 2017 年提出。", chunks,
    )
    assert grounded is False
    assert len(ungrounded) == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_citations.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/citations.py
"""声明级引文验证：LLM 提案、代码裁决（spec §12.2）。"""
from __future__ import annotations

import re
from typing import Any

from raganything.retrieval.json_utils import call_json_object

_SENT_SPLIT = re.compile(r"[。！？!?\n]+|(?<=[a-zA-Z0-9])\.\s")
_MIN_CLAIM_LEN = 6

_VERIFY_PROMPT = """\
You verify answer claims against retrieved evidence chunks.
For EACH claim below, quote the supporting span VERBATIM from the chunks,
or mark it unsupported. Output JSON only:
{{"claims": [{{"id": 0, "quote": "<verbatim span or empty>", "supported": true|false}}]}}

Chunks:
{chunks}

Question: {query}

Claims:
{claims}
"""


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", text)


def split_claims(answer: str, min_len: int = _MIN_CLAIM_LEN) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(answer) if p and p.strip()]
    return [p for p in parts if len(p) >= min_len]


async def verify_citations(
    model_pool: Any, query: str, answer: str, chunks: list[dict],
) -> tuple[bool, list[str]]:
    claims = split_claims(answer)
    if not claims:
        return False, [query]
    chunk_text = "\n---\n".join(str(c.get("content", ""))[:1500] for c in chunks[:20])
    normalized_corpus = _norm(chunk_text)
    prompt = _VERIFY_PROMPT.format(
        chunks=chunk_text, query=query,
        claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims)),
    )
    try:
        parsed = await call_json_object(
            lambda p, **kw: model_pool.call("checker", p, **kw), prompt, max_tokens=1024,
        )
    except Exception:
        return False, claims  # checker 失效 → 保守判全不支持
    verdicts = {int(c.get("id", -1)): c for c in parsed.get("claims", []) if isinstance(c, dict)}
    ungrounded: list[str] = []
    for i, claim in enumerate(claims):
        v = verdicts.get(i, {})
        quote = _norm(str(v.get("quote", "")))
        # 代码裁决：声称 supported 必须有真实引文（归一化后包含于语料）
        if not (v.get("supported") and quote and quote in normalized_corpus):
            ungrounded.append(claim)
    return (len(ungrounded) == 0), ungrounded
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_citations.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/citations.py tests/agent/test_citations.py
git commit -m "feat(agent): citation verification with normalized quote matching"
```

---

### Task 6: SessionMemory + SessionStore

**Files:**
- Create: `raganything/agent/session.py`
- Test: `tests/agent/test_session.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_session.py
from raganything.agent.session import SessionMemory, SessionStore


def test_active_entities_cap_evicts_oldest():
    s = SessionMemory(session_id="s1", workspace_id="w1")
    for i in range(14):
        s.register_entities([{"name": f"e{i}", "note": "", "last_turn": i}])
    assert len(s.active_entities) == 12
    assert all(e["name"] != "e0" for e in s.active_entities)


def test_chunk_cache_lru():
    s = SessionMemory(session_id="s1", workspace_id="w1", cache_max=2)
    s.cache_chunks([{"chunk_id": "c1", "content": "a"}, {"chunk_id": "c2", "content": "b"}])
    s.get_cached(["c1"])  # touch c1
    s.cache_chunks([{"chunk_id": "c3", "content": "c"}])
    assert set(s.chunk_cache) == {"c1", "c3"}


def test_store_get_create_and_ttl(monkeypatch):
    store = SessionStore(ttl_seconds=100, max_sessions=2)
    a = store.get("w1", "s1")
    assert store.get("w1", "s1") is a
    now = [1000.0]
    monkeypatch.setattr("raganything.agent.session._now", lambda: now[0])
    a.touch()
    now[0] += 200
    store.sweep()
    assert store.get("w1", "s1") is not a  # 过期重建


def test_drop_chunks_by_workspace():
    store = SessionStore()
    s = store.get("w1", "s1")
    s.cache_chunks([{"chunk_id": "c1", "content": "x"}])
    other = store.get("w2", "s2")
    other.cache_chunks([{"chunk_id": "c1", "content": "x"}])
    store.drop_chunks("w1", ["c1"])  # 治理删除联动 §5.6
    assert "c1" not in s.chunk_cache and "c1" in other.chunk_cache


def test_dump_load_roundtrip():
    s = SessionMemory(session_id="s1", workspace_id="w1")
    s.history_summary = "摘要"
    s.recent_turns.append({"q": "a", "a": "b", "cancelled": False})
    data = s.dump()
    s2 = SessionMemory.load(data)
    assert s2.history_summary == "摘要" and len(s2.recent_turns) == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_session.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/session.py
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
            "history_summary": self.history_summary, "chunk_cache": dict(self.chunk_cache),
        }

    @classmethod
    def load(cls, data: dict) -> "SessionMemory":
        s = cls(session_id=data["session_id"], workspace_id=data["workspace_id"])
        s.active_entities = list(data.get("active_entities", []))
        s.recent_turns = list(data.get("recent_turns", []))
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_session.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/session.py tests/agent/test_session.py
git commit -m "feat(agent): session memory with TTL store and governance-linked invalidation"
```

---

### Task 7: ToolRegistry + agent 检索 profile

**Files:**
- Create: `raganything/agent/tools.py`
- Test: `tests/agent/test_tools.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_tools.py
import pytest
from raganything.agent.tools import (
    ToolRegistry, ToolSpec, ParamSpec, build_default_registry, register_agent_profiles,
)
from raganything.retrieval.profiles import PROFILE_REGISTRY


def test_agent_profiles_registered_without_rerank():
    register_agent_profiles()
    for name in ["agent_sparse", "agent_dense", "agent_hybrid", "agent_graph", "agent_ppr"]:
        assert name in PROFILE_REGISTRY
        assert PROFILE_REGISTRY[name].enable_rerank is False  # rerank 移到池准入口 §5.1


def test_default_registry_costs_match_spec():
    reg = build_default_registry()
    costs = {n: reg.get(n).cost for n in reg.names()}
    assert costs["search_sparse"] == 1 and costs["search_hybrid"] == 2
    assert costs["search_ppr"] == 4 and costs["decompose_search"] == 8
    assert costs["answer"] == 0


def test_param_clamp_via_spec():
    spec = ToolSpec(name="t", cost=1, description="", profile="",
                    params={"top_k": ParamSpec(default=10, min=1, max=50)})
    assert spec.clamp({"top_k": 999}) == {"top_k": 50}
    assert spec.clamp({"top_k": "abc", "bogus": 1}) == {"top_k": 10}  # 非法回默认、未知丢弃


def test_expand_allowed_per_tool():
    reg = build_default_registry()
    assert "hyde" in reg.get("search_dense").allowed_expand   # hyde 仅 dense §7.4
    assert "hyde" not in reg.get("search_sparse").allowed_expand
    assert "mqe" in reg.get("search_hybrid").allowed_expand


def test_card_text_static():
    reg = build_default_registry()
    text = reg.card_text()
    assert "search_ppr" in text and "4" in text  # 成本进卡片 §4.2
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_tools.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/tools.py
"""工具注册表 + agent 专用检索 profile（spec §7）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from raganything.retrieval.profiles import PROFILE_REGISTRY, RetrievalProfile


@dataclass
class ParamSpec:
    default: object
    min: float | None = None
    max: float | None = None


@dataclass
class ToolSpec:
    name: str
    cost: float
    description: str
    profile: str                       # 对应 PROFILE_REGISTRY 键；非检索工具为 ""
    params: dict[str, ParamSpec] = field(default_factory=dict)
    allowed_expand: tuple[str, ...] = ("none",)

    def clamp(self, raw: dict) -> dict:
        out = {}
        for key, spec in self.params.items():
            value = raw.get(key, spec.default)
            if isinstance(spec.default, (int, float)):
                try:
                    value = type(spec.default)(value)
                except (TypeError, ValueError):
                    value = spec.default
                if spec.min is not None:
                    value = max(spec.min, value)
                if spec.max is not None:
                    value = min(spec.max, value)
            out[key] = value
        return out


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def card_text(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name} (cost {t.cost:g}): {t.description}"
                         + (f" | expand: {','.join(t.allowed_expand)}" if t.allowed_expand != ("none",) else ""))
        return "\n".join(lines)


def register_agent_profiles() -> None:
    """注册 enable_rerank=False 的单路 profile；幂等。"""
    defs = [
        ("agent_sparse", ["qdrant_sparse"]),
        ("agent_dense", ["naive"]),
        ("agent_hybrid", ["qdrant_hybrid"]),
        ("agent_graph", ["local_kg", "global_kg"]),
        ("agent_ppr", ["ppr"]),
    ]
    for name, paths in defs:
        if name in PROFILE_REGISTRY:
            continue
        PROFILE_REGISTRY[name] = RetrievalProfile(
            name=name, description=f"agent tool path: {'+'.join(paths)}",
            paths=paths, rrf_weights={p: 1.0 for p in paths}, enable_rerank=False,
        )


_TOPK = lambda d: {"top_k": ParamSpec(default=d, min=1, max=60), "query": ParamSpec(default="")}


def build_default_registry() -> ToolRegistry:
    register_agent_profiles()
    reg = ToolRegistry()
    reg.register(ToolSpec("search_sparse", 1, "BM25 精确词项检索；型号/ID/专名首选", "agent_sparse",
                          _TOPK(10), ("none", "mqe")))
    reg.register(ToolSpec("search_dense", 1, "稠密语义检索；语义型问题首选", "agent_dense",
                          _TOPK(10), ("none", "mqe", "hyde")))
    reg.register(ToolSpec("rewrite_query", 1, "重写当前检索查询", "",
                          {"query": ParamSpec(default="")}))
    reg.register(ToolSpec("search_hybrid", 2, "稠密+BM25 RRF 混合；标准武器", "agent_hybrid",
                          _TOPK(15), ("none", "mqe")))
    reg.register(ToolSpec("search_graph", 2, "知识图谱 local+global 检索", "agent_graph", _TOPK(15)))
    reg.register(ToolSpec("inspect_image", 2, "VLM 定向看图，提取文字事实入池", "",
                          {"chunk_ids": ParamSpec(default=[]), "question": ParamSpec(default="")}))
    reg.register(ToolSpec("search_ppr", 4, "全图 PPR 多跳检索；仅缺桥事实时使用", "agent_ppr", _TOPK(20)))
    reg.register(ToolSpec("decompose_search", 8, "问题分解+并行混合检索；最后手段", "agent_hybrid",
                          _TOPK(15)))
    reg.register(ToolSpec("answer", 0, "终结：生成答案", "",
                          {"generation_mode": ParamSpec(default="direct")}))
    return reg
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_tools.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/tools.py tests/agent/test_tools.py
git commit -m "feat(agent): tool registry with cost tiers and rerank-free agent profiles"
```

---

### Task 8: Decision 归一化 + 重复守卫

**Files:**
- Create: `raganything/agent/decision.py`
- Test: `tests/agent/test_decision.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_decision.py
from raganything.agent.decision import Decision, normalize_decision, decision_signature
from raganything.agent.tools import build_default_registry

REG = build_default_registry()


def test_unknown_action_difflib_matched():
    d = normalize_decision({"thought": "t", "action": "serch_hybird", "params": {}}, REG, "默认查询")
    assert d.action == "search_hybrid"
    assert d.params["query"] == "默认查询"  # 缺 query 回填 §4.3


def test_params_clamped_and_unknown_dropped():
    d = normalize_decision(
        {"thought": "t", "action": "search_dense", "params": {"top_k": 9999, "evil": 1}}, REG, "q")
    assert d.params["top_k"] == 60 and "evil" not in d.params


def test_answer_overrides_stop_flag():
    d = normalize_decision({"thought": "t", "action": "answer", "stop": False,
                            "params": {"generation_mode": "cot_reflect"}}, REG, "q")
    assert d.stop is True  # stop 与 action 不一致以 action 为准 §4.1


def test_unmatchable_action_raises():
    import pytest
    with pytest.raises(ValueError):
        normalize_decision({"thought": "t", "action": "zzzzzz", "params": {}}, REG, "q")


def test_signature_normalizes_query():
    a = decision_signature(Decision("t", "search_dense", {"query": " 大 模型 ", "top_k": 10}))
    b = decision_signature(Decision("t", "search_dense", {"query": "大模型", "top_k": 10}))
    assert a == b  # 空白差异不绕过重复守卫
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_decision.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/decision.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_decision.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/decision.py tests/agent/test_decision.py
git commit -m "feat(agent): decision normalization with difflib matching and repeat signature"
```

---

### Task 9: Planner（改写+分类合并调用 + 画像预设 + 快速通道）

**Files:**
- Create: `raganything/agent/planner.py`
- Test: `tests/agent/test_planner.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_planner.py
import json
import pytest
from raganything.agent.planner import make_plan, ARCHETYPE_PRESETS, PlanResult
from raganything.agent.session import SessionMemory


class FakePool:
    def __init__(self, payload):
        self.payload, self.calls = payload, []
    async def call(self, role, prompt, **kw):
        self.calls.append((role, prompt))
        return json.dumps(self.payload)


PAYLOAD = {
    "standalone_query": "Transformer attention 与 CNN 的区别",
    "archetype": "comparison", "confidence": 0.9,
    "exact_terms": [], "suggested_expand": "none", "visual_intent": False,
    "entities_referenced": [{"name": "CNN", "note": "对比对象", "last_turn": 2}],
}


@pytest.mark.asyncio
async def test_merged_call_rewrites_and_classifies():
    s = SessionMemory(session_id="s", workspace_id="w")
    s.recent_turns.append({"q": "attention 是什么", "a": "是加权求和", "cancelled": False})
    pool = FakePool(PAYLOAD)
    plan = await make_plan(pool, "它和 CNN 有什么区别", s)
    assert plan.standalone_query.startswith("Transformer")
    assert pool.calls[0][0] == "rewriter"
    assert "attention 是什么" in pool.calls[0][1]  # 历史进改写 prompt §6.2
    assert any(e["name"] == "CNN" for e in s.active_entities)  # 实体顺手登记


@pytest.mark.asyncio
async def test_fast_path_only_high_confidence_factoid():
    s = SessionMemory(session_id="s", workspace_id="w")
    p1 = await make_plan(FakePool({**PAYLOAD, "archetype": "factoid", "confidence": 0.9}), "q1", s)
    assert p1.fast_path is True
    p2 = await make_plan(FakePool({**PAYLOAD, "archetype": "factoid", "confidence": 0.5}), "q2", s)
    assert p2.fast_path is False
    p3 = await make_plan(FakePool({**PAYLOAD, "archetype": "summary", "confidence": 0.99}), "q3", s)
    assert p3.fast_path is False  # 仅 factoid §4.5


@pytest.mark.asyncio
async def test_plan_cached_per_session():
    s = SessionMemory(session_id="s", workspace_id="w")
    pool = FakePool(PAYLOAD)
    await make_plan(pool, "同一个问题", s)
    await make_plan(pool, "同一个问题", s)
    assert len(pool.calls) == 1


@pytest.mark.asyncio
async def test_unknown_on_parse_failure():
    class Broken:
        async def call(self, role, prompt, **kw):
            return "not json at all" * 3
    plan = await make_plan(Broken(), "问题", SessionMemory(session_id="s", workspace_id="w"))
    assert plan.archetype == "unknown" and plan.standalone_query == "问题"


def test_presets_match_spec_table():
    assert ARCHETYPE_PRESETS["summary"]["tool"] == "search_hybrid"
    assert ARCHETYPE_PRESETS["summary"]["expand"] == "mqe"
    assert ARCHETYPE_PRESETS["summary"]["top_k"] == 25
    assert ARCHETYPE_PRESETS["multihop"]["generation_mode"] == "cot_reflect"
    assert ARCHETYPE_PRESETS["factoid"]["top_k"] == 5
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_planner.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/planner.py
"""轮初合并调用：改写+分类+实体登记（spec §9）。"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from raganything.retrieval.json_utils import call_json_object
from raganything.agent.session import SessionMemory

logger = logging.getLogger(__name__)

ARCHETYPES = {"factoid", "summary", "multihop", "comparison", "unknown"}
FAST_PATH_CONFIDENCE = 0.8

# spec §9.2 画像→策略矩阵
ARCHETYPE_PRESETS: dict[str, dict] = {
    "factoid":    {"tool": "search_sparse", "top_k": 5,  "expand": "none", "generation_mode": "direct"},
    "summary":    {"tool": "search_hybrid", "top_k": 25, "expand": "mqe",  "generation_mode": "map_reduce"},
    "multihop":   {"tool": "search_hybrid", "top_k": 15, "expand": "none", "generation_mode": "cot_reflect"},
    "comparison": {"tool": "search_hybrid", "top_k": 10, "expand": "none", "generation_mode": "direct"},
    "unknown":    {"tool": "search_hybrid", "top_k": 15, "expand": "none", "generation_mode": "direct"},
}

_PLAN_PROMPT = """\
You prepare a user question for retrieval. Given conversation context, output JSON only:
{{"standalone_query": "<self-contained rewrite of the current question, resolve all references>",
  "archetype": "factoid|summary|multihop|comparison|unknown",
  "confidence": 0.0,
  "exact_terms": ["<IDs, model numbers, proper nouns needing exact match>"],
  "suggested_expand": "none|mqe|hyde",
  "visual_intent": false,
  "entities_referenced": [{{"name": "...", "note": "<role in conversation>", "last_turn": 0}}]}}

archetype rules: factoid=single specific fact; summary=broad survey/summarize;
multihop=requires chaining facts across documents; comparison=compare two+ entities;
unknown=unclear. visual_intent=true only if answering requires inspecting image pixels
beyond textual descriptions (read chart values, layout, colors).

History summary: {summary}
Active entities: {entities}
Recent turns:
{turns}

Current question: {query}
"""


@dataclass
class PlanResult:
    standalone_query: str
    archetype: str
    confidence: float
    exact_terms: list[str] = field(default_factory=list)
    suggested_expand: str = "none"
    visual_intent: bool = False
    fast_path: bool = False
    preset: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("standalone_query", "archetype", "confidence", "exact_terms",
                 "suggested_expand", "visual_intent", "fast_path", "preset")}


def _cache_key(query: str) -> str:
    return re.sub(r"\s+", "", query)


async def make_plan(model_pool: Any, query: str, session: SessionMemory) -> PlanResult:
    key = _cache_key(query)
    if key in session.plan_cache:
        return PlanResult(**session.plan_cache[key])

    turns = "\n".join(f"U: {t['q']}\nA: {str(t['a'])[:300]}" for t in session.recent_turns) or "(none)"
    prompt = _PLAN_PROMPT.format(
        summary=session.history_summary or "(none)",
        entities=", ".join(e["name"] for e in session.active_entities) or "(none)",
        turns=turns, query=query,
    )
    try:
        parsed = await call_json_object(
            lambda p, **kw: model_pool.call("rewriter", p, **kw), prompt, max_tokens=512)
    except Exception:
        logger.warning("plan call failed; defaulting to unknown archetype", exc_info=True)
        parsed = {}

    archetype = str(parsed.get("archetype", "unknown"))
    if archetype not in ARCHETYPES:
        archetype = "unknown"
    confidence = float(parsed.get("confidence") or 0.0)
    if confidence < 0.6 and archetype != "unknown":
        archetype = "unknown"  # 低置信走稳妥默认 §9.2
    entities = [e for e in parsed.get("entities_referenced", []) if isinstance(e, dict)]
    if entities:
        session.register_entities(entities)

    plan = PlanResult(
        standalone_query=str(parsed.get("standalone_query") or query),
        archetype=archetype,
        confidence=confidence,
        exact_terms=[str(t) for t in parsed.get("exact_terms", [])],
        suggested_expand=str(parsed.get("suggested_expand", "none")),
        visual_intent=bool(parsed.get("visual_intent", False)),
        fast_path=(archetype == "factoid" and confidence >= FAST_PATH_CONFIDENCE),
        preset=dict(ARCHETYPE_PRESETS[archetype]),
    )
    if plan.exact_terms and plan.archetype == "factoid":
        plan.preset["tool"] = "search_sparse"
    elif plan.archetype == "factoid":
        plan.preset["tool"] = "search_dense"  # 语义型 factoid §9.2
    session.plan_cache[key] = plan.to_dict()
    return plan
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_planner.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/planner.py tests/agent/test_planner.py
git commit -m "feat(agent): merged rewrite+classify planner with archetype presets and fast path"
```

---

### Task 10: LedgerGrader（增量评估 + 条件终审触发）

**Files:**
- Create: `raganything/agent/grading.py`
- Test: `tests/agent/test_grading.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_grading.py
import json
import pytest
from raganything.agent.evidence import EvidencePool, FactLedger
from raganything.agent.grading import LedgerGrader, should_final_review


class FakePool:
    def __init__(self, payload):
        self.payload, self.prompts = payload, []
    async def call(self, role, prompt, **kw):
        self.prompts.append(prompt)
        return json.dumps(self.payload)


def _pool_with(*cids):
    p = EvidencePool()
    p.add([{"chunk_id": c, "content": f"content of {c}"} for c in cids],
          step=0, tool="t", sub_query="q")
    return p


@pytest.mark.asyncio
async def test_grade_updates_ledger_and_marks_supports():
    pool = _pool_with("c1", "c2")
    ledger = FactLedger()
    grader = LedgerGrader(FakePool({"sufficient": False, "facts": [
        {"id": "f1", "text": "事实一", "status": "found", "chunks": ["c1"]},
        {"id": "f2", "text": "事实二", "status": "missing", "chunks": []},
    ]}))
    result = await grader.grade("q", ledger, pool, new_entries=list(pool.entries.values()))
    assert result["sufficient"] is False
    assert ledger.coverage == 0.5
    assert "f1" in pool.entries["c1"].supports  # supports 反写 §5.3


@pytest.mark.asyncio
async def test_missing_related_old_chunks_in_window():
    pool = _pool_with("c1", "c2", "c3")
    pool.set_scores({"c1": 0.9, "c2": 0.5, "c3": 0.1})
    ledger = FactLedger()
    ledger.update({"facts": [{"id": "f1", "text": "缺失事实", "status": "missing", "chunks": []}]})
    fake = FakePool({"sufficient": False, "facts": []})
    grader = LedgerGrader(fake)
    await grader.grade("q", ledger, pool, new_entries=[])  # 无新 chunk
    # 盲区修复：missing 存在时旧 chunk 进窗口 §5.3
    assert "content of c1" in fake.prompts[0]


def test_final_review_triggers():
    led = FactLedger()
    led.update({"facts": [{"id": "f1", "text": "x", "status": "found", "chunks": ["c1"]}]})
    pool = _pool_with("c1")
    pool.set_scores({"c1": 0.3})  # 单 chunk 支撑且 <0.4 → 触发 §5.4
    assert should_final_review(ledger_steps=1, ledger=led, pool=pool, recent_dup_rates=[0.0]) is True
    pool.set_scores({"c1": 0.9})
    assert should_final_review(ledger_steps=1, ledger=led, pool=pool, recent_dup_rates=[0.0]) is False
    assert should_final_review(ledger_steps=3, ledger=led, pool=pool, recent_dup_rates=[0.0]) is True
    assert should_final_review(ledger_steps=1, ledger=led, pool=pool, recent_dup_rates=[0.6, 0.7]) is True
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_grading.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/grading.py
"""账本式增量评估 + 条件终审（spec §5.3/5.4）。"""
from __future__ import annotations

import json
import logging
from typing import Any

from raganything.retrieval.json_utils import call_json_object
from raganything.agent.evidence import EvidencePool, FactLedger, PoolEntry

logger = logging.getLogger(__name__)

_RELATED_OLD_LIMIT = 5
_FINAL_REVIEW_STEPS = 3
_FINAL_REVIEW_LOW_SCORE = 0.4
_FINAL_REVIEW_DUP_RATE = 0.5

_GRADE_PROMPT = """\
You maintain a fact ledger for answering a question. Update it incrementally.

Question: {query}

Current ledger (facts needed to answer; found facts list supporting chunk ids):
{ledger}

Evidence chunks to evaluate (new + relevant old):
{chunks}

Task: re-derive the COMPLETE fact list needed to answer the question.
For each fact: status "found" (with chunk ids from evidence above that support it) or "missing".
Keep fact ids stable when the fact is unchanged. Add new facts if discovered necessary.
Output JSON only:
{{"sufficient": true|false, "facts": [{{"id": "f1", "text": "...", "status": "found|missing", "chunks": ["..."]}}]}}
sufficient=true only when every necessary fact is found.
"""


def _format_chunks(entries: list[PoolEntry]) -> str:
    return "\n---\n".join(f"[{e.chunk_id}] {e.content[:800]}" for e in entries) or "(none)"


class LedgerGrader:
    def __init__(self, model_pool: Any) -> None:
        self._pool = model_pool

    async def grade(
        self, query: str, ledger: FactLedger, pool: EvidencePool, *,
        new_entries: list[PoolEntry],
    ) -> dict:
        window = list(new_entries)
        if ledger.missing():
            # 盲区修复：后发现的 fact 必须能拿早期 chunk 核对 §5.3
            seen = {e.chunk_id for e in window}
            window += [e for e in pool.top(_RELATED_OLD_LIMIT) if e.chunk_id not in seen]
        prompt = _GRADE_PROMPT.format(
            query=query, ledger=json.dumps(ledger.to_dict(), ensure_ascii=False),
            chunks=_format_chunks(window),
        )
        try:
            parsed = await call_json_object(
                lambda p, **kw: self._pool.call("grader", p, **kw), prompt, max_tokens=1536)
        except Exception:
            logger.warning("LedgerGrader failed; keeping ledger unchanged", exc_info=True)
            return {"sufficient": False, "facts": []}
        ledger.update(parsed)
        for cid, fact_ids in ledger.supported_chunks().items():
            if cid in pool.entries:
                pool.entries[cid].supports |= fact_ids
        return {"sufficient": bool(parsed.get("sufficient", False)),
                "facts": parsed.get("facts", [])}

    async def final_review(self, query: str, pool: EvidencePool, top_n: int = 20) -> dict:
        """无账本全池终审：fresh grade（spec §5.4）。"""
        fresh = FactLedger()
        return await self.grade(query, fresh, pool, new_entries=pool.top(top_n)) | {
            "fresh_ledger": fresh.to_dict()}


def should_final_review(
    *, ledger_steps: int, ledger: FactLedger, pool: EvidencePool,
    recent_dup_rates: list[float],
) -> bool:
    if ledger_steps >= _FINAL_REVIEW_STEPS:
        return True
    if len(recent_dup_rates) >= 2 and all(r > _FINAL_REVIEW_DUP_RATE for r in recent_dup_rates[-2:]):
        return True
    for f in ledger.facts.values():
        if f["status"] == "found" and len(f["chunks"]) == 1:
            entry = pool.entries.get(f["chunks"][0])
            score = entry.canonical_score if entry and entry.canonical_score is not None else 0.0
            if score < _FINAL_REVIEW_LOW_SCORE:
                return True
    return False
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_grading.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/grading.py tests/agent/test_grading.py
git commit -m "feat(agent): ledger grader with blind-spot window and conditional final review"
```

---

### Task 11: Generate（装填器三道门 + 三种生成模式）

**Files:**
- Create: `raganything/agent/generate.py`
- Test: `tests/agent/test_generate.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_generate.py
import pytest
from raganything.agent.evidence import EvidencePool, FactLedger
from raganything.agent.generate import pack_context, generate_answer, estimate_tokens


def _pool(n, content_len=400, with_image=False):
    p = EvidencePool()
    chunks = []
    for i in range(n):
        body = f"chunk {i} " + "字" * content_len
        if with_image and i == 0:
            body += "\nImage Path: img/fig.jpg"
        chunks.append({"chunk_id": f"c{i}", "content": body})
    p.add(chunks, step=0, tool="t", sub_query="q")
    p.set_scores({f"c{i}": 1.0 - i * 0.01 for i in range(n)})
    return p


def test_pack_respects_token_budget_and_score_order():
    pool = _pool(50)
    packed = pack_context(pool, FactLedger(), max_context_tokens=1000)
    assert 0 < len(packed.chunks) < 50
    total = sum(estimate_tokens(e.content) for e in packed.chunks)
    assert total <= 1000
    scores = [e.canonical_score for e in packed.chunks]
    assert scores == sorted(scores, reverse=True)


def test_fact_supporters_packed_first():
    pool = _pool(10)
    pool.entries["c9"].supports.add("f1")  # 最低分但支撑事实
    packed = pack_context(pool, FactLedger(), max_context_tokens=600)
    assert any(e.chunk_id == "c9" for e in packed.chunks)


def test_image_gates():
    pool = _pool(3, with_image=True)
    no_intent = pack_context(pool, FactLedger(), max_context_tokens=5000, visual_intent=False)
    assert no_intent.images == []  # 门1 §11.2
    pool.entries["c0"].supports.add("f1")
    with_intent = pack_context(pool, FactLedger(), max_context_tokens=5000, visual_intent=True)
    assert with_intent.images == ["img/fig.jpg"]  # 门2 过（支撑事实）


@pytest.mark.asyncio
async def test_map_reduce_groups_by_file():
    calls = []
    class Pool:
        async def call(self, role, prompt, **kw):
            calls.append((role, prompt))
            return "部分总结" if "summarize" in prompt.lower() else "最终答案"
    p = EvidencePool()
    p.add([{"chunk_id": "a", "content": "x" * 4000, "file_path": "doc1.md"},
           {"chunk_id": "b", "content": "y" * 4000, "file_path": "doc2.md"}],
          step=0, tool="t", sub_query="q")
    p.set_scores({"a": 0.9, "b": 0.8})
    answer = await generate_answer(Pool(), "总结全部", p, FactLedger(),
                                   mode="map_reduce", max_context_tokens=500)
    assert answer == "最终答案"
    assert len(calls) == 3  # 2 map + 1 reduce


@pytest.mark.asyncio
async def test_cot_reflect_includes_ledger_scaffold():
    prompts = []
    class Pool:
        async def call(self, role, prompt, **kw):
            prompts.append(prompt)
            return "答案"
    led = FactLedger()
    led.update({"facts": [{"id": "f1", "text": "已证实的桥事实", "status": "found", "chunks": ["c0"]}]})
    await generate_answer(Pool(), "q", _pool(2), led, mode="cot_reflect", max_context_tokens=2000)
    assert "已证实的桥事实" in prompts[0]  # 账本作脚手架 §10
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_generate.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/generate.py
"""上下文装填 + direct/map_reduce/cot_reflect（spec §10/§11.2）。"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from raganything.agent.evidence import EvidencePool, FactLedger, PoolEntry

DEFAULT_MAX_CONTEXT_TOKENS = 12_000
MAX_IMAGES_PER_CALL = 6  # per-generation-call 上限 §11.2
_IMAGE_SCORE_FLOOR = 0.3


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


@dataclass
class PackedContext:
    chunks: list[PoolEntry] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n---\n".join(f"[{e.chunk_id}] ({e.file_path}) {e.content}" for e in self.chunks)


def pack_context(
    pool: EvidencePool, ledger: FactLedger, *,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    visual_intent: bool = False, max_images: int = MAX_IMAGES_PER_CALL,
) -> PackedContext:
    supporters = [e for e in pool.entries.values() if e.supports]
    others = sorted((e for e in pool.entries.values() if not e.supports),
                    key=lambda e: e.sort_key(), reverse=True)
    packed = PackedContext()
    budget = max_context_tokens
    for entry in supporters + others:  # 支撑事实优先 §5.3/§14 评审5
        cost = estimate_tokens(entry.content)
        if cost > budget:
            continue
        packed.chunks.append(entry)
        budget -= cost
    packed.chunks.sort(key=lambda e: e.sort_key(), reverse=True)
    if visual_intent:  # 三道门：意图门已过，再过相关门 §11.2
        for e in packed.chunks:
            for path in e.image_paths:
                if len(packed.images) >= max_images:
                    break
                if e.supports or (e.canonical_score or 0.0) >= _IMAGE_SCORE_FLOOR:
                    packed.images.append(path)
    return packed


_DIRECT_PROMPT = """\
Answer based ONLY on the evidence below. If evidence is insufficient, say what is missing.

Evidence:
{context}

Question: {query}
"""

_MAP_PROMPT = """\
Summarize the evidence below ONLY as it relates to the question. Max 300 tokens.

Question: {query}

Evidence:
{context}
"""

_REDUCE_PROMPT = """\
Synthesize the per-document summaries into a final answer to the question.

Question: {query}

Summaries:
{summaries}
"""

_COT_PROMPT = """\
Answer step by step, anchoring EVERY reasoning step on the verified facts below.
Do not introduce claims unsupported by the facts or evidence.

Verified facts:
{facts}

Unverifiable details (state these as unconfirmed if relevant):
{unverifiable}

Evidence:
{context}

Question: {query}
"""


async def generate_answer(
    model_pool: Any, query: str, pool: EvidencePool, ledger: FactLedger, *,
    mode: str = "direct", max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    visual_intent: bool = False,
) -> str:
    packed = pack_context(pool, ledger, max_context_tokens=max_context_tokens,
                          visual_intent=visual_intent)
    if mode == "map_reduce":
        total = sum(estimate_tokens(e.content) for e in pool.entries.values())
        if total > max_context_tokens:
            return await _map_reduce(model_pool, query, pool, max_context_tokens)
        mode = "direct"  # 预算内退化 §10
    if mode == "cot_reflect":
        facts = "\n".join(f"- {f['text']} (chunks: {','.join(f['chunks'])})"
                          for f in ledger.facts.values() if f["status"] == "found") or "(none)"
        unver = "\n".join(f"- {f['text']}" for f in ledger.unverifiable()) or "(none)"
        prompt = _COT_PROMPT.format(facts=facts, unverifiable=unver,
                                    context=packed.text(), query=query)
    else:
        prompt = _DIRECT_PROMPT.format(context=packed.text(), query=query)
    return str(await model_pool.call("generator", prompt))


async def _map_reduce(model_pool: Any, query: str, pool: EvidencePool,
                      max_context_tokens: int) -> str:
    groups: dict[str, list[PoolEntry]] = {}
    for e in sorted(pool.entries.values(), key=lambda e: e.sort_key(), reverse=True):
        groups.setdefault(e.file_path or "(unknown)", []).append(e)
    per_group = max(1000, max_context_tokens // max(1, len(groups)))

    async def _map_one(entries: list[PoolEntry]) -> str:
        budget, parts = per_group, []
        for e in entries:
            cost = estimate_tokens(e.content)
            if cost > budget:
                continue
            parts.append(e.content)
            budget -= cost
        return str(await model_pool.call(
            "generator", _MAP_PROMPT.format(query=query, context="\n---\n".join(parts))))

    summaries = await asyncio.gather(*[_map_one(v) for v in groups.values()])
    body = "\n\n".join(f"[{name}]\n{s}" for name, s in zip(groups, summaries))
    return str(await model_pool.call(
        "generator", _REDUCE_PROMPT.format(query=query, summaries=body)))
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_generate.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/generate.py tests/agent/test_generate.py
git commit -m "feat(agent): context packer with image gates and three generation modes"
```

---

### Task 12: Trace（v2 兼容输出）

**Files:**
- Create: `raganything/agent/trace.py`
- Test: `tests/agent/test_trace.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_trace.py
from raganything.agent.trace import TraceBuilder


def test_v2_compatible_top_level_keys():
    tb = TraceBuilder(profile="agent", query="q")
    tb.add_retrieval_step(step_type="initial", query="q", tool="search_dense",
                          chunks=3, trace={"paths_activated": ["naive"]}, cycle=0)
    tb.add_grader_event({"sufficient": False, "coverage": 0.3}, cycle=0)
    tb.add_decision(thought="先便宜的", action="search_dense", params={"top_k": 10},
                    budget_snapshot={"remaining_points": 5}, fallback=False)
    tb.add_hallucination_event({"grounded": True}, cycle=0)
    tb.add_reclassify("factoid", "multihop", cycle=1)
    out = tb.build(terminal_reason="grounded", grounded=True)
    # v2 同构键（agent_graph_v2.py trace 契约）
    for key in ("retrieval_steps", "grader_events", "hallucination_events",
                "rewrite_history", "profile", "terminal_reason", "grounded"):
        assert key in out
    # v3 新增键
    assert out["agent_decisions"][0]["action"] == "search_dense"
    assert out["reclassify_events"][0]["to"] == "multihop"
    assert out["used_fallback"] is False


def test_used_fallback_flag_for_ab_stratification():
    tb = TraceBuilder(profile="agent", query="q")
    tb.add_decision(thought="", action="search_hybrid", params={},
                    budget_snapshot={}, fallback=True)
    assert tb.build(terminal_reason="insufficient", grounded=False)["used_fallback"] is True
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_trace.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/trace.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_trace.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add raganything/agent/trace.py tests/agent/test_trace.py
git commit -m "feat(agent): v2-compatible trace builder with decision chain and fallback flag"
```

---

### Task 13: AgentLoop（主循环集成）

**Files:**
- Create: `raganything/agent/loop.py`
- Test: `tests/agent/test_loop.py`

实现要点（评审提示）：取消用 `asyncio.wait({tool_task, cancel_waiter}, return_when=FIRST_COMPLETED)` 包裹每次工具/LLM I/O，取消事件先到则 `tool_task.cancel()`，`CancelledError` 传播进底层 driver 中止 I/O——工具实现零侵入，秒级中断，不必等步边界。

- [ ] **Step 1: 写失败测试**

```python
# tests/agent/test_loop.py
import asyncio
import json
import pytest
from raganything.agent.budget import Budget
from raganything.agent.loop import AgentLoop, AgentResult
from raganything.agent.models import ModelPool
from raganything.agent.session import SessionMemory
from raganything.agent.tools import build_default_registry


def make_llm(script):
    """script: list of(matcher_substring, reply_json_or_text)，按序消费匹配项。"""
    async def llm(prompt, **kw):
        for i, (needle, reply) in enumerate(script):
            if needle in prompt:
                script.pop(i)
                return reply if isinstance(reply, str) else json.dumps(reply)
        return json.dumps({"thought": "stop", "action": "answer",
                           "params": {"generation_mode": "direct"}})
    return llm


def make_retrieve(chunks):
    calls = []
    async def retrieve(tool_name, params):
        calls.append((tool_name, dict(params)))
        return list(chunks), {"paths_activated": [tool_name], "chunks_per_path": {tool_name: len(chunks)}}
    retrieve.calls = calls
    return retrieve


PLAN = {"standalone_query": "规范查询", "archetype": "factoid", "confidence": 0.9,
        "exact_terms": [], "suggested_expand": "none", "visual_intent": False,
        "entities_referenced": []}
GRADE_OK = {"sufficient": True, "facts": [
    {"id": "f1", "text": "事实", "status": "found", "chunks": ["c1"]}]}
VERIFY_OK = {"claims": [{"id": 0, "quote": "证据内容证据内容", "supported": True}]}


def _loop(script, chunks):
    llm = make_llm(script)
    return AgentLoop(
        model_pool=ModelPool(main_func=llm),
        registry=build_default_registry(),
        retrieve_fn=make_retrieve(chunks),
        rerank_fn=None,
        vision_fn=None,
    )


@pytest.mark.asyncio
async def test_fast_path_zero_decision_calls():
    chunks = [{"chunk_id": "c1", "content": "证据内容证据内容"}]
    script = [("prepare a user question", PLAN),
              ("fact ledger", GRADE_OK),
              ("Answer based ONLY", "答案：证据内容证据内容。"),
              ("verify answer claims", VERIFY_OK)]
    result = await _loop(script, chunks).run(
        "问题", SessionMemory(session_id="s", workspace_id="w"))
    assert isinstance(result, AgentResult)
    assert result.grounded is True and result.answer
    assert result.trace["agent_decisions"] == []  # 快速通道零决策 §4.5


@pytest.mark.asyncio
async def test_budget_exhaustion_returns_structured_refusal():
    chunks = [{"chunk_id": "c1", "content": "无关内容"}]
    grade_bad = {"sufficient": False, "facts": [
        {"id": "f1", "text": "找不到的事实", "status": "missing", "chunks": []}]}
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9})]
    loop = _loop(script, chunks)
    loop._grade_override = grade_bad  # 测试钩子：grader 恒不充分
    result = await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"),
                            budget=Budget(points=2, max_seconds=None))
    assert result.answer is None and result.refusal  # 结构化拒答 §8.3
    assert "找不到的事实" in json.dumps(result.refusal, ensure_ascii=False)


@pytest.mark.asyncio
async def test_repeat_action_rejected_and_noted():
    chunks = [{"chunk_id": "c1", "content": "x"}]
    decision = {"thought": "重复", "action": "search_dense",
                "params": {"query": "同查询", "top_k": 10}}
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9}),
              ("decide the next action", decision),
              ("decide the next action", decision)]  # 第二次重复
    loop = _loop(script, chunks)
    loop._grade_override = {"sufficient": False, "facts": []}
    result = await loop.run("问题", SessionMemory(session_id="s", workspace_id="w"),
                            budget=Budget(points=3, max_seconds=None))
    assert loop.retrieve_fn.calls.count(("search_dense", {"query": "同查询", "top_k": 10})) <= 1


@pytest.mark.asyncio
async def test_cancellation_mid_tool_call():
    session = SessionMemory(session_id="s", workspace_id="w")
    async def slow_retrieve(tool_name, params):
        await asyncio.sleep(10)
        return [], {}
    script = [("prepare a user question", {**PLAN, "archetype": "unknown", "confidence": 0.9}),
              ("decide the next action",
               {"thought": "", "action": "search_dense", "params": {"query": "q"}})]
    loop = _loop(script, [])
    loop.retrieve_fn = slow_retrieve
    task = asyncio.create_task(loop.run("问题", session, budget=Budget(points=5, max_seconds=None)))
    await asyncio.sleep(0.1)
    session.cancel_event.set()
    result = await asyncio.wait_for(task, timeout=2)  # 秒级中断，不等 10s
    assert result.cancelled is True
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_loop.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# raganything/agent/loop.py
"""Agent 主循环（spec §4/§6.5/§8.3/§9.3）。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import asyncio

from raganything.retrieval.json_utils import call_json_object
from raganything.retrieval.recovery_policy import RecoveryPolicy

from raganything.agent.budget import Budget
from raganything.agent.citations import verify_citations
from raganything.agent.decision import Decision, decision_signature, normalize_decision
from raganything.agent.evidence import EvidencePool, FactLedger
from raganything.agent.generate import generate_answer
from raganything.agent.grading import LedgerGrader, should_final_review
from raganything.agent.models import ModelPool
from raganything.agent.planner import PlanResult, make_plan
from raganything.agent.session import SessionMemory
from raganything.agent.tools import ToolRegistry
from raganything.agent.trace import TraceBuilder

logger = logging.getLogger(__name__)

MAX_STEPS = 8  # 硬上限 §4.3
# RecoveryPolicy profile → 工具名映射（降级大脑 §4.4）
_FALLBACK_TOOL = {"precise": "search_sparse", "semantic": "search_dense",
                  "multihop": "search_ppr", "full_v2": "search_hybrid",
                  "global": "search_graph", "local": "search_graph"}

_DECIDE_PROMPT = """\
You decide the next action for an evidence-gathering agent. Tools:
{cards}

Rules: start cheap, escalate only on evidence; prefer enlarging top_k on the same
tool before switching; targeted retrieval for missing facts beats query expansion;
when budget is low, prefer answering with current best evidence.
Archetype is a prior — override it when evidence disagrees (set "reclassify").

decide the next action. Output JSON only:
{{"thought": "<one sentence>", "action": "<tool name>", "params": {{...}},
  "stop": false, "reclassify": null}}

Archetype: {archetype}
Query: {query}
Evidence pool: {pool_summary}
Fact ledger: {ledger}
Action history:
{history}
Tool status: {tool_status}
Budget: {budget}
"""

RetrieveFn = Callable[[str, dict], Awaitable[tuple[list[dict], dict]]]
RerankFn = Callable[[str, list[str]], Awaitable[list[float]]]


@dataclass
class AgentResult:
    answer: str | None
    grounded: bool
    refusal: dict | None
    ledger: dict
    trace: dict
    cancelled: bool = False


@dataclass
class AgentLoop:
    model_pool: ModelPool
    registry: ToolRegistry
    retrieve_fn: RetrieveFn                  # (tool_name, params) -> (chunks, trace)
    rerank_fn: RerankFn | None = None        # None → canonical=发现分数最大值（降级可用）
    vision_fn: Callable | None = None        # None → inspect_image 不可用
    recovery: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    max_context_tokens: int = 12_000
    _grade_override: dict | None = None      # 单测钩子

    async def run(self, query: str, session: SessionMemory, *,
                  budget: Budget | None = None, **qp_kwargs: Any) -> AgentResult:
        plan = await make_plan(self.model_pool, query, session)
        budget = budget or Budget.for_archetype(plan.archetype)
        pool, ledger = EvidencePool(), FactLedger()
        grader = LedgerGrader(self.model_pool)
        tb = TraceBuilder(profile=f"agent:{plan.archetype}", query=query)
        tb.add_rewrite(plan.standalone_query)
        tried: set[tuple] = set()
        dup_rates: list[float] = []
        ledger_steps = 0
        cq = plan.standalone_query

        # ---- 快速通道 §4.5 ----
        if plan.fast_path:
            await self._execute_search(plan.preset["tool"],
                                       {"query": cq, "top_k": plan.preset["top_k"]},
                                       pool, session, cq, tb, step=0, budget=budget)
            grade = await self._grade(grader, cq, ledger, pool,
                                      list(pool.entries.values()))
            tb.add_grader_event(grade, cycle=0)
            if grade["sufficient"]:
                return await self._finish(cq, plan, pool, ledger, tb, budget, session,
                                          generation_mode=plan.preset["generation_mode"])

        # ---- 主循环 ----
        for step in range(MAX_STEPS):
            if session.cancel_event.is_set():
                return self._cancelled_result(ledger, tb, session, query)
            reason = budget.exhausted()
            if reason:
                return await self._exhausted(cq, plan, pool, ledger, tb, budget,
                                             session, reason)
            decision = await self._decide(plan, cq, pool, ledger, tb, budget,
                                          tried, step)
            if decision is None:  # 连降级都给不出动作
                return await self._exhausted(cq, plan, pool, ledger, tb, budget,
                                             session, "no_action")
            if decision.reclassify and decision.reclassify != plan.archetype:
                if budget.upgrade(decision.reclassify):
                    tb.add_reclassify(plan.archetype, decision.reclassify, cycle=step)
                    plan.archetype = decision.reclassify
            if decision.action == "answer":
                return await self._finish(
                    cq, plan, pool, ledger, tb, budget, session,
                    generation_mode=str(decision.params.get("generation_mode", "direct")))
            sig = decision_signature(decision)
            if sig in tried:
                continue  # 重复守卫：拒绝执行 §4.3（下一步 observation 含历史，模型可见）
            tried.add(sig)
            if decision.action == "rewrite_query":
                cq = str(decision.params.get("query") or cq)
                tb.add_rewrite(cq)
                budget.charge(points=self.registry.get(decision.action).cost)
                continue
            new_entries = await self._execute_search(
                decision.action, decision.params, pool, session, cq, tb,
                step=step, budget=budget)
            if new_entries is None:  # 取消
                return self._cancelled_result(ledger, tb, session, query)
            dup_rates.append(pool.last_dup_rate)
            for fact in ledger.missing():
                ledger.record_attempt(fact["id"], decision.action)
            grade = await self._grade(grader, cq, ledger, pool, new_entries)
            ledger_steps += 1
            tb.add_grader_event(grade, cycle=step)
            if grade["sufficient"]:
                if should_final_review(ledger_steps=ledger_steps, ledger=ledger,
                                       pool=pool, recent_dup_rates=dup_rates):
                    review = await grader.final_review(cq, pool)
                    tb.add_grader_event({**review, "final_review": True}, cycle=step)
                    if not review["sufficient"]:
                        fresh = FactLedger()
                        fresh.update({"facts": review.get("facts", [])})
                        ledger = fresh  # 分歧→重建账本回 loop §5.4
                        continue
                return await self._finish(
                    cq, plan, pool, ledger, tb, budget, session,
                    generation_mode=plan.preset["generation_mode"])
        return await self._exhausted(cq, plan, pool, ledger, tb, budget, session, "max_steps")

    # ---- 内部 ----

    async def _cancellable(self, coro, session: SessionMemory):
        task = asyncio.ensure_future(coro)
        waiter = asyncio.ensure_future(session.cancel_event.wait())
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            waiter.cancel()
            return task.result()
        task.cancel()  # CancelledError 传播进底层 I/O，秒级中断
        return None

    async def _execute_search(self, tool_name, params, pool, session, cq, tb, *,
                              step, budget):
        spec = self.registry.get(tool_name)
        budget.charge(points=spec.cost)
        result = await self._cancellable(self.retrieve_fn(tool_name, dict(params)), session)
        if result is None:
            return None
        chunks, rtrace = result
        new_entries = pool.add(chunks, step=step, tool=tool_name,
                               sub_query=str(params.get("query", cq)))
        session.cache_chunks(chunks)
        if new_entries:
            if self.rerank_fn is not None:
                scores = await self.rerank_fn(cq, [e.content for e in new_entries])
                pool.set_scores({e.chunk_id: s for e, s in zip(new_entries, scores)})
            else:
                pool.set_scores({e.chunk_id: max((p["rrf_score"] for p in e.provenance),
                                                 default=0.0)
                                 for e in new_entries})
        pool.evict()
        tb.add_retrieval_step(step_type=tool_name, query=str(params.get("query", cq)),
                              tool=tool_name, chunks=len(chunks), trace=rtrace, cycle=step)
        return new_entries

    async def _grade(self, grader, cq, ledger, pool, new_entries) -> dict:
        if self._grade_override is not None:
            ledger.update(self._grade_override)
            return dict(self._grade_override)
        return await grader.grade(cq, ledger, pool, new_entries=new_entries)

    async def _decide(self, plan, cq, pool, ledger, tb, budget, tried, step) -> Decision | None:
        history = "\n".join(
            f"{i + 1}. {d['action']}({d['params']}) " for i, d in
            enumerate(tb._trace["agent_decisions"])) or "(none)"
        tool_status = "search_ppr: ready" if self.vision_fn or True else ""
        prompt = _DECIDE_PROMPT.format(
            cards=self.registry.card_text(), archetype=plan.archetype, query=cq,
            pool_summary=pool.summary(), ledger=str(ledger.to_dict())[:1500],
            history=history, tool_status=tool_status, budget=budget.snapshot())
        for attempt in range(2):
            try:
                raw = await call_json_object(
                    lambda p, **kw: self.model_pool.call("planner", p, **kw),
                    prompt, max_tokens=256)
                d = normalize_decision(raw, self.registry, cq)
                tb.add_decision(thought=d.thought, action=d.action, params=d.params,
                                budget_snapshot=budget.snapshot(), fallback=False)
                return d
            except Exception as exc:
                prompt += f"\nPrevious output invalid: {exc}. Output ONLY the JSON object."
        # 确定性降级：RecoveryPolicy §4.4
        failure = ledger.missing()[0]["text"] if ledger.missing() else "partial_evidence"
        action = self.recovery.choose(
            failure_type="partial_evidence", original_profile="semantic",
            original_query=cq, tried_profiles=set(), tried_signatures=set())
        if action is None:
            return None
        tool = "decompose_search" if action.action_type == "decompose" else \
            _FALLBACK_TOOL.get(action.profile, "search_hybrid")
        d = Decision(thought=f"fallback:{failure[:50]}", action=tool,
                     params=self.registry.get(tool).clamp({"query": cq}), fallback=True)
        tb.add_decision(thought=d.thought, action=d.action, params=d.params,
                        budget_snapshot=budget.snapshot(), fallback=True)
        return d

    async def _finish(self, cq, plan, pool, ledger, tb, budget, session, *,
                      generation_mode) -> AgentResult:
        answer = await self._cancellable(
            generate_answer(self.model_pool, cq, pool, ledger, mode=generation_mode,
                            max_context_tokens=self.max_context_tokens,
                            visual_intent=plan.visual_intent), session)
        if answer is None:
            return self._cancelled_result(ledger, tb, session, cq)
        chunks = [{"chunk_id": e.chunk_id, "content": e.content} for e in pool.top(20)]
        grounded, ungrounded = await verify_citations(self.model_pool, cq, answer, chunks)
        tb.add_hallucination_event({"grounded": grounded,
                                    "ungrounded_claims": ungrounded}, cycle=0)
        if not grounded and generation_mode == "cot_reflect" and not budget.exhausted():
            # 生成修复 1 次 §10
            repair_q = " ".join(ungrounded)[:300]
            await self._execute_search("search_dense", {"query": repair_q, "top_k": 10},
                                       pool, session, cq, tb, step=MAX_STEPS, budget=budget)
            answer = await generate_answer(self.model_pool, cq, pool, ledger,
                                           mode=generation_mode,
                                           max_context_tokens=self.max_context_tokens)
            grounded, ungrounded = await verify_citations(self.model_pool, cq, answer, chunks)
            tb.add_hallucination_event({"grounded": grounded,
                                        "ungrounded_claims": ungrounded}, cycle=1)
        session.add_turn(cq, answer if grounded else "")
        unver = ledger.unverifiable()
        if unver and grounded:
            answer += "\n\n（以下细节在语料中无法证实：" + "；".join(f["text"] for f in unver) + "）"
        return AgentResult(answer=answer if grounded else answer,
                           grounded=grounded, refusal=None,
                           ledger=ledger.to_dict(),
                           trace=tb.build(terminal_reason="grounded" if grounded else "ungrounded",
                                          grounded=grounded))

    async def _exhausted(self, cq, plan, pool, ledger, tb, budget, session,
                         reason) -> AgentResult:
        if ledger.coverage >= 0.5 and pool.entries:
            answer = await generate_answer(
                self.model_pool, cq, pool, ledger, mode="direct",
                max_context_tokens=self.max_context_tokens)
            answer += "\n\n（基于不完整证据作答，未覆盖：" + \
                "；".join(f["text"] for f in ledger.missing()) + "）"
            session.add_turn(cq, answer)
            return AgentResult(answer=answer, grounded=False, refusal=None,
                               ledger=ledger.to_dict(),
                               trace=tb.build(terminal_reason=reason, grounded=False))
        refusal = {"reason": reason,
                   "missing_facts": [f["text"] for f in ledger.missing()],
                   "unverifiable": [f["text"] for f in ledger.unverifiable()],
                   "attempts": [d["action"] for d in tb._trace["agent_decisions"]]}
        session.add_turn(cq, "")
        return AgentResult(answer=None, grounded=False, refusal=refusal,
                           ledger=ledger.to_dict(),
                           trace=tb.build(terminal_reason=reason, grounded=False))

    def _cancelled_result(self, ledger, tb, session, query) -> AgentResult:
        session.add_turn(query, "", cancelled=True)
        session.cancel_event.clear()
        return AgentResult(answer=None, grounded=False,
                           refusal={"reason": "cancelled"},
                           ledger=ledger.to_dict(),
                           trace=tb.build(terminal_reason="cancelled", grounded=False),
                           cancelled=True)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_loop.py -v`
Expected: 4 PASS（注意 test_cancellation 必须 <2s 完成）

- [ ] **Step 5: 全量回归**

Run: `pytest tests/agent/ -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add raganything/agent/loop.py tests/agent/test_loop.py
git commit -m "feat(agent): main agent loop with cancellation, fallback and budget exhaustion paths"
```

---

### Task 14: Server 集成（/agent/chat + cancel + 409 + 治理联动）

**Files:**
- Modify: `server/app.py`（追加路由与 wiring；执行时先读现有 lifespan/`get_service` 结构再插入）
- Test: `tests/agent/test_server_routes.py`

执行时核实点（不可跳过）：
1. 读 `raganything/services/local_rag.py`，确认 service 暴露的文本 LLM callable 名称（`LedgerGrader`/planner 走它）与 reranker callable（CrossEncoder 封装），绑定到 `ModelPool(main_func=...)` 与 `rerank_fn=...`。
2. 读 `server/app.py` 的 `DELETE /workspace/{ws}/document/{doc_id}` handler，确认治理 provenance 查询方法签名（`GovernanceService.lookup`），在删除成功后追加 `session_store.drop_chunks(ws, chunk_ids)`。
3. `retrieve_fn` 绑定：`RetrievalRouter.route(query, QueryParam, profile_name=spec.profile)`，tool→profile 映射取自 `ToolSpec.profile`；`decompose_search` 复用 v2 `_decompose` 的 prompt 思路在 executor 内实现（LLM 拆分后 gather agent_hybrid）。
4. `vision_fn` 本期传 `None`（inspect_image 自动不可用）；接通 VLM 适配器列入后续计划。

- [ ] **Step 1: 写失败测试（路由层用 FastAPI TestClient + 假 loop）**

```python
# tests/agent/test_server_routes.py
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from raganything.agent.session import SessionStore
from server.agent_routes import build_agent_router  # 新模块，挂到主 app


class FakeLoop:
    def __init__(self, delay=0.0):
        self.delay = delay
    async def run(self, query, session, **kw):
        from raganything.agent.loop import AgentResult
        await asyncio.sleep(self.delay)
        return AgentResult(answer="答", grounded=True, refusal=None,
                           ledger={}, trace={"terminal_reason": "grounded"})


def make_app(loop=None):
    app = FastAPI()
    store = SessionStore()
    app.include_router(build_agent_router(store, loop or FakeLoop()))
    app.state.session_store = store
    return app, store


def test_chat_returns_answer_and_trace():
    app, _ = make_app()
    client = TestClient(app)
    r = client.post("/agent/chat", json={"workspace_id": "w", "session_id": "s", "query": "q"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "答" and "trace" in body


def test_concurrent_same_session_409():
    app, store = make_app(FakeLoop(delay=1.0))
    client = TestClient(app)
    import threading
    results = {}
    def first():
        results["a"] = client.post("/agent/chat",
                                   json={"workspace_id": "w", "session_id": "s", "query": "q1"})
    t = threading.Thread(target=first); t.start()
    import time; time.sleep(0.2)
    r2 = client.post("/agent/chat", json={"workspace_id": "w", "session_id": "s", "query": "q2"})
    t.join()
    assert r2.status_code == 409  # §6.4
    assert "cancel" in r2.json()["detail"]["hint"]


def test_cancel_endpoint_sets_event():
    app, store = make_app()
    client = TestClient(app)
    session = store.get("w", "s")
    r = client.post("/agent/sessions/s/cancel", params={"workspace_id": "w"})
    assert r.status_code == 200
    assert session.cancel_event.is_set()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/agent/test_server_routes.py -v`
Expected: FAIL（ModuleNotFoundError: server.agent_routes）

- [ ] **Step 3: 实现路由模块**

```python
# server/agent_routes.py
"""Agent 端点：/agent/chat、/agent/sessions/{id}/cancel（spec §6.4/6.5）。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from raganything.agent.session import SessionStore


class AgentChatRequest(BaseModel):
    workspace_id: str
    session_id: str
    query: str
    top_k: int | None = None
    max_seconds: float | None = None


def build_agent_router(store: SessionStore, agent_loop: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/agent/chat")
    async def agent_chat(req: AgentChatRequest):
        session = store.get(req.workspace_id, req.session_id)
        if session.lock.locked():
            raise HTTPException(status_code=409, detail={
                "error": "session_busy",
                "running_query": session.recent_turns[-1]["q"] if session.recent_turns else "",
                "hint": "wait or POST /agent/sessions/{id}/cancel first",
            })
        async with session.lock:
            session.cancel_event.clear()
            kwargs: dict = {}
            if req.max_seconds is not None:
                from raganything.agent.budget import Budget
                kwargs["budget"] = Budget(points=10, max_seconds=req.max_seconds)
            result = await agent_loop.run(req.query, session, **kwargs)
        return {
            "answer": result.answer, "grounded": result.grounded,
            "refusal": result.refusal, "ledger": result.ledger,
            "trace": result.trace, "cancelled": result.cancelled,
        }

    @router.post("/agent/sessions/{session_id}/cancel")
    async def agent_cancel(session_id: str, workspace_id: str):
        session = store.get(workspace_id, session_id)
        session.cancel_event.set()
        return {"status": "cancelling"}

    return router
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/agent/test_server_routes.py -v`
Expected: 3 PASS

- [ ] **Step 5: 接入主 app（执行时按核实点 1-4 落地）**

在 `server/app.py` 的 lifespan 中（PG pool 创建之后）添加：

```python
# --- agent v3 wiring（在 lifespan 内，app.state 赋值区域）---
from raganything.agent.loop import AgentLoop
from raganything.agent.models import ModelPool
from raganything.agent.session import SessionStore
from raganything.agent.tools import build_default_registry
from server.agent_routes import build_agent_router
import os

app.state.session_store = SessionStore()
_registry = build_default_registry()

async def _retrieve_fn(tool_name: str, params: dict):
    # 核实点 3：按 ToolSpec.profile 调 RetrievalRouter；decompose 走 executor 分支
    ...  # 执行时实现，绑定 service 内 router 实例

_judge = None
if os.getenv("RAGANYTHING_JUDGE_API_BASE"):
    _judge = ...  # 用 AsyncOpenAI(base_url=..., model=RAGANYTHING_JUDGE_MODEL) 包装

app.state.agent_loop = AgentLoop(
    model_pool=ModelPool(main_func=..., judge_func=_judge),  # 核实点 1
    registry=_registry,
    retrieve_fn=_retrieve_fn,
    rerank_fn=...,   # 核实点 1；拿不到则 None（降级可用）
    vision_fn=None,  # 核实点 4
)
app.include_router(build_agent_router(app.state.session_store, app.state.agent_loop))
```

并在 `DELETE /workspace/{ws}/document/{doc_id}` handler 删除成功路径追加（核实点 2）：

```python
prov_rows = await governance.lookup(workspace_id, doc_id)  # 执行时核实方法名/签名
chunk_ids = [r.chunk_id for r in prov_rows if getattr(r, "chunk_id", None)]
app.state.session_store.drop_chunks(workspace_id, chunk_ids)
```

在 `DELETE /workspace/{workspace_id}` 成功路径追加：

```python
app.state.session_store.invalidate_workspace(workspace_id)
```

- [ ] **Step 6: 手动冒烟（需后端依赖已启动）**

```bash
uvicorn server.app:app --port 9621 &
curl -s -X POST localhost:9621/agent/chat -H "Content-Type: application/json" \
  -d '{"workspace_id":"<已有workspace>","session_id":"smoke1","query":"<语料内的事实性问题>"}' | python -m json.tool
```
Expected: 返回 JSON 含 `answer`（非空）、`trace.agent_decisions`、`ledger.coverage`。
再连发两条共指问题（"它…"），确认第二条的 `trace.rewrite_history[1]` 是自包含改写。

- [ ] **Step 7: 全量回归 + Commit**

Run: `pytest tests/ -v`（agent 全部 + 既有测试无回归）

```bash
git add server/agent_routes.py server/app.py tests/agent/test_server_routes.py
git commit -m "feat(agent): /agent/chat and cancel endpoints with 409 busy-session contract"
```

---

## Self-Review 记录

- **Spec 覆盖**：§4 决策（T8/T13）、§5 池与账本与终审（T3/T4/T10）、§6 session/取消/409（T6/T13/T14）、§7 工具与 expand 声明（T7；MQE/HyDE 执行逻辑在 retrieve_fn 绑定层，见缺口说明）、§8 预算（T1/T13）、§9 画像（T9）、§10 生成（T11）、§11 多模态门（T11；inspect_image 执行体后续接 vision_fn）、§12 ModelPool/引文（T2/T5）、§13 trace（T12）、§14 评测（范围外，独立后续计划；`used_fallback` 分层标记已在 T12 落地）。
- **已知缺口（刻意延后，不阻塞可用性）**：① MQE/HyDE 的 expand 执行逻辑与 decompose executor 在核实点 3 的 `_retrieve_fn` 实现中完成，工具声明与参数契约已就位；② inspect_image 在 `vision_fn=None` 时不可用，三道门装填已生效；③ PPR 预热挂载、A/B 评测脚本、前端 stop 按钮为后续独立计划。
- **类型一致性**：`AgentResult` 字段、`PoolEntry.sort_key()`、`ToolSpec.clamp()`、`ledger.to_dict()` 跨任务签名已核对一致；`retrieve_fn(tool_name, params) -> (chunks, trace)` 契约在 T13 测试与 T14 绑定层一致。
