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
async def test_found_fact_with_phantom_chunk_demoted_and_not_sufficient():
    # Grader declares sufficiency but cites a chunk id that is not in the pool
    # (fabricated/empty support) — the recall guard demotes it and blocks sufficiency.
    pool = _pool_with("c1")
    ledger = FactLedger()
    grader = LedgerGrader(FakePool({"sufficient": True, "facts": [
        {"id": "f1", "text": "真有支撑", "status": "found", "chunks": ["c1"]},
        {"id": "f2", "text": "幻觉支撑", "status": "found", "chunks": ["c999"]},
    ]}))
    result = await grader.grade("q", ledger, pool, new_entries=list(pool.entries.values()))
    assert result["sufficient"] is False           # invariant: a missing fact blocks it
    assert ledger.facts["f2"]["status"] == "missing"
    assert "f1" in pool.entries["c1"].supports


@pytest.mark.asyncio
async def test_all_found_real_support_is_sufficient():
    pool = _pool_with("c1", "c2")
    ledger = FactLedger()
    grader = LedgerGrader(FakePool({"sufficient": True, "facts": [
        {"id": "f1", "text": "事实一", "status": "found", "chunks": ["c1"]},
        {"id": "f2", "text": "事实二", "status": "found", "chunks": ["c2"]},
    ]}))
    result = await grader.grade("q", ledger, pool, new_entries=list(pool.entries.values()))
    assert result["sufficient"] is True


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


@pytest.mark.asyncio
async def test_grade_exception_leaves_ledger_unchanged():
    class Broken:
        async def call(self, role, prompt, **kw):
            raise RuntimeError("grader endpoint down")
    pool = _pool_with("c1")
    ledger = FactLedger()
    ledger.update({"facts": [{"id": "f1", "text": "原事实", "status": "found", "chunks": ["c1"]}]})
    before = ledger.to_dict()
    result = await LedgerGrader(Broken()).grade("q", ledger, pool, new_entries=list(pool.entries.values()))
    assert result == {"sufficient": False, "facts": []}
    assert ledger.to_dict() == before  # 异常不改账本


@pytest.mark.asyncio
async def test_ledger_prompt_capped_on_large_ledger():
    pool = _pool_with("c1")
    ledger = FactLedger()
    ledger.update({"facts": [
        {"id": f"f{i}", "text": f"事实{i}", "status": "found", "chunks": ["c1"]}
        for i in range(60)
    ]})
    fake = FakePool({"sufficient": True, "facts": []})
    await LedgerGrader(fake).grade("q", ledger, pool, new_entries=list(pool.entries.values()))
    # prompt 中账本事实数被截断（不应包含全部 60 条）
    assert fake.prompts[0].count('"text"') <= 41  # <=cap(40)+余量
