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
