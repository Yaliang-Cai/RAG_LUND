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
