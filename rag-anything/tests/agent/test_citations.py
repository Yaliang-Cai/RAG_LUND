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
