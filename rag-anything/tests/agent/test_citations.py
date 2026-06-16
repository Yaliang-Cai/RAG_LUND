import json
import pytest
from raganything.agent.citations import split_claims, verify_citations, verify_answer


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


@pytest.mark.asyncio
async def test_cross_chunk_spliced_quote_rejected():
    chunks = [{"chunk_id": "a", "content": "太阳能板很常见"},
              {"chunk_id": "b", "content": "发电效率逐年提升"}]
    payload = {"claims": [
        {"id": 0, "quote": "太阳能板很常见发电效率逐年提升", "supported": True},  # 跨块拼接 → 拒
    ]}
    grounded, ungrounded = await verify_citations(
        FakePool(payload), "q", "太阳能板很常见发电效率逐年提升。", chunks)
    assert grounded is False and len(ungrounded) == 1


@pytest.mark.asyncio
async def test_supported_with_empty_quote_rejected():
    chunks = [{"chunk_id": "c1", "content": "任何内容"}]
    payload = {"claims": [{"id": 0, "quote": "", "supported": True}]}
    grounded, ungrounded = await verify_citations(
        FakePool(payload), "q", "这是一个足够长的声明。", chunks)
    assert grounded is False


@pytest.mark.asyncio
async def test_verify_answer_returns_supporting_spans():
    chunks = [{"chunk_id": "c1", "content": "华为2023年5G营收为1234亿元。"},
              {"chunk_id": "c2", "content": "无关内容。"}]
    payload = {"claims": [
        {"id": 0, "quote": "华为2023年5G营收为1234亿元", "supported": True},
        {"id": 1, "quote": "捏造引文", "supported": True},
    ]}
    res = await verify_answer(
        FakePool(payload), "q", "华为2023年5G营收为1234亿元。另一个无法支撑的断言。", chunks)
    assert res.grounded is False                      # one claim fabricated
    assert len(res.citations) == 1                    # only the real span survives
    assert res.citations[0]["chunk_id"] == "c1"       # mapped to the matching chunk
    assert "1234" in res.citations[0]["quote"]


def test_markdown_structure_excluded_from_claims():
    answer = (
        "华为2023年5G营收约1234亿元。\n"
        "### 业务概述\n"
        "| 指标 | 数值 |\n"
        "| --- | --- |\n"
        "- **营收**：同比增长5% [c1]\n"
        "公式为 $E = mc^2$ 表示能量。\n"
        "---\n"
    )
    claims = split_claims(answer)
    # Real prose survives; headings / separators / list scaffolding / inline citation
    # tokens / LaTeX spans do not become "claims".
    assert any("华为2023年5G营收" in c for c in claims)
    assert any("营收" in c and "同比增长5" in c for c in claims)
    assert not any("###" in c or "业务概述" == c for c in claims)
    assert not any(set(c) <= set("|-: ") for c in claims)   # no table separator rows
    assert not any("[c1]" in c for c in claims)              # citation token stripped
    assert not any("mc^2" in c for c in claims)              # LaTeX span stripped


def test_duplicate_claims_deduplicated():
    # The same sentence appearing as both lead and a restated bullet collapses to one.
    answer = "结论是营收增长。\n- 结论是营收增长 [c1]\n"
    assert split_claims(answer) == ["结论是营收增长"]


def test_known_behavior_abbreviation_splitting():
    # 已知局限（启发式分句）："Fig. " 缩写被当作句界切开；短片段被 min_len 过滤。
    # 固化当前行为——未来改进分句器时需有意识更新本测试。
    # 实际输出：['3 shows the result clearly']（"Fig" 仅 3 字符，被过滤）
    claims = split_claims("Fig. 3 shows the result clearly.")
    assert "3 shows the result clearly" in claims
    assert all(len(c) >= 6 for c in claims)
