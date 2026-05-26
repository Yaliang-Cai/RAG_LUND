from __future__ import annotations

import pytest


class _WhitespaceTokenizer:
    def encode(self, text):
        return str(text).split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.mark.asyncio
async def test_table_processor_truncates_large_table_body_for_prompt_and_chunk(
    monkeypatch,
):
    monkeypatch.setenv("RAGANYTHING_MULTIMODAL_PROMPT_MAX_INPUT_TOKENS", "500")
    monkeypatch.setenv("RAGANYTHING_MULTIMODAL_CHUNK_MAX_TOKENS", "300")
    from raganything.modalprocessors import TableModalProcessor

    body = "\n".join(
        f"row-{index},value-{index}," + ("x" * 120) for index in range(200)
    )
    processor = object.__new__(TableModalProcessor)
    captured = {}

    async def fake_caption_func(prompt, system_prompt=None):
        captured["prompt"] = prompt
        return "{}"

    async def fake_create_entity_and_chunk(
        modal_chunk,
        entity_info,
        file_path,
        batch_mode,
        doc_id,
        chunk_order_index,
        page_idx=0,
    ):
        captured["chunk"] = modal_chunk
        return modal_chunk, entity_info, []

    processor.modal_caption_func = fake_caption_func
    processor._parse_table_response = lambda response, entity_name=None: (
        "short analysis",
        {"entity_name": "table", "entity_type": "table", "summary": "summary"},
    )
    processor._create_entity_and_chunk = fake_create_entity_and_chunk

    await TableModalProcessor.generate_description_only(
        processor,
        {"table_body": body, "table_caption": ["caption"]},
        "table",
    )
    await TableModalProcessor.process_multimodal_content(
        processor,
        {"table_body": body, "table_caption": ["caption"]},
        "table",
    )

    assert len(captured["prompt"]) < len(body)
    assert "truncated" in captured["prompt"].lower()
    assert len(captured["chunk"]) < len(body)
    assert "truncated" in captured["chunk"].lower()


def test_processor_chunk_template_truncates_large_table_body(monkeypatch):
    monkeypatch.setenv("RAGANYTHING_MULTIMODAL_CHUNK_MAX_TOKENS", "300")
    from raganything.processor import ProcessorMixin

    body = "\n".join(
        f"row-{index},value-{index}," + ("x" * 120) for index in range(200)
    )
    processor = object.__new__(ProcessorMixin)

    chunk = processor._apply_chunk_template(
        "table",
        {"table_body": body, "table_caption": ["caption"]},
        "short analysis",
    )

    assert len(chunk) < len(body)
    assert "truncated" in chunk.lower()


@pytest.mark.asyncio
async def test_table_processor_fits_full_prompt_to_context_budget(monkeypatch):
    monkeypatch.setenv("RAGANYTHING_LLM_CONTEXT_MAX_TOKENS", "800")
    monkeypatch.setenv("RAGANYTHING_LLM_CONTEXT_RESERVED_TOKENS", "20")
    monkeypatch.setenv("RAGANYTHING_INGEST_MAX_TOKENS", "40")
    monkeypatch.setenv("RAGANYTHING_MULTIMODAL_PROMPT_MAX_INPUT_TOKENS", "1000")
    from raganything.modalprocessors import TableModalProcessor
    from raganything.prompt import PROMPTS

    processor = object.__new__(TableModalProcessor)
    processor.tokenizer = _WhitespaceTokenizer()
    captured = {}

    async def fake_caption_func(prompt, system_prompt=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return "{}"

    processor.modal_caption_func = fake_caption_func
    processor._parse_table_response = lambda response, entity_name=None: (
        "short analysis",
        {"entity_name": "table", "entity_type": "table", "summary": "summary"},
    )

    body = " ".join(f"cell{index}" for index in range(1000))
    await TableModalProcessor.generate_description_only(
        processor,
        {"table_body": body, "table_caption": ["caption"]},
        "table",
    )

    full_input = f"{PROMPTS['TABLE_ANALYSIS_SYSTEM']}\n{captured['prompt']}"
    assert len(processor.tokenizer.encode(full_input)) <= 740
    assert "truncated" in captured["prompt"].lower()
