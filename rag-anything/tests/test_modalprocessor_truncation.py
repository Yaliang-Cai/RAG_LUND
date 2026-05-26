from __future__ import annotations

import pytest


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
