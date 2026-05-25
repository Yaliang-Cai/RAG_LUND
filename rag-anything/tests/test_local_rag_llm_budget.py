from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import pytest


def _stub_sentence_transformers(monkeypatch):
    stub = types.ModuleType("sentence_transformers")

    class _DummyCrossEncoder:
        pass

    class _DummySentenceTransformer:
        pass

    stub.CrossEncoder = _DummyCrossEncoder
    stub.SentenceTransformer = _DummySentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)


class _WhitespaceTokenizer:
    def encode(self, text):
        return str(text).split()


@pytest.mark.asyncio
async def test_llm_model_func_reduces_max_tokens_when_context_is_nearly_full(
    monkeypatch,
):
    _stub_sentence_transformers(monkeypatch)
    from raganything.services.local_rag import LocalRagSettings, build_llm_model_func

    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    settings = LocalRagSettings(
        ingest_max_tokens=8192,
        llm_context_max_tokens=100,
        llm_context_reserved_tokens=10,
    )
    fn = build_llm_model_func(
        settings,
        client,
        logging.getLogger("test_llm_budget"),
        "model",
        tokenizer=_WhitespaceTokenizer(),
    )
    system_prompt = "knowledge graph specialist extracting entities and relationships"
    prompt = (
        "extract entities and relationships from the input text "
        + " ".join("token" for _ in range(80))
    )

    result = await fn(prompt, system_prompt=system_prompt)

    assert result == "ok"
    assert captured["max_tokens"] < 8192
    assert captured["max_tokens"] >= 1
