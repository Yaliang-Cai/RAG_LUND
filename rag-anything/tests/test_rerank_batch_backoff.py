import logging
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")

    class _DummyCrossEncoder:
        pass

    class _DummySentenceTransformer:
        pass

    sentence_transformers_stub.CrossEncoder = _DummyCrossEncoder
    sentence_transformers_stub.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers_stub

from raganything.services.local_rag import LocalRagSettings, build_rerank_func


class _RecordingReranker:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, int]] = []

    def predict(self, pairs, batch_size=32):
        self.calls.append({"pairs": len(pairs), "batch_size": int(batch_size)})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _settings(
    *,
    rerank_batch_size: int = 32,
    rerank_enable_oom_backoff: bool = True,
    rerank_min_batch_size: int = 4,
):
    return SimpleNamespace(
        rerank_batch_size=rerank_batch_size,
        rerank_enable_oom_backoff=rerank_enable_oom_backoff,
        rerank_min_batch_size=rerank_min_batch_size,
    )


def test_local_rag_settings_reads_rerank_backoff_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAGANYTHING_RERANK_BATCH_SIZE", "64")
    monkeypatch.setenv("RAGANYTHING_RERANK_ENABLE_OOM_BACKOFF", "false")
    monkeypatch.setenv("RAGANYTHING_RERANK_MIN_BATCH_SIZE", "8")

    settings = LocalRagSettings.from_env()

    assert settings.rerank_batch_size == 64
    assert settings.rerank_enable_oom_backoff is False
    assert settings.rerank_min_batch_size == 8


@pytest.mark.asyncio
async def test_rerank_uses_configured_batch_size():
    reranker = _RecordingReranker([[0.2, 0.9, 0.1]])
    rerank_func = build_rerank_func(
        _settings(),
        reranker,
        logging.getLogger(__name__),
    )

    results = await rerank_func("query", ["a", "b", "c"], top_n=2)

    assert reranker.calls == [{"pairs": 3, "batch_size": 32}]
    assert results == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]


@pytest.mark.asyncio
async def test_rerank_oom_backoff_retries_from_scratch(caplog: pytest.LogCaptureFixture):
    reranker = _RecordingReranker(
        [
            RuntimeError("CUDA out of memory while scoring rerank batch"),
            [0.3, 0.8, 0.1],
        ]
    )
    rerank_func = build_rerank_func(
        _settings(),
        reranker,
        logging.getLogger(__name__),
    )

    with caplog.at_level(logging.WARNING):
        results = await rerank_func("query", ["a", "b", "c"], top_n=3)

    assert reranker.calls == [
        {"pairs": 3, "batch_size": 32},
        {"pairs": 3, "batch_size": 16},
    ]
    assert results[0] == {"index": 1, "relevance_score": 0.8}
    assert any("Rerank OOM backoff" in message for message in caplog.messages)
    assert any("Retrying full rerank from scratch" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_rerank_oom_falls_back_after_min_batch(caplog: pytest.LogCaptureFixture):
    reranker = _RecordingReranker(
        [
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA out of memory"),
        ]
    )
    rerank_func = build_rerank_func(
        _settings(),
        reranker,
        logging.getLogger(__name__),
    )

    with caplog.at_level(logging.WARNING):
        results = await rerank_func("query", ["a", "b"], top_n=2)

    assert reranker.calls == [
        {"pairs": 2, "batch_size": 32},
        {"pairs": 2, "batch_size": 16},
        {"pairs": 2, "batch_size": 8},
        {"pairs": 2, "batch_size": 4},
    ]
    assert results == []
    assert any("Rerank OOM fallback" in message for message in caplog.messages)
    assert any("Falling back to original retrieved items without rerank" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_rerank_non_oom_error_does_not_retry(caplog: pytest.LogCaptureFixture):
    reranker = _RecordingReranker([ValueError("tokenizer mismatch")])
    rerank_func = build_rerank_func(
        _settings(),
        reranker,
        logging.getLogger(__name__),
    )

    with caplog.at_level(logging.ERROR):
        results = await rerank_func("query", ["a", "b"], top_n=2)

    assert reranker.calls == [{"pairs": 2, "batch_size": 32}]
    assert results == []
    assert any("Rerank Error" in message for message in caplog.messages)
