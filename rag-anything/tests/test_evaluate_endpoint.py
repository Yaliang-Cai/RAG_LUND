import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

for _mod in [
    "sentence_transformers",
    "sentence_transformers.cross_encoder",
    "torch",
    "raganything.processor",
    "raganything.batch",
    "raganything.batch_parser",
    "raganything.raganything",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import raganything as _ra
_ra.RAGAnything = MagicMock()
_ra.RAGAnythingConfig = MagicMock()


def _make_service():
    from raganything.services.local_rag import LocalRagService, LocalRagSettings
    settings = MagicMock(spec=LocalRagSettings)
    service = LocalRagService.__new__(LocalRagService)
    service.settings = settings
    service.logger = MagicMock()
    return service


@pytest.mark.asyncio
async def test_evaluate_answer_returns_score_and_gap():
    """evaluate_answer must call AnswerEvaluator.evaluate and return its result."""
    service = _make_service()

    fake_rag = MagicMock()
    fake_rag.lightrag.llm_model_func = AsyncMock()
    service.get_rag = AsyncMock(return_value=fake_rag)

    with patch("raganything.retrieval.evaluator.AnswerEvaluator") as MockEval:
        instance = MockEval.return_value
        instance.evaluate = AsyncMock(return_value={"score": 0.87, "gap": ""})

        result = await service.evaluate_answer("ws1", "What is X?", "X is Y.")

    MockEval.assert_called_once_with(fake_rag.lightrag.llm_model_func)
    instance.evaluate.assert_awaited_once_with("What is X?", "X is Y.")
    assert result == {"score": 0.87, "gap": ""}


@pytest.mark.asyncio
async def test_evaluate_answer_with_gap():
    """Gap string is forwarded from the evaluator."""
    service = _make_service()

    fake_rag = MagicMock()
    fake_rag.lightrag.llm_model_func = AsyncMock()
    service.get_rag = AsyncMock(return_value=fake_rag)

    with patch("raganything.retrieval.evaluator.AnswerEvaluator") as MockEval:
        instance = MockEval.return_value
        instance.evaluate = AsyncMock(
            return_value={"score": 0.45, "gap": "Missing accuracy numbers"}
        )

        result = await service.evaluate_answer("ws1", "What is the accuracy?", "It is good.")

    assert result["score"] == 0.45
    assert "accuracy" in result["gap"]
