"""Unit tests for recognition memory components (operate.py).

All LLM calls are mocked — no real API calls in this suite.
"""
import difflib
import pytest
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# _min_max_norm — import by copying the function under test directly, so
# tests don't depend on operate.py import chain (which needs Neo4j/Qdrant).
# ---------------------------------------------------------------------------

def _min_max_norm(scores: dict) -> dict:
    """Copied verbatim from operate.py for isolated testing."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        uniform = 1.0 if hi > 0.0 else 0.0
        return {k: uniform for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


class TestMinMaxNorm:
    def test_empty_dict_returns_empty(self):
        assert _min_max_norm({}) == {}

    def test_single_entry_normalises_to_1(self):
        result = _min_max_norm({"a": 0.7})
        # hi == lo, value > 0 → uniform 1.0
        assert result == {"a": 1.0}

    def test_all_zero_returns_zero(self):
        result = _min_max_norm({"a": 0.0, "b": 0.0})
        assert result == {"a": 0.0, "b": 0.0}

    def test_all_equal_nonzero_returns_one(self):
        # hi == lo == 0.9 → should NOT collapse to 0.0
        result = _min_max_norm({"a": 0.9, "b": 0.9, "c": 0.9})
        assert result == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_normal_range(self):
        result = _min_max_norm({"a": 0.0, "b": 0.5, "c": 1.0})
        assert abs(result["a"] - 0.0) < 1e-9
        assert abs(result["b"] - 0.5) < 1e-9
        assert abs(result["c"] - 1.0) < 1e-9

    def test_output_range_is_zero_to_one(self):
        import random
        scores = {f"e{i}": random.uniform(0.5, 0.95) for i in range(20)}
        result = _min_max_norm(scores)
        assert all(0.0 <= v <= 1.0 for v in result.values())
        assert min(result.values()) == pytest.approx(0.0)
        assert max(result.values()) == pytest.approx(1.0)
