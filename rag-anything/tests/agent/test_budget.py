# tests/agent/test_budget.py
import time
from raganything.agent.budget import Budget, ARCHETYPE_POINTS


def test_archetype_points_table():
    assert ARCHETYPE_POINTS == {
        "factoid": 6, "comparison": 10, "summary": 12, "multihop": 16, "unknown": 10,
    }


def test_charge_and_exhaustion():
    b = Budget.for_archetype("factoid", max_tokens=100, max_seconds=None)
    assert b.exhausted() is None
    b.charge(points=6)
    assert b.exhausted() == "points"
    b2 = Budget.for_archetype("factoid", max_tokens=100, max_seconds=None)
    b2.charge(tokens=100)
    assert b2.exhausted() == "tokens"


def test_wall_clock_guardrail_optional():
    b = Budget(points=5, max_seconds=0.01)
    time.sleep(0.02)
    assert b.exhausted() == "wall_clock"
    b2 = Budget(points=5, max_seconds=None)  # 评测模式关护栏 §8.1
    assert b2.exhausted() is None


def test_low_soft_threshold():
    b = Budget(points=10, max_seconds=None)
    b.charge(points=8.5)
    assert b.low() is True  # 剩 1.5/10 ≤ 20%


def test_upgrade_once_only():
    b = Budget.for_archetype("factoid", max_seconds=None)
    assert b.upgrade("multihop") is True
    assert b.points == ARCHETYPE_POINTS["multihop"]
    assert b.upgrade("summary") is False  # 每轮最多升一次 §9.3


def test_snapshot_keys():
    snap = Budget(points=10).snapshot()
    assert set(snap) >= {"remaining_points", "spent_tokens", "elapsed_seconds"}
