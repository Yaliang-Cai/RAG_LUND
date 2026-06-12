# raganything/agent/budget.py
"""成本点预算 + 墙钟/token 双护栏（spec §8）。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

ARCHETYPE_POINTS: dict[str, int] = {
    "factoid": 6, "comparison": 10, "summary": 12, "multihop": 16, "unknown": 10,
}
DEFAULT_MAX_TOKENS = 30_000
DEFAULT_MAX_SECONDS = 60.0
SOFT_RATIO = 0.2
SOFT_TIME_RATIO = 0.75


@dataclass
class Budget:
    points: float
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_seconds: float | None = DEFAULT_MAX_SECONDS
    spent_points: float = 0.0
    spent_tokens: int = 0
    _upgraded: bool = field(default=False, init=False, repr=False)
    _start: float = field(default_factory=time.monotonic, init=False, repr=False)

    @classmethod
    def for_archetype(cls, archetype: str, **kwargs) -> "Budget":
        return cls(points=ARCHETYPE_POINTS.get(archetype, ARCHETYPE_POINTS["unknown"]), **kwargs)

    def charge(self, *, points: float = 0.0, tokens: int = 0) -> None:
        self.spent_points += points
        self.spent_tokens += tokens

    @property
    def remaining_points(self) -> float:
        return self.points - self.spent_points

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def exhausted(self) -> str | None:
        if self.remaining_points <= 0:
            return "points"
        if self.spent_tokens >= self.max_tokens:
            return "tokens"
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            return "wall_clock"
        return None

    def low(self) -> bool:
        if self.remaining_points <= self.points * SOFT_RATIO:
            return True
        if self.max_seconds is not None and self.elapsed >= self.max_seconds * SOFT_TIME_RATIO:
            return True  # 60s 护栏 → 45s 软阈值
        return False

    def upgrade(self, archetype: str) -> bool:
        """改判升档：补到新画像额度，每轮一次（spec §9.3）。"""
        if self._upgraded:
            return False
        self._upgraded = True
        self.points = max(self.points, ARCHETYPE_POINTS.get(archetype, self.points))
        return True

    def snapshot(self) -> dict:
        return {
            "remaining_points": round(self.remaining_points, 2),
            "spent_points": round(self.spent_points, 2),
            "spent_tokens": self.spent_tokens,
            "elapsed_seconds": round(self.elapsed, 2),
            "low": self.low(),
        }
