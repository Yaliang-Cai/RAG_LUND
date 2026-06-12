"""角色→端点映射 + 回退/熔断（spec §12）。"""
from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

LLMFunc = Callable[..., Awaitable[str]]
JUDGE_ROLES = frozenset({"grader", "checker", "rewriter", "summarizer"})
ALL_ROLES = JUDGE_ROLES | {"planner", "generator"}


class ModelPool:
    def __init__(
        self,
        main_func: LLMFunc,
        judge_func: LLMFunc | None = None,
        *,
        breaker_threshold: int = 5,
        probe_interval: float = 60.0,
    ) -> None:
        self._main = main_func
        self._judge = judge_func
        self._threshold = breaker_threshold
        self._probe_interval = probe_interval
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def breaker_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._probe_interval:
            return False  # 半开：放一次探测 §12.3
        return True

    async def call(self, role: str, prompt: str, **kwargs: Any) -> str:
        if role in JUDGE_ROLES and self._judge is not None and not self.breaker_open:
            try:
                result = await self._judge(prompt, **kwargs)
                self._consecutive_failures = 0
                self._opened_at = None
                return result
            except Exception:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._threshold:
                    self._opened_at = time.monotonic()
                logger.warning("judge endpoint failed (role=%s), fallback to main", role, exc_info=True)
        return await self._main(prompt, **kwargs)
