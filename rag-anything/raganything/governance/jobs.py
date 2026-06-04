"""In-process JobRunner — one asyncio.Task per job, state delegated to GovernanceService.

The submit/cancel/stop interface is the swap point for a future ARQ + Redis backend.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict
from uuid import UUID

logger = logging.getLogger(__name__)

CoroFactory = Callable[[], Awaitable[None]]


class JobRunner:
    def __init__(self, gov, max_concurrent: int = 1):
        self._gov = gov
        self._sem = asyncio.Semaphore(max_concurrent)
        self._tasks: Dict[UUID, asyncio.Task] = {}
        self._stopping = False

    async def start(self) -> None:
        """Reserved for symmetry with future Redis-backed runners. No-op today."""
        logger.info("JobRunner started (max_concurrent=%d)", self._sem._value)

    async def submit(self, job_id: UUID, coro_factory: CoroFactory) -> None:
        if self._stopping:
            raise RuntimeError("JobRunner is shutting down; refusing new submissions")
        task = asyncio.create_task(self._run(job_id, coro_factory))
        self._tasks[job_id] = task

    async def _run(self, job_id: UUID, coro_factory: CoroFactory) -> None:
        async with self._sem:
            try:
                await self._gov.mark_job_running(job_id)
                await coro_factory()
                await self._gov.mark_job_done(job_id)
            except asyncio.CancelledError:
                await self._gov.mark_job_failed(job_id, "cancelled")
                raise
            except Exception as exc:
                logger.exception("job %s failed", job_id)
                await self._gov.mark_job_failed(job_id, str(exc))
            finally:
                self._tasks.pop(job_id, None)

    async def cancel(self, job_id: UUID) -> bool:
        t = self._tasks.get(job_id)
        if t and not t.done():
            t.cancel()
            return True
        return False

    async def stop(self, grace_period: int) -> None:
        self._stopping = True
        if not self._tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=grace_period,
            )
        except asyncio.TimeoutError:
            logger.warning("JobRunner.stop: %d tasks exceeded grace; cancelling",
                           len(self._tasks))
            for t in list(self._tasks.values()):
                t.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
