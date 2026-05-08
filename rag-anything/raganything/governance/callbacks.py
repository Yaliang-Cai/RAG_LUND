"""Bridge LightRAG callbacks into governance state."""
from __future__ import annotations

import asyncio
import logging
import time
from uuid import UUID

from raganything.callbacks import ProcessingCallback

logger = logging.getLogger(__name__)


class IngestProvenanceCallback(ProcessingCallback):
    """Manually-callable chunk-id collector.

    NOTE: LightRAG's upstream ProcessingCallback does NOT dispatch a
    'on_chunk_indexed' event, so this callback is never invoked during
    real ingest. Provenance capture in production happens via snapshot
    diff in GovernanceService.run_ingest. This class is retained for
    unit tests and as an opt-in manual API.
    """

    def __init__(self, workspace_id: str, doc_id: UUID):
        super().__init__()
        self.workspace_id = workspace_id
        self.doc_id = doc_id
        self.chunk_ids: list[str] = []

    async def on_chunk_indexed(self, chunk_id: str, **_: object) -> None:  # type: ignore[override]
        self.chunk_ids.append(chunk_id)


class JobProgressCallback(ProcessingCallback):
    """Debounced writer of `{parsed, indexed, total}` to ingest_jobs.progress."""

    def __init__(self, gov, job_id: UUID, *, interval_s: int = 5, chunk_interval: int = 10):
        super().__init__()
        self._gov = gov
        self._job_id = job_id
        self._interval_s = interval_s
        self._chunk_interval = chunk_interval
        self._state = {"parsed": 0, "indexed": 0, "total": 0}
        self._last_flush = time.monotonic()
        self._chunks_since_flush = 0
        self._lock = asyncio.Lock()

    async def _maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or self._chunks_since_flush >= self._chunk_interval \
                 or (now - self._last_flush) >= self._interval_s:
            async with self._lock:
                await self._gov.update_job_progress(self._job_id, dict(self._state))
            self._last_flush = now
            self._chunks_since_flush = 0

    async def on_progress(self, **kw: object) -> None:  # type: ignore[override]
        for key in ("parsed", "indexed", "total"):
            if key in kw and isinstance(kw[key], int):
                self._state[key] = kw[key]  # type: ignore[assignment]
        await self._maybe_flush()

    async def on_chunk_indexed(self, chunk_id: str, **_: object) -> None:  # type: ignore[override]
        self._state["indexed"] = self._state.get("indexed", 0) + 1
        self._chunks_since_flush += 1
        await self._maybe_flush()

    async def flush(self) -> None:
        await self._maybe_flush(force=True)
