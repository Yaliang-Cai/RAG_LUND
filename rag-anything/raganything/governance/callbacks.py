"""Bridge LightRAG callbacks into governance state.

ProcessingCallback dispatches synchronous methods only. Our overrides
must be `def` (not `async def`) — see CallbackManager.dispatch which calls
the handler without awaiting it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from uuid import UUID

from raganything.callbacks import ProcessingCallback

logger = logging.getLogger(__name__)


class IngestProvenanceCallback(ProcessingCallback):
    """Capture LightRAG's internal `full_doc_id` when ingest completes.

    LightRAG identifies docs by content-hash (`doc-<sha256>`) internally,
    distinct from our governance doc_id (UUID). We need the LightRAG id so
    we can query `lr.text_chunks` after ingest and find chunks belonging
    to the doc we just ingested.

    `on_document_complete` is synchronous per ProcessingCallback's contract.
    """

    def __init__(self, workspace_id: str, doc_id: UUID):
        super().__init__()
        self.workspace_id = workspace_id
        self.doc_id = doc_id
        self.lightrag_doc_id: Optional[str] = None
        self.file_path: Optional[str] = None

    def on_document_complete(  # type: ignore[override]
        self,
        file_path: str,
        doc_id: str = "",
        duration_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        if doc_id:
            self.lightrag_doc_id = doc_id


class JobProgressCallback(ProcessingCallback):
    """Write progress to ingest_jobs.progress on real LightRAG events.

    Hooks the synchronous ProcessingCallback events that LightRAG actually
    dispatches. Each handler is sync; it schedules an async PG write via
    asyncio.run_coroutine_threadsafe(...) on the calling loop, or via
    create_task when already on the loop. Writes are debounced.
    """

    def __init__(
        self,
        gov,
        job_id: UUID,
        *,
        interval_s: int = 5,
    ):
        super().__init__()
        self._gov = gov
        self._job_id = job_id
        self._interval_s = interval_s
        self._state = {"parsed": 0, "indexed": 0, "total": 0}
        self._last_flush = time.monotonic()
        self._lock = asyncio.Lock()
        # Capture the loop on construction so sync callbacks can dispatch back to it.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _schedule_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_flush) < self._interval_s:
            return
        self._last_flush = now
        snapshot = dict(self._state)
        if self._loop is None:
            return
        async def _do_flush():
            async with self._lock:
                try:
                    await self._gov.update_job_progress(self._job_id, snapshot)
                except Exception:
                    logger.exception("progress flush failed for job %s", self._job_id)
        try:
            if asyncio.get_running_loop() is self._loop:
                self._loop.create_task(_do_flush())
                return
        except RuntimeError:
            pass
        asyncio.run_coroutine_threadsafe(_do_flush(), self._loop)

    def on_parse_complete(  # type: ignore[override]
        self,
        file_path: str,
        content_blocks: int = 0,
        doc_id: str = "",
        duration_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._state["total"] = content_blocks
        self._state["parsed"] = content_blocks
        self._schedule_flush()

    def on_text_insert_complete(  # type: ignore[override]
        self, file_path: str, duration_seconds: float = 0.0, **kwargs: Any
    ) -> None:
        self._state["indexed"] = self._state.get("total", 0)
        self._schedule_flush()

    def on_multimodal_item_complete(  # type: ignore[override]
        self,
        file_path: str,
        item_index: int = 0,
        item_type: str = "",
        total_items: int = 0,
        **kwargs: Any,
    ) -> None:
        self._state["indexed"] = max(self._state.get("indexed", 0), item_index + 1)
        if total_items and total_items > self._state.get("total", 0):
            self._state["total"] = total_items
        self._schedule_flush()

    def on_document_complete(  # type: ignore[override]
        self, file_path: str, doc_id: str = "", duration_seconds: float = 0.0, **kwargs: Any
    ) -> None:
        if self._state.get("total", 0) > 0:
            self._state["indexed"] = self._state["total"]
        self._schedule_flush(force=True)

    async def flush(self) -> None:
        """Async final flush — call after rag.ingest() returns."""
        async with self._lock:
            try:
                await self._gov.update_job_progress(self._job_id, dict(self._state))
            except Exception:
                logger.exception("final progress flush failed for job %s", self._job_id)
        self._last_flush = time.monotonic()
