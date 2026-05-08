import asyncio
import pytest
from uuid import uuid4

from raganything.governance.callbacks import IngestProvenanceCallback, JobProgressCallback

pytestmark = pytest.mark.asyncio


async def test_provenance_callback_collects_chunk_ids():
    cb = IngestProvenanceCallback("w1", uuid4())
    await cb.on_chunk_indexed("c1")
    await cb.on_chunk_indexed("c2")
    assert cb.chunk_ids == ["c1", "c2"]


class FakeGov:
    def __init__(self):
        self.writes: list[dict] = []
    async def update_job_progress(self, jid, progress):
        self.writes.append(dict(progress))


async def test_progress_callback_debounces_by_chunk_count():
    gov = FakeGov()
    cb = JobProgressCallback(gov, uuid4(), interval_s=999, chunk_interval=3)
    await cb.on_chunk_indexed("a")
    await cb.on_chunk_indexed("b")
    assert gov.writes == []  # not yet
    await cb.on_chunk_indexed("c")
    assert len(gov.writes) == 1
    assert gov.writes[0]["indexed"] == 3


async def test_progress_callback_flush_forces_write():
    gov = FakeGov()
    cb = JobProgressCallback(gov, uuid4(), interval_s=999, chunk_interval=999)
    await cb.on_progress(parsed=5, total=10)
    assert gov.writes == []
    await cb.flush()
    assert len(gov.writes) == 1
    assert gov.writes[0]["parsed"] == 5
    assert gov.writes[0]["total"] == 10
