import asyncio
import pytest
from uuid import uuid4

from raganything.governance.callbacks import IngestProvenanceCallback, JobProgressCallback

pytestmark = pytest.mark.asyncio


async def test_provenance_callback_captures_lightrag_doc_id():
    cb = IngestProvenanceCallback("w1", uuid4())
    assert cb.lightrag_doc_id is None
    # on_document_complete is sync per ProcessingCallback contract
    cb.on_document_complete(
        file_path="/tmp/demo.md",
        doc_id="doc-abc123",
        duration_seconds=1.5,
    )
    assert cb.lightrag_doc_id == "doc-abc123"
    assert cb.file_path == "/tmp/demo.md"


async def test_provenance_callback_ignores_empty_doc_id():
    cb = IngestProvenanceCallback("w1", uuid4())
    cb.on_document_complete(file_path="/tmp/demo.md", doc_id="")
    assert cb.lightrag_doc_id is None


class FakeGov:
    def __init__(self):
        self.writes: list[dict] = []

    async def update_job_progress(self, jid, progress):
        self.writes.append(dict(progress))


async def test_progress_callback_writes_on_parse_complete():
    gov = FakeGov()
    cb = JobProgressCallback(gov, uuid4(), interval_s=0)
    cb.on_parse_complete(file_path="/tmp/x.md", content_blocks=8)
    await asyncio.sleep(0.05)
    assert len(gov.writes) >= 1
    assert gov.writes[-1]["total"] == 8
    assert gov.writes[-1]["parsed"] == 8


async def test_progress_callback_writes_on_text_insert_complete():
    gov = FakeGov()
    cb = JobProgressCallback(gov, uuid4(), interval_s=0)
    cb.on_parse_complete(file_path="/tmp/x.md", content_blocks=8)
    cb.on_text_insert_complete(file_path="/tmp/x.md")
    await asyncio.sleep(0.05)
    assert gov.writes[-1]["indexed"] == 8


async def test_progress_callback_final_flush_is_async():
    gov = FakeGov()
    cb = JobProgressCallback(gov, uuid4(), interval_s=999)
    cb.on_parse_complete(file_path="/tmp/x.md", content_blocks=4)
    await asyncio.sleep(0.05)
    initial = len(gov.writes)
    await cb.flush()
    assert len(gov.writes) == initial + 1
    assert gov.writes[-1]["total"] == 4
