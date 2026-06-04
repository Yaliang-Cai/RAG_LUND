import asyncio
import pytest
from uuid import uuid4

from raganything.governance.jobs import JobRunner

pytestmark = pytest.mark.asyncio


class FakeGov:
    def __init__(self):
        self.events = []
    async def mark_job_running(self, jid): self.events.append(("run", jid))
    async def mark_job_done(self, jid): self.events.append(("done", jid))
    async def mark_job_failed(self, jid, err): self.events.append(("fail", jid, err))


async def test_submit_runs_to_completion():
    gov = FakeGov()
    runner = JobRunner(gov, max_concurrent=2)
    await runner.start()
    jid = uuid4()
    done = asyncio.Event()
    async def work():
        done.set()
    await runner.submit(jid, work)
    await asyncio.wait_for(done.wait(), timeout=2)
    # Allow the task to flush its done callback
    await asyncio.sleep(0)
    assert ("run", jid) in gov.events
    assert ("done", jid) in gov.events


async def test_submit_records_failure():
    gov = FakeGov()
    runner = JobRunner(gov)
    jid = uuid4()
    async def boom():
        raise ValueError("nope")
    await runner.submit(jid, boom)
    for _ in range(20):
        await asyncio.sleep(0.05)
        if any(e[0] == "fail" for e in gov.events):
            break
    fails = [e for e in gov.events if e[0] == "fail"]
    assert fails and fails[0][2] == "nope"


async def test_cancel_in_flight():
    gov = FakeGov()
    runner = JobRunner(gov)
    jid = uuid4()
    started = asyncio.Event()
    async def slow():
        started.set()
        await asyncio.sleep(10)
    await runner.submit(jid, slow)
    await started.wait()
    cancelled = await runner.cancel(jid)
    assert cancelled is True
    for _ in range(20):
        await asyncio.sleep(0.05)
        if any(e[0] == "fail" for e in gov.events):
            break
    fails = [e for e in gov.events if e[0] == "fail"]
    assert fails and fails[0][2] == "cancelled"


async def test_stop_waits_for_in_flight_then_cancels():
    gov = FakeGov()
    runner = JobRunner(gov)
    jid = uuid4()
    started = asyncio.Event()
    async def slow():
        started.set()
        await asyncio.sleep(5)
    await runner.submit(jid, slow)
    await started.wait()
    await runner.stop(grace_period=1)
    assert any(e[0] == "fail" for e in gov.events)
