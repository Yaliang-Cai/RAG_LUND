import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from raganything.agent.session import SessionStore
from server.agent_routes import build_agent_router


class FakeLoop:
    def __init__(self, delay=0.0):
        self.delay = delay
    async def run(self, query, session, **kw):
        from raganything.agent.loop import AgentResult
        await asyncio.sleep(self.delay)
        return AgentResult(answer="答", grounded=True, refusal=None,
                           ledger={}, trace={"terminal_reason": "grounded"})


def make_app(loop=None):
    app = FastAPI()
    store = SessionStore()
    app.include_router(build_agent_router(store, loop or FakeLoop()))
    app.state.session_store = store
    return app, store


def test_chat_returns_answer_and_trace():
    app, _ = make_app()
    client = TestClient(app)
    r = client.post("/agent/chat", json={"workspace_id": "w", "session_id": "s", "query": "q"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "答" and "trace" in body


def test_concurrent_same_session_409():
    app, store = make_app(FakeLoop(delay=1.0))
    client = TestClient(app)
    import threading
    results = {}
    def first():
        results["a"] = client.post("/agent/chat",
                                   json={"workspace_id": "w", "session_id": "s", "query": "q1"})
    t = threading.Thread(target=first); t.start()
    import time; time.sleep(0.2)
    r2 = client.post("/agent/chat", json={"workspace_id": "w", "session_id": "s", "query": "q2"})
    t.join()
    assert r2.status_code == 409  # §6.4
    assert "cancel" in r2.json()["detail"]["hint"]


def test_cancel_endpoint_sets_event():
    app, store = make_app()
    client = TestClient(app)
    session = store.get("w", "s")
    r = client.post("/agent/sessions/s/cancel", params={"workspace_id": "w"})
    assert r.status_code == 200
    assert session.cancel_event.is_set()


@pytest.mark.asyncio
async def test_rerank_adapter_remaps_to_input_order():
    from server.agent_routes import WorkspaceAgentRunner

    class FakeService:
        async def rerank_func(self, query, documents, top_n):
            # 模拟 reranker：按相关性乱序返回，index 指回原始输入下标
            return [{"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5}]
            # 注意：下标 1 缺失 → 该文档分数应回退 0.0

    runner = WorkspaceAgentRunner(FakeService())
    rerank_fn = runner._make_rerank_fn()
    scores = await rerank_fn("q", ["doc0", "doc1", "doc2"])
    assert scores == [0.5, 0.0, 0.9]  # 按输入顺序，缺失项 0.0


@pytest.mark.asyncio
async def test_rerank_adapter_none_when_service_lacks_reranker():
    from server.agent_routes import WorkspaceAgentRunner

    class NoRerank:
        pass

    assert WorkspaceAgentRunner(NoRerank())._make_rerank_fn() is None
