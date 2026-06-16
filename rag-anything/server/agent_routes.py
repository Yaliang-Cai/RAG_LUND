# server/agent_routes.py
"""Agent 端点：/agent/chat、/agent/sessions/{id}/cancel（spec §6.4/6.5）。"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from raganything.agent.session import SessionStore

logger = logging.getLogger(__name__)


class AgentChatRequest(BaseModel):
    workspace_id: str
    session_id: str
    query: str
    max_seconds: float | None = None


def _result_payload(result: Any) -> dict:
    """AgentResult → JSON the frontend consumes (non-stream body and SSE done event)."""
    return {
        "answer": result.answer, "grounded": result.grounded,
        "refusal": result.refusal, "ledger": result.ledger,
        "trace": result.trace, "cancelled": result.cancelled,
        "chunks": result.chunks,
        "citations": getattr(result, "citations", []),
        "metrics": getattr(result, "metrics", {}),
    }


def _busy_detail(session: Any) -> dict:
    return {
        "error": "session_busy",
        "running_query": session.recent_turns[-1]["q"] if session.recent_turns else "",
        "hint": "wait or POST /agent/sessions/{id}/cancel first",
    }


def _run_kwargs(req: AgentChatRequest) -> dict:
    if req.max_seconds is None:
        return {}
    from raganything.agent.budget import Budget
    return {"budget": Budget(points=10, max_seconds=req.max_seconds)}


def _token_pieces(text: str, size: int = 28) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i:i + size]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def build_agent_router(store: SessionStore, agent_loop: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/agent/chat")
    async def agent_chat(req: AgentChatRequest):
        session = store.get(req.workspace_id, req.session_id)
        if session.lock.locked():
            raise HTTPException(status_code=409, detail=_busy_detail(session))
        async with session.lock:
            session.cancel_event.clear()
            result = await agent_loop.run(req.query, session, **_run_kwargs(req))
        return _result_payload(result)

    @router.post("/agent/chat/stream")
    async def agent_chat_stream(req: AgentChatRequest):
        """SSE variant: streams live ``phase`` events (the agent's thinking chain) as
        the loop runs, then the answer as ``token`` events, then a final ``done``
        event carrying chunks / citations / metrics."""
        session = store.get(req.workspace_id, req.session_id)
        if session.lock.locked():
            raise HTTPException(status_code=409, detail=_busy_detail(session))

        async def event_stream():
            queue: asyncio.Queue = asyncio.Queue()

            async def cb(ev: dict) -> None:
                await queue.put(ev)

            async def runner():
                try:
                    async with session.lock:
                        session.cancel_event.clear()
                        result = await agent_loop.run(
                            req.query, session, event_cb=cb, **_run_kwargs(req))
                    await queue.put({"__result__": result})
                except Exception as exc:  # pragma: no cover - surfaced to client
                    logger.exception("agent stream turn failed")
                    await queue.put({"type": "error", "message": str(exc)})
                finally:
                    await queue.put(None)

            task = asyncio.create_task(runner())
            try:
                while True:
                    ev = await queue.get()
                    if ev is None:
                        break
                    if "__result__" in ev:
                        result = ev["__result__"]
                        for piece in _token_pieces(result.answer or ""):
                            yield _sse({"type": "token", "text": piece})
                            await asyncio.sleep(0.012)  # gentle typewriter cadence
                        yield _sse({"type": "done", **_result_payload(result)})
                    else:
                        yield _sse(ev)
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @router.post("/agent/sessions/{session_id}/cancel")
    async def agent_cancel(session_id: str, workspace_id: str):
        session = store.get(workspace_id, session_id)
        session.cancel_event.set()
        return {"status": "cancelling"}

    return router


# Tool name → agent profile. ToolSpec.profile is authoritative; this is only a
# defensive fallback for the minimal toolset.
_TOOL_PROFILE_FALLBACK = {
    "search": "agent_search",
    "search_multihop": "agent_multihop",
}


class WorkspaceAgentRunner:
    """把单一 ``.run(query, session, **kw)`` 入口路由到按 workspace 懒构建的
    AgentLoop。AgentLoop 在 startup 只能构造一次，但每个请求的 workspace 不同，
    retrieve_fn/rerank_fn 都要绑定到该 workspace 的 lightrag 实例，因此在这里
    按 ``session.workspace_id`` 缓存 per-workspace 的 AgentLoop（VP1/VP3）。"""

    def __init__(self, service: Any) -> None:
        self._service = service
        self._loops: dict[str, Any] = {}
        self._build_locks: dict[str, Any] = {}

    async def run(self, query: str, session: Any, **kw: Any):
        loop = await self._get_loop(session.workspace_id)
        return await loop.run(query, session, **kw)

    async def _get_loop(self, workspace_id: str):
        cached = self._loops.get(workspace_id)
        if cached is not None:
            return cached
        lock = self._build_locks.setdefault(workspace_id, asyncio.Lock())
        async with lock:
            cached = self._loops.get(workspace_id)  # 双检：等锁期间可能已被另一请求构建
            if cached is not None:
                return cached

            from raganything.agent.loop import AgentLoop
            from raganything.agent.models import ModelPool
            from raganything.agent.tools import build_default_registry

            rag = await self._service.get_rag(workspace_id)
            await rag._ensure_lightrag_initialized()
            lightrag = rag.lightrag

            model_pool = ModelPool(main_func=self._service.llm_model_func)
            registry = build_default_registry()  # minimal toolset: search / search_multihop / answer
            retrieve_fn = self._make_retrieve_fn(lightrag, registry)
            rerank_fn = self._make_rerank_fn()

            loop = AgentLoop(
                model_pool=model_pool,
                registry=registry,
                retrieve_fn=retrieve_fn,
                rerank_fn=rerank_fn,
                # Generation-time VLM: when the planner flags visual_intent and
                # relevant images survive pack_context's gates, the final answer is
                # generated by the vision model (restores agentic-v2 behavior).
                vision_fn=getattr(self._service, "vision_model_func", None),
            )
            self._loops[workspace_id] = loop
            return loop

    def _make_retrieve_fn(self, lightrag, registry):
        from lightrag import QueryParam
        from raganything.retrieval.router import RetrievalRouter

        router = RetrievalRouter(lightrag, llm_func=self._service.llm_model_func)

        async def retrieve_fn(tool_name: str, params: dict):
            try:
                spec = registry.get(tool_name)
                profile = spec.profile or _TOOL_PROFILE_FALLBACK.get(tool_name, "agent_search")
            except KeyError:
                profile = _TOOL_PROFILE_FALLBACK.get(tool_name, "agent_search")
            query = str(params.get("query", ""))
            top_k = int(params.get("top_k", 12) or 12)
            param = QueryParam(top_k=top_k, chunk_top_k=top_k, enable_rerank=False)
            chunks, trace = await router.route(query, param, profile_name=profile)
            return chunks, trace

        return retrieve_fn

    def _make_rerank_fn(self):
        rerank_func = getattr(self._service, "rerank_func", None)
        if rerank_func is None:
            return None

        async def rerank_fn(query: str, documents: list[str]) -> list[float]:
            # service.rerank_func 返回按分排序的 [{"index", "relevance_score"}]，
            # 需还原成与输入 documents 同序的分数列表。
            results = await rerank_func(query, documents, None)
            scores = [0.0] * len(documents)
            for r in results:
                idx = r.get("index")
                if isinstance(idx, int) and 0 <= idx < len(scores):
                    scores[idx] = float(r.get("relevance_score", 0.0))
            return scores

        return rerank_fn
