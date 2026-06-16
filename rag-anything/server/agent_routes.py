# server/agent_routes.py
"""Agent 端点：/agent/chat、/agent/sessions/{id}/cancel（spec §6.4/6.5）。"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from raganything.agent.session import SessionStore

logger = logging.getLogger(__name__)


class AgentChatRequest(BaseModel):
    workspace_id: str
    session_id: str
    query: str
    max_seconds: float | None = None


def build_agent_router(store: SessionStore, agent_loop: Any) -> APIRouter:
    router = APIRouter()

    @router.post("/agent/chat")
    async def agent_chat(req: AgentChatRequest):
        session = store.get(req.workspace_id, req.session_id)
        if session.lock.locked():
            raise HTTPException(status_code=409, detail={
                "error": "session_busy",
                "running_query": session.recent_turns[-1]["q"] if session.recent_turns else "",
                "hint": "wait or POST /agent/sessions/{id}/cancel first",
            })
        async with session.lock:
            session.cancel_event.clear()
            kwargs: dict = {}
            if req.max_seconds is not None:
                from raganything.agent.budget import Budget
                kwargs["budget"] = Budget(points=10, max_seconds=req.max_seconds)
            result = await agent_loop.run(req.query, session, **kwargs)
        return {
            "answer": result.answer, "grounded": result.grounded,
            "refusal": result.refusal, "ledger": result.ledger,
            "trace": result.trace, "cancelled": result.cancelled,
        }

    @router.post("/agent/sessions/{session_id}/cancel")
    async def agent_cancel(session_id: str, workspace_id: str):
        session = store.get(workspace_id, session_id)
        session.cancel_event.set()
        return {"status": "cancelling"}

    return router


# 工具名 → agent profile（spec §7；ToolSpec.profile 是权威来源，此处仅作降级兜底）。
_TOOL_PROFILE_FALLBACK = {
    "search_sparse": "agent_sparse",
    "search_dense": "agent_dense",
    "search_hybrid": "agent_hybrid",
    "search_graph": "agent_graph",
    "search_ppr": "agent_ppr",
    "decompose_search": "agent_hybrid",  # 完整 decompose 执行器按计划延后；v1 退化为 hybrid
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
            registry = build_default_registry()
            retrieve_fn = self._make_retrieve_fn(lightrag, registry)
            rerank_fn = self._make_rerank_fn()

            loop = AgentLoop(
                model_pool=model_pool,
                registry=registry,
                retrieve_fn=retrieve_fn,
                rerank_fn=rerank_fn,
                vision_fn=None,  # VP4：inspect_image 本阶段禁用
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
                profile = spec.profile or _TOOL_PROFILE_FALLBACK.get(tool_name, "agent_hybrid")
            except KeyError:
                profile = _TOOL_PROFILE_FALLBACK.get(tool_name, "agent_hybrid")
            query = str(params.get("query", ""))
            top_k = int(params.get("top_k", 10) or 10)
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
