"""Lazy singleton wrapper around gfmrag.GFMRetriever."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_instance: Optional["GFMRetrieverWrapper"] = None


class GFMRetrieverWrapper:
    def __init__(self, retriever) -> None:
        self._retriever = retriever

    @classmethod
    def get_instance(
        cls,
        data_dir: str,
        data_name: str,
        model_path: str,
    ) -> "GFMRetrieverWrapper":
        global _instance
        if _instance is not None:
            return _instance

        if not data_name:
            raise RuntimeError(
                "GFM_DATA_NAME is not configured. "
                "Run scripts/export_lightrag_to_gfm.py first, "
                "then set GFM_DATA_NAME in your .env file."
            )

        try:
            from gfmrag import GFMRetriever
        except ImportError as exc:
            raise ImportError(
                "gfmrag is not installed. Install it with: pip install gfmrag"
            ) from exc

        logger.info(
            "Initializing GFMRetriever (data_dir=%s, data_name=%s)", data_dir, data_name
        )
        retriever = GFMRetriever.from_index(data_dir, data_name, model_path)
        _instance = cls(retriever)
        return _instance

    async def retrieve(
        self,
        query: str,
        top_k: int,
        text_chunks_kv,
    ) -> list[dict]:
        """Retrieve chunks via GFM graph reasoning, fetch content from LightRAG KV."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._retriever.retrieve, query, top_k
        )
        documents = result.get("document", [])

        chunks: list[dict] = []
        for doc in documents:
            chunk_id = doc.get("id", "")
            score = float(doc.get("score", 0.0))
            if not chunk_id:
                continue
            chunk_data = await text_chunks_kv.get_by_id(chunk_id)
            if chunk_data is None:
                logger.warning("GFM returned chunk_id not found in KV store: %s", chunk_id)
                continue
            content = (
                chunk_data.get("content", "")
                if isinstance(chunk_data, dict)
                else str(chunk_data)
            )
            chunks.append({"chunk_id": chunk_id, "content": content, "score": score})

        return chunks
