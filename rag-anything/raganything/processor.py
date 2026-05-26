"""
Document processing functionality for RAGAnything

Contains methods for parsing documents and processing multimodal content
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional, Callable, Awaitable
from pathlib import Path

from raganything.base import DocStatus
from raganything.parser import MineruParser, DoclingParser, MineruExecutionError
from raganything.utils import (
    separate_content,
    insert_text_content,
    insert_text_content_with_multimodal_content,
    get_processor_for_type,
)
import asyncio
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import get_storage_keyed_lock
from lightrag.utils import (
    apply_source_ids_limit,
    compute_mdhash_id,
    compute_entity_id,
    compute_entity_vdb_id,
    merge_source_ids,
)
from raganything.constants import (
    DEFAULT_MULTIMODAL_ITEM_TIMEOUT_SECONDS,
    DEFAULT_MULTIMODAL_BATCH_WATCHDOG_SECONDS,
    DEFAULT_MULTIMODAL_CANCEL_GRACE_SECONDS,
)

_HTTPX_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = ()
_OPENAI_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = ()


def _content_list_filename(file_stem: str) -> str:
    return f"{file_stem}_content_list.json"


def _iter_content_list_matches(root: Path, file_stem: str) -> list[Path]:
    if not root.is_dir():
        return []
    expected_name = _content_list_filename(file_stem)
    return [
        path
        for path in root.rglob("*_content_list.json")
        if path.name == expected_name
    ]


try:
    import httpx

    _HTTPX_RETRYABLE_EXCEPTIONS = (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    )
except Exception:
    pass

try:
    import openai

    _OPENAI_RETRYABLE_EXCEPTIONS = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
except Exception:
    pass


class MultimodalPartialFailureError(RuntimeError):
    """Partial multimodal failure that keeps failed item metadata for targeted retry."""

    def __init__(self, message: str, failed_items: List[Dict[str, Any]]):
        super().__init__(message)
        self.failed_items = failed_items


_MM_STAGE_CHUNKS_STORED = "chunks_stored"
_MM_STAGE_MAIN_ENTITY_FAILED = "stage3_5_main_entity_failed"
_MM_STAGE_ER_FAILED = "stage4_er_extract_failed"
_MM_STAGE_BELONGS_TO_FAILED = "stage5_belongs_to_failed"
_MM_STAGE_MERGE_FAILED = "stage6_merge_failed"
_MM_STAGE_PARTIAL_FAILED = "partial_failed"
_MM_STAGE_COMPLETED = "completed"


class ProcessorMixin:
    """ProcessorMixin class containing document processing functionality for RAGAnything"""

    def _get_multimodal_retry_policy(self) -> Tuple[int, float, float]:
        """Return (max_attempts, base_delay_seconds, max_delay_seconds)."""
        addon_params = getattr(self.lightrag, "addon_params", {}) or {}
        attempts = int(
            addon_params.get(
                "multimodal_transient_retry_attempts",
                addon_params.get("resilience_max_attempts", 3),
            )
        )
        base_delay = float(
            addon_params.get(
                "multimodal_transient_retry_base_delay",
                addon_params.get("ingest_retry_base_delay", 1.0),
            )
        )
        max_delay = float(
            addon_params.get(
                "multimodal_transient_retry_max_delay",
                addon_params.get("resilience_max_delay", 20.0),
            )
        )
        attempts = max(2, min(3, attempts))
        base_delay = max(0.0, base_delay)
        max_delay = max(base_delay, max_delay)
        return attempts, base_delay, max_delay

    def _build_multimodal_failed_result(
        self,
        *,
        index: int,
        content_type: str,
        item: Optional[Dict[str, Any]],
        category: str,
        error: Any,
    ) -> Dict[str, Any]:
        """Create a normalized failed-item record used by runtime and doc_status."""
        normalized_category = (
            category
            if category in {"timeout", "parse", "model", "cancelled", "other"}
            else "other"
        )
        error_class = type(error).__name__
        error_message = str(error)
        return {
            "status": "failed",
            "index": index,
            "item_index": index,
            "type": content_type or "unknown",
            "item": item,
            "category": normalized_category,
            "error": error_message,
            "error_class": error_class,
            "error_message": error_message,
        }

    def _normalize_multimodal_task_result(
        self,
        *,
        result: Any,
        index: int,
        content_type: str,
        item: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalize gather/task output to either a successful record or a failed record."""
        if isinstance(result, BaseException):
            return self._build_multimodal_failed_result(
                index=index,
                content_type=content_type,
                item=item,
                category=self._categorize_multimodal_failure(result),
                error=result,
            )

        if isinstance(result, dict):
            status = str(result.get("status", ""))
            if status == "ok":
                return result
            if status == "failed":
                return self._build_multimodal_failed_result(
                    index=int(result.get("index", index)),
                    content_type=str(result.get("type", content_type)),
                    item=result.get("item", item),
                    category=str(result.get("category", "other")),
                    error=result.get("error", ""),
                )

        return self._build_multimodal_failed_result(
            index=index,
            content_type=content_type,
            item=item,
            category="other",
            error=f"Unexpected task result type: {type(result).__name__}",
        )

    def _resolve_multimodal_batch_guardrails(
        self, *, total_items: int
    ) -> Dict[str, Any]:
        """
        Resolve per-document multimodal batch guardrails.

        Keep guardrails simple and predictable:
        - item timeout: per-item timeout
        - batch watchdog: configured timeout floor (>= item timeout)
        - parallelism: runtime per-doc multimodal concurrency
        """
        _ = total_items
        addon_params = getattr(self.lightrag, "addon_params", {}) or {}
        item_timeout_seconds = max(
            1.0,
            float(
                addon_params.get(
                    "multimodal_item_timeout_seconds",
                    DEFAULT_MULTIMODAL_ITEM_TIMEOUT_SECONDS,
                )
            ),
        )
        configured_watchdog_seconds = max(
            item_timeout_seconds,
            float(
                addon_params.get(
                    "multimodal_batch_watchdog_seconds",
                    DEFAULT_MULTIMODAL_BATCH_WATCHDOG_SECONDS,
                )
            ),
        )
        cancel_grace_seconds = max(
            0.0,
            float(
                addon_params.get(
                    "multimodal_cancel_grace_seconds",
                    DEFAULT_MULTIMODAL_CANCEL_GRACE_SECONDS,
                )
            ),
        )
        configured_parallelism = addon_params.get("multimodal_item_parallelism")
        if configured_parallelism is not None:
            parallelism = max(1, int(configured_parallelism))
        else:
            parallelism = max(1, int(getattr(self.lightrag, "max_parallel_insert", 2)))

        retry_attempts, retry_base_delay, retry_max_delay = (
            self._get_multimodal_retry_policy()
        )

        return {
            "item_timeout_seconds": item_timeout_seconds,
            "batch_watchdog_seconds": configured_watchdog_seconds,
            "cancel_grace_seconds": cancel_grace_seconds,
            "parallelism": parallelism,
            "retry_attempts": retry_attempts,
            "retry_base_delay": retry_base_delay,
            "retry_max_delay": retry_max_delay,
        }

    def _categorize_multimodal_failure(self, exc: BaseException) -> str:
        """Map an exception to one of: timeout/parse/model/cancelled/other."""
        if isinstance(exc, asyncio.CancelledError):
            return "cancelled"
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return "timeout"
        if isinstance(exc, (ValueError, json.JSONDecodeError)):
            text = str(exc).lower()
            if "json" in text or "parse" in text or "field" in text:
                return "parse"
        text = str(exc).lower()
        if any(
            key in text
            for key in (
                "connection",
                "timeout",
                "rate limit",
                "api",
                "vllm",
                "openai",
                "httpx",
            )
        ):
            return "model"
        return "other"

    def _is_transient_multimodal_exception(self, exc: BaseException) -> bool:
        """Classify transient errors that are safe to retry."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True
        if _HTTPX_RETRYABLE_EXCEPTIONS and isinstance(
            exc, _HTTPX_RETRYABLE_EXCEPTIONS
        ):
            return True
        if _OPENAI_RETRYABLE_EXCEPTIONS and isinstance(
            exc, _OPENAI_RETRYABLE_EXCEPTIONS
        ):
            return True

        message = str(exc).lower()
        transient_markers = (
            "timed out",
            "timeout",
            "connection",
            "rate limit",
            "temporarily unavailable",
            "503",
            "429",
            "broken pipe",
        )
        return any(marker in message for marker in transient_markers)

    async def _run_multimodal_call_with_retry(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Run one multimodal operation with transient-error retry."""
        max_attempts, base_delay, max_delay = self._get_multimodal_retry_policy()
        for attempt in range(1, max_attempts + 1):
            try:
                return await operation()
            except Exception as exc:
                can_retry = self._is_transient_multimodal_exception(exc)
                if attempt >= max_attempts or not can_retry:
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                self.logger.warning(
                    "Multimodal transient retry: %s attempt=%d/%d delay=%.1fs reason=%s",
                    operation_name,
                    attempt,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    def _serialize_multimodal_item(self, item: Dict[str, Any]) -> str:
        """Build a stable signature for one multimodal item."""
        try:
            return json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            return repr(item)

    def _select_multimodal_retry_subset(
        self,
        multimodal_items: List[Dict[str, Any]],
        existing_doc_status: Optional[Dict[str, Any]],
        doc_id: str,
    ) -> List[Dict[str, Any]]:
        """If doc_status contains failed multimodal items, retry only that subset."""
        if not isinstance(existing_doc_status, dict):
            return multimodal_items

        failed_entries = existing_doc_status.get("multimodal_failed_items")
        if not isinstance(failed_entries, list) or not failed_entries:
            return multimodal_items

        failed_candidates = [
            entry.get("item")
            for entry in failed_entries
            if isinstance(entry, dict) and isinstance(entry.get("item"), dict)
        ]
        if not failed_candidates:
            return multimodal_items

        source_map: Dict[str, Dict[str, Any]] = {}
        for item in multimodal_items:
            if isinstance(item, dict):
                key = self._serialize_multimodal_item(item)
                source_map.setdefault(key, item)

        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in failed_candidates:
            key = self._serialize_multimodal_item(candidate)
            if key in seen:
                continue
            selected.append(source_map.get(key, candidate))
            seen.add(key)

        if not selected:
            return multimodal_items

        self.logger.info(
            "Resuming multimodal with failed subset: %d/%d items (doc_id=%s)",
            len(selected),
            len(multimodal_items),
            doc_id,
        )
        return selected

    def _get_multimodal_chunk_ids_from_status(
        self, doc_status: Optional[Dict[str, Any]]
    ) -> List[str]:
        if not isinstance(doc_status, dict):
            return []
        chunk_ids = doc_status.get("multimodal_chunk_ids")
        if not isinstance(chunk_ids, list):
            return []
        normalized: List[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str):
                continue
            value = chunk_id.strip()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        return normalized

    def _should_resume_multimodal_from_stored_chunks(
        self, doc_status: Optional[Dict[str, Any]]
    ) -> bool:
        if not isinstance(doc_status, dict):
            return False
        if bool(doc_status.get("multimodal_processed", False)):
            return False
        if doc_status.get("multimodal_failed_items"):
            return False
        stage = str(doc_status.get("multimodal_stage", "") or "").strip()
        if stage not in {
            _MM_STAGE_CHUNKS_STORED,
            _MM_STAGE_MAIN_ENTITY_FAILED,
            _MM_STAGE_ER_FAILED,
            _MM_STAGE_BELONGS_TO_FAILED,
            _MM_STAGE_MERGE_FAILED,
        }:
            return False
        return len(self._get_multimodal_chunk_ids_from_status(doc_status)) > 0

    async def _load_lightrag_chunks_from_storage(
        self, chunk_ids: List[str]
    ) -> Dict[str, Any]:
        if not chunk_ids:
            return {}
        chunk_rows = await self.lightrag.text_chunks.get_by_ids(chunk_ids)
        if not isinstance(chunk_rows, list):
            return {}
        chunks: Dict[str, Any] = {}
        for idx, row in enumerate(chunk_rows):
            if not isinstance(row, dict):
                continue
            # Different KV backends may return either "id" or "_id", and some
            # rely on call-site ordering without embedding the ID in payload.
            chunk_id = row.get("id") or row.get("_id")
            if (not isinstance(chunk_id, str) or not chunk_id) and idx < len(chunk_ids):
                chunk_id = chunk_ids[idx]
            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            normalized_row = dict(row)
            normalized_row.setdefault("id", chunk_id)
            normalized_row.setdefault("_id", chunk_id)
            chunks[chunk_id] = normalized_row
        return chunks

    @staticmethod
    def _split_source_ids(source_id_value: Any) -> List[str]:
        if source_id_value is None:
            return []
        return [
            chunk_id
            for chunk_id in str(source_id_value).split(GRAPH_FIELD_SEP)
            if chunk_id
        ]

    def _limit_entity_source_ids(
        self, entity_id: str, source_ids: List[str]
    ) -> List[str]:
        if not source_ids:
            return []
        max_source_ids_raw = getattr(self.lightrag, "max_source_ids_per_entity", 0)
        try:
            max_source_ids = int(max_source_ids_raw)
        except (TypeError, ValueError):
            max_source_ids = 0

        if max_source_ids <= 0:
            return list(source_ids)

        limit_method = getattr(self.lightrag, "source_ids_limit_method", None)
        return apply_source_ids_limit(
            source_ids,
            max_source_ids,
            limit_method,
            identifier=f"`{entity_id}`",
        )

    async def _upsert_multimodal_main_entities_to_core_storage(
        self, entities_to_store: Dict[str, Dict[str, Any]]
    ) -> None:
        if not entities_to_store:
            return

        did_upsert_entities_vdb = False
        workspace = getattr(self.lightrag, "workspace", "") or ""
        graph_lock_namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"

        for entity_vdb_id, entity_data in entities_to_store.items():
            entity_name = str(entity_data.get("entity_name", "")).strip()
            entity_type = str(entity_data.get("entity_type", "multimodal")).strip()
            if not entity_type:
                entity_type = "multimodal"

            composite_id = str(entity_data.get("entity_id", "")).strip()
            if not composite_id and entity_name:
                _disambig = getattr(self.lightrag, "enable_entity_disambiguation", True)
                composite_id = compute_entity_id(entity_name, entity_type, _disambig)
            if not composite_id:
                continue

            file_path = str(entity_data.get("file_path", "unknown_source"))
            description = str(entity_data.get("content", "")).strip()
            new_source_id = str(entity_data.get("source_id", "")).strip()

            async with get_storage_keyed_lock(
                [composite_id], namespace=graph_lock_namespace, enable_logging=False
            ):
                existing_node = await self.lightrag.chunk_entity_relation_graph.get_node(
                    composite_id
                )

                existing_full_source_ids: List[str] = []
                if self.lightrag.entity_chunks:
                    stored_chunks = await self.lightrag.entity_chunks.get_by_id(
                        composite_id
                    )
                    if isinstance(stored_chunks, dict):
                        existing_full_source_ids = [
                            chunk_id
                            for chunk_id in stored_chunks.get("chunk_ids", [])
                            if chunk_id
                        ]
                if not existing_full_source_ids and isinstance(existing_node, dict):
                    existing_full_source_ids = self._split_source_ids(
                        existing_node.get("source_id", "")
                    )

                merged_full_source_ids = merge_source_ids(
                    existing_full_source_ids, [new_source_id]
                )
                limited_source_ids = self._limit_entity_source_ids(
                    composite_id, merged_full_source_ids
                )
                limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

                node_created_at = (
                    existing_node.get("created_at")
                    if isinstance(existing_node, dict)
                    and existing_node.get("created_at") is not None
                    else int(time.time())
                )

                node_data = {
                    "entity_id": composite_id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": limited_source_id_str,
                    "file_path": file_path,
                    "created_at": node_created_at,
                    "truncate": (
                        existing_node.get("truncate", "")
                        if isinstance(existing_node, dict)
                        else ""
                    ),
                }
                await self.lightrag.chunk_entity_relation_graph.upsert_node(
                    composite_id, node_data
                )

                if self.lightrag.entity_chunks and merged_full_source_ids:
                    await self.lightrag.entity_chunks.upsert(
                        {
                            composite_id: {
                                "chunk_ids": merged_full_source_ids,
                                "count": len(merged_full_source_ids),
                            }
                        }
                    )

                await self.lightrag.entities_vdb.upsert(
                    {
                        entity_vdb_id: {
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "entity_id": composite_id,
                            "content": description,
                            "source_id": limited_source_id_str,
                            "file_path": file_path,
                        }
                    }
                )
                did_upsert_entities_vdb = True

        if did_upsert_entities_vdb:
            await self.lightrag.entities_vdb.index_done_callback()

    async def _store_multimodal_main_entities_from_stored_chunks(
        self, lightrag_chunks: Dict[str, Any], doc_id: str
    ) -> None:
        entities_to_store: Dict[str, Dict[str, Any]] = {}
        _disambig = getattr(self.lightrag, "enable_entity_disambiguation", True)
        for chunk_id, chunk_data in lightrag_chunks.items():
            if not isinstance(chunk_data, dict):
                continue
            if not chunk_data.get("is_multimodal"):
                continue
            entity_name = str(chunk_data.get("modal_entity_name", "")).strip()
            if not entity_name:
                continue
            entity_type = str(chunk_data.get("original_type", "multimodal")).strip()
            if not entity_type:
                entity_type = "multimodal"
            file_path = str(chunk_data.get("file_path", "unknown_source"))
            content = str(chunk_data.get("content", "")).strip()
            entity_id = compute_entity_id(entity_name, entity_type, _disambig)
            entity_vdb_id = compute_entity_vdb_id(entity_name, entity_type, _disambig)
            entities_to_store[entity_vdb_id] = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "content": content,
                "source_id": chunk_id,
                "file_path": file_path,
            }

        if not entities_to_store:
            return

        await self._upsert_multimodal_main_entities_to_core_storage(entities_to_store)

        current_doc_entities = await self.lightrag.full_entities.get_by_id(doc_id)
        if not current_doc_entities:
            entity_names = sorted(
                entity_data["entity_id"] for entity_data in entities_to_store.values()
            )
            doc_entities_data = {
                "entity_names": entity_names,
                "count": len(entity_names),
                "update_time": int(time.time()),
            }
        else:
            existing_entity_names = list(current_doc_entities.get("entity_names", []))
            seen_entity_names = set(existing_entity_names)
            for entity_data in entities_to_store.values():
                entity_id = entity_data["entity_id"]
                if entity_id not in seen_entity_names:
                    existing_entity_names.append(entity_id)
                    seen_entity_names.add(entity_id)
            doc_entities_data = {
                **current_doc_entities,
                "entity_names": existing_entity_names,
                "count": len(existing_entity_names),
                "update_time": int(time.time()),
            }

        await self.lightrag.full_entities.upsert({doc_id: doc_entities_data})
        await self.lightrag.full_entities.index_done_callback()

    async def _resume_multimodal_from_stored_chunks(
        self, doc_id: str, chunk_ids: List[str], file_path: str
    ) -> None:
        self.logger.info(
            "Resuming multimodal from stored chunks: doc_id=%s chunks=%d",
            doc_id,
            len(chunk_ids),
        )
        lightrag_chunks = await self._load_lightrag_chunks_from_storage(chunk_ids)
        if not lightrag_chunks:
            raise RuntimeError(
                f"{_MM_STAGE_CHUNKS_STORED}: no stored chunks found for doc_id={doc_id}"
            )
        try:
            await self._store_multimodal_main_entities_from_stored_chunks(
                lightrag_chunks, doc_id
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_MAIN_ENTITY_FAILED}: {exc}") from exc

        try:
            chunk_results = await self._batch_extract_entities_lightrag_style_type_aware(
                lightrag_chunks
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_ER_FAILED}: {exc}") from exc

        try:
            enhanced_chunk_results = (
                await self._batch_add_belongs_to_relations_from_stored_chunks(
                    chunk_results, lightrag_chunks
                )
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_BELONGS_TO_FAILED}: {exc}") from exc

        try:
            await self._batch_merge_lightrag_style_type_aware(
                enhanced_chunk_results, file_path, doc_id
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_MERGE_FAILED}: {exc}") from exc

        await self._finalize_multimodal_doc_status(
            doc_id, list(lightrag_chunks.keys()), mark_processed=True
        )

    def _infer_multimodal_stage_from_error(self, error_msg: str) -> str:
        msg = str(error_msg or "")
        if msg.startswith(f"{_MM_STAGE_MAIN_ENTITY_FAILED}:"):
            return _MM_STAGE_MAIN_ENTITY_FAILED
        if msg.startswith(f"{_MM_STAGE_ER_FAILED}:"):
            return _MM_STAGE_ER_FAILED
        if msg.startswith(f"{_MM_STAGE_BELONGS_TO_FAILED}:"):
            return _MM_STAGE_BELONGS_TO_FAILED
        if msg.startswith(f"{_MM_STAGE_MERGE_FAILED}:"):
            return _MM_STAGE_MERGE_FAILED
        return _MM_STAGE_CHUNKS_STORED

    def _get_file_reference(self, file_path: str) -> str:
        """
        Get file reference based on use_full_path configuration.

        Args:
            file_path: Path to the file (can be absolute or relative)

        Returns:
            str: Full path if use_full_path is True, otherwise basename
        """
        if self.config.use_full_path:
            return str(file_path)
        else:
            return os.path.basename(file_path)

    def _generate_cache_key(
        self, file_path: Path, parse_method: str = None, **kwargs
    ) -> str:
        """
        Generate cache key based on file path and parsing configuration

        Args:
            file_path: Path to the file
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            str: Cache key for the file and configuration
        """

        # Get file modification time
        mtime = file_path.stat().st_mtime

        # Create configuration dict for cache key
        config_dict = {
            "file_path": str(file_path.absolute()),
            "mtime": mtime,
            "parser": self.config.parser,
            "parse_method": parse_method or self.config.parse_method,
        }

        # Add relevant kwargs to config
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "lang",
                "device",
                "start_page",
                "end_page",
                "formula",
                "table",
                "backend",
                "source",
            ]
        }
        config_dict.update(relevant_kwargs)

        # Generate hash from config
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """
        Generate doc_id based on document content

        Args:
            content_list: Parsed content list

        Returns:
            str: Content-based document ID with doc- prefix
        """
        from lightrag.utils import compute_mdhash_id

        # Extract key content for ID generation
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # For text content, use the text
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # For other content types, use key identifiers
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # For other types, use string representation
                    content_hash_data.append(str(item))

        # Create a content signature
        content_signature = "\n".join(content_hash_data)

        # Generate doc_id from content
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def _get_cached_result(
        self, cache_key: str, file_path: Path, parse_method: str = None, **kwargs
    ) -> tuple[List[Dict[str, Any]], str] | None:
        """
        Get cached parsing result if available and valid

        Args:
            cache_key: Cache key to look up
            file_path: Path to the file for mtime check
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            tuple[List[Dict[str, Any]], str] | None: (content_list, doc_id) or None if not found/invalid
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return None

        try:
            cached_data = await self.parse_cache.get_by_id(cache_key)
            if not cached_data:
                return None

            # Check file modification time
            current_mtime = file_path.stat().st_mtime
            cached_mtime = cached_data.get("mtime", 0)

            if current_mtime != cached_mtime:
                self.logger.debug(f"Cache invalid - file modified: {cache_key}")
                return None

            # Check parsing configuration
            cached_config = cached_data.get("parse_config", {})
            current_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to current config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            current_config.update(relevant_kwargs)

            if cached_config != current_config:
                self.logger.debug(f"Cache invalid - config changed: {cache_key}")
                return None

            content_list = cached_data.get("content_list", [])
            doc_id = cached_data.get("doc_id")

            if content_list and doc_id:
                self.logger.debug(
                    f"Found valid cached parsing result for key: {cache_key}"
                )
                return content_list, doc_id
            else:
                self.logger.debug(
                    f"Cache incomplete - missing content or doc_id: {cache_key}"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error accessing parse cache: {e}")

        return None

    async def _store_cached_result(
        self,
        cache_key: str,
        content_list: List[Dict[str, Any]],
        doc_id: str,
        file_path: Path,
        parse_method: str = None,
        **kwargs,
    ) -> None:
        """
        Store parsing result in cache

        Args:
            cache_key: Cache key to store under
            content_list: Content list to cache
            doc_id: Content-based document ID
            file_path: Path to the file for mtime storage
            parse_method: Parse method used
            **kwargs: Additional parser parameters
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return

        try:
            # Get file modification time
            file_mtime = file_path.stat().st_mtime

            # Create parsing configuration
            parse_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            parse_config.update(relevant_kwargs)

            cache_data = {
                cache_key: {
                    "content_list": content_list,
                    "doc_id": doc_id,
                    "mtime": file_mtime,
                    "parse_config": parse_config,
                    "cached_at": time.time(),
                    "cache_version": "1.0",
                }
            }
            await self.parse_cache.upsert(cache_data)
            # Ensure data is persisted to disk
            await self.parse_cache.index_done_callback()
            self.logger.info(f"Stored parsing result in cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error storing to parse cache: {e}")

    def _resolve_mineru_method(
        self,
        parse_method: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Resolve MinerU output directory name from parse method and backend hint.
        """
        method = parse_method or self.config.parse_method
        backend = kwargs.get("backend") or ""
        if isinstance(backend, str):
            if backend.startswith("vlm-"):
                return "vlm"
            if backend.startswith("hybrid-"):
                return "hybrid_auto"
        return method

    def _find_latest_mineru_json(
        self,
        base_output_dir: Path,
        file_stem: str,
    ) -> Optional[Path]:
        """
        Find the latest MinerU content_list JSON under output directory.
        """
        candidates: list[Path] = []

        direct_json = base_output_dir / _content_list_filename(file_stem)
        if direct_json.exists():
            candidates.append(direct_json)

        safe_stem = (
            "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in file_stem.strip())
            or "file"
        )
        for subdir_name in dict.fromkeys((file_stem, safe_stem)):
            stem_subdir = base_output_dir / subdir_name
            if stem_subdir.is_dir():
                candidates.extend(_iter_content_list_matches(stem_subdir, file_stem))

        if base_output_dir.is_dir():
            primary_keys = {str(path) for path in candidates}
            for candidate in _iter_content_list_matches(base_output_dir, file_stem):
                if str(candidate) not in primary_keys:
                    candidates.append(candidate)

        deduped_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped_candidates.append(candidate)
        candidates = deduped_candidates

        if not candidates:
            return None

        valid_candidates: list[Path] = []
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.warning(
                    "Skipping unreadable MinerU content_list during reuse: %s (%s)",
                    candidate,
                    exc,
                )
                continue
            if isinstance(payload, list) and payload:
                valid_candidates.append(candidate)
            else:
                self.logger.warning(
                    "Skipping empty or invalid MinerU content_list during reuse: %s",
                    candidate,
                )

        if not valid_candidates:
            return None

        try:
            return max(valid_candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            # Fallback to the first existing candidate if stat fails unexpectedly.
            return valid_candidates[0]

    async def _try_load_existing_mineru_output(
        self,
        file_path: Path,
        output_dir: str,
        parse_method: str,
        **kwargs,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Try loading already-generated MinerU output from disk.

        This allows reuse of parser artifacts even when parse_cache is missing
        (e.g., after deleting rag_storage/workspace).
        """
        if self.config.parser != "mineru":
            return None

        # Reuse only for file types that are handled via MinerU output artifacts.
        if file_path.suffix.lower() not in {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".gif",
            ".webp",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
            ".txt",
            ".md",
            ".html",
            ".htm",
            ".xhtml",
        }:
            return None

        base_output_dir = Path(output_dir) if output_dir else file_path.parent / "mineru_output"
        if not base_output_dir.exists():
            return None

        file_stem = file_path.stem
        latest_json = self._find_latest_mineru_json(base_output_dir, file_stem)
        if latest_json is None:
            return None

        try:
            source_mtime = file_path.stat().st_mtime
            if latest_json.stat().st_mtime < source_mtime:
                self.logger.warning(
                    "Reusing MinerU output older than source file: %s. "
                    "Delete the artifact to force reparsing after source content changes.",
                    latest_json,
                )
        except Exception as exc:
            self.logger.debug("Failed to compare mtime for MinerU reuse: %s", exc)

        resolved_method = self._resolve_mineru_method(parse_method, **kwargs)
        parser = MineruParser()
        read_base_output_dir = latest_json.parent
        content_list, _ = parser._read_output_files(
            read_base_output_dir,
            file_stem,
            method=resolved_method,
        )
        if not content_list:
            return None

        self.logger.info(
            "Reusing existing MinerU output without re-parsing: %s", latest_json
        )
        return content_list

    async def parse_document(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        **kwargs,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Parse document with caching support

        Args:
            file_path: Path to the file to parse
            output_dir: Output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)

        Returns:
            tuple[List[Dict[str, Any]], str]: (content_list, doc_id)
        """
        # Use config defaults if not provided
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting document parsing: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        callback_file = str(file_path)
        callback_manager = getattr(self, "callback_manager", None)
        parse_start_time = time.time()
        if callback_manager is not None:
            callback_manager.dispatch(
                "on_parse_start",
                file_path=callback_file,
                parser=self.config.parser,
            )

        # Generate cache key based on file and configuration
        cache_key = self._generate_cache_key(file_path, parse_method, **kwargs)

        # Check cache first
        cached_result = await self._get_cached_result(
            cache_key, file_path, parse_method, **kwargs
        )
        if cached_result is not None:
            content_list, doc_id = cached_result
            self.logger.info(f"Using cached parsing result for: {file_path}")
            if display_stats:
                self.logger.info(
                    f"* Total blocks in cached content_list: {len(content_list)}"
                )
            if callback_manager is not None:
                duration = time.time() - parse_start_time
                callback_manager.dispatch(
                    "on_parse_complete",
                    file_path=callback_file,
                    content_blocks=len(content_list),
                    doc_id=doc_id,
                    duration_seconds=duration,
                )
            return content_list, doc_id

        # Parse-cache miss: try to recover from existing MinerU output files.
        reused_content = await self._try_load_existing_mineru_output(
            file_path=file_path,
            output_dir=output_dir,
            parse_method=parse_method,
            **kwargs,
        )
        if reused_content:
            doc_id = self._generate_content_based_doc_id(reused_content)
            await self._store_cached_result(
                cache_key, reused_content, doc_id, file_path, parse_method, **kwargs
            )
            if display_stats:
                self.logger.info(
                    f"* Total blocks in reused MinerU content_list: {len(reused_content)}"
                )
            return reused_content, doc_id

        # Choose appropriate parsing method based on file extension
        ext = file_path.suffix.lower()

        try:
            doc_parser = (
                DoclingParser() if self.config.parser == "docling" else MineruParser()
            )

            # Log parser and method information
            self.logger.info(
                f"Using {self.config.parser} parser with method: {parse_method}"
            )

            if ext in [".pdf"]:
                self.logger.info("Detected PDF file, using parser for PDF...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    output_dir=output_dir,
                    method=parse_method,
                    **kwargs,
                )
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            ]:
                self.logger.info("Detected image file, using parser for images...")
                # Use the selected parser's image parsing capability
                if hasattr(doc_parser, "parse_image"):
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                else:
                    # Fallback to MinerU for image parsing if current parser doesn't support it
                    self.logger.warning(
                        f"{self.config.parser} parser doesn't support image parsing, falling back to MinerU"
                    )
                    content_list = MineruParser().parse_image(
                        image_path=file_path, output_dir=output_dir, **kwargs
                    )
            elif ext in [
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".html",
                ".htm",
                ".xhtml",
            ]:
                self.logger.info(
                    "Detected Office or HTML document, using parser for Office/HTML..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=file_path,
                    output_dir=output_dir,
                    **kwargs,
                )
            else:
                # For other or unknown formats, use generic parser
                self.logger.info(
                    f"Using generic parser for {ext} file (method={parse_method})..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_document,
                    file_path=file_path,
                    method=parse_method,
                    output_dir=output_dir,
                    **kwargs,
                )

        except MineruExecutionError as e:
            self.logger.error(f"Mineru command failed: {e}")
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_parse_error",
                    file_path=callback_file,
                    error=e,
                    parser=self.config.parser,
                )
            raise
        except Exception as e:
            self.logger.error(
                f"Error during parsing with {self.config.parser} parser: {str(e)}"
            )
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_parse_error",
                    file_path=callback_file,
                    error=e,
                    parser=self.config.parser,
                )
            raise e

        msg = f"Parsing {file_path} complete! Extracted {len(content_list)} content blocks"
        self.logger.info(msg)

        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content was extracted")

        # Generate doc_id based on content
        doc_id = self._generate_content_based_doc_id(content_list)

        # Store result in cache
        await self._store_cached_result(
            cache_key, content_list, doc_id, file_path, parse_method, **kwargs
        )

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        if callback_manager is not None:
            duration = time.time() - parse_start_time
            callback_manager.dispatch(
                "on_parse_complete",
                file_path=callback_file,
                content_blocks=len(content_list),
                doc_id=doc_id,
                duration_seconds=duration,
            )

        return content_list, doc_id

    async def _process_multimodal_content(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        pipeline_status: Optional[Any] = None,
        pipeline_status_lock: Optional[Any] = None,
    ):
        """
        Process multimodal content (using specialized processors)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
            pipeline_status: Pipeline status object
            pipeline_status_lock: Pipeline status lock
        """

        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        callback_manager = getattr(self, "callback_manager", None)
        mm_start_time = time.time()

        # Check multimodal processing status - handle LightRAG's early DocStatus.PROCESSED marking
        existing_doc_status: Optional[Dict[str, Any]] = None
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if existing_doc_status:
                # Check if multimodal content is already processed
                multimodal_processed = existing_doc_status.get(
                    "multimodal_processed", False
                )

                if multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} multimodal content is already processed"
                    )
                    return

                # Even if status is DocStatus.PROCESSED (text processing done),
                # we still need to process multimodal content if not yet done
                doc_status = existing_doc_status.get("status", "")
                if doc_status == DocStatus.PROCESSED and not multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} text processing is complete, but multimodal content still needs processing"
                    )
                    # Continue with multimodal processing
                elif doc_status == DocStatus.PROCESSED and multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} is fully processed (text + multimodal)"
                    )
                    return

        except Exception as e:
            self.logger.debug(f"Error checking document status for {doc_id}: {e}")
            # Continue with processing if cache check fails

        if self._should_resume_multimodal_from_stored_chunks(existing_doc_status):
            chunk_ids = self._get_multimodal_chunk_ids_from_status(existing_doc_status)
            try:
                await self._resume_multimodal_from_stored_chunks(
                    doc_id=doc_id,
                    chunk_ids=chunk_ids,
                    file_path=file_path,
                )
            except Exception as resume_error:
                stage = self._infer_multimodal_stage_from_error(str(resume_error))
                try:
                    await self._mark_multimodal_doc_status_failed(
                        doc_id=doc_id,
                        error_msg=str(resume_error),
                        stage=stage,
                    )
                except Exception as status_error:
                    raise RuntimeError(
                        "Multimodal resume failure and failed to persist "
                        f"doc_status failure state: {status_error}"
                    ) from resume_error
                raise
            return

        multimodal_items = self._select_multimodal_retry_subset(
            multimodal_items, existing_doc_status, doc_id
        )
        if callback_manager is not None:
            callback_manager.dispatch(
                "on_multimodal_start",
                file_path=file_path,
                item_count=len(multimodal_items),
                doc_id=doc_id,
            )

        # Use ProcessorMixin's own batch processing that can handle multiple content types
        log_message = "Starting multimodal content processing..."
        self.logger.info(log_message)
        if pipeline_status_lock and pipeline_status:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        try:
            # Ensure LightRAG is initialized
            await self._ensure_lightrag_initialized()

            try:
                await self._process_multimodal_content_batch_type_aware(
                    multimodal_items=multimodal_items,
                    file_path=file_path,
                    doc_id=doc_id,
                )
            except MultimodalPartialFailureError as partial_error:
                try:
                    await self._mark_multimodal_doc_status_failed(
                        doc_id=doc_id,
                        error_msg=str(partial_error),
                        failed_items=partial_error.failed_items,
                        stage=_MM_STAGE_PARTIAL_FAILED,
                    )
                except Exception as status_error:
                    raise RuntimeError(
                        "Multimodal batch partial failure and failed to persist "
                        f"doc_status failure state: {status_error}"
                    ) from partial_error
                raise
            except Exception as batch_error:
                stage = self._infer_multimodal_stage_from_error(str(batch_error))
                try:
                    await self._mark_multimodal_doc_status_failed(
                        doc_id=doc_id,
                        error_msg=str(batch_error),
                        stage=stage,
                    )
                except Exception as status_error:
                    raise RuntimeError(
                        "Multimodal batch failure and failed to persist "
                        f"doc_status failure state: {status_error}"
                    ) from batch_error
                raise

            log_message = "Multimodal content processing complete"
            self.logger.info(log_message)
            if pipeline_status_lock and pipeline_status:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

            if callback_manager is not None:
                duration = time.time() - mm_start_time
                callback_manager.dispatch(
                    "on_multimodal_complete",
                    file_path=file_path,
                    processed_count=len(multimodal_items),
                    duration_seconds=duration,
                    doc_id=doc_id,
                )
        except Exception as e:
            self.logger.error(f"Error in multimodal processing: {e}")
            raise

    async def _process_multimodal_content_individual(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Process multimodal content individually (fallback method)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
        """
        # Use full path or basename based on config
        file_name = self._get_file_reference(file_path)

        # Collect all chunk results for batch processing (similar to text content processing)
        all_chunk_results = []
        multimodal_chunk_ids = []

        # Get current text chunks count to set proper order indexes for multimodal chunks
        existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
        existing_chunks_count = (
            existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
        )
        addon_params = getattr(self.lightrag, "addon_params", {}) or {}
        item_timeout_seconds = max(
            1.0,
            float(
                addon_params.get(
                    "multimodal_item_timeout_seconds",
                    DEFAULT_MULTIMODAL_ITEM_TIMEOUT_SECONDS,
                )
            ),
        )
        failed_items: List[Dict[str, Any]] = []

        for i, item in enumerate(multimodal_items):
            try:
                content_type = item.get("type", "unknown")
                self.logger.info(
                    f"Processing item {i+1}/{len(multimodal_items)}: {content_type} content"
                )

                # Select appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Prepare item info for context extraction
                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": i,
                        "type": content_type,
                    }

                    # Process content and get chunk results instead of immediately merging
                    async def _process_once() -> Tuple[str, Dict[str, Any], List[Any]]:
                        return await asyncio.wait_for(
                            processor.process_multimodal_content(
                                modal_content=item,
                                content_type=content_type,
                                file_path=file_name,
                                item_info=item_info,
                                batch_mode=True,
                                doc_id=doc_id,
                                chunk_order_index=existing_chunks_count + i,
                            ),
                            timeout=item_timeout_seconds,
                        )

                    process_result = await self._run_multimodal_call_with_retry(
                        operation_name=f"{content_type} item {i}",
                        operation=_process_once,
                    )

                    if not isinstance(process_result, tuple):
                        raise TypeError(
                            "process_multimodal_content must return a 3-tuple, got "
                            f"{type(process_result).__name__}"
                        )
                    if len(process_result) != 3:
                        raise ValueError(
                            "process_multimodal_content must return "
                            "(description, entity_info, chunk_results), got tuple length "
                            f"{len(process_result)} for {content_type}"
                        )
                    enhanced_caption, entity_info, chunk_results = process_result
                    if not chunk_results:
                        raise RuntimeError(
                            f"{content_type} item {i} returned empty chunk_results"
                        )

                    # Collect chunk results for batch processing
                    all_chunk_results.extend(chunk_results)

                    # Extract chunk ID from the entity_info (actual chunk_id created by processor)
                    if entity_info and "chunk_id" in entity_info:
                        chunk_id = entity_info["chunk_id"]
                        multimodal_chunk_ids.append(chunk_id)

                    self.logger.info(
                        f"{content_type} processing complete: {entity_info.get('entity_name', 'Unknown')}"
                    )
                else:
                    self.logger.warning(
                        f"No suitable processor found for {content_type} type content"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                self.logger.debug("Exception details:", exc_info=True)
                failed_items.append(
                    self._build_multimodal_failed_result(
                        index=i,
                        content_type=str(item.get("type", "unknown")),
                        item=item,
                        category=self._categorize_multimodal_failure(e),
                        error=e,
                    )
                )
                continue

        # Batch merge all multimodal content results (similar to text content processing)
        if all_chunk_results:
            from lightrag.operate import merge_nodes_and_edges
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            # Get pipeline status and lock from shared storage
            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            await merge_nodes_and_edges(
                chunk_results=all_chunk_results,
                knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
                entity_vdb=self.lightrag.entities_vdb,
                relationships_vdb=self.lightrag.relationships_vdb,
                global_config=self.lightrag.__dict__,
                full_entities_storage=self.lightrag.full_entities,
                full_relations_storage=self.lightrag.full_relations,
                doc_id=doc_id,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.lightrag.llm_response_cache,
                entity_chunks_storage=self.lightrag.entity_chunks,
                relation_chunks_storage=self.lightrag.relation_chunks,
                current_file_number=1,
                total_files=1,
                file_path=file_name,
            )

            await self.lightrag._insert_done()

        self.logger.info("Individual multimodal content processing complete")

        if failed_items:
            error_msg = (
                "Multimodal fallback completed with failures: "
                f"{len(failed_items)}/{len(multimodal_items)} items failed"
            )
            raise MultimodalPartialFailureError(error_msg, failed_items)

        await self._finalize_multimodal_doc_status(doc_id, multimodal_chunk_ids)

    async def _process_multimodal_content_batch_type_aware(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Type-aware batch processing that selects correct processors based on content type.
        This is the corrected implementation that handles different modality types properly.

        Args:
            multimodal_items: List of multimodal items with different types
            file_path: File path for citation
            doc_id: Document ID for proper association
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        total_items = len(multimodal_items)
        guardrails = self._resolve_multimodal_batch_guardrails(total_items=total_items)
        item_timeout_seconds = float(guardrails["item_timeout_seconds"])
        batch_watchdog_seconds = float(guardrails["batch_watchdog_seconds"])
        cancel_grace_seconds = float(guardrails["cancel_grace_seconds"])
        item_parallelism = int(guardrails["parallelism"])
        retry_attempts = int(guardrails["retry_attempts"])

        # Get existing chunks count for proper order indexing
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            existing_chunks_count = (
                existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
            )
        except Exception:
            existing_chunks_count = 0

        # Use per-document item parallelism derived from LightRAG runtime settings.
        semaphore = asyncio.Semaphore(item_parallelism)

        # Progress tracking variables
        completed_count = 0
        progress_lock = asyncio.Lock()
        progress_log_interval = max(1, total_items // 10)

        # Log processing start
        self.logger.info(
            "Starting multimodal batch: items=%d parallelism=%d item_timeout=%.1fs "
            "retry_attempts=%d batch_watchdog=%.1fs doc_id=%s",
            total_items,
            item_parallelism,
            item_timeout_seconds,
            retry_attempts,
            batch_watchdog_seconds,
            doc_id,
        )

        async def _record_progress() -> None:
            nonlocal completed_count
            async with progress_lock:
                completed_count += 1
                if (
                    completed_count % progress_log_interval == 0
                    or completed_count == total_items
                ):
                    progress_percent = (completed_count / total_items) * 100
                    self.logger.info(
                        "Multimodal chunk generation progress: %d/%d (%.1f%%)",
                        completed_count,
                        total_items,
                        progress_percent,
                    )

        # Stage 1: Concurrent generation of descriptions using correct processors for each type
        async def process_single_item_with_correct_processor(
            item: Dict[str, Any], index: int, file_path: str
        ):
            """Process single item using the correct processor for its type"""
            async with semaphore:
                content_type = str(item.get("type", "unknown"))
                try:
                    # Select the correct processor based on content type
                    processor = get_processor_for_type(
                        self.modal_processors, content_type
                    )

                    if not processor:
                        error = f"No processor found for type: {content_type}"
                        self.logger.warning(error)
                        await _record_progress()
                        return self._build_multimodal_failed_result(
                            index=index,
                            content_type=content_type,
                            item=item,
                            category="other",
                            error=error,
                        )

                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": index,
                        "type": content_type,
                    }

                    # Call the correct processor's description generation method
                    async def _generate_once() -> Tuple[str, Dict[str, Any]]:
                        return await asyncio.wait_for(
                            processor.generate_description_only(
                                modal_content=item,
                                content_type=content_type,
                                item_info=item_info,
                                entity_name=None,
                                raise_on_error=True,
                            ),
                            timeout=item_timeout_seconds,
                        )

                    description, entity_info = (
                        await self._run_multimodal_call_with_retry(
                            operation_name=f"{content_type} item {index}",
                            operation=_generate_once,
                        )
                    )

                    await _record_progress()

                    return {
                        "status": "ok",
                        "index": index,
                        "type": content_type,
                        "description": description,
                        "entity_info": entity_info,
                        "original_item": item,
                        "item_info": item_info,
                        "chunk_order_index": existing_chunks_count + index,
                        "processor": processor,  # Keep reference to the processor used
                        "file_path": file_path,  # Add file_path to the result
                    }

                except Exception as e:
                    await _record_progress()
                    self.logger.error(
                        f"Error generating description for {content_type} item {index}: {e}"
                    )
                    return self._build_multimodal_failed_result(
                        index=index,
                        content_type=content_type,
                        item=item,
                        category=self._categorize_multimodal_failure(e),
                        error=e,
                    )

        # Process all items concurrently with correct processors
        task_specs = []
        for index, item in enumerate(multimodal_items):
            task_specs.append(
                {
                    "index": index,
                    "type": str(item.get("type", "unknown")),
                    "item": item,
                    "task": asyncio.create_task(
                        process_single_item_with_correct_processor(item, index, file_path)
                    ),
                }
            )

        try:
            raw_results = await asyncio.wait_for(
                asyncio.gather(
                    *[spec["task"] for spec in task_specs], return_exceptions=True
                ),
                timeout=batch_watchdog_seconds,
            )
            results = [
                self._normalize_multimodal_task_result(
                    result=result,
                    index=int(spec["index"]),
                    content_type=str(spec["type"]),
                    item=spec["item"],
                )
                for spec, result in zip(task_specs, raw_results)
            ]
        except asyncio.TimeoutError:
            self.logger.error(
                "Multimodal batch watchdog timeout after %.1fs: items=%d parallelism=%d doc_id=%s. Cancelling pending tasks.",
                batch_watchdog_seconds,
                total_items,
                item_parallelism,
                doc_id,
            )
            for spec in task_specs:
                task = spec["task"]
                if not task.done():
                    task.cancel()
            if cancel_grace_seconds > 0:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[spec["task"] for spec in task_specs],
                            return_exceptions=True,
                        ),
                        timeout=cancel_grace_seconds,
                    )
                except Exception as cancel_exc:
                    self.logger.warning(
                        "Timed out while waiting cancelled multimodal tasks to exit (grace=%.1fs): %s",
                        cancel_grace_seconds,
                        cancel_exc,
                    )
            results = []
            for spec in task_specs:
                task = spec["task"]
                if not task.done():
                    results.append(
                        self._build_multimodal_failed_result(
                            index=int(spec["index"]),
                            content_type=str(spec["type"]),
                            item=spec["item"],
                            category="timeout",
                            error="pending task not finished before watchdog cleanup",
                        )
                    )
                    continue
                if task.cancelled():
                    results.append(
                        self._build_multimodal_failed_result(
                            index=int(spec["index"]),
                            content_type=str(spec["type"]),
                            item=spec["item"],
                            category="cancelled",
                            error="task cancelled by watchdog",
                        )
                    )
                    continue
                try:
                    results.append(
                        self._normalize_multimodal_task_result(
                            result=task.result(),
                            index=int(spec["index"]),
                            content_type=str(spec["type"]),
                            item=spec["item"],
                        )
                    )
                except BaseException as exc:
                    results.append(
                        self._build_multimodal_failed_result(
                            index=int(spec["index"]),
                            content_type=str(spec["type"]),
                            item=spec["item"],
                            category=self._categorize_multimodal_failure(exc),
                            error=exc,
                        )
                    )

        # Filter successful results and keep structured failure records.
        multimodal_data_list: List[Dict[str, Any]] = []
        failed_items: List[Dict[str, Any]] = []
        failure_stats = {
            "timeout": 0,
            "parse": 0,
            "model": 0,
            "cancelled": 0,
            "other": 0,
        }
        for result in results:
            if isinstance(result, dict):
                status = result.get("status", "")
                if status == "ok":
                    multimodal_data_list.append(result)
                    continue
                if status == "failed":
                    category = str(result.get("category", "other"))
                    if category not in failure_stats:
                        category = "other"
                    failure_stats[category] += 1
                    failed_items.append(result)
                    self.logger.error(
                        "Multimodal task failed: index=%s type=%s category=%s error=%s",
                        result.get("index", -1),
                        result.get("type", "unknown"),
                        category,
                        result.get("error", ""),
                    )
                    continue

            failure_stats["other"] += 1
            failed_items.append(
                self._build_multimodal_failed_result(
                    index=-1,
                    content_type="unknown",
                    item=None,
                    category="other",
                    error=f"Unexpected task result type: {type(result).__name__}",
                )
            )

        if not multimodal_data_list:
            self.logger.warning(
                "No valid multimodal descriptions generated: failed=%d/%d details=%s",
                len(failed_items),
                len(results),
                failure_stats,
            )
            raise MultimodalPartialFailureError(
                "No valid multimodal descriptions generated", failed_items
            )

        self.logger.info(
            f"Generated descriptions for {len(multimodal_data_list)}/{len(multimodal_items)} multimodal items using correct processors"
        )

        # Stage 2: Convert to LightRAG chunks format
        lightrag_chunks = self._convert_to_lightrag_chunks_type_aware(
            multimodal_data_list, file_path, doc_id
        )

        # Stage 3: Store chunks to LightRAG storage
        await self._store_chunks_to_lightrag_storage_type_aware(lightrag_chunks)

        # Track chunk IDs for doc_status update
        chunk_ids = list(lightrag_chunks.keys())
        await self._finalize_multimodal_doc_status(
            doc_id=doc_id,
            new_chunk_ids=chunk_ids,
            mark_processed=False,
            stage=_MM_STAGE_CHUNKS_STORED,
        )

        # Stage 3.5: Store multimodal main entities to entities_vdb and full_entities
        try:
            await self._store_multimodal_main_entities(
                multimodal_data_list, lightrag_chunks, file_path, doc_id
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_MAIN_ENTITY_FAILED}: {exc}") from exc

        # Stage 4: Use LightRAG's batch entity relation extraction
        try:
            chunk_results = await self._batch_extract_entities_lightrag_style_type_aware(
                lightrag_chunks
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_ER_FAILED}: {exc}") from exc

        # Stage 5: Add belongs_to relations (multimodal-specific)
        try:
            enhanced_chunk_results = (
                await self._batch_add_belongs_to_relations_type_aware(
                    chunk_results, multimodal_data_list
                )
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_BELONGS_TO_FAILED}: {exc}") from exc

        # Stage 6: Use LightRAG's batch merge
        try:
            await self._batch_merge_lightrag_style_type_aware(
                enhanced_chunk_results, file_path, doc_id
            )
        except Exception as exc:
            raise RuntimeError(f"{_MM_STAGE_MERGE_FAILED}: {exc}") from exc

        # Stage 7: Finalize multimodal doc_status.
        if failed_items:
            self.logger.warning(
                "Multimodal batch partial failures: failed=%d/%d details=%s",
                len(failed_items),
                len(results),
                failure_stats,
            )
            # Persist successful chunk progress first, then let upper layer retry failed items.
            await self._finalize_multimodal_doc_status(
                doc_id,
                chunk_ids,
                mark_processed=False,
                stage=_MM_STAGE_PARTIAL_FAILED,
            )
            raise MultimodalPartialFailureError(
                f"Multimodal batch failed for {len(failed_items)}/{len(results)} items",
                failed_items,
            )

        await self._finalize_multimodal_doc_status(doc_id, chunk_ids)

    def _convert_to_lightrag_chunks_type_aware(
        self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str
    ) -> Dict[str, Any]:
        """Convert multimodal data to LightRAG standard chunks format"""

        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["type"]
            original_item = data["original_item"]

            # Apply the appropriate chunk template based on content type
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Calculate tokens
            tokens = len(self.lightrag.tokenizer.encode(formatted_chunk_content))

            # Use full path or basename based on config
            file_ref = self._get_file_reference(file_path)

            # Build LightRAG standard chunk format
            chunks[chunk_id] = {
                "content": formatted_chunk_content,  # Now uses the templated content
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": file_ref,
                "llm_cache_list": [],  # LightRAG will populate this field
                # Multimodal-specific metadata
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": data["type"],
                "page_idx": data["item_info"].get("page_idx", 0),
            }

        self.logger.debug(
            f"Converted {len(chunks)} multimodal items to multimodal chunks format"
        )
        return chunks

    def _apply_chunk_template(
        self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """
        Apply the appropriate chunk template based on content type

        Args:
            content_type: Type of content (image, table, equation, generic)
            original_item: Original multimodal item data
            description: Enhanced description generated by the processor

        Returns:
            Formatted chunk content using the appropriate template
        """
        from raganything.prompt import PROMPTS
        from raganything.modalprocessors import _chunk_modal_text

        try:
            if content_type == "image":
                image_path = original_item.get("img_path", "")
                captions = original_item.get(
                    "image_caption", original_item.get("img_caption", [])
                )
                footnotes = original_item.get(
                    "image_footnote", original_item.get("img_footnote", [])
                )

                return PROMPTS["image_chunk"].format(
                    image_path=image_path,
                    captions=_chunk_modal_text(captions, "image_caption")
                    if captions
                    else "None",
                    footnotes=_chunk_modal_text(footnotes, "image_footnote")
                    if footnotes
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_caption = original_item.get("table_caption", [])
                table_body = original_item.get("table_body", "")
                table_footnote = original_item.get("table_footnote", [])

                return PROMPTS["table_chunk"].format(
                    table_img_path=table_img_path,
                    table_caption=_chunk_modal_text(table_caption, "table_caption")
                    if table_caption
                    else "None",
                    table_body=_chunk_modal_text(table_body, "table_body"),
                    table_footnote=_chunk_modal_text(
                        table_footnote, "table_footnote"
                    )
                    if table_footnote
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "equation":
                equation_text = original_item.get("text", "")
                equation_format = original_item.get("text_format", "")

                return PROMPTS["equation_chunk"].format(
                    equation_text=_chunk_modal_text(equation_text, "equation_text"),
                    equation_format=equation_format,
                    enhanced_caption=description,
                )

            else:  # generic or unknown types
                content = str(original_item.get("content", original_item))

                return PROMPTS["generic_chunk"].format(
                    content_type=content_type.title(),
                    content=_chunk_modal_text(content, "generic_content"),
                    enhanced_caption=description,
                )

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            # Fallback to just the description if template fails
            return description

    async def _store_chunks_to_lightrag_storage_type_aware(
        self, chunks: Dict[str, Any]
    ):
        """Store chunks to storage"""
        try:
            # Store in text_chunks storage (required for extract_entities)
            await self.lightrag.text_chunks.upsert(chunks)

            # Store in chunks vector database for retrieval
            await self.lightrag.chunks_vdb.upsert(chunks)

            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")

        except Exception as e:
            self.logger.error(f"Error storing chunks to storage: {e}")
            raise

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        lightrag_chunks: Dict[str, Any],
        file_path: str,
        doc_id: str = None,
    ):
        """
        Store multimodal main entities to entities_vdb and full_entities.
        This ensures that entities like "TableName (table)" are properly indexed.

        Args:
            multimodal_data_list: List of processed multimodal data with entity info
            lightrag_chunks: Chunks in LightRAG format (already formatted with templates)
            file_path: File path for the entities
            doc_id: Document ID for full_entities storage
        """
        if not multimodal_data_list:
            return

        # Create entities_vdb entries for all multimodal main entities
        entities_to_store = {}

        # Use full path or basename based on config
        file_ref = self._get_file_reference(file_path)

        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            description = data["description"]
            content_type = data["type"]
            original_item = data["original_item"]

            # Apply the same chunk template to get the formatted content
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id using the formatted content (same as in _convert_to_lightrag_chunks)
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Generate entity_id using LightRAG's standard format
            _etype = entity_info.get("entity_type", content_type)
            _disambig = getattr(self.lightrag, "enable_entity_disambiguation", True)
            entity_id = compute_entity_vdb_id(entity_name, _etype, _disambig)

            # Create entity data in LightRAG format
            entity_data = {
                "entity_name": entity_name,
                "entity_type": _etype,
                "entity_id": compute_entity_id(entity_name, _etype, _disambig),
                "content": entity_info.get("summary", description),
                "source_id": chunk_id,
                "file_path": file_ref,
            }

            entities_to_store[entity_id] = entity_data

        if entities_to_store:
            try:
                await self._upsert_multimodal_main_entities_to_core_storage(
                    entities_to_store
                )

                # NEW: Store multimodal main entities in full_entities storage
                if doc_id and self.lightrag.full_entities:
                    await self._store_multimodal_entities_to_full_entities(
                        entities_to_store, doc_id
                    )

                self.logger.debug(
                    f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
                )

            except Exception as e:
                self.logger.error(f"Error storing multimodal main entities: {e}")
                raise

    async def _store_multimodal_entities_to_full_entities(
        self, entities_to_store: Dict[str, Any], doc_id: str
    ):
        """
        Store multimodal main entities to full_entities storage.

        Args:
            entities_to_store: Dictionary of entities to store
            doc_id: Document ID for grouping entities
        """
        try:
            # Get current full_entities data for this document
            current_doc_entities = await self.lightrag.full_entities.get_by_id(doc_id)
            incoming_entity_ids = [
                str(entity_data.get("entity_id", "")).strip()
                for entity_data in entities_to_store.values()
            ]
            incoming_entity_ids = [
                entity_id for entity_id in incoming_entity_ids if entity_id
            ]

            if current_doc_entities is None:
                # Create new document entry
                entity_names = incoming_entity_ids
                doc_entities_data = {
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            else:
                # Update existing document entry while preserving original metadata fields.
                existing_entity_names = list(
                    current_doc_entities.get("entity_names", [])
                )
                seen_entity_names = set(existing_entity_names)
                for entity_id in incoming_entity_ids:
                    if entity_id not in seen_entity_names:
                        existing_entity_names.append(entity_id)
                        seen_entity_names.add(entity_id)
                doc_entities_data = {
                    **current_doc_entities,
                    "entity_names": existing_entity_names,
                    "count": len(existing_entity_names),
                    "update_time": int(time.time()),
                }

            # Store updated data
            await self.lightrag.full_entities.upsert({doc_id: doc_entities_data})
            await self.lightrag.full_entities.index_done_callback()

            self.logger.debug(
                f"Added {len(entities_to_store)} multimodal main entities to full_entities for doc {doc_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Error storing multimodal entities to full_entities: {e}"
            )
            raise

    async def _batch_extract_entities_lightrag_style_type_aware(
        self, lightrag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """Use LightRAG's extract_entities for batch entity relation extraction"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import extract_entities

        # Get pipeline status (consistent with LightRAG)
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Directly use LightRAG's extract_entities
        chunk_results = await extract_entities(
            chunks=lightrag_chunks,
            global_config=self.lightrag.__dict__,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        self.logger.info(
            f"Extracted entities from {len(lightrag_chunks)} multimodal chunks"
        )
        return chunk_results

    async def _batch_add_belongs_to_relations_type_aware(
        self, chunk_results: List[Tuple], multimodal_data_list: List[Dict[str, Any]]
    ) -> List[Tuple]:
        """Add belongs_to relations for multimodal entities"""
        # Create mapping from chunk_id to modal_entity_name
        chunk_to_modal_entity = {}
        chunk_to_file_path = {}

        for data in multimodal_data_list:
            description = data["description"]
            content_type = data["type"]
            original_item = data["original_item"]

            # Use the same template formatting as in _convert_to_lightrag_chunks_type_aware
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            entity_info = data["entity_info"]
            modal_entity_type = (
                str(entity_info.get("entity_type", "multimodal") or "multimodal").strip()
                or "multimodal"
            )
            chunk_to_modal_entity[chunk_id] = (entity_info["entity_name"], modal_entity_type)
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")

        return await self._batch_add_belongs_to_relations_by_chunk_mapping(
            chunk_results=chunk_results,
            chunk_to_modal_entity=chunk_to_modal_entity,
            chunk_to_file_path=chunk_to_file_path,
        )

    async def _batch_add_belongs_to_relations_from_stored_chunks(
        self, chunk_results: List[Tuple], lightrag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        chunk_to_modal_entity: Dict[str, Tuple[str, str]] = {}
        chunk_to_file_path: Dict[str, str] = {}
        for chunk_id, chunk_data in lightrag_chunks.items():
            if not isinstance(chunk_data, dict):
                continue
            modal_entity_name = str(chunk_data.get("modal_entity_name", "")).strip()
            if not modal_entity_name:
                continue
            modal_entity_type = (
                str(chunk_data.get("original_type", "multimodal")).strip() or "multimodal"
            )
            chunk_to_modal_entity[chunk_id] = (modal_entity_name, modal_entity_type)
            chunk_to_file_path[chunk_id] = str(
                chunk_data.get("file_path", "multimodal_content")
            )

        return await self._batch_add_belongs_to_relations_by_chunk_mapping(
            chunk_results=chunk_results,
            chunk_to_modal_entity=chunk_to_modal_entity,
            chunk_to_file_path=chunk_to_file_path,
        )

    async def _batch_add_belongs_to_relations_by_chunk_mapping(
        self,
        chunk_results: List[Tuple],
        chunk_to_modal_entity: Dict[str, Tuple[str, str] | str],
        chunk_to_file_path: Dict[str, str],
    ) -> List[Tuple]:
        enhanced_chunk_results = []
        belongs_to_count = 0

        for maybe_nodes, maybe_edges in chunk_results:
            # Find corresponding modal_entity_name for this chunk
            chunk_id = None
            for nodes_dict in maybe_nodes.values():
                if nodes_dict:
                    chunk_id = nodes_dict[0].get("source_id")
                    break
            if not chunk_id:
                for edge_records in maybe_edges.values():
                    if edge_records:
                        chunk_id = edge_records[0].get("source_id")
                        if chunk_id:
                            break

            if chunk_id and chunk_id in chunk_to_modal_entity:
                modal_entity_value = chunk_to_modal_entity[chunk_id]
                if isinstance(modal_entity_value, (tuple, list)):
                    modal_entity_name = str(modal_entity_value[0]).strip()
                    raw_modal_entity_type = (
                        modal_entity_value[1]
                        if len(modal_entity_value) > 1
                        else "multimodal"
                    )
                    modal_entity_type = (
                        str(raw_modal_entity_type).strip()
                        or "multimodal"
                    )
                else:
                    modal_entity_name = str(modal_entity_value).strip()
                    modal_entity_type = "multimodal"
                if not modal_entity_name:
                    enhanced_chunk_results.append((maybe_nodes, maybe_edges))
                    continue
                file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")

                # Add belongs_to relations for all extracted entities
                for entity_name in list(maybe_nodes.keys()):
                    if entity_name != modal_entity_name:  # Avoid self-relation
                        belongs_to_relation = {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "source_id": chunk_id,
                            "weight": 1.0,
                            "file_path": file_path,
                        }

                        # Add to maybe_edges
                        edge_key = (entity_name, modal_entity_name)
                        if edge_key not in maybe_edges:
                            maybe_edges[edge_key] = []
                        maybe_edges[edge_key].append(belongs_to_relation)
                        belongs_to_count += 1

                # Inject a stub entry for the modal entity into maybe_nodes so that
                # merge_nodes_and_edges includes its composite ID in
                # doc_relation_endpoint_ids. Without this, strict endpoint matching
                # drops every belongs_to edge because the modal entity was written to
                # the graph in stage 3.5 (bypassing the normal extraction path) and
                # therefore never appears in processed_entities.
                if modal_entity_name not in maybe_nodes:
                    maybe_nodes[modal_entity_name] = []
                maybe_nodes[modal_entity_name].append({
                    "entity_name": modal_entity_name,
                    "entity_type": modal_entity_type,
                    "description": "",
                    "source_id": chunk_id,
                    "file_path": file_path,
                })

            enhanced_chunk_results.append((maybe_nodes, maybe_edges))

        self.logger.info(
            f"Added {belongs_to_count} belongs_to relations for multimodal entities"
        )
        return enhanced_chunk_results

    async def _batch_merge_lightrag_style_type_aware(
        self, enhanced_chunk_results: List[Tuple], file_path: str, doc_id: str = None
    ):
        """Use LightRAG's merge_nodes_and_edges for batch merge"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import merge_nodes_and_edges

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Use full path or basename based on config
        file_ref = self._get_file_reference(file_path)

        await merge_nodes_and_edges(
            chunk_results=enhanced_chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=self.lightrag.__dict__,
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            entity_chunks_storage=self.lightrag.entity_chunks,
            relation_chunks_storage=self.lightrag.relation_chunks,
            current_file_number=1,
            total_files=1,
            file_path=file_ref,
        )

        await self.lightrag._insert_done()

    async def _finalize_multimodal_doc_status(
        self,
        doc_id: str,
        new_chunk_ids: List[str],
        mark_processed: bool = True,
        stage: Optional[str] = None,
    ) -> None:
        """Finalize multimodal doc_status with one atomic upsert."""
        try:
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not current_doc_status:
                raise RuntimeError(f"Missing doc_status for document {doc_id}")

            existing_chunks_list = current_doc_status.get("chunks_list", [])
            merged_chunks_list = list(
                dict.fromkeys([*existing_chunks_list, *new_chunk_ids])
            )
            existing_multimodal_chunks = current_doc_status.get(
                "multimodal_chunk_ids", []
            )
            if not isinstance(existing_multimodal_chunks, list):
                existing_multimodal_chunks = []
            merged_multimodal_chunks = list(
                dict.fromkeys([*existing_multimodal_chunks, *new_chunk_ids])
            )
            updated_doc_status = {
                **current_doc_status,
                "chunks_list": merged_chunks_list,
                "chunks_count": len(merged_chunks_list),
                "multimodal_chunk_ids": merged_multimodal_chunks,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            }
            if mark_processed:
                updated_doc_status["multimodal_processed"] = True
                updated_doc_status["multimodal_stage"] = _MM_STAGE_COMPLETED
                updated_doc_status["multimodal_error_msg"] = None
                updated_doc_status["multimodal_failed_items"] = []
            else:
                updated_doc_status["multimodal_processed"] = False
                if stage:
                    updated_doc_status["multimodal_stage"] = stage
            await self.lightrag.doc_status.upsert({doc_id: updated_doc_status})
            await self.lightrag.doc_status.index_done_callback()
            self.logger.info(
                "Finalized multimodal doc_status for %s: chunks=%d (added=%d, mm_chunks=%d, processed=%s, stage=%s)",
                doc_id,
                len(merged_chunks_list),
                len(new_chunk_ids),
                len(merged_multimodal_chunks),
                mark_processed,
                updated_doc_status.get("multimodal_stage", ""),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to finalize multimodal doc_status for {doc_id}: {e}"
            ) from e

    async def _mark_multimodal_doc_status_failed(
        self,
        doc_id: str,
        error_msg: str,
        failed_items: Optional[List[Dict[str, Any]]] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Persist multimodal failure state; raise if persistence fails."""
        current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
        if not current_doc_status:
            raise RuntimeError(
                f"Missing doc_status for document {doc_id} while persisting multimodal failure"
            )

        updated_doc_status = {
            **current_doc_status,
            "multimodal_processed": False,
            "multimodal_error_msg": str(error_msg)[:4096],
            "multimodal_failed_items": failed_items or [],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        }
        if stage:
            updated_doc_status["multimodal_stage"] = stage
        await self.lightrag.doc_status.upsert({doc_id: updated_doc_status})
        await self.lightrag.doc_status.index_done_callback()

    async def is_document_fully_processed(self, doc_id: str) -> bool:
        """
        Check if a document is fully processed (both text and multimodal content).

        Args:
            doc_id: Document ID to check

        Returns:
            bool: True if both text and multimodal content are processed
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return False

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)

            return text_processed and multimodal_processed

        except Exception as e:
            self.logger.error(
                f"Error checking document processing status for {doc_id}: {e}"
            )
            return False

    async def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get detailed processing status for a document.

        Args:
            doc_id: Document ID to check

        Returns:
            Dict with processing status details
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return {
                    "exists": False,
                    "text_processed": False,
                    "multimodal_processed": False,
                    "fully_processed": False,
                    "chunks_count": 0,
                }

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)
            fully_processed = text_processed and multimodal_processed

            return {
                "exists": True,
                "text_processed": text_processed,
                "multimodal_processed": multimodal_processed,
                "fully_processed": fully_processed,
                "chunks_count": doc_status.get("chunks_count", 0),
                "chunks_list": doc_status.get("chunks_list", []),
                "status": doc_status.get("status", ""),
                "updated_at": doc_status.get("updated_at", ""),
                "raw_status": doc_status,
            }

        except Exception as e:
            self.logger.error(
                f"Error getting document processing status for {doc_id}: {e}"
            )
            return {
                "exists": False,
                "error": str(e),
                "text_processed": False,
                "multimodal_processed": False,
                "fully_processed": False,
                "chunks_count": 0,
            }

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        file_name: str | None = None,
        **kwargs,
    ):
        """
        Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        callback_manager = getattr(self, "callback_manager", None)
        doc_start_time = time.time()
        stage = "parse"

        try:
            # Ensure LightRAG is initialized
            await self._ensure_lightrag_initialized()

            # Use config defaults if not provided
            if output_dir is None:
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                parse_method = self.config.parse_method
            if display_stats is None:
                display_stats = self.config.display_content_stats

            self.logger.info(f"Starting complete document processing: {file_path}")

            # Step 1: Parse document
            content_list, content_based_doc_id = await self.parse_document(
                file_path, output_dir, parse_method, display_stats, **kwargs
            )

            # Use provided doc_id or fall back to content-based doc_id
            if doc_id is None:
                doc_id = content_based_doc_id

            # Step 2: Separate text and multimodal content
            text_content, multimodal_items = separate_content(content_list)

            # Step 2.5: Set content source for context extraction in multimodal processing
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # Step 3: Insert pure text content with all parameters
            stage = "text_insert"
            if text_content.strip():
                if file_name is None:
                    # Use full path or basename based on config
                    file_name = self._get_file_reference(file_path)
                if callback_manager is not None:
                    callback_manager.dispatch(
                        "on_text_insert_start",
                        file_path=file_name,
                        text_length=len(text_content),
                        doc_id=doc_id,
                    )
                insert_start = time.time()
                await insert_text_content(
                    self.lightrag,
                    input=text_content,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                )
                if callback_manager is not None:
                    insert_duration = time.time() - insert_start
                    callback_manager.dispatch(
                        "on_text_insert_complete",
                        file_path=file_name,
                        duration_seconds=insert_duration,
                        doc_id=doc_id,
                    )
            else:
                # Determine file reference even if no text content
                if file_name is None:
                    file_name = self._get_file_reference(file_path)

            # Step 4: Process multimodal content (using specialized processors)
            stage = "multimodal"
            if multimodal_items:
                await self._process_multimodal_content(
                    multimodal_items, file_name, doc_id
                )
            else:
                # If no multimodal content, mark multimodal processing as complete
                # This ensures the document status properly reflects completion of all processing
                await self._finalize_multimodal_doc_status(doc_id, [])
                self.logger.debug(
                    f"No multimodal content found in document {doc_id}, "
                    "marked multimodal processing as complete",
                )

        except Exception as exc:
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_document_error",
                    file_path=str(file_path),
                    doc_id=doc_id,
                    stage=stage,
                    error=exc,
                )
            raise

        self.logger.info(f"Document {file_path} processing complete!")
        if callback_manager is not None:
            duration = time.time() - doc_start_time
            callback_manager.dispatch(
                "on_document_complete",
                file_path=str(file_path),
                doc_id=doc_id,
                duration_seconds=duration,
            )

    async def process_document_complete_lightrag_api(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        scheme_name: str | None = None,
        parser: str | None = None,
        **kwargs,
    ):
        """
        API exclusively for LightRAG calls: Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        # Use full path or basename based on config
        file_name = self._get_file_reference(file_path)
        doc_pre_id = f"doc-pre-{file_name}"
        pipeline_status = None
        pipeline_status_lock = None

        if parser:
            self.config.parser = parser

        current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)

        try:
            # Ensure LightRAG is initialized
            result = await self._ensure_lightrag_initialized()
            if not result["success"]:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": result["error"],
                        }
                    }
                )
                return False

            # Use config defaults if not provided
            if output_dir is None:
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                parse_method = self.config.parse_method
            if display_stats is None:
                display_stats = self.config.display_content_stats

            self.logger.info(f"Starting complete document processing: {file_path}")

            # Initialize doc status
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)
            if not current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            "status": DocStatus.READY,
                            "content": "",
                            "error_msg": "",
                            "content_summary": "",
                            "multimodal_content": [],
                            "scheme_name": scheme_name,
                            "content_length": 0,
                            "created_at": "",
                            "updated_at": "",
                            "file_path": file_name,
                        }
                    }
                )
                current_doc_status = await self.lightrag.doc_status.get_by_id(
                    doc_pre_id
                )

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            # Set processing status
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": True})
                pipeline_status["history_messages"].append("Now is not allowed to scan")

            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.HANDLING,
                        "error_msg": "",
                    }
                }
            )

            content_list = []
            content_based_doc_id = ""

            try:
                # Step 1: Parse document
                content_list, content_based_doc_id = await self.parse_document(
                    file_path, output_dir, parse_method, display_stats, **kwargs
                )
            except MineruExecutionError as e:
                error_message = e.error_msg
                if isinstance(e.error_msg, list):
                    error_message = "\n".join(e.error_msg)
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": error_message,
                        }
                    }
                )
                self.logger.info(
                    f"Error processing document {file_path}: MineruExecutionError"
                )
                return False
            except Exception as e:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": str(e),
                        }
                    }
                )
                self.logger.info(f"Error processing document {file_path}: {str(e)}")
                return False

            # Use provided doc_id or fall back to content-based doc_id
            if doc_id is None:
                doc_id = content_based_doc_id

            # Step 2: Separate text and multimodal content
            text_content, multimodal_items = separate_content(content_list)

            # Step 2.5: Set content source for context extraction in multimodal processing
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # Step 3: Insert pure text content and multimodal content with all parameters
            if text_content.strip():
                await insert_text_content_with_multimodal_content(
                    self.lightrag,
                    input=text_content,
                    multimodal_content=multimodal_items,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                    scheme_name=scheme_name,
                )

            self.logger.info(f"Document {file_path} processing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)

            # Update doc status to Failed
            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.FAILED,
                        "error_msg": str(e),
                    }
                }
            )
            await self.lightrag.doc_status.index_done_callback()

            # Update pipeline status
            if pipeline_status_lock and pipeline_status:
                try:
                    async with pipeline_status_lock:
                        pipeline_status.update({"scan_disabled": False})
                        error_msg = (
                            f"RAGAnything processing failed for {file_name}: {str(e)}"
                        )
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)
                        pipeline_status["history_messages"].append(
                            "Now is allowed to scan"
                        )
                except Exception as pipeline_update_error:
                    self.logger.error(
                        f"Failed to update pipeline status: {pipeline_update_error}"
                    )

            return False

        finally:
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": False})
                pipeline_status["latest_message"] = (
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append(
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append("Now is allowed to scan")

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        display_stats: bool = None,
    ):
        """
        Insert content list directly without document parsing

        Args:
            content_list: Pre-parsed content list containing text and multimodal items.
                         Each item should be a dictionary with the following structure:
                         - Text: {"type": "text", "text": "content", "page_idx": 0}
                         - Image: {"type": "image", "img_path": "/absolute/path/to/image.jpg",
                                  "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}
                         - Table: {"type": "table", "table_body": "markdown table",
                                  "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}
                         - Equation: {"type": "equation", "latex": "LaTeX formula",
                                     "text": "description", "page_idx": 3}
                         - Generic: {"type": "custom_type", "content": "any content", "page_idx": 4}
            file_path: Reference file path/name for citation (defaults to "unknown_document")
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)

        Note:
            - img_path must be an absolute path to the image file
            - page_idx represents the page number where the content appears (0-based indexing)
            - Items are processed in the order they appear in the list
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        # Use config defaults if not provided
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(
            f"Starting direct content list insertion for: {file_path} ({len(content_list)} items)"
        )

        # Generate doc_id based on content if not provided
        if doc_id is None:
            doc_id = self._generate_content_based_doc_id(content_list)

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        # Step 1: Separate text and multimodal content
        text_content, multimodal_items = separate_content(content_list)

        # Step 1.5: Set content source for context extraction in multimodal processing
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # Step 2: Insert pure text content with all parameters
        if text_content.strip():
            # Use full path or basename based on config
            file_ref = self._get_file_reference(file_path)
            await insert_text_content(
                self.lightrag,
                input=text_content,
                file_paths=file_ref,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )
        else:
            # Determine file reference even if no text content
            file_ref = self._get_file_reference(file_path)

        # Step 3: Process multimodal content (using specialized processors)
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_ref, doc_id)
        else:
            # If no multimodal content, mark multimodal processing as complete
            # This ensures the document status properly reflects completion of all processing
            await self._finalize_multimodal_doc_status(doc_id, [])
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        self.logger.info(f"Content list insertion complete for: {file_path}")
