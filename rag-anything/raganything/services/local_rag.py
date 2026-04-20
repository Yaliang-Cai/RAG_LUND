#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG-Anything Local Multimodal Pipeline (Optimized Version)

新开图谱 (处理整个文件夹)：
python raganything/services/local_rag.py -p ./data/my_paper_folder -i My_New_Graph

补充文件 (向已有图谱添加)：
python raganything/services/local_rag.py -p ./data/extra.pdf -i My_New_Graph
"""

import asyncio
import hashlib
import json
import logging
import logging.config
import os
import argparse
import ipaddress
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

import numpy as np
import httpx
import openai
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc, Tokenizer
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from raganything.callbacks import MetricsCallback, ProcessingCallback
from raganything.chunking import get_chunking_func
from raganything.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_TIKTOKEN_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_RERANK_MODEL_PATH,
    DEFAULT_TOKENIZER_MODEL_PATH,
    DEFAULT_VISION_MODEL_PATH,
    DEFAULT_WORKING_DIR_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_VLLM_API_BASE,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_TEXT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_VISION_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_TOKEN_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_QUERY_MAX_TOKENS,
    DEFAULT_INGEST_MAX_TOKENS,
    DEFAULT_VLM_ENABLE_JSON_SCHEMA,
    DEFAULT_IMAGE_TOKEN_ESTIMATE_METHOD,
    DEFAULT_IMAGE_WRAPPER_TOKENS_PER_IMAGE,
    DEFAULT_MULTIMODAL_ITEM_TIMEOUT_SECONDS,
    DEFAULT_MULTIMODAL_BATCH_WATCHDOG_SECONDS,
    DEFAULT_MULTIMODAL_CANCEL_GRACE_SECONDS,
    DEFAULT_MULTIMODAL_ENABLE_STRICT_FALLBACK,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_CHUNK_TOKEN_SIZE,
    DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE,
    DEFAULT_MINERU_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_LLM_MODEL_MAX_ASYNC,
    DEFAULT_ENTITY_EXTRACT_MAX_GLEANING,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_ENABLE_INLINE_CITATIONS,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_QUERY_MODE,
    DEFAULT_ENABLE_RERANK,
    DEFAULT_ENABLE_SYNONYM_LINKING,
    DEFAULT_SYNONYMY_THRESHOLD,
    DEFAULT_SYNONYMY_TOPK,
    DEFAULT_SYNONYMY_MIN_ENTITY_LEN,
    DEFAULT_ENABLE_MULTI_HOP,
    DEFAULT_MULTI_HOP_DEPTH,
    DEFAULT_PPR_DAMPING,
    DEFAULT_PPR_TOP_K,
    DEFAULT_PASSAGE_NODE_WEIGHT,
    DEFAULT_PPR_SYNONYM_WEIGHT_MODE,
    DEFAULT_EXCLUDE_SYNONYM_EDGES,
    DEFAULT_RECOGNITION_TOP_K,
    DEFAULT_LINKING_TOP_K,
    DEFAULT_PPR_QA_TOP_K,
    DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS,
    DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
    DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS,
    DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION,
    DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION,
    DEFAULT_ENTITY_UPPERCASE_ALLOWLIST,
    DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH,
    DEFAULT_SERIALIZE_INGEST_BY_WORKSPACE_ID,
    DEFAULT_MAX_ASYNC_INGEST,
    DEFAULT_ENABLE_RESILIENCE,
    DEFAULT_RESILIENCE_MAX_ATTEMPTS,
    DEFAULT_INGEST_RETRY_BASE_DELAY,
    DEFAULT_QUERY_RETRY_BASE_DELAY,
    DEFAULT_RESILIENCE_MAX_DELAY,
    DEFAULT_INGEST_BREAKER_FAILURE_THRESHOLD,
    DEFAULT_QUERY_BREAKER_FAILURE_THRESHOLD,
    DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS,
    DEFAULT_ENABLE_METRICS_CALLBACK,
    DEFAULT_ENABLE_CALLBACK_EVENT_LOG,
    DEFAULT_UPLOADS_DIR,
    DEFAULT_SUPPORTED_FILE_EXTENSIONS,
)
from raganything.resilience import CircuitBreaker, async_retry
from raganything.query_message_repack import repack_query_messages

# Inline citation toggle: read once at startup; env var ENABLE_INLINE_CITATIONS overrides constant
_INLINE_CITATIONS_ENABLED: bool = os.environ.get(
    "ENABLE_INLINE_CITATIONS", str(DEFAULT_ENABLE_INLINE_CITATIONS)
).lower() in ("true", "1", "yes")

_INLINE_CITATION_INSTRUCTION = (
    "MANDATORY INLINE CITATIONS: Every factual statement or claim MUST be immediately "
    "followed by an inline citation using the chunk's `id` field (e.g., `[DC1]`, `[DC2]`). "
    "Do NOT write any sentence containing a fact without an inline citation. "
    "Use ONLY the `id` values that appear in the Document Chunks above."
)

_MODEL_CACHE: Dict[str, Any] = {}
_INTERNAL_OPENAI_KWARGS = {
    "hashing_kv",
    "keyword_extraction",
    "enable_cot",
    "max_tokens",
}


@dataclass
class LocalRagSettings:
    tiktoken_cache_dir: str = DEFAULT_TIKTOKEN_CACHE_DIR
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH
    rerank_model_path: str = DEFAULT_RERANK_MODEL_PATH
    tokenizer_model_path: str = DEFAULT_TOKENIZER_MODEL_PATH
    vision_model_path: str = DEFAULT_VISION_MODEL_PATH

    working_dir_root: str = DEFAULT_WORKING_DIR_ROOT
    output_dir: str = DEFAULT_OUTPUT_DIR
    uploads_dir: str = DEFAULT_UPLOADS_DIR
    log_dir: str = DEFAULT_LOG_DIR

    vllm_api_base: str = DEFAULT_VLLM_API_BASE
    vllm_api_key: str = DEFAULT_VLLM_API_KEY
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    text_request_timeout_seconds: float = DEFAULT_TEXT_REQUEST_TIMEOUT_SECONDS
    vision_vllm_api_base: str = DEFAULT_VLLM_API_BASE
    vision_vllm_api_key: str = DEFAULT_VLLM_API_KEY
    vision_model_name: str = DEFAULT_LLM_MODEL_NAME
    vision_request_timeout_seconds: float = DEFAULT_VISION_REQUEST_TIMEOUT_SECONDS
    device: str = DEFAULT_DEVICE

    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    max_token_size: int = DEFAULT_MAX_TOKEN_SIZE

    temperature: float = DEFAULT_TEMPERATURE
    query_max_tokens: int = DEFAULT_QUERY_MAX_TOKENS
    ingest_max_tokens: int = DEFAULT_INGEST_MAX_TOKENS

    vlm_enable_json_schema: bool = DEFAULT_VLM_ENABLE_JSON_SCHEMA
    multimodal_item_timeout_seconds: float = DEFAULT_MULTIMODAL_ITEM_TIMEOUT_SECONDS
    multimodal_batch_watchdog_seconds: float = (
        DEFAULT_MULTIMODAL_BATCH_WATCHDOG_SECONDS
    )
    multimodal_cancel_grace_seconds: float = DEFAULT_MULTIMODAL_CANCEL_GRACE_SECONDS
    multimodal_enable_strict_fallback: bool = DEFAULT_MULTIMODAL_ENABLE_STRICT_FALLBACK
    image_token_estimate_method: str = DEFAULT_IMAGE_TOKEN_ESTIMATE_METHOD
    image_token_model_name_or_path: str = ""
    image_wrapper_tokens_per_image: int = DEFAULT_IMAGE_WRAPPER_TOKENS_PER_IMAGE

    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY
    chunk_token_size: int = DEFAULT_CHUNK_TOKEN_SIZE
    chunk_overlap_token_size: int = DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE
    serialize_ingest_by_workspace_id: bool = DEFAULT_SERIALIZE_INGEST_BY_WORKSPACE_ID
    mineru_vllm_gpu_memory_utilization: float = (
        DEFAULT_MINERU_VLLM_GPU_MEMORY_UTILIZATION
    )
    enable_resilience: bool = DEFAULT_ENABLE_RESILIENCE
    resilience_max_attempts: int = DEFAULT_RESILIENCE_MAX_ATTEMPTS
    ingest_retry_base_delay: float = DEFAULT_INGEST_RETRY_BASE_DELAY
    query_retry_base_delay: float = DEFAULT_QUERY_RETRY_BASE_DELAY
    resilience_max_delay: float = DEFAULT_RESILIENCE_MAX_DELAY
    ingest_breaker_failure_threshold: int = DEFAULT_INGEST_BREAKER_FAILURE_THRESHOLD
    query_breaker_failure_threshold: int = DEFAULT_QUERY_BREAKER_FAILURE_THRESHOLD
    breaker_reset_timeout_seconds: float = DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS
    enable_metrics_callback: bool = DEFAULT_ENABLE_METRICS_CALLBACK
    enable_callback_event_log: bool = DEFAULT_ENABLE_CALLBACK_EVENT_LOG
    enable_entity_disambiguation: bool = True
    enable_synonym_linking: bool = DEFAULT_ENABLE_SYNONYM_LINKING
    synonymy_threshold: float = DEFAULT_SYNONYMY_THRESHOLD
    synonymy_topk: int = DEFAULT_SYNONYMY_TOPK
    synonymy_min_entity_len: int = DEFAULT_SYNONYMY_MIN_ENTITY_LEN
    enable_multi_hop: bool = DEFAULT_ENABLE_MULTI_HOP
    multi_hop_depth: int = DEFAULT_MULTI_HOP_DEPTH
    ppr_damping: float = DEFAULT_PPR_DAMPING
    ppr_top_k: int = DEFAULT_PPR_TOP_K
    passage_node_weight: float = DEFAULT_PASSAGE_NODE_WEIGHT
    ppr_synonym_weight_mode: str = DEFAULT_PPR_SYNONYM_WEIGHT_MODE
    exclude_synonym_edges: bool | None = DEFAULT_EXCLUDE_SYNONYM_EDGES
    recognition_top_k: int = DEFAULT_RECOGNITION_TOP_K
    linking_top_k: int = DEFAULT_LINKING_TOP_K
    ppr_qa_top_k: int = DEFAULT_PPR_QA_TOP_K
    recognition_prompt_max_tokens: int = DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS
    recognition_prompt_output_max_tokens: int = (
        DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS
    )
    recognition_prompt_reserved_tokens: int = (
        DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS
    )
    enable_entity_surface_normalization: bool = (
        DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION
    )
    enable_keyword_case_normalization: bool = (
        DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION
    )
    entity_uppercase_allowlist: list[str] = field(
        default_factory=lambda: list(DEFAULT_ENTITY_UPPERCASE_ALLOWLIST)
    )
    strict_relation_endpoint_entity_match: bool = (
        DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH
    )
    qdrant_enable_sparse_bm25: bool = True
    qdrant_sparse_bm25_model: str = "Qdrant/bm25"

    @classmethod
    def from_env(cls) -> "LocalRagSettings":
        vllm_base = os.getenv("VLLM_API_BASE", DEFAULT_VLLM_API_BASE)
        vllm_key = os.getenv("VLLM_API_KEY", DEFAULT_VLLM_API_KEY)
        llm_name = os.getenv("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME)
        vision_model_path = os.getenv("VISION_MODEL_PATH", DEFAULT_VISION_MODEL_PATH)
        tokenizer_model_path = os.getenv("TOKENIZER_MODEL_PATH", vision_model_path)
        image_token_model_name_or_path = os.getenv(
            "RAGANYTHING_IMAGE_TOKEN_MODEL_PATH",
            os.getenv("IMAGE_TOKEN_MODEL_NAME_OR_PATH", vision_model_path),
        )

        return cls(
            tiktoken_cache_dir=os.getenv("TIKTOKEN_CACHE_DIR", DEFAULT_TIKTOKEN_CACHE_DIR),
            embedding_model_path=os.getenv("RAGANYTHING_EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL_PATH),
            rerank_model_path=os.getenv("RAGANYTHING_RERANK_MODEL_PATH", DEFAULT_RERANK_MODEL_PATH),
            tokenizer_model_path=tokenizer_model_path,
            vision_model_path=vision_model_path,
            log_dir=os.getenv("RAGANYTHING_LOG_DIR", DEFAULT_LOG_DIR),
            vllm_api_base=vllm_base,
            vllm_api_key=vllm_key,
            llm_model_name=llm_name,
            text_request_timeout_seconds=float(
                os.getenv(
                    "RAGANYTHING_TEXT_REQUEST_TIMEOUT_SECONDS",
                    str(DEFAULT_TEXT_REQUEST_TIMEOUT_SECONDS),
                )
            ),
            vision_vllm_api_base=os.getenv("VISION_VLLM_API_BASE", vllm_base),
            vision_vllm_api_key=os.getenv("VISION_VLLM_API_KEY", vllm_key),
            vision_model_name=os.getenv("VISION_MODEL_NAME", llm_name),
            vision_request_timeout_seconds=float(
                os.getenv(
                    "RAGANYTHING_VISION_REQUEST_TIMEOUT_SECONDS",
                    str(DEFAULT_VISION_REQUEST_TIMEOUT_SECONDS),
                )
            ),
            device=os.getenv("RAGANYTHING_DEVICE", DEFAULT_DEVICE),
            working_dir_root=os.getenv("RAGANYTHING_WORKDIR_ROOT", DEFAULT_WORKING_DIR_ROOT),
            output_dir=os.getenv("RAGANYTHING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
            uploads_dir=os.getenv("RAGANYTHING_UPLOADS_DIR", DEFAULT_UPLOADS_DIR),
            embedding_dim=int(os.getenv("RAGANYTHING_EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM))),
            max_token_size=int(os.getenv("RAGANYTHING_MAX_TOKEN_SIZE", str(DEFAULT_MAX_TOKEN_SIZE))),
            temperature=float(os.getenv("RAGANYTHING_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            query_max_tokens=int(
                os.getenv("RAGANYTHING_QUERY_MAX_TOKENS", str(DEFAULT_QUERY_MAX_TOKENS))
            ),
            ingest_max_tokens=int(
                os.getenv("RAGANYTHING_INGEST_MAX_TOKENS", str(DEFAULT_INGEST_MAX_TOKENS))
            ),
            vlm_enable_json_schema=os.getenv(
                "RAGANYTHING_VLM_ENABLE_JSON_SCHEMA", str(DEFAULT_VLM_ENABLE_JSON_SCHEMA)
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            multimodal_item_timeout_seconds=float(
                os.getenv(
                    "RAGANYTHING_MULTIMODAL_ITEM_TIMEOUT_SECONDS",
                    str(DEFAULT_MULTIMODAL_ITEM_TIMEOUT_SECONDS),
                )
            ),
            multimodal_batch_watchdog_seconds=float(
                os.getenv(
                    "RAGANYTHING_MULTIMODAL_BATCH_WATCHDOG_SECONDS",
                    str(DEFAULT_MULTIMODAL_BATCH_WATCHDOG_SECONDS),
                )
            ),
            multimodal_cancel_grace_seconds=float(
                os.getenv(
                    "RAGANYTHING_MULTIMODAL_CANCEL_GRACE_SECONDS",
                    str(DEFAULT_MULTIMODAL_CANCEL_GRACE_SECONDS),
                )
            ),
            multimodal_enable_strict_fallback=os.getenv(
                "RAGANYTHING_MULTIMODAL_ENABLE_STRICT_FALLBACK",
                str(DEFAULT_MULTIMODAL_ENABLE_STRICT_FALLBACK),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            image_token_estimate_method=os.getenv(
                "RAGANYTHING_IMAGE_TOKEN_ESTIMATE_METHOD",
                DEFAULT_IMAGE_TOKEN_ESTIMATE_METHOD,
            ),
            image_token_model_name_or_path=image_token_model_name_or_path,
            image_wrapper_tokens_per_image=int(
                os.getenv(
                    "RAGANYTHING_IMAGE_WRAPPER_TOKENS_PER_IMAGE",
                    str(DEFAULT_IMAGE_WRAPPER_TOKENS_PER_IMAGE),
                )
            ),
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", DEFAULT_CHUNKING_STRATEGY),
            chunk_token_size=int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_TOKEN_SIZE))),
            chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP_SIZE", str(DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE))),
            serialize_ingest_by_workspace_id=os.getenv(
                "RAGANYTHING_SERIALIZE_INGEST_BY_WORKSPACE_ID",
                str(DEFAULT_SERIALIZE_INGEST_BY_WORKSPACE_ID),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            mineru_vllm_gpu_memory_utilization=float(
                os.getenv(
                    "MINERU_VLLM_GPU_MEMORY_UTILIZATION",
                    str(DEFAULT_MINERU_VLLM_GPU_MEMORY_UTILIZATION),
                )
            ),
            enable_resilience=os.getenv(
                "RAGANYTHING_ENABLE_RESILIENCE",
                str(DEFAULT_ENABLE_RESILIENCE),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            resilience_max_attempts=int(
                os.getenv(
                    "RAGANYTHING_RESILIENCE_MAX_ATTEMPTS",
                    str(DEFAULT_RESILIENCE_MAX_ATTEMPTS),
                )
            ),
            ingest_retry_base_delay=float(
                os.getenv(
                    "RAGANYTHING_INGEST_RETRY_BASE_DELAY",
                    str(DEFAULT_INGEST_RETRY_BASE_DELAY),
                )
            ),
            query_retry_base_delay=float(
                os.getenv(
                    "RAGANYTHING_QUERY_RETRY_BASE_DELAY",
                    str(DEFAULT_QUERY_RETRY_BASE_DELAY),
                )
            ),
            resilience_max_delay=float(
                os.getenv(
                    "RAGANYTHING_RESILIENCE_MAX_DELAY",
                    str(DEFAULT_RESILIENCE_MAX_DELAY),
                )
            ),
            ingest_breaker_failure_threshold=int(
                os.getenv(
                    "RAGANYTHING_INGEST_BREAKER_FAILURE_THRESHOLD",
                    str(DEFAULT_INGEST_BREAKER_FAILURE_THRESHOLD),
                )
            ),
            query_breaker_failure_threshold=int(
                os.getenv(
                    "RAGANYTHING_QUERY_BREAKER_FAILURE_THRESHOLD",
                    str(DEFAULT_QUERY_BREAKER_FAILURE_THRESHOLD),
                )
            ),
            breaker_reset_timeout_seconds=float(
                os.getenv(
                    "RAGANYTHING_BREAKER_RESET_TIMEOUT_SECONDS",
                    str(DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS),
                )
            ),
            enable_metrics_callback=os.getenv(
                "RAGANYTHING_ENABLE_METRICS_CALLBACK",
                str(DEFAULT_ENABLE_METRICS_CALLBACK),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            enable_callback_event_log=os.getenv(
                "RAGANYTHING_ENABLE_CALLBACK_EVENT_LOG",
                str(DEFAULT_ENABLE_CALLBACK_EVENT_LOG),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            enable_entity_disambiguation=os.getenv(
                "RAGANYTHING_ENABLE_ENTITY_DISAMBIGUATION", "true"
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            enable_synonym_linking=os.getenv(
                "RAGANYTHING_ENABLE_SYNONYM_LINKING",
                str(DEFAULT_ENABLE_SYNONYM_LINKING),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            synonymy_threshold=float(
                os.getenv(
                    "RAGANYTHING_SYNONYMY_THRESHOLD",
                    str(DEFAULT_SYNONYMY_THRESHOLD),
                )
            ),
            synonymy_topk=int(
                os.getenv(
                    "RAGANYTHING_SYNONYMY_TOPK",
                    str(DEFAULT_SYNONYMY_TOPK),
                )
            ),
            synonymy_min_entity_len=int(
                os.getenv(
                    "RAGANYTHING_SYNONYMY_MIN_ENTITY_LEN",
                    str(DEFAULT_SYNONYMY_MIN_ENTITY_LEN),
                )
            ),
            enable_multi_hop=os.getenv(
                "RAGANYTHING_ENABLE_MULTI_HOP",
                str(DEFAULT_ENABLE_MULTI_HOP),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            multi_hop_depth=int(
                os.getenv(
                    "RAGANYTHING_MULTI_HOP_DEPTH",
                    str(DEFAULT_MULTI_HOP_DEPTH),
                )
            ),
            ppr_damping=float(
                os.getenv(
                    "RAGANYTHING_PPR_DAMPING",
                    str(DEFAULT_PPR_DAMPING),
                )
            ),
            ppr_top_k=int(
                os.getenv(
                    "RAGANYTHING_PPR_TOP_K",
                    str(DEFAULT_PPR_TOP_K),
                )
            ),
            passage_node_weight=float(
                os.getenv(
                    "RAGANYTHING_PASSAGE_NODE_WEIGHT",
                    str(DEFAULT_PASSAGE_NODE_WEIGHT),
                )
            ),
            ppr_synonym_weight_mode=_normalize_ppr_synonym_weight_mode(
                os.getenv(
                    "RAGANYTHING_PPR_SYNONYM_WEIGHT_MODE",
                    DEFAULT_PPR_SYNONYM_WEIGHT_MODE,
                )
            ),
            exclude_synonym_edges=_parse_optional_bool_env(
                "RAGANYTHING_EXCLUDE_SYNONYM_EDGES",
                DEFAULT_EXCLUDE_SYNONYM_EDGES,
            ),
            recognition_top_k=int(
                os.getenv(
                    "RAGANYTHING_RECOGNITION_TOP_K",
                    str(DEFAULT_RECOGNITION_TOP_K),
                )
            ),
            linking_top_k=int(
                os.getenv(
                    "RAGANYTHING_LINKING_TOP_K",
                    str(DEFAULT_LINKING_TOP_K),
                )
            ),
            ppr_qa_top_k=int(
                os.getenv(
                    "RAGANYTHING_PPR_QA_TOP_K",
                    str(DEFAULT_PPR_QA_TOP_K),
                )
            ),
            recognition_prompt_max_tokens=int(
                os.getenv(
                    "RAGANYTHING_RECOGNITION_PROMPT_MAX_TOKENS",
                    str(DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS),
                )
            ),
            recognition_prompt_output_max_tokens=int(
                os.getenv(
                    "RAGANYTHING_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS",
                    str(DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS),
                )
            ),
            recognition_prompt_reserved_tokens=int(
                os.getenv(
                    "RAGANYTHING_RECOGNITION_PROMPT_RESERVED_TOKENS",
                    str(DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS),
                )
            ),
            enable_entity_surface_normalization=os.getenv(
                "RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION",
                str(DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            enable_keyword_case_normalization=os.getenv(
                "RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION",
                str(DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            entity_uppercase_allowlist=_parse_uppercase_allowlist(
                os.getenv("RAGANYTHING_ENTITY_UPPERCASE_ALLOWLIST")
            ),
            strict_relation_endpoint_entity_match=os.getenv(
                "RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH",
                str(DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH),
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            qdrant_enable_sparse_bm25=os.getenv(
                "QDRANT_ENABLE_SPARSE_BM25", "true"
            ).lower()
            in {"1", "true", "yes", "y", "on"},
            qdrant_sparse_bm25_model=os.getenv(
                "QDRANT_SPARSE_BM25_MODEL", "Qdrant/bm25"
            ),
        )


def configure_logging(settings: LocalRagSettings) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = Path(settings.log_dir) / f"run_{timestamp}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "detailed",
                    "filename": str(log_file_path),
                    "maxBytes": DEFAULT_LOG_MAX_BYTES,
                    "backupCount": DEFAULT_LOG_BACKUP_COUNT,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "": {"handlers": ["console", "file"], "level": "INFO"},
            },
        }
    )
    return logging.getLogger(__name__)


def _strip_internal_openai_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    # 过滤 LightRAG 内部参数，避免透传到 OpenAI 兼容接口。
    return {k: v for k, v in kwargs.items() if k not in _INTERNAL_OPENAI_KWARGS}


def _model_cache_key(settings: LocalRagSettings) -> str:
    return f"{settings.embedding_model_path}|{settings.rerank_model_path}|{settings.device}"


def _resolve_ingest_serialize_by_workspace_id(
    default_serialize_by_workspace_id: bool,
    serialize_by_workspace_id: Optional[bool],
    chunking_strategy: Optional[str],
    default_chunking_strategy: str,
) -> tuple[bool, bool]:
    effective_serialize = (
        default_serialize_by_workspace_id
        if serialize_by_workspace_id is None
        else bool(serialize_by_workspace_id)
    )
    forced_to_serialize = (
        not effective_serialize
        and bool(chunking_strategy)
        and chunking_strategy != default_chunking_strategy
    )
    if forced_to_serialize:
        return True, True
    return effective_serialize, False


def load_models(settings: LocalRagSettings) -> tuple[SentenceTransformer, CrossEncoder]:
    key = _model_cache_key(settings)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    st_model = SentenceTransformer(
        settings.embedding_model_path,
        trust_remote_code=True,
        device=settings.device,
    )
    reranker_model = CrossEncoder(
        settings.rerank_model_path,
        device=settings.device,
        trust_remote_code=True,
    )
    _MODEL_CACHE[key] = (st_model, reranker_model)
    return st_model, reranker_model


class _HFTokenizerAdapter:
    def __init__(self, hf_tokenizer: Any):
        self._hf_tokenizer = hf_tokenizer

    def encode(self, content: str) -> list[int]:
        return self._hf_tokenizer.encode(content, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._hf_tokenizer.decode(tokens, skip_special_tokens=False)


def build_lightrag_tokenizer(
    model_name_or_path: str, logger: logging.Logger
) -> Tokenizer:
    model_name_or_path = (model_name_or_path or "").strip()
    if not model_name_or_path:
        raise ValueError(
            "Tokenizer model path is empty. Set TOKENIZER_MODEL_PATH."
        )

    try:
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Qwen tokenizer from '{model_name_or_path}'. "
            "Set TOKENIZER_MODEL_PATH to your local Qwen model directory."
        ) from exc

    logger.info("LightRAG tokenizer initialized from: %s", model_name_or_path)
    return Tokenizer(
        model_name=model_name_or_path,
        tokenizer=_HFTokenizerAdapter(hf_tokenizer),
    )


def build_embedding_func(
    settings: LocalRagSettings, st_model: SentenceTransformer
) -> EmbeddingFunc:
    raw_model_path = str(settings.embedding_model_path or "").strip()
    normalized_model_path = raw_model_path.rstrip("/\\")
    model_name = Path(normalized_model_path).name if normalized_model_path else ""
    if not model_name:
        model_name = "embedding_model"

    async def _compute_embedding(texts: list[str]) -> np.ndarray:
        return st_model.encode(texts, normalize_embeddings=True)

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.max_token_size,
        model_name=model_name,
        func=_compute_embedding,
    )


def build_rerank_func(reranker_model: CrossEncoder, logger: logging.Logger):
    async def rerank_func(query: str, documents: list[str], top_n: int) -> list[dict]:
        if not documents:
            return []
        try:
            pairs = [[query, doc] for doc in documents]
            scores = reranker_model.predict(pairs)
            results = [
                {"index": i, "relevance_score": float(score)}
                for i, score in enumerate(scores)
            ]
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:top_n]
        except Exception as exc:
            logger.error(f"Rerank Error: {exc}")
            return []

    return rerank_func


def _build_modal_analysis_response_format() -> dict[str, Any]:
    # 统一的结构化输出 schema（保持兼容优先）。
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "modal_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "detailed_description": {"type": "string"},
                    "entity_info": {
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string"},
                            "entity_type": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": [
                            "entity_name",
                            "entity_type",
                            "summary",
                        ],
                        "additionalProperties": True,
                    },
                },
                "required": ["detailed_description", "entity_info"],
                "additionalProperties": True,
            },
        },
    }


def _infer_ingest_task_type(
    prompt: Any,
    system_prompt: Any,
    image_data: Any,
    messages: Any,
) -> str:
    if messages is not None:
        return "query"
    if image_data is not None:
        return "image"

    text = f"{str(system_prompt or '')}\n{str(prompt or '')}".lower()
    is_structured_ingest_prompt = (
        "provide a json response with the following structure" in text
        or "response must be a valid json object" in text
        or "entity_info" in text
    )
    if not is_structured_ingest_prompt:
        return "query"

    if "table" in text:
        return "table"
    if "equation" in text or "latex" in text:
        return "equation"
    if "image" in text:
        return "image"
    return "generic"


def _should_use_ingest_schema(
    settings: LocalRagSettings,
    prompt: Any,
    system_prompt: Any,
    image_data: Any,
    messages: Any,
) -> tuple[bool, str, bool, bool]:
    task_type = _infer_ingest_task_type(prompt, system_prompt, image_data, messages)
    prompt_has_json = "json" in str(prompt or "").lower()
    is_ingest_task = task_type in {"image", "table", "equation", "generic"}
    use_schema = settings.vlm_enable_json_schema and is_ingest_task and prompt_has_json
    return use_schema, task_type, prompt_has_json, is_ingest_task


def _is_entity_extraction_call(system_prompt: Any, prompt: Any) -> bool:
    # LightRAG 索引抽取链路：extract + glean 两类 user prompt 共用同一 system role。
    sys_text = str(system_prompt or "").lower()
    user_text = str(prompt or "").lower()
    return (
        "knowledge graph specialist" in sys_text
        and "extracting entities and relationships" in sys_text
    ) or (
        "extract entities and relationships from the input text" in user_text
        or "based on the last extraction task" in user_text
    )


def _is_transient_llm_exception(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timed out",
            "timeout",
            "connection",
            "rate limit",
            "temporarily unavailable",
            "broken pipe",
            "429",
            "503",
        )
    )


def _resolve_llm_retry_policy(
    settings: LocalRagSettings, *, ingest_path: bool
) -> tuple[int, float, float]:
    if not settings.enable_resilience:
        return 1, 0.0, 0.0
    attempts = max(1, min(3, int(settings.resilience_max_attempts)))
    base_delay = (
        float(settings.ingest_retry_base_delay)
        if ingest_path
        else float(settings.query_retry_base_delay)
    )
    base_delay = max(0.0, base_delay)
    max_delay = max(base_delay, float(settings.resilience_max_delay))
    return attempts, base_delay, max_delay


async def _run_llm_with_transient_retry(
    *,
    logger: logging.Logger,
    operation_name: str,
    call: Callable[[], Awaitable[Any]],
    max_attempts: int,
    base_delay: float,
    max_delay: float,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except Exception as exc:
            retryable = _is_transient_llm_exception(exc)
            if attempt >= max_attempts or not retryable:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "LLM transient retry: op=%s attempt=%d/%d delay=%.1fs reason=%s",
                operation_name,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            if delay > 0:
                await asyncio.sleep(delay)


def build_llm_model_func(
    settings: LocalRagSettings,
    client: AsyncOpenAI,
    logger: logging.Logger,
    model_name: str,
):
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        history_messages = history_messages or []
        keyword_extraction = bool(kwargs.pop("keyword_extraction", False))
        is_streaming = bool(kwargs.pop("stream", False))
        max_tokens_override_raw = kwargs.pop("max_tokens", None)
        cleaned_kwargs = _strip_internal_openai_kwargs(kwargs)
        is_ingest_call = _is_entity_extraction_call(system_prompt, prompt)
        if is_ingest_call:
            max_tokens = settings.ingest_max_tokens
        else:
            max_tokens = settings.query_max_tokens
        if max_tokens_override_raw is not None:
            try:
                parsed_override = int(max_tokens_override_raw)
                if parsed_override > 0:
                    max_tokens = parsed_override
            except Exception:
                logger.warning(
                    "Ignoring invalid max_tokens override in llm_model_func: %r",
                    max_tokens_override_raw,
                )
        retry_attempts, retry_base_delay, retry_max_delay = _resolve_llm_retry_policy(
            settings,
            ingest_path=is_ingest_call,
        )
        if keyword_extraction:
            cleaned_kwargs["response_format"] = GPTKeywordExtractionFormat
        messages = []
        repacked = repack_query_messages(
            system_prompt,
            prompt,
            has_images=False,
        )
        if repacked:
            final_system, final_user = repacked
            if final_system:
                messages.append({"role": "system", "content": final_system})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": final_user})
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

        # Streaming path: return async generator so LightRAG wraps it as response_iterator
        if is_streaming:
            async def _token_stream():
                try:
                    stream = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=settings.temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **cleaned_kwargs,
                    )
                    async for chunk in stream:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                yield delta.content
                except Exception as exc:
                    logger.error(f"LLM streaming error: {exc}")
                    if is_ingest_call:
                        raise
                    raise RuntimeError(f"LLM streaming query failed: {exc}") from exc
            return _token_stream()

        # Non-streaming path (existing behaviour)
        try:
            async def _request_once():
                if "response_format" in cleaned_kwargs:
                    return await client.chat.completions.parse(
                        model=model_name,
                        messages=messages,
                        temperature=settings.temperature,
                        max_tokens=max_tokens,
                        **cleaned_kwargs,
                    )
                return await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=max_tokens,
                    **cleaned_kwargs,
                )

            response = await _run_llm_with_transient_retry(
                logger=logger,
                operation_name="ingest" if is_ingest_call else "query",
                call=_request_once,
                max_attempts=retry_attempts,
                base_delay=retry_base_delay,
                max_delay=retry_max_delay,
            )
            message = response.choices[0].message
            if hasattr(message, "parsed") and message.parsed is not None:
                parsed = message.parsed
                if hasattr(parsed, "model_dump_json"):
                    return parsed.model_dump_json()
                return json.dumps(parsed, ensure_ascii=False)
            return getattr(message, "content", "") or ""
        except Exception as exc:
            logger.error(f"LLM Error: {exc}")
            if is_ingest_call:
                raise
            raise RuntimeError(f"LLM query failed: {exc}") from exc

    return llm_model_func


def build_vision_model_func(
    settings: LocalRagSettings,
    client: AsyncOpenAI,
    logger: logging.Logger,
    model_name: str,
):
    llm_fallback = build_llm_model_func(settings, client, logger, model_name)
    schema_stats = {"total": 0, "success": 0, "fallback": 0}
    item_timeout = max(1.0, float(settings.multimodal_item_timeout_seconds))
    query_retry_attempts, query_retry_base_delay, query_retry_max_delay = (
        _resolve_llm_retry_policy(settings, ingest_path=False)
    )
    ingest_retry_attempts, ingest_retry_base_delay, ingest_retry_max_delay = (
        _resolve_llm_retry_policy(settings, ingest_path=True)
    )

    async def _vision_once(
        *,
        messages_payload: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
        task_type: str,
        strict_mode: str,
        attempt: int,
    ):
        try:
            logger.info(
                "Vision ingest request: task_type=%s strict_mode=%s timeout=%.1fs attempt=%d",
                task_type,
                strict_mode,
                item_timeout,
                attempt,
            )

            async def _request_once():
                return await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_name,
                        messages=messages_payload,
                        temperature=settings.temperature,
                        max_tokens=settings.ingest_max_tokens,
                        **request_kwargs,
                    ),
                    timeout=item_timeout,
                )

            return await _run_llm_with_transient_retry(
                logger=logger,
                operation_name=f"vision_ingest:{task_type}",
                call=_request_once,
                max_attempts=ingest_retry_attempts,
                base_delay=ingest_retry_base_delay,
                max_delay=ingest_retry_max_delay,
            )
        except Exception as exc:
            logger.warning(
                "Vision ingest request failed: task_type=%s strict_mode=%s timeout=%.1fs attempt=%d error_class=%s error=%s",
                task_type,
                strict_mode,
                item_timeout,
                attempt,
                type(exc).__name__,
                exc,
            )
            raise

    async def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        history_messages = history_messages or []
        cleaned_kwargs = _strip_internal_openai_kwargs(kwargs)

        # 问答阶段：直接透传 query.py 构建好的多模态消息给 VLM
        # （图片数量已在检索阶段由 multimodal_top_k 控制，
        #   路径实体已在索引阶段被过滤，无需查询时二次处理）
        if messages:
            try:
                async def _query_once():
                    return await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=settings.temperature,
                        max_tokens=settings.query_max_tokens,
                        **cleaned_kwargs,
                    )

                response = await _run_llm_with_transient_retry(
                    logger=logger,
                    operation_name="vision_query",
                    call=_query_once,
                    max_attempts=query_retry_attempts,
                    base_delay=query_retry_base_delay,
                    max_delay=query_retry_max_delay,
                )
                return response.choices[0].message.content
            except Exception as exc:
                logger.error(f"Vision LLM Error (Query): {exc}")
                raise

        use_schema, task_type, prompt_has_json, is_ingest_task = _should_use_ingest_schema(
            settings,
            prompt,
            system_prompt,
            image_data,
            messages,
        )

        # 入库阶段（image / table / equation / generic）：统一走 ingest 请求路径。
        if is_ingest_task:
            import base64

            base64_image = None
            if image_data is not None:
                if isinstance(image_data, bytes):
                    base64_image = base64.b64encode(image_data).decode("utf-8")
                else:
                    base64_image = str(image_data)

            if base64_image:
                content_payload = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
                msgs = [{"role": "user", "content": content_payload}]
            else:
                msgs = [{"role": "user", "content": str(prompt)}]

            if system_prompt:
                msgs.insert(0, {"role": "system", "content": system_prompt})

            request_kwargs = dict(cleaned_kwargs)
            logger.info(
                "Vision ingest schema check: task_type=%s, enabled=%s, prompt_has_json=%s, using_schema=%s",
                task_type,
                settings.vlm_enable_json_schema,
                prompt_has_json,
                use_schema,
            )
            if use_schema:
                schema_stats["total"] += 1
                request_kwargs["response_format"] = _build_modal_analysis_response_format()
                logger.info(
                    "Vision ingest calling with response_format=json_schema (strict=%s)",
                    request_kwargs["response_format"]["json_schema"].get("strict"),
                )

            try:
                response = await _vision_once(
                    messages_payload=msgs,
                    request_kwargs=request_kwargs,
                    task_type=task_type,
                    strict_mode="strict=true" if "response_format" in request_kwargs else "none",
                    attempt=1,
                )
                if use_schema:
                    schema_stats["success"] += 1
                    success_rate = schema_stats["success"] / max(schema_stats["total"], 1)
                    fallback_rate = schema_stats["fallback"] / max(schema_stats["total"], 1)
                    logger.info(
                        "Vision ingest json_schema call succeeded: schema_success_rate=%.4f, schema_fallback_rate=%.4f",
                        success_rate,
                        fallback_rate,
                    )
                return response.choices[0].message.content
            except Exception as exc:
                can_fallback = (
                    "response_format" in request_kwargs
                    and settings.multimodal_enable_strict_fallback
                )
                if can_fallback:
                    schema_stats["fallback"] += 1
                    logger.warning(
                        "Vision ingest json_schema call failed, retrying without schema: %s",
                        exc,
                    )
                    request_kwargs.pop("response_format", None)
                    try:
                        response = await _vision_once(
                            messages_payload=msgs,
                            request_kwargs=request_kwargs,
                            task_type=task_type,
                            strict_mode="strict=false",
                            attempt=2,
                        )
                        success_rate = schema_stats["success"] / max(schema_stats["total"], 1)
                        fallback_rate = schema_stats["fallback"] / max(schema_stats["total"], 1)
                        logger.info(
                            "Vision ingest fallback retry succeeded: schema_success_rate=%.4f, schema_fallback_rate=%.4f",
                            success_rate,
                            fallback_rate,
                        )
                        return response.choices[0].message.content
                    except Exception as retry_exc:
                        logger.warning("Vision ingest fallback retry failed: %s", retry_exc)
                        exc = retry_exc

                logger.error("Vision LLM Error (Ingest): %s", exc)
                raise

        return await llm_fallback(
            prompt,
            system_prompt,
            history_messages,
            **kwargs,
        )

    return vision_model_func


def _resolve_text_endpoint(settings: LocalRagSettings) -> tuple[str, str, str]:
    # 文本端为空时，回退到 vision 端配置。
    vision_base = (settings.vision_vllm_api_base or "").strip()
    vision_key = (settings.vision_vllm_api_key or "").strip()
    vision_model = (settings.vision_model_name or "").strip()

    text_base = (settings.vllm_api_base or "").strip() or vision_base
    text_key = (settings.vllm_api_key or "").strip() or vision_key
    text_model = (settings.llm_model_name or "").strip() or vision_model
    return text_base, text_key, text_model


def _resolve_vision_endpoint(settings: LocalRagSettings) -> tuple[str, str, str]:
    # vision 端为空时，回退到文本端配置。
    vision_base = (settings.vision_vllm_api_base or "").strip() or (
        settings.vllm_api_base or ""
    ).strip()
    vision_key = (settings.vision_vllm_api_key or "").strip() or (
        settings.vllm_api_key or ""
    ).strip()
    vision_model = (settings.vision_model_name or "").strip() or (
        settings.llm_model_name or ""
    ).strip()
    return vision_base, vision_key, vision_model


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    if not normalized:
        return False
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _collect_loopback_hosts_from_base_urls(*base_urls: str) -> list[str]:
    hosts: set[str] = set()
    for base_url in base_urls:
        try:
            parsed = urlparse((base_url or "").strip())
        except Exception:
            continue
        host = (parsed.hostname or "").strip()
        if _is_loopback_host(host):
            hosts.update({"localhost", "127.0.0.1", "::1"})
    return sorted(hosts)


def _merge_no_proxy_env_hosts(hosts: list[str]) -> bool:
    if not hosts:
        return False
    changed = False
    for env_key in ("NO_PROXY", "no_proxy"):
        raw = os.getenv(env_key, "")
        parts = [token.strip() for token in raw.split(",") if token.strip()]
        normalized = {token.lower() for token in parts}
        appended = False
        for host in hosts:
            if host.lower() in normalized:
                continue
            parts.append(host)
            normalized.add(host.lower())
            appended = True
        if appended:
            os.environ[env_key] = ",".join(parts)
            changed = True
    return changed


def _maybe_enable_internvl2_prompts(settings: LocalRagSettings, logger: logging.Logger) -> None:
    # 命中 InternVL2 模型名时启用对应 prompt 覆盖。
    text_model = (settings.llm_model_name or "").lower()
    vision_model = (settings.vision_model_name or "").lower()
    if "internvl2" not in text_model and "internvl2" not in vision_model:
        return

    try:
        from raganything.prompt import PROMPTS
        from raganything.prompt_internvl2 import PROMPTS as PROMPTS_INTERNVL2

        PROMPTS.update(PROMPTS_INTERNVL2)
        logger.info("InternVL2 prompt overrides enabled.")
    except Exception as exc:
        logger.warning(f"Failed to enable InternVL2 prompt overrides: {exc}")

def _safe_workspace_id(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    if cleaned:
        return cleaned
    return hashlib.md5(name.encode("utf-8")).hexdigest()


def _normalize_qwen_image_token_method(method: str | None) -> str:
    normalized = (method or "").strip().lower()
    if not normalized:
        normalized = DEFAULT_IMAGE_TOKEN_ESTIMATE_METHOD
    if normalized == "qwen_vl":
        return normalized
    raise ValueError(
        "Only Qwen-VL image token estimation is supported in this project. "
        "Set RAGANYTHING_IMAGE_TOKEN_ESTIMATE_METHOD=qwen_vl."
    )


def _parse_optional_bool_env(name: str, default: bool | None = None) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_ppr_synonym_weight_mode(mode: str | None) -> str:
    normalized = (mode or "").strip().lower()
    if normalized in {"raw", "plus_one"}:
        return normalized
    return DEFAULT_PPR_SYNONYM_WEIGHT_MODE


def _parse_uppercase_allowlist(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return list(DEFAULT_ENTITY_UPPERCASE_ALLOWLIST)

    stripped = raw_value.strip()
    if not stripped:
        return []

    parsed_items: list[str] = []
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                parsed_items = [str(item) for item in parsed]
            else:
                parsed_items = [stripped]
        except json.JSONDecodeError:
            parsed_items = [item.strip() for item in stripped.split(",")]
    else:
        parsed_items = [item.strip() for item in stripped.split(",")]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in parsed_items:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


class LocalRagService:
    def __init__(self, settings: Optional[LocalRagSettings] = None):
        # 初始化双客户端：text/vision 可同端口也可分离部署。
        self.settings = settings or LocalRagSettings.from_env()
        self.logger = configure_logging(self.settings)
        self.settings.image_token_estimate_method = _normalize_qwen_image_token_method(
            self.settings.image_token_estimate_method
        )
        self.settings.ppr_synonym_weight_mode = _normalize_ppr_synonym_weight_mode(
            self.settings.ppr_synonym_weight_mode
        )
        normalized_allowlist: list[str] = []
        seen_allowlist: set[str] = set()
        for item in (self.settings.entity_uppercase_allowlist or []):
            cleaned = str(item).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_allowlist:
                continue
            seen_allowlist.add(key)
            normalized_allowlist.append(cleaned)
        self.settings.entity_uppercase_allowlist = normalized_allowlist
        if not self.settings.image_token_model_name_or_path:
            self.settings.image_token_model_name_or_path = (
                self.settings.vision_model_path
            )
        if not self.settings.tokenizer_model_path:
            self.settings.tokenizer_model_path = (
                self.settings.image_token_model_name_or_path
                or self.settings.vision_model_path
            )

        os.environ["IMAGE_TOKEN_ESTIMATE_METHOD"] = (
            self.settings.image_token_estimate_method
        )
        os.environ["IMAGE_TOKEN_MODEL_NAME_OR_PATH"] = (
            self.settings.image_token_model_name_or_path
        )
        os.environ["IMAGE_WRAPPER_TOKENS_PER_IMAGE"] = str(
            self.settings.image_wrapper_tokens_per_image
        )
        os.environ["MINERU_VLLM_GPU_MEMORY_UTILIZATION"] = str(
            self.settings.mineru_vllm_gpu_memory_utilization
        )
        os.environ["QDRANT_ENABLE_SPARSE_BM25"] = (
            "true" if self.settings.qdrant_enable_sparse_bm25 else "false"
        )
        os.environ["QDRANT_SPARSE_BM25_MODEL"] = (
            self.settings.qdrant_sparse_bm25_model
        )
        self.logger.info(
            "Qdrant sparse indexing configured: enabled=%s, model=%s",
            self.settings.qdrant_enable_sparse_bm25,
            self.settings.qdrant_sparse_bm25_model,
        )
        self.lightrag_tokenizer = build_lightrag_tokenizer(
            self.settings.tokenizer_model_path,
            self.logger,
        )
        _maybe_enable_internvl2_prompts(self.settings, self.logger)

        text_base, text_key, text_model = _resolve_text_endpoint(self.settings)
        vision_base, vision_key, vision_model = _resolve_vision_endpoint(self.settings)
        local_proxy_bypass_hosts = _collect_loopback_hosts_from_base_urls(
            text_base, vision_base
        )
        if local_proxy_bypass_hosts:
            env_changed = _merge_no_proxy_env_hosts(local_proxy_bypass_hosts)
            self.logger.info(
                "Loopback LLM endpoint detected; %s NO_PROXY hosts: %s",
                "updated" if env_changed else "using existing",
                ",".join(local_proxy_bypass_hosts),
            )
        self.logger.info(
            "Model endpoints resolved: text_model=%s, vision_model=%s",
            text_model,
            vision_model,
        )
        self.logger.info(
            "Token caps configured: query_max_tokens=%s, ingest_max_tokens=%s, vlm_enable_json_schema=%s",
            self.settings.query_max_tokens,
            self.settings.ingest_max_tokens,
            self.settings.vlm_enable_json_schema,
        )
        self.logger.info(
            "Recognition caps configured: top_k=%s, prompt_max=%s, output_max=%s, reserved=%s",
            self.settings.recognition_top_k,
            self.settings.recognition_prompt_max_tokens,
            self.settings.recognition_prompt_output_max_tokens,
            self.settings.recognition_prompt_reserved_tokens,
        )
        self.logger.info(
            "Query synonym-edge filter default: exclude_synonym_edges=%s (None=auto: PPR=False, non-PPR=True)",
            self.settings.exclude_synonym_edges,
        )
        self.logger.info(
            "Text request timeout configured: %.1fs",
            self.settings.text_request_timeout_seconds,
        )
        self.logger.info(
            "Vision request timeout configured: %.1fs; multimodal item timeout: %.1fs",
            self.settings.vision_request_timeout_seconds,
            self.settings.multimodal_item_timeout_seconds,
        )
        self.logger.info(
            "Image token estimate configured: method=qwen_vl, model_path=%s, wrapper_tokens=%s",
            self.settings.image_token_model_name_or_path,
            self.settings.image_wrapper_tokens_per_image,
        )

        self.text_model_name = text_model
        text_request_timeout = max(
            1.0, float(self.settings.text_request_timeout_seconds)
        )
        self.text_client = AsyncOpenAI(
            api_key=text_key,
            base_url=text_base,
            timeout=text_request_timeout,
        )
        vision_request_timeout = max(
            1.0, float(self.settings.vision_request_timeout_seconds)
        )
        self.vision_client = AsyncOpenAI(
            api_key=vision_key,
            base_url=vision_base,
            timeout=vision_request_timeout,
        )
        self._rag_instances: Dict[str, RAGAnything] = {}
        self._init_lock = asyncio.Lock()
        self._registered_callbacks: list[ProcessingCallback] = []
        # Per-workspace-id locks: serialise ingest for the same workspace so that the
        # temporary chunking_func swap never races with a concurrent ingest.
        self._ingest_locks: Dict[str, asyncio.Lock] = {}
        self._workspace_warmup_locks: Dict[str, asyncio.Lock] = {}
        self._warmed_workspaces: set[str] = set()

        st_model, reranker_model = load_models(self.settings)
        self.embedding_func = build_embedding_func(self.settings, st_model)
        self.rerank_func = build_rerank_func(reranker_model, self.logger)
        self.llm_model_func = build_llm_model_func(
            self.settings,
            self.text_client,
            self.logger,
            text_model,
        )
        self.vision_model_func = build_vision_model_func(
            self.settings,
            self.vision_client,
            self.logger,
            vision_model,
        )
        self._safe_ingest_call, self._safe_query_call = self._build_resilience_wrappers()
        if self.settings.enable_metrics_callback:
            self.register_callback(MetricsCallback())

    def _on_resilience_retry(
        self,
        op_name: str,
        exc: BaseException,
        attempt: int,
        delay: float,
    ) -> None:
        self.logger.warning(
            "Resilience retry (%s): attempt=%d delay=%.1fs reason=%s",
            op_name,
            attempt,
            delay,
            exc,
        )

    def _build_resilience_wrappers(
        self,
    ) -> tuple[
        Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]],
        Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]],
    ]:
        async def _plain_wrapper(func: Callable[[], Awaitable[Any]]) -> Any:
            return await func()

        if not self.settings.enable_resilience:
            self.logger.info("Service resilience is disabled.")
            return _plain_wrapper, _plain_wrapper

        ingest_breaker = CircuitBreaker(
            failure_threshold=max(1, self.settings.ingest_breaker_failure_threshold),
            reset_timeout=max(0.0, self.settings.breaker_reset_timeout_seconds),
            name="local_rag_ingest",
        )
        query_breaker = CircuitBreaker(
            failure_threshold=max(1, self.settings.query_breaker_failure_threshold),
            reset_timeout=max(0.0, self.settings.breaker_reset_timeout_seconds),
            name="local_rag_query",
        )

        @ingest_breaker.async_call
        @async_retry(
            max_attempts=max(1, self.settings.resilience_max_attempts),
            base_delay=max(0.0, self.settings.ingest_retry_base_delay),
            max_delay=max(0.0, self.settings.resilience_max_delay),
            on_retry=lambda exc, attempt, delay: self._on_resilience_retry(
                "ingest", exc, attempt, delay
            ),
        )
        async def _safe_ingest(func: Callable[[], Awaitable[Any]]) -> Any:
            return await func()

        @query_breaker.async_call
        @async_retry(
            max_attempts=max(1, self.settings.resilience_max_attempts),
            base_delay=max(0.0, self.settings.query_retry_base_delay),
            max_delay=max(0.0, self.settings.resilience_max_delay),
            on_retry=lambda exc, attempt, delay: self._on_resilience_retry(
                "query", exc, attempt, delay
            ),
        )
        async def _safe_query(func: Callable[[], Awaitable[Any]]) -> Any:
            return await func()

        self.logger.info(
            "Service resilience enabled: max_attempts=%d ingest_delay=%.1fs query_delay=%.1fs max_delay=%.1fs",
            max(1, self.settings.resilience_max_attempts),
            max(0.0, self.settings.ingest_retry_base_delay),
            max(0.0, self.settings.query_retry_base_delay),
            max(0.0, self.settings.resilience_max_delay),
        )
        return _safe_ingest, _safe_query

    def _register_callbacks_to_rag(self, rag: RAGAnything) -> None:
        for callback in self._registered_callbacks:
            rag.callback_manager.register(callback)
        if self.settings.enable_callback_event_log:
            rag.callback_manager.enable_event_log(True)

    def register_callback(self, callback: ProcessingCallback) -> None:
        if not isinstance(callback, ProcessingCallback):
            raise TypeError(
                f"Expected ProcessingCallback instance, got {type(callback).__name__}"
            )
        if callback in self._registered_callbacks:
            return
        self._registered_callbacks.append(callback)
        for rag in self._rag_instances.values():
            rag.callback_manager.register(callback)
            if self.settings.enable_callback_event_log:
                rag.callback_manager.enable_event_log(True)

    def unregister_callback(self, callback: ProcessingCallback) -> None:
        if callback in self._registered_callbacks:
            self._registered_callbacks.remove(callback)
        for rag in self._rag_instances.values():
            try:
                rag.callback_manager.unregister(callback)
            except ValueError:
                pass

    def get_metrics_summary(self) -> str:
        lines: list[str] = []
        for callback in self._registered_callbacks:
            if isinstance(callback, MetricsCallback):
                lines.append(callback.summary())
        return "\n\n".join(lines)

    def get_callback_events(self, workspace_id: str) -> list[dict[str, Any]]:
        rag = self._rag_instances.get(workspace_id)
        if rag is None:
            return []
        return [event.to_dict() for event in rag.callback_manager.event_log]

    async def cleanup_workspace_instance(self, workspace_id: str) -> None:
        rag = self._rag_instances.get(workspace_id)
        if rag is not None:
            await rag.finalize_storages()
            self._rag_instances.pop(workspace_id, None)
        self._warmed_workspaces.discard(workspace_id)
        self._workspace_warmup_locks.pop(workspace_id, None)

    def _apply_multimodal_guardrails_to_rag(self, rag: RAGAnything) -> None:
        """
        Inject multimodal guardrail settings into LightRAG addon_params without
        overwriting existing upstream addon keys (e.g. language/entity_types).
        """
        lightrag_inst = getattr(rag, "lightrag", None)
        if lightrag_inst is None:
            return
        addon_params = getattr(lightrag_inst, "addon_params", None)
        if not isinstance(addon_params, dict):
            addon_params = {}
            lightrag_inst.addon_params = addon_params
        addon_params["multimodal_item_timeout_seconds"] = float(
            self.settings.multimodal_item_timeout_seconds
        )
        addon_params["multimodal_batch_watchdog_seconds"] = float(
            self.settings.multimodal_batch_watchdog_seconds
        )
        addon_params["multimodal_cancel_grace_seconds"] = float(
            self.settings.multimodal_cancel_grace_seconds
        )
        addon_params["multimodal_transient_retry_attempts"] = int(
            max(2, min(3, self.settings.resilience_max_attempts))
        )
        addon_params["multimodal_transient_retry_base_delay"] = float(
            max(0.0, self.settings.ingest_retry_base_delay)
        )
        addon_params["multimodal_transient_retry_max_delay"] = float(
            max(0.0, self.settings.resilience_max_delay)
        )

    def _build_rag(self, working_dir: str, workspace_id: str) -> RAGAnything:
        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        return RAGAnything(
            config=config,
            llm_model_func=self.llm_model_func,
            vision_model_func=self.vision_model_func,
            embedding_func=self.embedding_func,
            lightrag_kwargs={
                # workspace 使 Neo4j 和 Qdrant 按工作空间隔离数据；
                # 不传则 LightRAG 默认读 WORKSPACE 环境变量，空字符串时
                # Neo4JStorage 回退为 "base"，导致所有工作空间共用同一标签。
                "workspace": workspace_id,
                "rerank_model_func": self.rerank_func,
                "tokenizer": self.lightrag_tokenizer,
                "chunk_token_size": self.settings.chunk_token_size,
                "chunk_overlap_token_size": self.settings.chunk_overlap_token_size,
                "chunking_func": get_chunking_func(self.settings.chunking_strategy),
                # --- Indexing 并发与质量参数 ---
                # 以下参数默认值见 raganything/constants.py 的
                # "Indexing concurrency & quality" 区块。
                # entity extraction LLM 最大并发（LightRAG 默认 4）
                "llm_model_max_async": DEFAULT_LLM_MODEL_MAX_ASYNC,
                # gleaning 补充提取轮数（0=禁用，1=多提取一次；LightRAG 默认 1）
                "entity_extract_max_gleaning": DEFAULT_ENTITY_EXTRACT_MAX_GLEANING,
                # 文档级最大并发插入数（LightRAG 默认 2）
                "max_parallel_insert": DEFAULT_MAX_PARALLEL_INSERT,
                # embedding 单批最大文本数（LightRAG 默认 10）
                "embedding_batch_num": DEFAULT_EMBEDDING_BATCH_NUM,
                # embedding 调用最大并发数（LightRAG 默认 8）
                "embedding_func_max_async": DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
                # rerank 后保留 chunk 的最低分数（LightRAG 默认 0.0 = 不过滤）
                # BGE-reranker-v2-m3 相关 chunk 典型得分 >0.5，不相关 <0.3
                "min_rerank_score": DEFAULT_MIN_RERANK_SCORE,
                # V1/V2/V3 ablation toggles
                "enable_entity_disambiguation": self.settings.enable_entity_disambiguation,
                "enable_synonym_linking": self.settings.enable_synonym_linking,
                "synonymy_threshold": self.settings.synonymy_threshold,
                "synonymy_topk": self.settings.synonymy_topk,
                "synonymy_min_entity_len": self.settings.synonymy_min_entity_len,
                "enable_entity_surface_normalization": self.settings.enable_entity_surface_normalization,
                "enable_keyword_case_normalization": self.settings.enable_keyword_case_normalization,
                "entity_uppercase_allowlist": self.settings.entity_uppercase_allowlist,
                "strict_relation_endpoint_entity_match": self.settings.strict_relation_endpoint_entity_match,
                "recognition_prompt_max_tokens": self.settings.recognition_prompt_max_tokens,
                "recognition_prompt_output_max_tokens": self.settings.recognition_prompt_output_max_tokens,
                "recognition_prompt_reserved_tokens": self.settings.recognition_prompt_reserved_tokens,
                # V3 knobs are query-time only (QueryParam) and should not be
                # passed into LightRAG.__init__ for compatibility.
                "graph_storage": "Neo4JStorage",
                "vector_storage": "QdrantVectorDBStorage",
            },
        )

    async def get_rag(self, workspace_id: str, working_dir: str | None = None) -> RAGAnything:
        async with self._init_lock:
            if workspace_id in self._rag_instances:
                return self._rag_instances[workspace_id]
            effective_working_dir = working_dir or str(
                Path(self.settings.working_dir_root) / workspace_id
            )
            rag = self._build_rag(effective_working_dir, workspace_id)
            self._register_callbacks_to_rag(rag)
            self._rag_instances[workspace_id] = rag
            return rag

    async def _ensure_workspace_warmed(self, workspace_id: str) -> None:
        if workspace_id in self._warmed_workspaces:
            return
        if workspace_id not in self._workspace_warmup_locks:
            self._workspace_warmup_locks[workspace_id] = asyncio.Lock()
        async with self._workspace_warmup_locks[workspace_id]:
            if workspace_id in self._warmed_workspaces:
                return
            rag = await self.get_rag(workspace_id)
            init_result = await rag._ensure_lightrag_initialized()
            if not isinstance(init_result, dict) or not bool(
                init_result.get("success", False)
            ):
                error_msg = (
                    str(init_result.get("error", "")).strip()
                    if isinstance(init_result, dict)
                    else ""
                )
                raise RuntimeError(
                    f"LightRAG warmup failed for workspace '{workspace_id}': "
                    f"{error_msg or 'unknown error'}"
                )
            if getattr(rag, "lightrag", None) is None:
                raise RuntimeError(
                    f"LightRAG warmup failed for workspace '{workspace_id}': runtime not initialized"
                )
            self._apply_multimodal_guardrails_to_rag(rag)
            self._warmed_workspaces.add(workspace_id)
            self.logger.info("Workspace warmup complete: %s", workspace_id)

    async def ingest(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        workspace_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        serialize_by_workspace_id: Optional[bool] = None,
    ) -> str:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Input not found: {file_path}")

        workspace_id = workspace_id or _safe_workspace_id(file_path_obj.stem)
        rag = await self.get_rag(workspace_id)
        output_dir = output_dir or self.settings.output_dir
        effective_serialize, forced_to_serialize = _resolve_ingest_serialize_by_workspace_id(
            default_serialize_by_workspace_id=self.settings.serialize_ingest_by_workspace_id,
            serialize_by_workspace_id=serialize_by_workspace_id,
            chunking_strategy=chunking_strategy,
            default_chunking_strategy=self.settings.chunking_strategy,
        )
        if forced_to_serialize:
            self.logger.warning(
                "serialize_by_workspace_id is forced to True because chunking_strategy "
                "override requires temporary chunking_func swap (workspace_id=%s, "
                "default=%s, override=%s).",
                workspace_id,
                self.settings.chunking_strategy,
                chunking_strategy,
            )

        async def _run_ingest() -> None:
            old_chunking_func = None
            if chunking_strategy and chunking_strategy != self.settings.chunking_strategy:
                new_func = get_chunking_func(chunking_strategy)
                # lightrag is created lazily inside process_document_complete, so we
                # must swap in lightrag_kwargs (read during LightRAG.__init__).
                # If lightrag is already initialized, also patch the live instance.
                old_chunking_func = rag.lightrag_kwargs.get("chunking_func")
                rag.lightrag_kwargs["chunking_func"] = new_func
                if rag.lightrag is not None:
                    rag.lightrag.chunking_func = new_func
            try:
                if file_path_obj.is_file():
                    await rag.process_document_complete(
                        file_path=str(file_path_obj),
                        output_dir=output_dir,
                        parse_method="auto",
                    )
                else:
                    await rag.process_folder_complete(
                        str(file_path_obj),
                        output_dir=output_dir,
                        recursive=False,
                    )
            finally:
                if old_chunking_func is not None:
                    # Restore both the kwargs and the live instance (if now initialized)
                    rag.lightrag_kwargs["chunking_func"] = old_chunking_func
                    if rag.lightrag is not None:
                        rag.lightrag.chunking_func = old_chunking_func

        await self._ensure_workspace_warmed(workspace_id)

        if effective_serialize:
            # Per-workspace-id lock: serialise ingest for the same workspace so the
            # temporary chunking_func swap never races with a concurrent ingest.
            if workspace_id not in self._ingest_locks:
                self._ingest_locks[workspace_id] = asyncio.Lock()
            async with self._ingest_locks[workspace_id]:
                await self._safe_ingest_call(_run_ingest)
        else:
            await self._safe_ingest_call(_run_ingest)

        return workspace_id

    async def lightrag_ainsert(
        self,
        workspace_id: str,
        *,
        input: str | list[str],
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> Any:
        """Resilience-wrapped proxy to LightRAG.ainsert for prebuilt-chunk ingest paths."""
        rag = await self.get_rag(workspace_id)
        await self._ensure_workspace_warmed(workspace_id)

        async def _run_insert() -> Any:
            return await rag.lightrag.ainsert(
                input=input,
                ids=ids,
                file_paths=file_paths,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
            )

        if self.settings.serialize_ingest_by_workspace_id:
            if workspace_id not in self._ingest_locks:
                self._ingest_locks[workspace_id] = asyncio.Lock()
            async with self._ingest_locks[workspace_id]:
                return await self._safe_ingest_call(_run_insert)

        return await self._safe_ingest_call(_run_insert)

    async def lightrag_adelete_by_doc_id(
        self,
        workspace_id: str,
        doc_id: str,
        *,
        delete_llm_cache: bool = False,
    ) -> Any:
        """Resilience-wrapped proxy to LightRAG.adelete_by_doc_id."""
        rag = await self.get_rag(workspace_id)
        await self._ensure_workspace_warmed(workspace_id)

        async def _run_delete() -> Any:
            return await rag.lightrag.adelete_by_doc_id(
                doc_id, delete_llm_cache=delete_llm_cache
            )

        return await self._safe_ingest_call(_run_delete)

    async def lightrag_aquery_data(
        self,
        workspace_id: str,
        query: str,
        *,
        param: Any,
    ) -> dict[str, Any]:
        """Resilience-wrapped proxy to LightRAG.aquery_data for retrieval-only evaluations."""
        rag = await self.get_rag(workspace_id)
        await self._ensure_workspace_warmed(workspace_id)

        async def _run_query_data() -> dict[str, Any]:
            result = await rag.lightrag.aquery_data(query, param=param)
            if isinstance(result, dict):
                return result
            return {"result": result}

        return await self._safe_query_call(_run_query_data)

    async def query(self, workspace_id: str, query: str, **kwargs) -> str:
        rag = await self.get_rag(workspace_id)
        normalized_kwargs = dict(kwargs)
        normalized_kwargs.setdefault(
            "image_token_estimate_method",
            self.settings.image_token_estimate_method,
        )
        normalized_kwargs.setdefault(
            "image_token_model_name_or_path",
            self.settings.image_token_model_name_or_path,
        )
        normalized_kwargs.setdefault(
            "image_wrapper_tokens_per_image",
            self.settings.image_wrapper_tokens_per_image,
        )
        normalized_kwargs["image_token_estimate_method"] = _normalize_qwen_image_token_method(
            str(normalized_kwargs.get("image_token_estimate_method", ""))
        )
        normalized_kwargs.setdefault("enable_multi_hop", self.settings.enable_multi_hop)
        normalized_kwargs.setdefault("multi_hop_depth", self.settings.multi_hop_depth)
        normalized_kwargs.setdefault("ppr_damping", self.settings.ppr_damping)
        normalized_kwargs.setdefault("ppr_top_k", self.settings.ppr_top_k)
        normalized_kwargs.setdefault("passage_node_weight", self.settings.passage_node_weight)
        normalized_kwargs.setdefault("recognition_top_k", self.settings.recognition_top_k)
        normalized_kwargs.setdefault("linking_top_k", self.settings.linking_top_k)
        normalized_kwargs.setdefault("ppr_qa_top_k", self.settings.ppr_qa_top_k)
        normalized_kwargs.setdefault(
            "exclude_synonym_edges", self.settings.exclude_synonym_edges
        )
        normalized_kwargs.setdefault(
            "ppr_synonym_weight_mode", self.settings.ppr_synonym_weight_mode
        )
        normalized_kwargs["ppr_synonym_weight_mode"] = _normalize_ppr_synonym_weight_mode(
            str(normalized_kwargs.get("ppr_synonym_weight_mode", ""))
        )
        normalized_kwargs.setdefault(
            "user_prompt",
            _INLINE_CITATION_INSTRUCTION if _INLINE_CITATIONS_ENABLED else "",
        )

        async def _run_query() -> str:
            return await rag.aquery(query, **normalized_kwargs)

        return await self._safe_query_call(_run_query)

    async def query_with_trace(
        self, workspace_id: str, query: str, working_dir: str | None = None, **kwargs
    ) -> dict[str, Any]:
        rag = await self.get_rag(workspace_id, working_dir=working_dir)
        normalized_kwargs = dict(kwargs)
        normalized_kwargs.setdefault(
            "image_token_estimate_method",
            self.settings.image_token_estimate_method,
        )
        normalized_kwargs.setdefault(
            "image_token_model_name_or_path",
            self.settings.image_token_model_name_or_path,
        )
        normalized_kwargs.setdefault(
            "image_wrapper_tokens_per_image",
            self.settings.image_wrapper_tokens_per_image,
        )
        normalized_kwargs["image_token_estimate_method"] = _normalize_qwen_image_token_method(
            str(normalized_kwargs.get("image_token_estimate_method", ""))
        )
        normalized_kwargs.setdefault("enable_multi_hop", self.settings.enable_multi_hop)
        normalized_kwargs.setdefault("multi_hop_depth", self.settings.multi_hop_depth)
        normalized_kwargs.setdefault("ppr_damping", self.settings.ppr_damping)
        normalized_kwargs.setdefault("ppr_top_k", self.settings.ppr_top_k)
        normalized_kwargs.setdefault("passage_node_weight", self.settings.passage_node_weight)
        normalized_kwargs.setdefault("recognition_top_k", self.settings.recognition_top_k)
        normalized_kwargs.setdefault("linking_top_k", self.settings.linking_top_k)
        normalized_kwargs.setdefault("ppr_qa_top_k", self.settings.ppr_qa_top_k)
        normalized_kwargs.setdefault(
            "exclude_synonym_edges", self.settings.exclude_synonym_edges
        )
        normalized_kwargs.setdefault(
            "ppr_synonym_weight_mode", self.settings.ppr_synonym_weight_mode
        )
        normalized_kwargs["ppr_synonym_weight_mode"] = _normalize_ppr_synonym_weight_mode(
            str(normalized_kwargs.get("ppr_synonym_weight_mode", ""))
        )
        normalized_kwargs["return_trace"] = True

        async def _run_query_with_trace() -> dict[str, Any]:
            result = await rag.aquery(query, **normalized_kwargs)
            if isinstance(result, dict):
                return {
                    "answer": str(result.get("answer", "")),
                    "trace": result.get("trace", {}),
                }
            return {"answer": str(result), "trace": {}}

        return await self._safe_query_call(_run_query_with_trace)

    async def stream_query(
        self,
        workspace_id: str,
        query: str,
        mode: str = DEFAULT_QUERY_MODE,
        top_k: int = DEFAULT_TOP_K,
        chunk_top_k: int = DEFAULT_CHUNK_TOP_K,
        enable_rerank: bool = DEFAULT_ENABLE_RERANK,
        enable_multi_hop: bool | None = None,
        multi_hop_depth: int | None = None,
        ppr_damping: float | None = None,
        ppr_top_k: int | None = None,
        passage_node_weight: float | None = None,
        recognition_top_k: int | None = None,
        ppr_synonym_weight_mode: str | None = None,
        exclude_synonym_edges: bool | None = None,
    ):
        """Async generator — yields structured events via LightRAG aquery_llm().

        Event types:
          {"type": "meta", "data": {...}, "metadata": {...}}   # citations + entities + chunks
          {"type": "chunk", "text": str}                       # LLM token
          {"type": "error", "text": str}                       # error
        """
        try:
            from lightrag import QueryParam
            rag_instance = await self.get_rag(workspace_id)
            await rag_instance._ensure_lightrag_initialized()
            param = QueryParam(
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                enable_rerank=enable_rerank,
                rerank_score_scope="all",
                stream=True,
                include_references=True,
                user_prompt=_INLINE_CITATION_INSTRUCTION if _INLINE_CITATIONS_ENABLED else "",
                enable_multi_hop=enable_multi_hop if enable_multi_hop is not None else self.settings.enable_multi_hop,
                multi_hop_depth=multi_hop_depth if multi_hop_depth is not None else self.settings.multi_hop_depth,
                ppr_damping=ppr_damping if ppr_damping is not None else self.settings.ppr_damping,
                ppr_top_k=ppr_top_k if ppr_top_k is not None else self.settings.ppr_top_k,
                passage_node_weight=passage_node_weight
                if passage_node_weight is not None
                else self.settings.passage_node_weight,
                recognition_top_k=(
                    recognition_top_k
                    if recognition_top_k is not None
                    else self.settings.recognition_top_k
                ),
                exclude_synonym_edges=(
                    exclude_synonym_edges
                    if exclude_synonym_edges is not None
                    else self.settings.exclude_synonym_edges
                ),
                ppr_synonym_weight_mode=_normalize_ppr_synonym_weight_mode(
                    ppr_synonym_weight_mode
                    if ppr_synonym_weight_mode is not None
                    else self.settings.ppr_synonym_weight_mode
                ),
            )
            result = await rag_instance.lightrag.aquery_llm(query, param=param)

            # Yield meta first (citations, entities, chunks, relationships)
            yield {
                "type": "meta",
                "data": result.get("data", {}),
                "metadata": result.get("metadata", {}),
            }

            # Then yield LLM tokens
            llm_response = result.get("llm_response", {})
            if llm_response.get("is_streaming"):
                async for token in llm_response["response_iterator"]:
                    if token:
                        yield {"type": "chunk", "text": token}
            else:
                # Cache hit or non-streaming fallback
                content = llm_response.get("content") or "No answer returned."
                yield {"type": "chunk", "text": content}

        except Exception as exc:
            self.logger.error("stream_query error: %s", exc)
            yield {"type": "error", "text": str(exc)}


if __name__ == "__main__":
    # 加载 .env 文件中的环境变量
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG 后台管理工具")
    parser.add_argument("--path", "-p", required=True, help="要入库的文件或文件夹路径")
    parser.add_argument("--id", "-i", required=True, help="工作空间名称 (workspace_id)")
    parser.add_argument(
        "--max-async-ingest",
        type=int,
        default=DEFAULT_MAX_ASYNC_INGEST,
        help=f"文件夹模式下，同时并发入库的最大文件数（默认 {DEFAULT_MAX_ASYNC_INGEST}）",
    )
    args = parser.parse_args()

    async def main():
        print(f"正在初始化 RAG 服务...")
        settings = LocalRagSettings.from_env()
        service = LocalRagService(settings)

        target_path = args.path
        workspace_name = args.id
        max_async_ingest: int = max(1, args.max_async_ingest)

        print(f"开始处理: {target_path}")
        print(f"目标工作区: {settings.working_dir_root}/{workspace_name}")

        # output_dir 与 Web UI 保持一致：output/{workspace_id}/
        workspace_output = str(Path(settings.output_dir) / workspace_name)

        # 将源文件复制到 uploads/{workspace_id}/，使 /uploads 端点可见
        # 使用 settings.uploads_dir 保证与服务器读取的路径一致
        upload_workspace_dir = Path(settings.uploads_dir) / workspace_name
        upload_workspace_dir.mkdir(parents=True, exist_ok=True)
        supported_exts = {ext.strip().lower() for ext in DEFAULT_SUPPORTED_FILE_EXTENSIONS.split(",")}
        target_path_obj = Path(target_path)
        if target_path_obj.is_file():
            files_to_register = [target_path_obj]
        else:
            # 只注册顶层文件（与 process_folder_complete recursive=False 保持一致）
            files_to_register = [
                p for p in target_path_obj.glob("*")
                if p.is_file() and p.suffix.lower() in supported_exts
            ]
        for src in files_to_register:
            dest = upload_workspace_dir / src.name
            if not dest.exists():
                shutil.copy2(str(src), str(dest))

        if target_path_obj.is_file():
            # 单文件：直接入库
            try:
                await service.ingest(
                    file_path=target_path,
                    workspace_id=workspace_name,
                    output_dir=workspace_output,
                )
                print(f"\n 入库成功！")
                print(f"知识图谱已更新: {settings.working_dir_root}/{workspace_name}/graph_chunk_entity_relation.graphml")
                print(f"Markdown 已生成: {workspace_output}/")
            except Exception as e:
                print(f"\n 发生错误: {e}")
        else:
            # 文件夹：参考 evaluate_shared.py 的并发批处理模式
            # serialize_by_workspace_id=False 避免 workspace 级锁串行化；
            # LightRAG 内部的 llm_model_max_async 信号量仍会限制最大并发 LLM 调用数。
            ingest_jobs = files_to_register
            if not ingest_jobs:
                print(f"未找到可入库的文件：{target_path}")
                return

            print(f"共 {len(ingest_jobs)} 个文件，每批并发 {max_async_ingest} 个")
            success_count = 0
            failed_files: list[str] = []

            for batch_start in range(0, len(ingest_jobs), max_async_ingest):
                batch = ingest_jobs[batch_start : batch_start + max_async_ingest]
                batch_label = f"{batch_start + 1}-{batch_start + len(batch)}"
                print(f"[批次 {batch_label}/{len(ingest_jobs)}] 开始并发入库...")

                async def _ingest_one(file_path: Path) -> str:
                    await service.ingest(
                        file_path=str(file_path),
                        workspace_id=workspace_name,
                        output_dir=workspace_output,
                        serialize_by_workspace_id=False,
                    )
                    return str(file_path)

                batch_results = await asyncio.gather(
                    *[_ingest_one(f) for f in batch],
                    return_exceptions=True,
                )

                for f, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        print(f"  [失败] {f.name}: {result}")
                        failed_files.append(str(f))
                    else:
                        print(f"  [完成] {f.name}")
                        success_count += 1

            print(f"\n 入库完成：成功 {success_count} / 共 {len(ingest_jobs)} 个文件")
            if failed_files:
                print(f" 失败文件：")
                for fp in failed_files:
                    print(f"  - {fp}")
            print(f"知识图谱已更新: {settings.working_dir_root}/{workspace_name}/graph_chunk_entity_relation.graphml")
            print(f"Markdown 已生成: {workspace_output}/")

    asyncio.run(main())
