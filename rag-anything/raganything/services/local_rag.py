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
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from raganything.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_TIKTOKEN_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_RERANK_MODEL_PATH,
    DEFAULT_WORKING_DIR_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_VLLM_API_BASE,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_LLM_MODEL_NAME,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_TOKEN_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_QUERY_MAX_TOKENS,
    DEFAULT_INGEST_MAX_TOKENS,
    DEFAULT_VLM_ENABLE_JSON_SCHEMA,
)

_MODEL_CACHE: Dict[str, Any] = {}
_INTERNAL_OPENAI_KWARGS = {"hashing_kv", "keyword_extraction", "enable_cot"}


@dataclass
class LocalRagSettings:
    tiktoken_cache_dir: str = DEFAULT_TIKTOKEN_CACHE_DIR
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH
    rerank_model_path: str = DEFAULT_RERANK_MODEL_PATH

    working_dir_root: str = DEFAULT_WORKING_DIR_ROOT
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_dir: str = DEFAULT_LOG_DIR

    vllm_api_base: str = DEFAULT_VLLM_API_BASE
    vllm_api_key: str = DEFAULT_VLLM_API_KEY
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME
    vision_vllm_api_base: str = DEFAULT_VLLM_API_BASE
    vision_vllm_api_key: str = DEFAULT_VLLM_API_KEY
    vision_model_name: str = DEFAULT_LLM_MODEL_NAME
    device: str = DEFAULT_DEVICE

    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    max_token_size: int = DEFAULT_MAX_TOKEN_SIZE

    temperature: float = DEFAULT_TEMPERATURE
    query_max_tokens: int = DEFAULT_QUERY_MAX_TOKENS
    ingest_max_tokens: int = DEFAULT_INGEST_MAX_TOKENS

    vlm_enable_json_schema: bool = DEFAULT_VLM_ENABLE_JSON_SCHEMA

    @classmethod
    def from_env(cls) -> "LocalRagSettings":
        vllm_base = os.getenv("VLLM_API_BASE", DEFAULT_VLLM_API_BASE)
        vllm_key = os.getenv("VLLM_API_KEY", DEFAULT_VLLM_API_KEY)
        llm_name = os.getenv("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME)

        return cls(
            tiktoken_cache_dir=os.getenv("TIKTOKEN_CACHE_DIR", DEFAULT_TIKTOKEN_CACHE_DIR),
            embedding_model_path=os.getenv("RAGANYTHING_EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL_PATH),
            rerank_model_path=os.getenv("RAGANYTHING_RERANK_MODEL_PATH", DEFAULT_RERANK_MODEL_PATH),
            log_dir=os.getenv("RAGANYTHING_LOG_DIR", DEFAULT_LOG_DIR),
            vllm_api_base=vllm_base,
            vllm_api_key=vllm_key,
            llm_model_name=llm_name,
            vision_vllm_api_base=os.getenv("VISION_VLLM_API_BASE", vllm_base),
            vision_vllm_api_key=os.getenv("VISION_VLLM_API_KEY", vllm_key),
            vision_model_name=os.getenv("VISION_MODEL_NAME", llm_name),
            device=os.getenv("RAGANYTHING_DEVICE", DEFAULT_DEVICE),
            working_dir_root=os.getenv("RAGANYTHING_WORKDIR_ROOT", DEFAULT_WORKING_DIR_ROOT),
            output_dir=os.getenv("RAGANYTHING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
            embedding_dim=int(os.getenv("RAGANYTHING_EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM))),
            max_token_size=int(os.getenv("RAGANYTHING_MAX_TOKEN_SIZE", str(DEFAULT_MAX_TOKEN_SIZE))),
            temperature=float(os.getenv("RAGANYTHING_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            query_max_tokens=int(
                os.getenv(
                    "RAGANYTHING_QUERY_MAX_TOKENS",
                    os.getenv(
                        "RAGANYTHING_VISION_MAX_TOKENS",
                        os.getenv("RAGANYTHING_MAX_TOKENS", str(DEFAULT_QUERY_MAX_TOKENS)),
                    ),
                )
            ),
            ingest_max_tokens=int(
                os.getenv("RAGANYTHING_INGEST_MAX_TOKENS", str(DEFAULT_INGEST_MAX_TOKENS))
            ),
            vlm_enable_json_schema=os.getenv(
                "RAGANYTHING_VLM_ENABLE_JSON_SCHEMA", str(DEFAULT_VLM_ENABLE_JSON_SCHEMA)
            ).lower()
            in {"1", "true", "yes", "y", "on"},
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


def build_embedding_func(
    settings: LocalRagSettings, st_model: SentenceTransformer
) -> EmbeddingFunc:
    async def _compute_embedding(texts: list[str]) -> np.ndarray:
        return st_model.encode(texts, normalize_embeddings=True)

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.max_token_size,
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


def _extract_last_context_segment(raw_text: str) -> tuple[str, str]:
    # 只取最后一个 ---Context---，避免命中示例或前序模板。
    context_idx = raw_text.rfind("---Context---")
    if context_idx < 0:
        return "", raw_text.strip()
    prefix = raw_text[:context_idx].strip()
    context = raw_text[context_idx:].strip()
    return prefix, context


def _drop_empty_additional_instructions(text: str) -> str:
    # 删除空的 "Additional Instructions" 行，避免空模板进入 system。
    out_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(
            r"^(?:\d+\.\s*)?Additional Instructions:\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            payload = match.group(1).strip()
            if payload.lower() in {"", "{user_prompt}", "none", "null", "n/a"}:
                continue
        out_lines.append(raw_line)
    return "\n".join(out_lines).strip()


def _compose_final_system(role_prefix: str, upstream_system: str) -> str:
    # 组合最终 system：优先保留 LightRAG 结构化指令，并把上游 system 前置。
    role_text = _drop_empty_additional_instructions(role_prefix.strip())
    upstream_text = upstream_system.strip()
    if role_text and upstream_text:
        return f"{upstream_text}\n\n{role_text}"
    return role_text or upstream_text


def _clean_context_for_user_text(context_text: str) -> str:
    # 清理上下文中残留问句/尾句，避免 query 重复。
    cleaned = context_text.strip()
    cleaned = re.sub(
        r"(?im)^\s*User Question\s*:\s*.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*Please answer based on the (?:provided )?context(?: and images)?\.?\s*$",
        "",
        cleaned,
    )
    return cleaned.strip()


def _try_repack_text_query(system_prompt: Any, prompt: Any) -> Optional[tuple[str, str]]:
    # 将 LightRAG 文本 query 从“全量 system”重组为“system 指令 + user 上下文/问题”。
    raw_system = str(system_prompt or "").strip()
    if "---Context---" not in raw_system:
        return None

    role_prefix, context_segment = _extract_last_context_segment(raw_system)
    if not role_prefix or not context_segment:
        return None

    final_system = _compose_final_system(role_prefix, "")
    cleaned_context = _clean_context_for_user_text(context_segment)
    if not cleaned_context:
        return None

    question = str(prompt or "").strip()
    if question:
        final_user = (
            f"{cleaned_context}\n\n"
            f"User Question: {question}\n\n"
            "Please answer based on the provided context."
        )
    else:
        final_user = cleaned_context
    return final_system, final_user


def _build_modal_analysis_response_format() -> dict[str, Any]:
    # 统一的结构化输出 schema（保持兼容优先）。
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "modal_analysis",
            "strict": False,
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
        cleaned_kwargs = _strip_internal_openai_kwargs(kwargs)
        if keyword_extraction:
            cleaned_kwargs["response_format"] = GPTKeywordExtractionFormat
        messages = []
        repacked = _try_repack_text_query(system_prompt, prompt)
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
        try:
            if "response_format" in cleaned_kwargs:
                response = await client.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.query_max_tokens,
                    **cleaned_kwargs,
                )
            else:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.query_max_tokens,
                    **cleaned_kwargs,
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
            return ""

    return llm_model_func


def build_vision_model_func(
    settings: LocalRagSettings,
    client: AsyncOpenAI,
    logger: logging.Logger,
    model_name: str,
):
    llm_fallback = build_llm_model_func(settings, client, logger, model_name)
    schema_stats = {"total": 0, "success": 0, "fallback": 0}

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
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.query_max_tokens,
                    **cleaned_kwargs,
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
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=settings.temperature,
                    max_tokens=settings.ingest_max_tokens,
                    **request_kwargs,
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
                if "response_format" in request_kwargs:
                    schema_stats["fallback"] += 1
                    logger.warning(
                        "Vision ingest json_schema call failed, retrying without schema: %s",
                        exc,
                    )
                    request_kwargs.pop("response_format", None)
                    try:
                        response = await client.chat.completions.create(
                            model=model_name,
                            messages=msgs,
                            temperature=settings.temperature,
                            max_tokens=settings.ingest_max_tokens,
                            **request_kwargs,
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
                fallback_seed = base64_image if base64_image else str(prompt)
                fallback_id = hashlib.md5(fallback_seed.encode("utf-8")).hexdigest()[:8]
                if prompt_has_json:
                    fallback_payload = {
                        "detailed_description": f"{task_type} description unavailable due to vision model error.",
                        "entity_info": {
                            "entity_name": f"{task_type}_{fallback_id}",
                            "entity_type": task_type,
                            "summary": f"{task_type} description unavailable due to vision model error.",
                        },
                    }
                    return json.dumps(fallback_payload, ensure_ascii=False)
                return f"{task_type} description unavailable due to vision model error."

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

def _safe_doc_id(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    if cleaned:
        return cleaned
    return hashlib.md5(name.encode("utf-8")).hexdigest()


class LocalRagService:
    def __init__(self, settings: Optional[LocalRagSettings] = None):
        # 初始化双客户端：text/vision 可同端口也可分离部署。
        self.settings = settings or LocalRagSettings.from_env()
        os.environ["TIKTOKEN_CACHE_DIR"] = self.settings.tiktoken_cache_dir
        self.logger = configure_logging(self.settings)
        _maybe_enable_internvl2_prompts(self.settings, self.logger)

        text_base, text_key, text_model = _resolve_text_endpoint(self.settings)
        vision_base, vision_key, vision_model = _resolve_vision_endpoint(self.settings)
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
        legacy_query_cap = getattr(self.settings, "max_tokens", None)
        legacy_vision_cap = getattr(self.settings, "vision_max_tokens", None)
        if legacy_query_cap is not None or legacy_vision_cap is not None:
            self.logger.warning(
                "Legacy token caps detected (max_tokens=%s, vision_max_tokens=%s). Please migrate to query_max_tokens/ingest_max_tokens.",
                legacy_query_cap,
                legacy_vision_cap,
            )

        self.text_client = AsyncOpenAI(api_key=text_key, base_url=text_base)
        self.vision_client = AsyncOpenAI(api_key=vision_key, base_url=vision_base)
        self._rag_instances: Dict[str, RAGAnything] = {}
        self._init_lock = asyncio.Lock()

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

    def _build_rag(self, working_dir: str) -> RAGAnything:
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
            lightrag_kwargs={"rerank_model_func": self.rerank_func},
        )

    async def get_rag(self, doc_id: str) -> RAGAnything:
        async with self._init_lock:
            if doc_id in self._rag_instances:
                return self._rag_instances[doc_id]
            working_dir = str(Path(self.settings.working_dir_root) / doc_id)
            rag = self._build_rag(working_dir)
            self._rag_instances[doc_id] = rag
            return rag

    async def ingest(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Input not found: {file_path}")

        doc_id = doc_id or _safe_doc_id(file_path_obj.stem)
        rag = await self.get_rag(doc_id)
        output_dir = output_dir or self.settings.output_dir

        if file_path_obj.is_file():
            await rag.process_document_complete(
                file_path=str(file_path_obj),
                output_dir=output_dir,
                parse_method="auto",
            )
        else:
            await rag.process_folder_complete(str(file_path_obj), recursive=False)

        return doc_id

    async def query(self, doc_id: str, query: str, **kwargs) -> str:
        rag = await self.get_rag(doc_id)
        return await rag.aquery(query, **kwargs)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RAG 后台管理工具")
    parser.add_argument("--path", "-p", required=True, help="要入库的文件或文件夹路径")
    parser.add_argument("--id", "-i", required=True, help="工作空间名称 (doc_id)")
    args = parser.parse_args()
    
    async def main():
        print(f"正在初始化 RAG 服务...")
        settings = LocalRagSettings.from_env()
        service = LocalRagService(settings)

        target_path = args.path
        workspace_name = args.id
        
        print(f"开始处理: {target_path}")
        print(f"目标工作区: {settings.working_dir_root}/{workspace_name}")

        try:
            await service.ingest(file_path=target_path, doc_id=workspace_name)

            print(f"\n 入库成功！")
            print(f"知识图谱已更新: {settings.working_dir_root}/{workspace_name}/graph_chunk_entity_relation.graphml")
            print(f"Markdown 已生成: {settings.output_dir}/{workspace_name}/")

        except Exception as e:
            print(f"\n 发生错误: {e}")
    
    asyncio.run(main())
