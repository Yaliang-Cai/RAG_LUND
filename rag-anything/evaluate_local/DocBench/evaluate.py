#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocBench Evaluation Script for RAG-Anything Local (Manual Server Mode)
======================================================================

Quick workflow:
---------
1. Start Qwen3-VL-30B-A3B-Instruct-FP8 service (port 8001)
   cd /data/y50056788/Yaliang/projects/rag-anything
   bash start_server_qwen3_vl.sh

2. Generate system answers (can run in background)
   python evaluate.py --mode generate
   or: nohup python evaluate.py --mode generate > run_generate.log 2>&1 &

3. Stop Qwen3-VL service, then start Qwen2.5-32B (port 8008)
   # press Ctrl+C in the server terminal
   bash start_server_qwen2.5_32b_awq.sh

4. Evaluate answers
   python evaluate.py --mode evaluate

5. Show stats
   python evaluate.py --mode stats
"""

import os

# ===== Limit MinerU internal vLLM GPU memory usage =====
# 0.10 = 4.74GB, 0.15 = 7.1GB, 0.20 = 9.47GB
# Recommended 0.15: MinerU ~= 7.1GB, GPU0 total ~= 12GB
os.environ['MINERU_VLLM_GPU_MEMORY_UTILIZATION'] = '0.1'
# =================================================

import json
import asyncio
import logging
import re
import sys
import gc
from pathlib import Path
from datetime import datetime
from typing import Any, TextIO
from openai import AsyncOpenAI

# Add parent directory to path to import local_rag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluate_local.ablation_flags import as_bool
from raganything.services.local_rag import LocalRagService, LocalRagSettings
from raganything.constants import (
    DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
    DEFAULT_PPR_QA_TOP_K,
    DEFAULT_PPR_TOP_K,
    DEFAULT_RECOGNITION_TOP_K,
)

# ==========================================
# Configuration (absolute paths for running from any directory)
# ==========================================

# Script directory
SCRIPT_DIR = Path("/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/DocBench")

# DocBench dataset directory
DATA_ROOT = Path("/data/y50056788/Yaliang/datasets_for_eval/data_for_DocBench")

# Output root directory (all results are saved here)
OUTPUT_DIR = SCRIPT_DIR / "docbench_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DUMP_DIR = OUTPUT_DIR / "prompt_dumps"
PROMPT_DUMP_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MESSAGES_DUMP_DIR = OUTPUT_DIR / "final_vlm_messages"
FINAL_MESSAGES_DUMP_DIR.mkdir(parents=True, exist_ok=True)
GENERATION_CONFIG_FILE = OUTPUT_DIR / "generation_config.json"
SINGLE_INGEST_MANIFEST_FILE = OUTPUT_DIR / "single_ingest_manifest.json"
SINGLE_INGEST_FAILURES_FILE = OUTPUT_DIR / "single_ingest_failures.jsonl"
DOCBENCH_INDEX_PROFILE_FILE = ".docbench_index_profile.json"

# RAG working directory (one isolated graph per document)
# Example output: docbench_results/rag_workspaces/docbench_0/, docbench_1/, ...
WORKING_DIR_ROOT = OUTPUT_DIR / "rag_workspaces"
WORKING_DIR_ROOT.mkdir(parents=True, exist_ok=True)

# MinerU output directory (isolated per document)
# Example output: docbench_results/mineru_outputs/docbench_0/{pdf_name}/hybrid_auto/, ...
OUTPUT_MD_DIR = OUTPUT_DIR / "mineru_outputs"
OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

# API settings
RAG_API_BASE = "http://localhost:8001/v1"      # Qwen3-VL-30B-A3B-Instruct-FP8 (answer generation)
JUDGE_API_BASE = "http://localhost:8008/v1"    # Qwen2.5-32B (evaluation)
RAG_API_KEY = "EMPTY"
RAG_VISION_MODEL_PATH = "/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8"

RAG_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
DOCBENCH_EVAL_PROMPT_FILENAME = "evaluation_prompt.txt"
RAGANYTHING_EVAL_PROMPT_FILENAME = "evaluation_prompt_RAG-Anything.txt"

# Query parameters (DocBench tuned)
DOCBENCH_QUERY_PARAMS = {
    "mode": "hybrid",
    "top_k": 40,
    "chunk_top_k": 20,
    "enable_rerank": True,
    "rerank_score_scope": "all",
    "vlm_enhanced": True,
    "multimodal_top_k": 3,
    "image_token_estimate_method": "qwen_vl",
    "image_token_model_name_or_path": RAG_VISION_MODEL_PATH,
    "image_wrapper_tokens_per_image": 2,
    "keyword_entity_rrf_k": 10,
    "keyword_relation_rrf_k": 20,
}
DOCBENCH_QUERY_MODE_CHOICES = (
    "local",
    "global",
    "hybrid",
    "naive",
    "mix",
    "bypass",
    "rrf",
    "ppr_local",
    "ppr",
)

ONE_SENTENCE_USER_PROMPT = (
    "Provide the final answer in exactly one sentence. "
    "Do not include headings, bullet points, numbering, code blocks, or a "
    "references section."
)

_BINARY_SCORE_RE = re.compile(r"(?<!\d)([01])(?!\d)")
_ACCURACY_FIELD_RE = re.compile(r'"accuracy"\s*:\s*([01])', flags=re.IGNORECASE)
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _build_docbench_query_params(
    one_sentence: bool = False,
    *,
    query_mode: str | None = None,
    recognition_top_k: int = DEFAULT_RECOGNITION_TOP_K,
    keyword_fanout_mode: str = "joined",
    keyword_entity_rrf_k: int = 10,
    keyword_relation_rrf_k: int = 20,
    entity_retrieval_mode: str = "dense",
    chunk_retrieval_mode: str = "dense",
    exclude_synonym_edges: bool | None = None,
    answer_context_mode: str = "kg_prompt",
    kg_chunk_selection_source: str = "truncated",
    max_total_tokens: int | None = None,
    multimodal_top_k: int | None = None,
    enable_rerank: bool = True,
    enable_kg_rerank: bool = True,
    ppr_top_k: int | None = None,
    ppr_qa_top_k: int | None = None,
    ppr_post_rerank_fusion: str = "none",
    ppr_post_rerank_rrf_k: int = 60,
    bypass_query_cache: bool = False,
    bypass_keywords_cache: bool = False,
) -> dict[str, Any]:
    query_params = dict(DOCBENCH_QUERY_PARAMS)
    if query_mode is not None:
        normalized_mode = str(query_mode).strip()
        if normalized_mode:
            query_params["mode"] = normalized_mode

    mode = str(query_params.get("mode", "")).strip()
    query_params["enable_rerank"] = bool(enable_rerank)
    query_params["enable_kg_rerank"] = bool(enable_kg_rerank)
    query_params["keyword_fanout_mode"] = str(keyword_fanout_mode).strip()
    query_params["keyword_entity_rrf_k"] = int(keyword_entity_rrf_k)
    query_params["keyword_relation_rrf_k"] = int(keyword_relation_rrf_k)
    query_params["entity_qdrant_retrieval_mode"] = str(entity_retrieval_mode).strip()
    query_params["chunk_qdrant_retrieval_mode"] = str(chunk_retrieval_mode).strip()
    query_params["kg_chunk_selection_source"] = str(kg_chunk_selection_source).strip()
    query_params["ppr_post_rerank_fusion"] = str(ppr_post_rerank_fusion).strip().lower()
    query_params["ppr_post_rerank_rrf_k"] = int(ppr_post_rerank_rrf_k)
    query_params["bypass_query_cache"] = bool(bypass_query_cache)
    query_params["bypass_keywords_cache"] = bool(bypass_keywords_cache)
    if max_total_tokens is not None:
        query_params["max_total_tokens"] = int(max_total_tokens)
    if multimodal_top_k is not None:
        query_params["multimodal_top_k"] = int(multimodal_top_k)

    if ppr_top_k is not None:
        query_params["ppr_top_k"] = int(ppr_top_k)
    else:
        query_params["ppr_top_k"] = int(query_params.get("ppr_top_k", DEFAULT_PPR_TOP_K))
    if ppr_qa_top_k is not None:
        query_params["ppr_qa_top_k"] = int(ppr_qa_top_k)
    else:
        query_params["ppr_qa_top_k"] = int(
            query_params.get("ppr_qa_top_k", DEFAULT_PPR_QA_TOP_K)
        )

    if mode == "ppr":
        query_params["recognition_top_k"] = max(0, int(recognition_top_k))
        query_params["answer_context_mode"] = "chunk_only_prompt"
        ppr_limit = int(query_params.get("ppr_top_k", 0) or 0)
        ppr_qa_limit = int(query_params.get("ppr_qa_top_k", 0) or 0)
        if ppr_limit <= 0:
            raise ValueError(f"ppr_top_k must be > 0, got {ppr_limit}")
        if ppr_qa_limit <= 0:
            raise ValueError(f"ppr_qa_top_k must be > 0, got {ppr_qa_limit}")
        if ppr_qa_limit > ppr_limit:
            raise ValueError(
                f"ppr_qa_top_k must be <= ppr_top_k, got {ppr_qa_limit} > {ppr_limit}"
            )
    else:
        query_params.pop("recognition_top_k", None)
        query_params["answer_context_mode"] = str(answer_context_mode).strip()

    query_params["exclude_synonym_edges"] = (
        mode != "ppr" if exclude_synonym_edges is None else bool(exclude_synonym_edges)
    )
    if one_sentence:
        query_params["user_prompt"] = ONE_SENTENCE_USER_PROMPT
        query_params["response_type"] = "Single Sentence"
    return query_params


def _resolve_eval_setup(use_raganything_eval_setup: bool) -> tuple[str, bool, str]:
    if use_raganything_eval_setup:
        return (
            "rag_anything",
            True,
            RAGANYTHING_EVAL_PROMPT_FILENAME,
        )
    return (
        "docbench_official",
        False,
        DOCBENCH_EVAL_PROMPT_FILENAME,
    )


def _load_eval_prompt(eval_prompt_filename: str) -> str:
    eval_prompt_file = SCRIPT_DIR / eval_prompt_filename
    if not eval_prompt_file.exists():
        raise FileNotFoundError(f"Evaluation prompt file not found: {eval_prompt_file}")
    with open(eval_prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def _save_generation_config(
    *,
    one_sentence: bool,
    max_async_docs: int,
    max_async_ingest_docs: int,
    max_async_query_docs: int,
    max_async_generate: int,
    doc_flush_every: int,
    profile_name: str,
    eval_prompt_filename: str,
    effective_query_params: dict[str, Any],
    index_profile: dict[str, Any],
    start_id: int,
    end_id: int,
    resume: bool,
) -> None:
    payload = {
        "profile_name": profile_name,
        "one_sentence": bool(one_sentence),
        "max_async_docs": int(max_async_docs),
        "max_async_ingest_docs": int(max_async_ingest_docs),
        "max_async_query_docs": int(max_async_query_docs),
        "max_async_generate": int(max_async_generate),
        "doc_flush_every": int(doc_flush_every),
        "eval_prompt_filename": eval_prompt_filename,
        "effective_query_params": dict(effective_query_params),
        "index_profile": dict(index_profile),
        "start_id": int(start_id),
        "end_id": int(end_id),
        "resume": bool(resume),
    }
    with open(GENERATION_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_generation_config() -> dict[str, Any] | None:
    if not GENERATION_CONFIG_FILE.exists():
        return None
    try:
        with open(GENERATION_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"Failed to load generation config: {exc}")
    return None


def _normalize_max_async(max_async: int, default: int = 4) -> int:
    """Clamp async worker count to a safe positive integer."""
    try:
        return max(1, int(max_async))
    except Exception:
        return default


def _normalize_flush_every(flush_every: int, default: int = 4) -> int:
    """Clamp flush interval to a safe non-negative integer."""
    try:
        value = int(flush_every)
        return value if value >= 0 else default
    except Exception:
        return default

# ==========================================
# Logging
# ==========================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure logs are visible in terminal
    ]
)
logger = logging.getLogger(__name__)

# Configure related module loggers as well
logging.getLogger("raganything").setLevel(logging.INFO)
logging.getLogger("raganything.processor").setLevel(logging.INFO)
logging.getLogger("raganything.parser").setLevel(logging.INFO)

_MASTER_LOG_PATH: Path | None = None


def _ensure_master_log_handler() -> None:
    """
    Keep one consolidated evaluate log file even if LocalRagService is recycled.
    """
    global _MASTER_LOG_PATH
    if _MASTER_LOG_PATH is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _MASTER_LOG_PATH = OUTPUT_DIR / "logs" / f"evaluate_generate_{ts}.log"
        _MASTER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base = getattr(handler, "baseFilename", "")
            if base and Path(base) == _MASTER_LOG_PATH:
                return

    file_handler = logging.FileHandler(_MASTER_LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    logger.info(f"Master log file: {_MASTER_LOG_PATH}")


def _append_jsonl_record(file_obj: TextIO, payload: dict[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file_obj.flush()


def _find_doc_files(folder_path: Path) -> tuple[Path | None, Path | None]:
    pdf_file: Path | None = None
    qa_file: Path | None = None
    for file in folder_path.iterdir():
        if file.suffix.lower() == ".pdf":
            pdf_file = file
        elif file.name.endswith("_qa.jsonl"):
            qa_file = file
    return pdf_file, qa_file


def _workspace_id_for_doc(doc_name: str) -> str:
    return f"docbench_{doc_name}"


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning(f"Failed to load JSON file {path}: {exc}")
    return None


def _save_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_docbench_index_profile(settings: LocalRagSettings) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "schema_version": "docbench_single_index_profile_v1",
        "profile_version": 1,
        "enable_entity_disambiguation": bool(settings.enable_entity_disambiguation),
        "enable_synonym_linking": bool(settings.enable_synonym_linking),
        "enable_multi_hop": bool(settings.enable_multi_hop),
        "enable_entity_surface_normalization": bool(
            settings.enable_entity_surface_normalization
        ),
        "enable_keyword_case_normalization": bool(
            settings.enable_keyword_case_normalization
        ),
        "strict_relation_endpoint_entity_match": bool(
            settings.strict_relation_endpoint_entity_match
        ),
        "qdrant_enable_sparse_bm25": bool(settings.qdrant_enable_sparse_bm25),
        "qdrant_retrieval_mode": str(settings.qdrant_retrieval_mode),
        "context_zero_window_content_types": str(
            DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES
        ),
    }
    if hasattr(settings, "synonymy_threshold"):
        profile["synonymy_threshold"] = float(settings.synonymy_threshold)
    if hasattr(settings, "synonymy_topk"):
        profile["synonymy_topk"] = int(settings.synonymy_topk)
    if hasattr(settings, "synonymy_min_entity_len"):
        profile["synonymy_min_entity_len"] = int(settings.synonymy_min_entity_len)
    return profile


def _workspace_profile_path(workspace_id: str) -> Path:
    return WORKING_DIR_ROOT / workspace_id / DOCBENCH_INDEX_PROFILE_FILE


def _workspace_has_artifacts(workspace_dir: Path) -> bool:
    if not workspace_dir.exists():
        return False
    for entry in workspace_dir.rglob("*"):
        if not entry.is_file():
            continue
        if entry.name == DOCBENCH_INDEX_PROFILE_FILE:
            continue
        return True
    return False


def _ensure_docbench_workspace_index_profile(
    *,
    workspace_id: str,
    index_profile: dict[str, Any],
) -> None:
    workspace_dir = WORKING_DIR_ROOT / workspace_id
    profile_path = _workspace_profile_path(workspace_id)
    existing_profile = _load_json_file(profile_path)
    if isinstance(existing_profile, dict):
        if existing_profile != index_profile:
            raise RuntimeError(
                "DocBench single-doc workspace index profile mismatch: "
                f"workspace_id={workspace_id}, existing={existing_profile}, "
                f"current={index_profile}. Use a new workspace or rebuild cleanly."
            )
        return

    if _workspace_has_artifacts(workspace_dir):
        raise RuntimeError(
            "DocBench single-doc workspace contains existing artifacts but no "
            f"{DOCBENCH_INDEX_PROFILE_FILE}: {workspace_dir}. Use a new workspace "
            "or rebuild cleanly to avoid mixed index settings."
        )

    _save_json_file(profile_path, index_profile)


def _load_single_ingest_manifest() -> dict[str, Any]:
    payload = _load_json_file(SINGLE_INGEST_MANIFEST_FILE)
    if isinstance(payload, dict):
        docs = payload.get("docs")
        if isinstance(docs, dict):
            return payload
    return {
        "schema_version": "docbench_single_ingest_manifest_v1",
        "docs": {},
    }


def _save_single_ingest_manifest(manifest: dict[str, Any]) -> None:
    manifest["schema_version"] = "docbench_single_ingest_manifest_v1"
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_json_file(SINGLE_INGEST_MANIFEST_FILE, manifest)


def _single_ingest_record_matches(
    *,
    manifest: dict[str, Any],
    doc_name: str,
    workspace_id: str,
    index_profile: dict[str, Any],
) -> bool:
    docs = manifest.get("docs")
    if not isinstance(docs, dict):
        return False
    record = docs.get(str(doc_name))
    if not isinstance(record, dict):
        return False
    if record.get("status") != "ok":
        return False
    if str(record.get("workspace_id", "")) != workspace_id:
        return False
    if record.get("index_profile") != index_profile:
        return False
    existing_profile = _load_json_file(_workspace_profile_path(workspace_id))
    if existing_profile is not None and existing_profile != index_profile:
        raise RuntimeError(
            "DocBench single-doc profile mismatch between manifest and workspace: "
            f"workspace_id={workspace_id}, manifest={index_profile}, "
            f"workspace={existing_profile}."
        )
    return True


def _validate_resume_processed_doc_profile(
    *,
    manifest: dict[str, Any],
    doc_name: str,
    workspace_id: str,
    index_profile: dict[str, Any],
) -> None:
    existing_profile = _load_json_file(_workspace_profile_path(workspace_id))
    if existing_profile != index_profile:
        raise RuntimeError(
            "DocBench resume processed-doc index profile mismatch: "
            f"doc_id={doc_name}, workspace_id={workspace_id}, "
            f"existing={existing_profile}, current={index_profile}. "
            "Use a new output file/workspace or rebuild cleanly."
        )

    if not _single_ingest_record_matches(
        manifest=manifest,
        doc_name=doc_name,
        workspace_id=workspace_id,
        index_profile=index_profile,
    ):
        raise RuntimeError(
            "DocBench resume processed-doc manifest mismatch: "
            f"doc_id={doc_name}, workspace_id={workspace_id}. "
            "Use a new output file/workspace or rebuild cleanly."
        )


def _record_single_ingest_success(
    *,
    manifest: dict[str, Any],
    doc_name: str,
    workspace_id: str,
    pdf_file: Path,
    index_profile: dict[str, Any],
) -> None:
    docs = manifest.setdefault("docs", {})
    docs[str(doc_name)] = {
        "status": "ok",
        "doc_id": str(doc_name),
        "workspace_id": workspace_id,
        "file_name": pdf_file.name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "index_profile": dict(index_profile),
    }
    _save_single_ingest_manifest(manifest)


def _append_single_ingest_failure(payload: dict[str, Any]) -> None:
    SINGLE_INGEST_FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SINGLE_INGEST_FAILURES_FILE, "a", encoding="utf-8") as f:
        _append_jsonl_record(f, payload)


def _validate_per_doc_workspace_env_isolation(workspace_ids: list[str]) -> None:
    unique_workspace_ids = {str(workspace_id).strip() for workspace_id in workspace_ids}
    unique_workspace_ids.discard("")
    if not unique_workspace_ids:
        return

    conflicts: list[str] = []
    for env_name in ("NEO4J_WORKSPACE", "QDRANT_WORKSPACE"):
        env_value_raw = os.getenv(env_name)
        if env_value_raw is None:
            continue
        env_value = str(env_value_raw).strip()
        if not env_value:
            continue
        if len(unique_workspace_ids) != 1 or env_value not in unique_workspace_ids:
            conflicts.append(
                f"{env_name}={env_value!r} conflicts with per-doc workspaces "
                f"{sorted(unique_workspace_ids)!r}"
            )
    if conflicts:
        raise ValueError(
            "Per-document workspace isolation check failed. Unset "
            "NEO4J_WORKSPACE/QDRANT_WORKSPACE or run a single matching doc. "
            f"Details: {'; '.join(conflicts)}"
        )


def _build_generation_result(
    doc_name: str,
    question: str,
    answer: str,
    qa_item: dict[str, Any],
) -> dict[str, Any]:
    return {
        "doc_id": doc_name,
        "question": question,
        "sys_ans": answer,
        "ref_ans": qa_item["answer"],
        "type": qa_item["type"],
        "evidence": qa_item["evidence"],
    }


def _build_eval_prompt(eval_prompt: str, item: dict[str, Any]) -> str:
    return (
        eval_prompt.replace("{{question}}", item["question"])
        .replace("{{sys_ans}}", item["sys_ans"])
        .replace("{{ref_ans}}", item["ref_ans"])
        .replace("{{ref_text}}", item["evidence"])
    )


def _parse_eval_score(eval_result: str) -> int:
    text = (eval_result or "").strip()
    if not text:
        return 0

    # Prefer strict JSON parsing for RAG-Anything-style prompt output.
    candidate = text
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()

    for payload in (candidate, text):
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = None

        if isinstance(parsed, dict) and "accuracy" in parsed:
            value = parsed["accuracy"]
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)) and value in (0, 1):
                return int(value)
            if isinstance(value, str) and value.strip() in {"0", "1"}:
                return int(value.strip())

    # Fallback: extract explicit accuracy field from text.
    accuracy_match = _ACCURACY_FIELD_RE.search(text)
    if accuracy_match:
        return int(accuracy_match.group(1))

    # Legacy fallback for old prompt outputs.
    head = text[:120]
    match = _BINARY_SCORE_RE.search(head)
    if match:
        return int(match.group(1))
    return 0


async def _cleanup_rag_instance(service: LocalRagService, rag_workspace_id: str) -> None:
    rag_instances = getattr(service, "_rag_instances", None)
    if not isinstance(rag_instances, dict):
        return

    rag_instance = rag_instances.get(rag_workspace_id)
    if rag_instance is None:
        logger.info(f"No RAG instance to clean for {rag_workspace_id}")
        return

    try:
        await rag_instance.finalize_storages()
        rag_instances.pop(rag_workspace_id, None)
        logger.info(f"Cleaned up RAG instance for {rag_workspace_id}")
    except Exception as exc:
        logger.warning(f"Failed to cleanup RAG instance {rag_workspace_id}: {exc}")


def _clear_cuda_cache(doc_id: int) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return

        mem_before = torch.cuda.memory_allocated(0) / 1024**3
        reserved_before = torch.cuda.memory_reserved(0) / 1024**3
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated(0) / 1024**3
        reserved_after = torch.cuda.memory_reserved(0) / 1024**3
        freed = reserved_before - reserved_after
        logger.info(
            f"GPU cache cleared after doc {doc_id}: "
            f"Allocated {mem_before:.2f}->{mem_after:.2f} GB, "
            f"Reserved {reserved_before:.2f}->{reserved_after:.2f} GB "
            f"(freed {freed:.2f} GB)"
        )
    except Exception as exc:
        logger.warning(f"Failed to clear GPU cache after doc {doc_id}: {exc}")


async def _finalize_local_rag_service(
    service: LocalRagService,
    *,
    clear_model_cache: bool = False,
) -> None:
    rag_instances = getattr(service, "_rag_instances", None)
    if isinstance(rag_instances, dict):
        for workspace_id in list(rag_instances.keys()):
            await _cleanup_rag_instance(service, workspace_id)

    if clear_model_cache:
        try:
            import raganything.services.local_rag as local_rag_module

            model_cache = getattr(local_rag_module, "_MODEL_CACHE", None)
            if isinstance(model_cache, dict):
                model_cache.clear()
        except Exception as exc:
            logger.warning(f"Failed to clear local model cache: {exc}")

    del service
    gc.collect()
    _clear_cuda_cache(-1)


async def _recycle_local_rag_service(
    service: LocalRagService,
    settings: LocalRagSettings,
    *,
    clear_model_cache: bool = False,
) -> LocalRagService:
    await _finalize_local_rag_service(
        service,
        clear_model_cache=clear_model_cache,
    )
    new_service = LocalRagService(settings)
    _ensure_master_log_handler()
    _bridge_lightrag_logs_to_run_file()
    return new_service


def _extract_reference_lines(raw_prompt: str) -> list[str]:
    """
    Extract lines under "Reference Document List" from raw prompt.
    """
    marker = (
        "Reference Document List (Each entry starts with a [reference_id] "
        "that corresponds to entries in the Document Chunks):"
    )
    marker_pos = raw_prompt.rfind(marker)
    if marker_pos < 0:
        return []

    # Parse only the fenced block after the final Reference Document List marker.
    tail = raw_prompt[marker_pos + len(marker) :]
    block_match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n```", tail, re.DOTALL)
    if not block_match:
        return []

    return [line.strip() for line in block_match.group(1).splitlines() if line.strip()]


def _dump_raw_prompt(
    doc_name: str,
    qa_idx: int,
    question: str,
    raw_prompt: str,
) -> Path:
    """
    Dump raw prompt for debugging reference construction.
    """
    doc_dump_dir = PROMPT_DUMP_DIR / f"docbench_{doc_name}"
    doc_dump_dir.mkdir(parents=True, exist_ok=True)
    out_file = doc_dump_dir / f"q{qa_idx + 1:03d}_raw_prompt.txt"

    ref_lines = _extract_reference_lines(raw_prompt)
    header_lines = [
        f"# doc_id: {doc_name}",
        f"# question_index: {qa_idx + 1}",
        f"# question: {question}",
        f"# reference_line_count: {len(ref_lines)}",
        "# reference_lines:",
    ]
    if ref_lines:
        header_lines.extend([f"#   {line}" for line in ref_lines])
    else:
        header_lines.append("#   <not found>")
    header_lines.extend(["", "===== RAW PROMPT START =====", ""])

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write(raw_prompt)

    return out_file


def _extract_reference_lines_from_messages(messages_payload: list[dict[str, Any]]) -> list[str]:
    """
    Extract reference lines from the final chat.completions messages payload.
    """
    user_text_parts: list[str] = []
    for msg in messages_payload:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        user_text_parts.append(str(text))
        elif isinstance(content, str):
            user_text_parts.append(content)

    if not user_text_parts:
        return []

    return _extract_reference_lines("\n".join(user_text_parts))


def _dump_final_messages(
    doc_name: str,
    qa_idx: int,
    question: str,
    captured_calls: list[dict[str, Any]],
) -> Path:
    """
    Dump captured chat.completions messages for this query.
    """
    doc_dump_dir = FINAL_MESSAGES_DUMP_DIR / f"docbench_{doc_name}"
    doc_dump_dir.mkdir(parents=True, exist_ok=True)
    out_file = doc_dump_dir / f"q{qa_idx + 1:03d}_final_messages.json"

    selected_messages = (
        captured_calls[-1].get("messages", [])
        if captured_calls
        else []
    )
    ref_lines = _extract_reference_lines_from_messages(selected_messages)
    payload = {
        "doc_id": doc_name,
        "question_index": qa_idx + 1,
        "question": question,
        "capture_count": len(captured_calls),
        "selected_client": captured_calls[-1].get("client") if captured_calls else None,
        "reference_line_count": len(ref_lines),
        "reference_lines": ref_lines,
        "messages": selected_messages,
        "captured_calls": captured_calls,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_file


def _set_doc_workspace(workspace: str) -> None:
    """
    Force per-document LightRAG namespace isolation.
    """
    os.environ["WORKSPACE"] = workspace
    try:
        from lightrag.kg.shared_storage import set_default_workspace

        set_default_workspace(workspace)
    except Exception as exc:
        logger.warning(f"Failed to set LightRAG workspace '{workspace}': {exc}")


def _bridge_lightrag_logs_to_run_file() -> None:
    """
    LightRAG uses its own logger with propagate=False and a console-only handler.
    Attach root file handlers so LightRAG warnings/info are persisted in run_*.log.
    """
    try:
        from lightrag.utils import logger as lightrag_logger
    except Exception as exc:
        logger.warning(f"Failed to import lightrag logger: {exc}")
        return

    root_logger = logging.getLogger()
    file_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    if not file_handlers:
        logger.warning("No root file handler found. LightRAG logs may be console-only.")
        return

    existing_files = {getattr(h, "baseFilename", None) for h in lightrag_logger.handlers}
    attached = 0
    for handler in file_handlers:
        file_path = getattr(handler, "baseFilename", None)
        if file_path and file_path not in existing_files:
            lightrag_logger.addHandler(handler)
            attached += 1

    lightrag_logger.setLevel(logging.INFO)
    logger.info(f"LightRAG log bridge ready (attached file handlers: {attached}).")


def _build_docbench_settings() -> LocalRagSettings:
    os.environ["ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE"] = "true"
    os.environ["CONTEXT_ZERO_WINDOW_CONTENT_TYPES"] = str(
        DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES
    )

    settings = LocalRagSettings.from_env()
    settings.working_dir_root = str(WORKING_DIR_ROOT)
    settings.output_dir = str(OUTPUT_MD_DIR)
    settings.log_dir = str(SCRIPT_DIR / "logs")
    settings.vllm_api_base = settings.vision_vllm_api_base = RAG_API_BASE
    settings.vllm_api_key = settings.vision_vllm_api_key = RAG_API_KEY
    settings.device = "cuda:0"
    settings.llm_model_name = settings.vision_model_name = RAG_MODEL_NAME
    settings.vision_model_path = RAG_VISION_MODEL_PATH
    settings.tokenizer_model_path = RAG_VISION_MODEL_PATH
    settings.image_token_estimate_method = "qwen_vl"
    settings.image_token_model_name_or_path = RAG_VISION_MODEL_PATH
    settings.image_wrapper_tokens_per_image = 2
    settings.temperature = 0.0
    settings.query_max_tokens = 2048
    settings.ingest_max_tokens = 8192
    settings.vlm_enable_json_schema = True
    settings.enable_entity_disambiguation = True
    settings.enable_synonym_linking = True
    settings.enable_multi_hop = False
    settings.enable_entity_surface_normalization = True
    settings.enable_keyword_case_normalization = True
    settings.strict_relation_endpoint_entity_match = True
    settings.qdrant_enable_sparse_bm25 = True
    settings.qdrant_retrieval_mode = "dense"
    return settings


def _build_raw_prompt_query_kwargs(
    query_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_query_params = dict(query_params or DOCBENCH_QUERY_PARAMS)
    raw_prompt_kwargs: dict[str, Any] = {
        "mode": effective_query_params["mode"],
        "top_k": effective_query_params["top_k"],
        "chunk_top_k": effective_query_params["chunk_top_k"],
        "rerank_score_scope": "all",
        "multimodal_top_k": effective_query_params.get("multimodal_top_k"),
        "image_token_estimate_method": effective_query_params[
            "image_token_estimate_method"
        ],
        "image_token_model_name_or_path": effective_query_params[
            "image_token_model_name_or_path"
        ],
        "image_wrapper_tokens_per_image": effective_query_params[
            "image_wrapper_tokens_per_image"
        ],
        "only_need_prompt": True,
    }
    passthrough_keys = (
        "max_total_tokens",
        "max_entity_tokens",
        "max_relation_tokens",
        "entity_qdrant_retrieval_mode",
        "chunk_qdrant_retrieval_mode",
        "exclude_synonym_edges",
        "answer_context_mode",
        "kg_chunk_selection_source",
        "enable_rerank",
        "enable_kg_rerank",
        "keyword_fanout_mode",
        "keyword_entity_rrf_k",
        "keyword_relation_rrf_k",
        "ppr_top_k",
        "ppr_qa_top_k",
        "recognition_top_k",
        "ppr_post_rerank_fusion",
        "ppr_post_rerank_rrf_k",
    )
    for key in passthrough_keys:
        if key in effective_query_params:
            raw_prompt_kwargs[key] = effective_query_params[key]
    return raw_prompt_kwargs


# ==========================================
# Step 1: Generate Answers
# ==========================================

async def generate_answers(
    start_id: int = 0,
    end_id: int = 229,
    resume: bool = True,
    dump_raw_prompt: bool = False,
    dump_final_messages: bool = False,
    max_async_generate: int = 1,
    max_async_docs: int | None = None,
    max_async_ingest_docs: int | None = None,
    max_async_query_docs: int | None = None,
    doc_flush_every: int = 4,
    one_sentence: bool = False,
    profile_name: str = "docbench_official",
    eval_prompt_filename: str = DOCBENCH_EVAL_PROMPT_FILENAME,
    query_mode: str | None = None,
    recognition_top_k: int = DEFAULT_RECOGNITION_TOP_K,
    keyword_fanout_mode: str = "joined",
    keyword_entity_rrf_k: int = 10,
    keyword_relation_rrf_k: int = 20,
    entity_retrieval_mode: str = "dense",
    chunk_retrieval_mode: str = "dense",
    exclude_synonym_edges: bool | None = None,
    answer_context_mode: str = "kg_prompt",
    kg_chunk_selection_source: str = "truncated",
    max_total_tokens: int | None = None,
    multimodal_top_k: int | None = None,
    enable_rerank: bool = True,
    enable_kg_rerank: bool = True,
    ppr_top_k: int | None = None,
    ppr_qa_top_k: int | None = None,
    ppr_post_rerank_fusion: str = "none",
    ppr_post_rerank_rrf_k: int = 60,
    bypass_query_cache: bool = False,
    bypass_keywords_cache: bool = False,
):
    """
    Generate system answers for DocBench.

    Prerequisite: Qwen3-VL-30B-A3B-Instruct-FP8 service is running on port 8001.

    Args:
        start_id: Start document ID (inclusive).
        end_id: End document ID (exclusive).
        resume: Skip already-processed documents.
        dump_raw_prompt: Dump per-question raw retrieval prompt for debugging.
        dump_final_messages: Dump final VLM messages for debugging.
        max_async_generate: Max concurrent questions per document.
        max_async_docs: Backward-compatible doc concurrency alias.
        max_async_ingest_docs: Max concurrent documents in ingest phase.
        max_async_query_docs: Max concurrent document workspaces in query phase.
        doc_flush_every: Recycle service every N docs; 0 disables recycle.
    """
    output_file = OUTPUT_DIR / "system_answers.jsonl"
    max_async_generate = _normalize_max_async(max_async_generate, default=1)
    if max_async_docs is not None:
        if max_async_ingest_docs is None:
            max_async_ingest_docs = max_async_docs
        if max_async_query_docs is None:
            max_async_query_docs = max_async_docs
    max_async_ingest_docs = _normalize_max_async(
        max_async_ingest_docs if max_async_ingest_docs is not None else 2,
        default=2,
    )
    max_async_query_docs = _normalize_max_async(
        max_async_query_docs if max_async_query_docs is not None else 1,
        default=1,
    )
    max_async_docs_config = (
        int(max_async_docs)
        if max_async_docs is not None
        else max(max_async_ingest_docs, max_async_query_docs)
    )
    doc_flush_every = _normalize_flush_every(doc_flush_every, default=4)
    if (dump_raw_prompt or dump_final_messages) and max_async_generate > 1:
        logger.warning(
            "dump_raw_prompt/dump_final_messages is enabled, force max_async_generate=1."
        )
        max_async_generate = 1
    if (dump_raw_prompt or dump_final_messages) and (
        max_async_ingest_docs > 1 or max_async_query_docs > 1
    ):
        logger.warning(
            "dump_raw_prompt/dump_final_messages is enabled, force doc concurrency=1."
        )
        max_async_ingest_docs = 1
        max_async_query_docs = 1

    query_params = _build_docbench_query_params(
        one_sentence=one_sentence,
        query_mode=query_mode,
        recognition_top_k=recognition_top_k,
        keyword_fanout_mode=keyword_fanout_mode,
        keyword_entity_rrf_k=keyword_entity_rrf_k,
        keyword_relation_rrf_k=keyword_relation_rrf_k,
        entity_retrieval_mode=entity_retrieval_mode,
        chunk_retrieval_mode=chunk_retrieval_mode,
        exclude_synonym_edges=exclude_synonym_edges,
        answer_context_mode=answer_context_mode,
        kg_chunk_selection_source=kg_chunk_selection_source,
        max_total_tokens=max_total_tokens,
        multimodal_top_k=multimodal_top_k,
        enable_rerank=enable_rerank,
        enable_kg_rerank=enable_kg_rerank,
        ppr_top_k=ppr_top_k,
        ppr_qa_top_k=ppr_qa_top_k,
        ppr_post_rerank_fusion=ppr_post_rerank_fusion,
        ppr_post_rerank_rrf_k=ppr_post_rerank_rrf_k,
        bypass_query_cache=bypass_query_cache,
        bypass_keywords_cache=bypass_keywords_cache,
    )

    settings = _build_docbench_settings()
    index_profile = _build_docbench_index_profile(settings)

    _save_generation_config(
        one_sentence=one_sentence,
        max_async_docs=max_async_docs_config,
        max_async_ingest_docs=max_async_ingest_docs,
        max_async_query_docs=max_async_query_docs,
        max_async_generate=max_async_generate,
        doc_flush_every=doc_flush_every,
        profile_name=profile_name,
        eval_prompt_filename=eval_prompt_filename,
        effective_query_params=query_params,
        index_profile=index_profile,
        start_id=start_id,
        end_id=end_id,
        resume=resume,
    )
    logger.info(f"Generation config saved: {GENERATION_CONFIG_FILE}")

    _ensure_master_log_handler()
    _bridge_lightrag_logs_to_run_file()
    
    # Load processed docs for resume mode.
    processed_docs = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_docs.add(data['doc_id'])
        logger.info(f"Resume: found {len(processed_docs)} processed documents")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Generating answers for documents {start_id} to {end_id-1}")
    logger.info(f"Output: {output_file}")
    logger.info(
        "max_async_ingest_docs=%d, max_async_query_docs=%d, "
        "max_async_generate=%d, doc_flush_every=%d",
        max_async_ingest_docs,
        max_async_query_docs,
        max_async_generate,
        doc_flush_every,
    )
    logger.info(f"Index profile: {index_profile}")
    logger.info(f"Query params: {query_params}")
    logger.info(f"{'='*80}\n")

    single_ingest_manifest = _load_single_ingest_manifest()
    doc_jobs: list[tuple[int, str, Path, Path]] = []
    for doc_id in range(start_id, end_id):
        doc_name = str(doc_id)

        if resume and doc_name in processed_docs:
            workspace_id = _workspace_id_for_doc(doc_name)
            _validate_resume_processed_doc_profile(
                manifest=single_ingest_manifest,
                doc_name=doc_name,
                workspace_id=workspace_id,
                index_profile=index_profile,
            )
            logger.info(f"[skip] [{doc_id}] Already processed")
            continue

        folder_path = DATA_ROOT / doc_name
        if not folder_path.exists():
            logger.warning(f"[{doc_id}] Folder not found: {folder_path}")
            continue

        pdf_file, qa_file = _find_doc_files(folder_path)
        if not pdf_file or not qa_file:
            logger.warning(f"[{doc_id}] Missing PDF or QA file")
            continue

        doc_jobs.append((doc_id, doc_name, pdf_file, qa_file))

    _validate_per_doc_workspace_env_isolation(
        [_workspace_id_for_doc(doc_name) for _, doc_name, _, _ in doc_jobs]
    )
    logger.info(f"Pending docs: {len(doc_jobs)}")

    manifest_lock = asyncio.Lock()
    ingested_docs: set[str] = set()
    ingested_docs_lock = asyncio.Lock()

    def _new_service(label: str) -> LocalRagService:
        service = LocalRagService(settings)
        logger.info("RAG service initialized for %s", label)
        _ensure_master_log_handler()
        _bridge_lightrag_logs_to_run_file()
        return service

    async def _ingest_single_doc(
        service: LocalRagService,
        doc_id: int,
        doc_name: str,
        pdf_file: Path,
    ) -> tuple[str, bool]:
        workspace_id = _workspace_id_for_doc(doc_name)

        logger.info(f"\n{'='*80}")
        logger.info(f"[{doc_id}/{end_id-1}] Ingesting: {pdf_file.name}")
        logger.info(f"{'='*80}")

        try:
            _ensure_docbench_workspace_index_profile(
                workspace_id=workspace_id,
                index_profile=index_profile,
            )

            if resume and _single_ingest_record_matches(
                manifest=single_ingest_manifest,
                doc_name=doc_name,
                workspace_id=workspace_id,
                index_profile=index_profile,
            ):
                logger.info("[%s] Single-doc ingest already done, skip", doc_name)
                return doc_name, True

            logger.info(f"Processing: {pdf_file.name} -> workspace_id: {workspace_id}")
            doc_output_dir = str(OUTPUT_MD_DIR / f"docbench_{doc_name}")

            logger.info("Ingesting document...")
            returned_workspace_id = await service.ingest(
                file_path=str(pdf_file),
                output_dir=doc_output_dir,
                workspace_id=workspace_id,
            )
            if settings.enable_synonym_linking:
                await service.finalize_workspace_synonyms(
                    workspace_id,
                    force=False,
                    reset_existing=True,
                )
            logger.info(f"Ingestion complete, workspace_id: {returned_workspace_id}")
            async with manifest_lock:
                _record_single_ingest_success(
                    manifest=single_ingest_manifest,
                    doc_name=doc_name,
                    workspace_id=workspace_id,
                    pdf_file=pdf_file,
                    index_profile=index_profile,
                )
            return doc_name, True
        except Exception as exc:
            logger.exception(f"[{doc_id}] Ingest error: {exc}")
            _append_single_ingest_failure(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "doc_id": doc_name,
                    "workspace_id": workspace_id,
                    "file_name": pdf_file.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "index_profile": dict(index_profile),
                }
            )
            return doc_name, False
        finally:
            await _cleanup_rag_instance(service, workspace_id)
            _clear_cuda_cache(doc_id)

    async def _query_single_doc(
        service: LocalRagService,
        doc_id: int,
        doc_name: str,
        qa_file: Path,
    ) -> tuple[str, list[dict[str, Any]]]:
        workspace_id = _workspace_id_for_doc(doc_name)

        logger.info(f"\n{'='*80}")
        logger.info(f"[{doc_id}/{end_id-1}] Querying: {workspace_id}")
        logger.info(f"{'='*80}")

        try:
            with open(qa_file, 'r', encoding='utf-8') as f_qa:
                qa_list = [json.loads(line) for line in f_qa]

            logger.info(
                f"\nAnswering {len(qa_list)} questions "
                f"(max_async_generate={max_async_generate})..."
            )
            question_semaphore = asyncio.Semaphore(max_async_generate)

            async def _answer_single_question(
                qa_idx: int, qa_item: dict[str, Any]
            ) -> tuple[int, dict[str, Any]]:
                question = qa_item["question"]
                answer = ""

                async with question_semaphore:
                    logger.info(
                        f"[{doc_name}][{qa_idx + 1}/{len(qa_list)}] {question[:60]}..."
                    )
                    try:
                        if dump_raw_prompt:
                            try:
                                from lightrag.base import QueryParam

                                rag = await service.get_rag(workspace_id)
                                raw_prompt_param = QueryParam(
                                    **_build_raw_prompt_query_kwargs(query_params)
                                )
                                raw_prompt_result = await rag.lightrag.aquery(
                                    question, param=raw_prompt_param
                                )
                                raw_prompt_text = (
                                    raw_prompt_result.content
                                    if hasattr(raw_prompt_result, "content")
                                    else str(raw_prompt_result)
                                )
                                dump_path = _dump_raw_prompt(
                                    doc_name, qa_idx, question, raw_prompt_text
                                )
                                ref_count = len(
                                    _extract_reference_lines(raw_prompt_text)
                                )
                                logger.info(
                                    f"      Prompt dumped: {dump_path} (reference lines: {ref_count})"
                                )
                            except Exception as dump_exc:
                                logger.warning(
                                    f"      Raw prompt dump failed: {dump_exc}"
                                )

                        captured_calls: list[dict[str, Any]] = []
                        patched_completions: list[tuple[Any, Any]] = []
                        if dump_final_messages:

                            def _patch_client(
                                client_name: str, client_obj: Any
                            ) -> None:
                                if client_obj is None:
                                    return
                                chat_obj = getattr(client_obj, "chat", None)
                                completions_api = getattr(chat_obj, "completions", None)
                                if completions_api is None:
                                    return
                                for api_obj, _ in patched_completions:
                                    if api_obj is completions_api:
                                        return

                                original_create = completions_api.create

                                async def _capture_create(
                                    *args,
                                    _orig=original_create,
                                    _client=client_name,
                                    **kwargs,
                                ):
                                    messages_payload = kwargs.get("messages")
                                    if isinstance(messages_payload, list):
                                        captured_calls.append(
                                            {
                                                "client": _client,
                                                "messages": messages_payload,
                                            }
                                        )
                                    return await _orig(*args, **kwargs)

                                completions_api.create = _capture_create
                                patched_completions.append(
                                    (completions_api, original_create)
                                )

                            _patch_client(
                                "vision", getattr(service, "vision_client", None)
                            )
                            _patch_client("text", getattr(service, "text_client", None))

                        try:
                            answer = await service.query(
                                workspace_id=workspace_id,
                                query=question,
                                **query_params,
                            )
                        finally:
                            if dump_final_messages:
                                for completions_api, original_create in patched_completions:
                                    completions_api.create = original_create

                        if dump_final_messages:
                            if captured_calls:
                                dump_path = _dump_final_messages(
                                    doc_name, qa_idx, question, captured_calls
                                )
                                final_messages = captured_calls[-1].get("messages", [])
                                ref_count = len(
                                    _extract_reference_lines_from_messages(
                                        final_messages
                                    )
                                )
                                logger.info(
                                    f"      Final messages dumped: {dump_path} (calls: {len(captured_calls)}, reference lines: {ref_count})"
                                )
                            else:
                                logger.warning(
                                    "      Final messages not captured for this query."
                                )

                        logger.info(f"      Answer: {answer[:80]}...")
                    except Exception as exc:
                        logger.error(f"      Error: {exc}")

                result = _build_generation_result(
                    doc_name=doc_name,
                    question=question,
                    answer=answer,
                    qa_item=qa_item,
                )
                return qa_idx, result

            answered_count = 0
            doc_records: list[dict[str, Any]] = []
            total_batches = max(
                1, (len(qa_list) + max_async_generate - 1) // max_async_generate
            )
            for batch_idx, qa_start in enumerate(
                range(0, len(qa_list), max_async_generate), start=1
            ):
                qa_batch = qa_list[qa_start : qa_start + max_async_generate]
                qa_tasks = [
                    asyncio.create_task(
                        _answer_single_question(qa_start + offset, qa_item)
                    )
                    for offset, qa_item in enumerate(qa_batch)
                ]
                qa_results = await asyncio.gather(*qa_tasks)
                ordered_batch = [
                    result for _, result in sorted(qa_results, key=lambda item: item[0])
                ]

                doc_records.extend(ordered_batch)
                answered_count += len(ordered_batch)
                logger.info(
                    f"[{doc_name}] Completed question batch {batch_idx}/{total_batches} "
                    f"({len(ordered_batch)} answers, total={answered_count})"
                )

                # Release allocator pressure during long query phase.
                gc.collect()
                _clear_cuda_cache(doc_id)

            logger.info(f"[{doc_id}] Completed: {answered_count} questions answered\n")
            return doc_name, doc_records
        except Exception as exc:
            logger.exception(f"[{doc_id}] Query error: {exc}")
            return doc_name, []
        finally:
            await _cleanup_rag_instance(service, workspace_id)
            _clear_cuda_cache(doc_id)

    # Phase A: all ingest first. Each worker owns one LocalRagService instance.
    logger.info(f"\n{'='*80}")
    logger.info("Phase A: ingest all pending documents")
    logger.info(f"{'='*80}")

    ingest_queue: asyncio.Queue[tuple[int, str, Path, Path]] = asyncio.Queue()
    for job in doc_jobs:
        ingest_queue.put_nowait(job)

    async def _ingest_worker(worker_id: int) -> None:
        service = _new_service(f"ingest-worker-{worker_id}")
        completed_since_flush = 0
        try:
            while True:
                try:
                    doc_id, doc_name, pdf_file, _ = ingest_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    completed_doc_name, ok = await _ingest_single_doc(
                        service, doc_id, doc_name, pdf_file
                    )
                    if ok:
                        async with ingested_docs_lock:
                            ingested_docs.add(completed_doc_name)
                        completed_since_flush += 1
                finally:
                    ingest_queue.task_done()

                if doc_flush_every > 0 and completed_since_flush >= doc_flush_every:
                    logger.info(
                        "[Ingest worker %d] Recycle LocalRagService after %d docs.",
                        worker_id,
                        completed_since_flush,
                    )
                    service = await _recycle_local_rag_service(
                        service,
                        settings,
                        clear_model_cache=False,
                    )
                    completed_since_flush = 0
        finally:
            await _finalize_local_rag_service(service, clear_model_cache=False)

    ingest_worker_count = min(max_async_ingest_docs, len(doc_jobs))
    if ingest_worker_count > 0:
        await asyncio.gather(
            *[
                asyncio.create_task(_ingest_worker(worker_id))
                for worker_id in range(ingest_worker_count)
            ]
        )
    logger.info(f"[Ingest] Completed docs: {len(ingested_docs)}/{len(doc_jobs)}")

    # Phase B: query after ingest. Default query doc concurrency is one, so only
    # questions from a single document graph run concurrently unless requested.
    query_jobs = [
        (doc_id, doc_name, qa_file)
        for doc_id, doc_name, _, qa_file in doc_jobs
        if doc_name in ingested_docs
    ]
    logger.info(f"\n{'='*80}")
    logger.info("Phase B: query all ingested documents")
    logger.info(f"{'='*80}")
    logger.info(f"Pending query docs: {len(query_jobs)}")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for batch_start in range(0, len(query_jobs), max_async_query_docs):
            batch = query_jobs[batch_start : batch_start + max_async_query_docs]
            logger.info(
                f"[Query] Running doc batch {batch_start + 1}-{batch_start + len(batch)}/{len(query_jobs)}"
            )
            services = [
                _new_service(f"query-doc-{doc_name}")
                for _, doc_name, _ in batch
            ]
            try:
                batch_results = await asyncio.gather(
                    *[
                        asyncio.create_task(
                            _query_single_doc(service, doc_id, doc_name, qa_file)
                        )
                        for service, (doc_id, doc_name, qa_file) in zip(services, batch)
                    ]
                )
            finally:
                await asyncio.gather(
                    *[
                        _finalize_local_rag_service(
                            service,
                            clear_model_cache=False,
                        )
                        for service in services
                    ],
                    return_exceptions=True,
                )

            for _, doc_records in batch_results:
                for record in doc_records:
                    _append_jsonl_record(f_out, record)
    
    logger.info(f"\n{'='*80}")
    logger.info("Answer generation complete!")
    logger.info(f"Saved to: {output_file}")
    logger.info(f"{'='*80}")


# ==========================================
# Step 2: Evaluate Answers
# ==========================================

async def evaluate_answers(
    resume: bool = True,
    max_async_judge: int = 4,
    eval_prompt_filename: str = DOCBENCH_EVAL_PROMPT_FILENAME,
):
    """
    Evaluate system answers with Qwen2.5-32B judge model.

    Prerequisite: Qwen2.5-32B service is running on port 8008.

    Args:
        resume: Skip already-evaluated records.
    """
    input_file = OUTPUT_DIR / "system_answers.jsonl"
    output_file = OUTPUT_DIR / "eval_results.jsonl"
    max_async_judge = _normalize_max_async(max_async_judge)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please run: python evaluate.py --mode generate first")
        return
    
    # Load evaluation prompt.
    try:
        eval_prompt = _load_eval_prompt(eval_prompt_filename)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return
    
    # Load generated system answers.
    with open(input_file, 'r', encoding='utf-8') as f:
        answers = [json.loads(line) for line in f]
    
    # Load evaluated keys for resume mode.
    evaluated_keys = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = f"{data['doc_id']}_{data['question']}"
                evaluated_keys.add(key)
        logger.info(f"Resume: found {len(evaluated_keys)} evaluated answers")
    
    # Initialize judge client.
    judge_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=JUDGE_API_BASE
    )
    
    pending_items: list[tuple[int, dict[str, Any]]] = []
    skipped = 0
    for i, item in enumerate(answers, 1):
        key = f"{item['doc_id']}_{item['question']}"
        if resume and key in evaluated_keys:
            skipped += 1
            continue
        pending_items.append((i, item))

    logger.info(f"\n{'='*80}")
    logger.info(
        f"Evaluating {len(pending_items)}/{len(answers)} answers using "
        f"{JUDGE_MODEL_NAME} (max_async_judge={max_async_judge})"
    )
    logger.info(f"Eval prompt: {eval_prompt_filename}")
    if skipped:
        logger.info(f"Skipped {skipped} already-evaluated answers")
    logger.info(f"{'='*80}\n")

    if not pending_items:
        logger.info("No pending answers to evaluate.")
        return

    semaphore = asyncio.Semaphore(max_async_judge)
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    done_count = 0
    total_pending = len(pending_items)

    async def _evaluate_one(i: int, item: dict[str, Any]) -> None:
        nonlocal done_count
        logger.info(f"\n[{i}/{len(answers)}] Doc {item['doc_id']}")
        logger.info(f"  Q: {item['question'][:60]}...")
        logger.info(f"  A: {item['sys_ans'][:60]}...")

        cur_prompt = _build_eval_prompt(eval_prompt, item)

        async with semaphore:
            try:
                response = await judge_client.chat.completions.create(
                    model=JUDGE_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful evaluator."},
                        {"role": "user", "content": cur_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=50
                )
                eval_result = response.choices[0].message.content.strip()
                score = _parse_eval_score(eval_result)
                logger.info(f"  Score: {score} | {eval_result[:40]}...")
                result = {
                    **item,
                    'eval': eval_result,
                    'score': score
                }
            except Exception as e:
                logger.error(f"  Error: {e}")
                result = {
                    **item,
                    'eval': f"[ERROR: {str(e)}]",
                    'score': 0
                }

        async with write_lock:
            _append_jsonl_record(f_out, result)

        async with progress_lock:
            done_count += 1
            if done_count == total_pending or done_count % max(1, total_pending // 10) == 0:
                logger.info(f"Progress: {done_count}/{total_pending}")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        tasks = [asyncio.create_task(_evaluate_one(i, item)) for i, item in pending_items]
        await asyncio.gather(*tasks)
    
    logger.info(f"\n{'='*80}")
    logger.info("Evaluation complete!")
    logger.info(f"Saved to: {output_file}")
    logger.info(f"{'='*80}")


# ==========================================
# Step 3: Calculate Statistics
# ==========================================

def _build_experiment_config(
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = _build_docbench_settings()
    effective_query_params = dict(DOCBENCH_QUERY_PARAMS)
    profile_name = "docbench_official"
    one_sentence = None
    max_async_docs = None
    max_async_ingest_docs = None
    max_async_query_docs = None
    max_async_generate = None
    doc_flush_every = None
    eval_prompt_filename = DOCBENCH_EVAL_PROMPT_FILENAME
    index_profile = None
    if generation_config:
        cfg_query_params = generation_config.get("effective_query_params")
        if isinstance(cfg_query_params, dict):
            effective_query_params = dict(cfg_query_params)
        if isinstance(generation_config.get("one_sentence"), bool):
            one_sentence = generation_config["one_sentence"]
        if isinstance(generation_config.get("max_async_docs"), int):
            max_async_docs = generation_config["max_async_docs"]
        if isinstance(generation_config.get("max_async_ingest_docs"), int):
            max_async_ingest_docs = generation_config["max_async_ingest_docs"]
        if isinstance(generation_config.get("max_async_query_docs"), int):
            max_async_query_docs = generation_config["max_async_query_docs"]
        if isinstance(generation_config.get("max_async_generate"), int):
            max_async_generate = generation_config["max_async_generate"]
        if isinstance(generation_config.get("doc_flush_every"), int):
            doc_flush_every = generation_config["doc_flush_every"]
        if isinstance(generation_config.get("profile_name"), str):
            profile_name = generation_config["profile_name"]
        if isinstance(generation_config.get("eval_prompt_filename"), str):
            eval_prompt_filename = generation_config["eval_prompt_filename"]
        if isinstance(generation_config.get("index_profile"), dict):
            index_profile = dict(generation_config["index_profile"])

    return {
        "rag_generation": {
            "api_base": settings.vision_vllm_api_base or settings.vllm_api_base,
            "model_name": settings.vision_model_name or settings.llm_model_name,
            "model_path": settings.vision_model_path,
            "api_key_is_empty": (
                (settings.vision_vllm_api_key or settings.vllm_api_key) == "EMPTY"
            ),
        },
        "judge_evaluation": {
            "api_base": JUDGE_API_BASE,
            "model_name": JUDGE_MODEL_NAME,
        },
        "generation_settings": {
            "device": settings.device,
            "temperature": settings.temperature,
            "query_max_tokens": settings.query_max_tokens,
            "ingest_max_tokens": settings.ingest_max_tokens,
            "vlm_enable_json_schema": settings.vlm_enable_json_schema,
            "tokenizer_model_path": settings.tokenizer_model_path,
            "image_token_estimate_method": settings.image_token_estimate_method,
            "image_token_model_name_or_path": settings.image_token_model_name_or_path,
            "image_wrapper_tokens_per_image": settings.image_wrapper_tokens_per_image,
        },
        "evaluation_profile": {
            "name": profile_name,
            "eval_prompt_file": eval_prompt_filename,
        },
        "one_sentence": one_sentence,
        "max_async_docs": max_async_docs,
        "max_async_ingest_docs": max_async_ingest_docs,
        "max_async_query_docs": max_async_query_docs,
        "max_async_generate": max_async_generate,
        "doc_flush_every": doc_flush_every,
        "query_params": effective_query_params,
        "index_profile": index_profile,
        "paths": {
            "script_dir": str(SCRIPT_DIR),
            "data_root": str(DATA_ROOT),
            "output_dir": str(OUTPUT_DIR),
            "working_dir_root": str(WORKING_DIR_ROOT),
            "output_md_dir": str(OUTPUT_MD_DIR),
        },
    }


TYPE_GROUP_ORDER = ("Txt.", "Mm.", "Una.")
TYPE_GROUP_MAPPING = {
    "text-only": "Txt.",
    "multimodal-f": "Mm.",
    "multimodal-t": "Mm.",
    "meta-data": "Mm.",
    "una-web": "Una.",
    "unanswerable": "Una.",
}


def _map_type_group(qtype: Any) -> str | None:
    normalized = str(qtype or "").strip().lower().replace("_", "-").replace(" ", "-")
    return TYPE_GROUP_MAPPING.get(normalized)


def calculate_statistics():
    """Calculate evaluation statistics."""
    result_file = OUTPUT_DIR / "eval_results.jsonl"
    
    if not result_file.exists():
        logger.error(f"Result file not found: {result_file}")
        logger.info("Please run: python evaluate.py --mode evaluate first")
        return
    
    # Load evaluation results.
    with open(result_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    generation_config = _load_generation_config()
    
    logger.info(f"\n{'='*80}")
    logger.info("DocBench Evaluation Statistics")
    logger.info(f"{'='*80}\n")
    
    # Overall accuracy.
    total = len(results)
    correct = sum(1 for r in results if r.get('score', 0) == 1)
    overall_acc = correct / total * 100 if total > 0 else 0
    
    logger.info(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    
    # Accuracy by question type.
    type_stats = {}
    for r in results:
        qtype = r['type']
        if qtype not in type_stats:
            type_stats[qtype] = {'correct': 0, 'total': 0}
        type_stats[qtype]['total'] += 1
        if r.get('score', 0) == 1:
            type_stats[qtype]['correct'] += 1
    
    logger.info("\nAccuracy by Question Type:")
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {qtype:20s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})")

    # Accuracy by type group (Txt. / Mm. / Una.).
    type_group_stats = {
        group: {"correct": 0, "total": 0}
        for group in TYPE_GROUP_ORDER
    }
    unknown_type_counts: dict[str, int] = {}
    for r in results:
        qtype_raw = r.get("type", "")
        group = _map_type_group(qtype_raw)
        if group is None:
            qtype_key = str(qtype_raw)
            unknown_type_counts[qtype_key] = unknown_type_counts.get(qtype_key, 0) + 1
            continue
        type_group_stats[group]["total"] += 1
        if r.get("score", 0) == 1:
            type_group_stats[group]["correct"] += 1

    logger.info("\nAccuracy by Type Group:")
    for group in TYPE_GROUP_ORDER:
        stats = type_group_stats[group]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        logger.info(
            f"  {group:20s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})"
        )
    if unknown_type_counts:
        logger.warning(f"Unknown type labels (not grouped): {unknown_type_counts}")
    
    # Domain-level statistics (DocBench official ranges).
    domain_ranges = {
        'Academic': range(0, 49),
        'Finance': range(49, 89),
        'Government': range(89, 133),
        'Law': range(133, 179),
        'News': range(179, 229)
    }
    
    domain_stats = {domain: {'correct': 0, 'total': 0} for domain in domain_ranges.keys()}
    for r in results:
        try:
            doc_num = int(r['doc_id'])
            for domain, id_range in domain_ranges.items():
                if doc_num in id_range:
                    domain_stats[domain]['total'] += 1
                    if r.get('score', 0) == 1:
                        domain_stats[domain]['correct'] += 1
                    break
        except Exception:
            continue
    
    logger.info("\nAccuracy by Domain:")
    for domain in ['Academic', 'Finance', 'Government', 'Law', 'News']:
        stats = domain_stats[domain]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            logger.info(f"  {domain:15s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})")
    
    # Save statistics to JSON.
    stats_output = {
        'experiment_config': _build_experiment_config(generation_config),
        'overall': {'accuracy': overall_acc, 'correct': correct, 'total': total},
        'by_type': {
            qtype: {
                'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for qtype, stats in type_stats.items()
        },
        'by_type_group': {
            group: {
                'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for group, stats in type_group_stats.items()
        },
        'by_domain': {
            domain: {
                'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for domain, stats in domain_stats.items() if stats['total'] > 0
        }
    }
    
    stats_file = OUTPUT_DIR / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nStatistics saved to: {stats_file}")
    logger.info(f"{'='*80}\n")


# ==========================================
# Main Entry Point
# ==========================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DocBench Evaluation Script (Manual Server Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --mode generate
  python evaluate.py --mode generate --start_id 0 --end_id 10
  python evaluate.py --mode evaluate
  python evaluate.py --mode stats
  nohup python evaluate.py --mode generate > run.log 2>&1 &
  tail -f run.log
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate", "evaluate", "stats"],
        help="Run mode.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Start document ID (inclusive).",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=229,
        help="End document ID (exclusive).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume mode.",
    )
    parser.add_argument(
        "--dump_raw_prompt",
        action="store_true",
        help="Dump raw retrieval prompt for each query.",
    )
    parser.add_argument(
        "--dump_final_messages",
        action="store_true",
        help="Dump final VLM messages for each query.",
    )
    parser.add_argument(
        "--raganything_eval_setup",
        action="store_true",
        help="Use RAG-Anything eval setup.",
    )
    parser.add_argument(
        "--max_async_judge",
        type=int,
        default=4,
        help="Max concurrent judge requests in evaluate mode.",
    )
    parser.add_argument(
        "--max_async_generate",
        type=int,
        default=1,
        help="Max concurrent question requests per document in generate mode.",
    )
    parser.add_argument(
        "--max_async_docs",
        type=int,
        default=None,
        help="Deprecated alias for both ingest/query document concurrency.",
    )
    parser.add_argument(
        "--max_async_ingest_docs",
        type=int,
        default=2,
        help="Max concurrent document graph builds in generate mode.",
    )
    parser.add_argument(
        "--max_async_query_docs",
        type=int,
        default=1,
        help="Max concurrent document workspaces in query phase.",
    )
    parser.add_argument(
        "--doc_flush_every",
        type=int,
        default=4,
        help="Recycle LocalRagService every N completed docs in generate mode; 0 disables.",
    )
    parser.add_argument(
        "--query_mode",
        choices=DOCBENCH_QUERY_MODE_CHOICES,
        default=DOCBENCH_QUERY_PARAMS["mode"],
    )
    parser.add_argument(
        "--entity_retrieval_mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
    )
    parser.add_argument(
        "--chunk_retrieval_mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
    )
    parser.add_argument(
        "--exclude_synonym_edges",
        choices=["true", "false", "none"],
        default="none",
        help=(
            "Query-time synonym-edge filtering. 'none' means auto: true for "
            "non-PPR, false for PPR."
        ),
    )
    parser.add_argument("--max_total_tokens", type=int, default=None)
    parser.add_argument(
        "--multimodal_top_k",
        type=int,
        default=DOCBENCH_QUERY_PARAMS["multimodal_top_k"],
    )
    parser.add_argument(
        "--enable_rerank",
        type=as_bool,
        default=DOCBENCH_QUERY_PARAMS["enable_rerank"],
    )
    parser.add_argument("--enable_kg_rerank", type=as_bool, default=True)
    parser.add_argument("--recognition_top_k", type=int, default=DEFAULT_RECOGNITION_TOP_K)
    parser.add_argument("--ppr_top_k", type=int, default=DEFAULT_PPR_TOP_K)
    parser.add_argument("--ppr_qa_top_k", type=int, default=DEFAULT_PPR_QA_TOP_K)
    parser.add_argument(
        "--ppr_post_rerank_fusion",
        choices=["none", "raw_rrf"],
        default="none",
    )
    parser.add_argument("--ppr_post_rerank_rrf_k", type=int, default=60)
    parser.add_argument(
        "--keyword_fanout_mode",
        choices=["joined", "per_keyword_rrf"],
        default="joined",
    )
    parser.add_argument(
        "--keyword_entity_rrf_k",
        "--keyword-entity-rrf-k",
        dest="keyword_entity_rrf_k",
        type=int,
        default=DOCBENCH_QUERY_PARAMS["keyword_entity_rrf_k"],
    )
    parser.add_argument(
        "--keyword_relation_rrf_k",
        "--keyword-relation-rrf-k",
        dest="keyword_relation_rrf_k",
        type=int,
        default=DOCBENCH_QUERY_PARAMS["keyword_relation_rrf_k"],
    )
    parser.add_argument(
        "--answer_context_mode",
        choices=["kg_prompt", "chunk_only_prompt"],
        default="kg_prompt",
    )
    parser.add_argument(
        "--kg_chunk_selection_source",
        choices=["truncated", "untruncated"],
        default="truncated",
    )
    parser.add_argument("--bypass_query_cache", action="store_true")
    parser.add_argument("--bypass_keywords_cache", action="store_true")

    args = parser.parse_args()
    
    profile_name, effective_one_sentence, eval_prompt_filename = _resolve_eval_setup(
        args.raganything_eval_setup
    )

    logger.info(f"\n{'='*80}")
    logger.info("DocBench Evaluation - Manual Server Mode")
    logger.info(f"{'='*80}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Range: {args.start_id} to {args.end_id-1}")
    logger.info(f"Resume: {not args.no_resume}")
    logger.info(f"DumpRawPrompt: {args.dump_raw_prompt}")
    logger.info(f"DumpFinalMessages: {args.dump_final_messages}")
    logger.info(f"EvalProfile: {profile_name}")
    logger.info(f"OneSentence: {effective_one_sentence}")
    logger.info(f"EvalPromptFile: {eval_prompt_filename}")
    logger.info(f"MaxAsyncJudge: {args.max_async_judge}")
    logger.info(f"MaxAsyncGenerate: {args.max_async_generate}")
    logger.info(f"MaxAsyncDocs: {args.max_async_docs}")
    logger.info(f"MaxAsyncIngestDocs: {args.max_async_ingest_docs}")
    logger.info(f"MaxAsyncQueryDocs: {args.max_async_query_docs}")
    logger.info(f"DocFlushEvery: {args.doc_flush_every}")
    logger.info(f"QueryMode: {args.query_mode}")
    logger.info(f"EntityRetrievalMode: {args.entity_retrieval_mode}")
    logger.info(f"ChunkRetrievalMode: {args.chunk_retrieval_mode}")
    logger.info(f"ExcludeSynonymEdges: {args.exclude_synonym_edges}")
    logger.info(f"{'='*80}\n")
    
    # Run selected mode.
    if args.mode == 'generate':
        logger.info("Please ensure Qwen3-VL-30B-A3B-Instruct-FP8 is running on port 8001")
        logger.info(f"   Check: curl http://localhost:8001/v1/models\n")
        await generate_answers(
            start_id=args.start_id,
            end_id=args.end_id,
            resume=not args.no_resume,
            dump_raw_prompt=args.dump_raw_prompt,
            dump_final_messages=args.dump_final_messages,
            max_async_generate=args.max_async_generate,
            max_async_docs=args.max_async_docs,
            max_async_ingest_docs=args.max_async_ingest_docs,
            max_async_query_docs=args.max_async_query_docs,
            doc_flush_every=args.doc_flush_every,
            one_sentence=effective_one_sentence,
            profile_name=profile_name,
            eval_prompt_filename=eval_prompt_filename,
            query_mode=args.query_mode,
            recognition_top_k=args.recognition_top_k,
            keyword_fanout_mode=args.keyword_fanout_mode,
            keyword_entity_rrf_k=args.keyword_entity_rrf_k,
            keyword_relation_rrf_k=args.keyword_relation_rrf_k,
            entity_retrieval_mode=args.entity_retrieval_mode,
            chunk_retrieval_mode=args.chunk_retrieval_mode,
            exclude_synonym_edges=(
                None
                if args.exclude_synonym_edges == "none"
                else args.exclude_synonym_edges == "true"
            ),
            answer_context_mode=args.answer_context_mode,
            kg_chunk_selection_source=args.kg_chunk_selection_source,
            max_total_tokens=args.max_total_tokens,
            multimodal_top_k=args.multimodal_top_k,
            enable_rerank=args.enable_rerank,
            enable_kg_rerank=args.enable_kg_rerank,
            ppr_top_k=args.ppr_top_k,
            ppr_qa_top_k=args.ppr_qa_top_k,
            ppr_post_rerank_fusion=args.ppr_post_rerank_fusion,
            ppr_post_rerank_rrf_k=args.ppr_post_rerank_rrf_k,
            bypass_query_cache=args.bypass_query_cache,
            bypass_keywords_cache=args.bypass_keywords_cache,
        )
    
    elif args.mode == 'evaluate':
        logger.info("Please ensure Qwen2.5-32B is running on port 8008")
        logger.info(f"   Check: curl http://localhost:8008/v1/models\n")
        await evaluate_answers(
            resume=not args.no_resume,
            max_async_judge=args.max_async_judge,
            eval_prompt_filename=eval_prompt_filename,
        )
    
    elif args.mode == 'stats':
        calculate_statistics()
    
    logger.info("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
