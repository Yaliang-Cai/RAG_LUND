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

3. Stop Qwen3-VL service, then start Qwen2.5-32B (port 8002)
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
from pathlib import Path
from typing import Any, TextIO
from openai import AsyncOpenAI

# Add parent directory to path to import local_rag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raganything.services.local_rag import LocalRagService, LocalRagSettings

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
JUDGE_API_BASE = "http://localhost:8002/v1"    # Qwen2.5-32B (evaluation)
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
    "vlm_enhanced": True,
    "multimodal_top_k": 5,
    "image_token_estimate_method": "qwen_vl",
    "image_token_model_name_or_path": RAG_VISION_MODEL_PATH,
    "image_wrapper_tokens_per_image": 2,
}

ONE_SENTENCE_USER_PROMPT = (
    "Provide the final answer in exactly one sentence. "
    "Do not include headings, bullet points, numbering, code blocks, or a "
    "references section."
)

_SENTENCE_END_RE = re.compile(r"[.!?。！？]")
_BINARY_SCORE_RE = re.compile(r"(?<!\d)([01])(?!\d)")


def _build_docbench_query_params(one_sentence: bool = False) -> dict[str, Any]:
    query_params = dict(DOCBENCH_QUERY_PARAMS)
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
    profile_name: str,
    eval_prompt_filename: str,
    effective_query_params: dict[str, Any],
    start_id: int,
    end_id: int,
    resume: bool,
) -> None:
    payload = {
        "profile_name": profile_name,
        "one_sentence": bool(one_sentence),
        "eval_prompt_filename": eval_prompt_filename,
        "effective_query_params": dict(effective_query_params),
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


def _coerce_one_sentence(text: str) -> str:
    """Best-effort one-sentence post-process for DocBench generation."""
    if not text:
        return ""

    cleaned = str(text)
    # Remove trailing references section if model still emits it.
    cleaned = re.split(r"(?im)^\s*#{1,6}\s*references\s*$", cleaned, maxsplit=1)[0]
    # Remove fenced code blocks and citation bullet lines.
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*[-*]\s*\[\d+\].*$", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    match = _SENTENCE_END_RE.search(cleaned)
    if match:
        return cleaned[: match.end()].strip()
    return cleaned


def _normalize_max_async(max_async: int, default: int = 4) -> int:
    """Clamp async worker count to a safe positive integer."""
    try:
        return max(1, int(max_async))
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
    head = (eval_result or "")[:20]
    match = _BINARY_SCORE_RE.search(head)
    if match:
        return int(match.group(1))
    return 0


async def _cleanup_rag_instance(service: LocalRagService, rag_doc_id: str) -> None:
    rag_instances = getattr(service, "_rag_instances", None)
    if not isinstance(rag_instances, dict):
        return

    rag_instance = rag_instances.get(rag_doc_id)
    if rag_instance is None:
        logger.info(f"No RAG instance to clean for {rag_doc_id}")
        return

    try:
        await rag_instance.finalize_storages()
        rag_instances.pop(rag_doc_id, None)
        logger.info(f"Cleaned up RAG instance for {rag_doc_id}")
    except Exception as exc:
        logger.warning(f"Failed to cleanup RAG instance {rag_doc_id}: {exc}")


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
            f"Allocated {mem_before:.2f}→{mem_after:.2f} GB, "
            f"Reserved {reserved_before:.2f}→{reserved_after:.2f} GB "
            f"(freed {freed:.2f} GB)"
        )
    except Exception as exc:
        logger.warning(f"Failed to clear GPU cache after doc {doc_id}: {exc}")


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
    return settings


def _build_raw_prompt_query_kwargs() -> dict[str, Any]:
    raw_prompt_kwargs: dict[str, Any] = {
        "mode": DOCBENCH_QUERY_PARAMS["mode"],
        "top_k": DOCBENCH_QUERY_PARAMS["top_k"],
        "chunk_top_k": DOCBENCH_QUERY_PARAMS["chunk_top_k"],
        "multimodal_top_k": DOCBENCH_QUERY_PARAMS.get("multimodal_top_k"),
        "image_token_estimate_method": DOCBENCH_QUERY_PARAMS[
            "image_token_estimate_method"
        ],
        "image_token_model_name_or_path": DOCBENCH_QUERY_PARAMS[
            "image_token_model_name_or_path"
        ],
        "image_wrapper_tokens_per_image": DOCBENCH_QUERY_PARAMS[
            "image_wrapper_tokens_per_image"
        ],
        "only_need_prompt": True,
    }
    for key in ("max_total_tokens", "max_entity_tokens", "max_relation_tokens"):
        if key in DOCBENCH_QUERY_PARAMS:
            raw_prompt_kwargs[key] = DOCBENCH_QUERY_PARAMS[key]
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
    one_sentence: bool = False,
    profile_name: str = "docbench_official",
    eval_prompt_filename: str = DOCBENCH_EVAL_PROMPT_FILENAME,
):
    """
    为 DocBench 生成系统答案
    
    前提：Qwen3-VL-30B-A3B-Instruct-FP8 服务已在 8001 端口运行
    
    Args:
        start_id: 起始文档 ID
        end_id: 结束文档 ID（不包含）
        resume: 是否跳过已处理的文档
        dump_raw_prompt: 是否落盘每题 raw_prompt（用于调试引用列表）
        dump_final_messages: 是否落盘最终发送给 VLM 的 messages（用于检查引用列表）
    """
    output_file = OUTPUT_DIR / "system_answers.jsonl"
    query_params = _build_docbench_query_params(one_sentence=one_sentence)
    _save_generation_config(
        one_sentence=one_sentence,
        profile_name=profile_name,
        eval_prompt_filename=eval_prompt_filename,
        effective_query_params=query_params,
        start_id=start_id,
        end_id=end_id,
        resume=resume,
    )
    logger.info(f"🧾 Generation config saved: {GENERATION_CONFIG_FILE}")
    
    # 显式配置路径，保证评测时每个文档目录和日志目录可控
    settings = _build_docbench_settings()
    
    # 只创建一次 service
    service = LocalRagService(settings)
    _bridge_lightrag_logs_to_run_file()
    logger.info(f"✅ RAG Service initialized")
    
    # 加载已处理的文档（用于断点续传）
    processed_docs = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_docs.add(data['doc_id'])
        logger.info(f"📁 Resume: found {len(processed_docs)} processed documents")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📝 Generating answers for documents {start_id} to {end_id-1}")
    logger.info(f"📂 Output: {output_file}")
    logger.info(f"{'='*80}\n")
    
    # 遍历每个文档
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for doc_id in range(start_id, end_id):
            doc_name = str(doc_id)
            
            # 跳过已处理
            if resume and doc_name in processed_docs:
                logger.info(f"⏭️  [{doc_id}] Already processed, skipping")
                continue
            
            # 检查文件夹是否存在
            folder_path = DATA_ROOT / doc_name
            if not folder_path.exists():
                logger.warning(f"⚠️  [{doc_id}] Folder not found: {folder_path}")
                continue
            
            # 查找 PDF 和 QA 文件
            pdf_file, qa_file = _find_doc_files(folder_path)
            
            if not pdf_file or not qa_file:
                logger.warning(f"⚠️  [{doc_id}] Missing PDF or QA file")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 [{doc_id}/{end_id-1}] Processing: {pdf_file.name}")
            logger.info(f"{'='*80}")
            rag_doc_id = f"docbench_{doc_name}"

            try:
                # 每个文档独立的 ID
                _set_doc_workspace(rag_doc_id)
                logger.info(f"🔧 Processing: {pdf_file.name} → doc_id: {rag_doc_id}")
                
                # MinerU 独立输出目录（避免 PDF 重名冲突）
                doc_output_dir = str(OUTPUT_MD_DIR / f"docbench_{doc_name}")
                
                # Ingest 文档
                logger.info(f"📖 Ingesting document...")
                returned_doc_id = await service.ingest(
                    file_path=str(pdf_file),
                    output_dir=doc_output_dir,  # 每个文档独立 MinerU 输出
                    doc_id=rag_doc_id
                )
                logger.info(f"✅ Ingestion complete, doc_id: {returned_doc_id}")
                
                # 加载问题
                with open(qa_file, 'r', encoding='utf-8') as f_qa:
                    qa_list = [json.loads(line) for line in f_qa]
                
                logger.info(f"\n❓ Answering {len(qa_list)} questions...")
                
                # 回答每个问题
                for qa_idx, qa_item in enumerate(qa_list):
                    question = qa_item['question']
                    logger.info(f"  [{qa_idx+1}/{len(qa_list)}] {question[:60]}...")
                    answer = ""

                    try:
                        if dump_raw_prompt:
                            try:
                                from lightrag.base import QueryParam

                                rag = await service.get_rag(rag_doc_id)
                                raw_prompt_param = QueryParam(
                                    **_build_raw_prompt_query_kwargs()
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
                                ref_count = len(_extract_reference_lines(raw_prompt_text))
                                logger.info(
                                    f"      🔎 Prompt dumped: {dump_path} (reference lines: {ref_count})"
                                )
                            except Exception as dump_exc:
                                logger.warning(f"      ⚠️ Raw prompt dump failed: {dump_exc}")

                        captured_calls: list[dict[str, Any]] = []
                        patched_completions: list[tuple[Any, Any]] = []
                        if dump_final_messages:
                            def _patch_client(client_name: str, client_obj: Any) -> None:
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
                                patched_completions.append((completions_api, original_create))

                            _patch_client("vision", getattr(service, "vision_client", None))
                            _patch_client("text", getattr(service, "text_client", None))

                        # 使用 RAG 查询
                        try:
                            answer = await service.query(
                                doc_id=rag_doc_id,
                                query=question,
                                **query_params
                            )
                        finally:
                            if dump_final_messages:
                                for completions_api, original_create in patched_completions:
                                    completions_api.create = original_create

                        if one_sentence:
                            one_sentence_answer = _coerce_one_sentence(answer)
                            if one_sentence_answer != answer:
                                logger.info("      One-sentence postprocess applied.")
                            answer = one_sentence_answer

                        if dump_final_messages:
                            if captured_calls:
                                dump_path = _dump_final_messages(
                                    doc_name, qa_idx, question, captured_calls
                                )
                                final_messages = captured_calls[-1].get("messages", [])
                                ref_count = len(
                                    _extract_reference_lines_from_messages(final_messages)
                                )
                                logger.info(
                                    f"      🧾 Final messages dumped: {dump_path} (calls: {len(captured_calls)}, reference lines: {ref_count})"
                                )
                            else:
                                logger.warning(
                                    "      ⚠️ Final messages not captured for this query."
                                )
                        
                        logger.info(f"      ✓ Answer: {answer[:80]}...")
                    except Exception as exc:
                        logger.error(f"      ✗ Error: {exc}")

                    result = _build_generation_result(
                        doc_name=doc_name,
                        question=question,
                        answer=answer,
                        qa_item=qa_item,
                    )
                    _append_jsonl_record(f_out, result)
                
                logger.info(f"✅ [{doc_id}] Completed: {len(qa_list)} questions answered\n")
            except Exception as exc:
                logger.exception(f"❌ [{doc_id}] Error: {exc}")
            finally:
                await _cleanup_rag_instance(service, rag_doc_id)
                _clear_cuda_cache(doc_id)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Answer generation complete!")
    logger.info(f"📁 Saved to: {output_file}")
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
    使用 Qwen2.5-32B 评估系统答案
    
    前提：Qwen2.5-32B 服务已在 8002 端口运行
    
    Args:
        resume: 是否跳过已评估的答案
    """
    input_file = OUTPUT_DIR / "system_answers.jsonl"
    output_file = OUTPUT_DIR / "eval_results.jsonl"
    max_async_judge = _normalize_max_async(max_async_judge)
    
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        logger.info("💡 Please run: python evaluate.py --mode generate first")
        return
    
    # 加载评估 prompt
    try:
        eval_prompt = _load_eval_prompt(eval_prompt_filename)
    except FileNotFoundError as exc:
        logger.error(f"❌ {exc}")
        return
    
    # 加载系统答案
    with open(input_file, 'r', encoding='utf-8') as f:
        answers = [json.loads(line) for line in f]
    
    # 加载已评估的答案（用于断点续传）
    evaluated_keys = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = f"{data['doc_id']}_{data['question']}"
                evaluated_keys.add(key)
        logger.info(f"📁 Resume: found {len(evaluated_keys)} evaluated answers")
    
    # 初始化 Judge 客户端
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
        f"⚖️  Evaluating {len(pending_items)}/{len(answers)} answers using "
        f"{JUDGE_MODEL_NAME} (max_async_judge={max_async_judge})"
    )
    logger.info(f"🧾 Eval prompt: {eval_prompt_filename}")
    if skipped:
        logger.info(f"⏭️  Skipped {skipped} already-evaluated answers")
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
                logger.info(f"  ⚖️  Score: {score} | {eval_result[:40]}...")
                result = {
                    **item,
                    'eval': eval_result,
                    'score': score
                }
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
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
    logger.info(f"✅ Evaluation complete!")
    logger.info(f"📁 Saved to: {output_file}")
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
    eval_prompt_filename = DOCBENCH_EVAL_PROMPT_FILENAME
    if generation_config:
        cfg_query_params = generation_config.get("effective_query_params")
        if isinstance(cfg_query_params, dict):
            effective_query_params = dict(cfg_query_params)
        if isinstance(generation_config.get("one_sentence"), bool):
            one_sentence = generation_config["one_sentence"]
        if isinstance(generation_config.get("profile_name"), str):
            profile_name = generation_config["profile_name"]
        if isinstance(generation_config.get("eval_prompt_filename"), str):
            eval_prompt_filename = generation_config["eval_prompt_filename"]

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
        "query_params": effective_query_params,
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
    """计算评估统计数据"""
    result_file = OUTPUT_DIR / "eval_results.jsonl"
    
    if not result_file.exists():
        logger.error(f"❌ Result file not found: {result_file}")
        logger.info("💡 Please run: python evaluate.py --mode evaluate first")
        return
    
    # 加载评估结果
    with open(result_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    generation_config = _load_generation_config()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 DocBench Evaluation Statistics")
    logger.info(f"{'='*80}\n")
    
    # 总体准确率
    total = len(results)
    correct = sum(1 for r in results if r.get('score', 0) == 1)
    overall_acc = correct / total * 100 if total > 0 else 0
    
    logger.info(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    
    # 按问题类型统计
    type_stats = {}
    for r in results:
        qtype = r['type']
        if qtype not in type_stats:
            type_stats[qtype] = {'correct': 0, 'total': 0}
        type_stats[qtype]['total'] += 1
        if r.get('score', 0) == 1:
            type_stats[qtype]['correct'] += 1
    
    logger.info(f"\n📋 Accuracy by Question Type:")
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  {qtype:20s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})")

    # 按归并类型统计（Txt. / Mm. / Una.）
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

    logger.info(f"\n🧩 Accuracy by Type Group:")
    for group in TYPE_GROUP_ORDER:
        stats = type_group_stats[group]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        logger.info(
            f"  {group:20s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})"
        )
    if unknown_type_counts:
        logger.warning(f"Unknown type labels (not grouped): {unknown_type_counts}")
    
    # 按领域统计（参考 DocBench 官方分布）
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
    
    logger.info(f"\n🌐 Accuracy by Domain:")
    for domain in ['Academic', 'Finance', 'Government', 'Law', 'News']:
        stats = domain_stats[domain]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            logger.info(f"  {domain:15s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})")
    
    # 保存统计到 JSON
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
    
    logger.info(f"\n📁 Statistics saved to: {stats_file}")
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
  # 生成所有文档的答案
  python evaluate.py --mode generate
  
  # 生成部分文档的答案
  python evaluate.py --mode generate --start_id 0 --end_id 10
  
  # 评估所有答案
  python evaluate.py --mode evaluate
  
  # 查看统计
  python evaluate.py --mode stats
  
  # 后台运行（推荐）
  nohup python evaluate.py --mode generate > run.log 2>&1 &
  tail -f run.log
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['generate', 'evaluate', 'stats'],
        help='运行模式'
    )
    parser.add_argument(
        '--start_id',
        type=int,
        default=0,
        help='起始文档 ID'
    )
    parser.add_argument(
        '--end_id',
        type=int,
        default=229,
        help='结束文档 ID（不包含）'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='不恢复之前的进度，从头开始'
    )
    parser.add_argument(
        '--dump_raw_prompt',
        action='store_true',
        help='生成模式下落盘每题 raw_prompt（用于检查 Reference Document List）'
    )
    parser.add_argument(
        '--dump_final_messages',
        action='store_true',
        help='生成模式下落盘最终发给 VLM 的 messages（用于检查真实输入）'
    )
    
    parser.add_argument(
        '--raganything_eval_setup',
        action='store_true',
        help='Use RAG-Anything eval setup: one-sentence generation + evaluation_prompt_RAG-Anything.txt.'
    )
    parser.add_argument(
        '--max_async_judge',
        type=int,
        default=4,
        help='Max concurrent judge requests in evaluate mode (default: 4).'
    )

    args = parser.parse_args()
    
    profile_name, effective_one_sentence, eval_prompt_filename = _resolve_eval_setup(
        args.raganything_eval_setup
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 DocBench Evaluation - Manual Server Mode")
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
    logger.info(f"{'='*80}\n")
    
    # 执行对应的步骤
    if args.mode == 'generate':
        logger.info("⚠️  Please ensure Qwen3-VL-30B-A3B-Instruct-FP8 is running on port 8001")
        logger.info(f"   Check: curl http://localhost:8001/v1/models\n")
        await generate_answers(
            start_id=args.start_id,
            end_id=args.end_id,
            resume=not args.no_resume,
            dump_raw_prompt=args.dump_raw_prompt,
            dump_final_messages=args.dump_final_messages,
            one_sentence=effective_one_sentence,
            profile_name=profile_name,
            eval_prompt_filename=eval_prompt_filename,
        )
    
    elif args.mode == 'evaluate':
        logger.info("⚠️  Please ensure Qwen2.5-32B is running on port 8002")
        logger.info(f"   Check: curl http://localhost:8002/v1/models\n")
        await evaluate_answers(
            resume=not args.no_resume,
            max_async_judge=args.max_async_judge,
            eval_prompt_filename=eval_prompt_filename,
        )
    
    elif args.mode == 'stats':
        calculate_statistics()
    
    logger.info(f"\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
