#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocBench Shared-Storage Evaluation Script
=========================================

Use one shared RAG workspace/doc_id for a document range (default: 0-48),
then evaluate generated answers.
"""

import os
import sys
import json
import re
import asyncio
import logging
import gc
from pathlib import Path
from typing import Any, TextIO

from openai import AsyncOpenAI

# Keep MinerU memory usage aligned with evaluate.py
os.environ.setdefault("MINERU_VLLM_GPU_MEMORY_UTILIZATION", "0.1")

# Add project root to import local_rag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raganything.services.local_rag import LocalRagService, LocalRagSettings


SCRIPT_DIR = Path("/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/DocBench")
DATA_ROOT = Path("/data/y50056788/Yaliang/datasets_for_eval/data_for_DocBench")

OUTPUT_DIR = SCRIPT_DIR / "docbench_shared_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORKING_DIR_ROOT = OUTPUT_DIR / "rag_workspaces"
WORKING_DIR_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_MD_DIR = OUTPUT_DIR / "mineru_outputs"
OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_ANSWERS_FILE = OUTPUT_DIR / "system_answers.jsonl"
EVAL_RESULTS_FILE = OUTPUT_DIR / "eval_results.jsonl"
STATS_FILE = OUTPUT_DIR / "statistics.json"
GENERATION_CONFIG_FILE = OUTPUT_DIR / "generation_config.json"
INGEST_MANIFEST_FILE = OUTPUT_DIR / "shared_ingest_manifest.json"

RAG_API_BASE = "http://localhost:8001/v1"
JUDGE_API_BASE = "http://localhost:8002/v1"
RAG_API_KEY = "EMPTY"
RAG_VISION_MODEL_PATH = "/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8"
RAG_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

DOCBENCH_EVAL_PROMPT_FILENAME = "evaluation_prompt.txt"
RAGANYTHING_EVAL_PROMPT_FILENAME = "evaluation_prompt_RAG-Anything.txt"
DISABLE_LOCATOR_CLEANUP_ENV = "LIGHTRAG_DISABLE_LOCATOR_ENTITY_CLEANUP"

DEFAULT_SHARED_DOC_ID = "docbench_shared_0_48"

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

_BINARY_SCORE_RE = re.compile(r"(?<!\d)([01])(?!\d)")
_ACCURACY_FIELD_RE = re.compile(r'"accuracy"\s*:\s*([01])', flags=re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logging.getLogger("raganything").setLevel(logging.INFO)
logging.getLogger("raganything.processor").setLevel(logging.INFO)
logging.getLogger("raganything.parser").setLevel(logging.INFO)


def _bridge_lightrag_logs_to_run_file() -> None:
    """
    Attach root file handlers to LightRAG logger so all LightRAG logs
    are persisted in the same run log file (consistent with evaluate.py).
    """
    try:
        from lightrag.utils import logger as lightrag_logger
    except Exception as exc:
        logger.warning("Failed to import lightrag logger: %s", exc)
        return

    root_logger = logging.getLogger()
    file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
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
    logger.info("LightRAG log bridge ready (attached file handlers: %d).", attached)


def _normalize_max_async(max_async: int, default: int = 4) -> int:
    try:
        return max(1, int(max_async))
    except Exception:
        return default


def _normalize_flush_every(flush_every: int, default: int = 6) -> int:
    try:
        value = int(flush_every)
        return value if value >= 0 else default
    except Exception:
        return default


def _resolve_eval_setup(use_raganything_eval_setup: bool) -> tuple[str, bool, str]:
    if use_raganything_eval_setup:
        return ("rag_anything", True, RAGANYTHING_EVAL_PROMPT_FILENAME)
    return ("docbench_official", False, DOCBENCH_EVAL_PROMPT_FILENAME)


def _build_query_params(one_sentence: bool = False) -> dict[str, Any]:
    params = dict(DOCBENCH_QUERY_PARAMS)
    if one_sentence:
        params["user_prompt"] = ONE_SENTENCE_USER_PROMPT
        params["response_type"] = "Single Sentence"
    return params


def _append_jsonl_record(file_obj: TextIO, payload: dict[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file_obj.flush()


def _find_doc_files(folder_path: Path) -> tuple[Path | None, Path | None]:
    pdf_file = None
    qa_file = None
    for file in folder_path.iterdir():
        if file.suffix.lower() == ".pdf":
            pdf_file = file
        elif file.name.endswith("_qa.jsonl"):
            qa_file = file
    return pdf_file, qa_file


def _load_eval_prompt(eval_prompt_filename: str) -> str:
    prompt_path = SCRIPT_DIR / eval_prompt_filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Evaluation prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


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

    accuracy_match = _ACCURACY_FIELD_RE.search(text)
    if accuracy_match:
        return int(accuracy_match.group(1))

    head = text[:120]
    match = _BINARY_SCORE_RE.search(head)
    if match:
        return int(match.group(1))
    return 0


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
    return None


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_ingest_manifest() -> dict[str, Any]:
    payload = _load_json(INGEST_MANIFEST_FILE)
    if isinstance(payload, dict):
        return payload
    return {"shared_doc_id": "", "ingested_doc_ids": []}


def _save_ingest_manifest(shared_doc_id: str, ingested_doc_ids: set[str]) -> None:
    _save_json(
        INGEST_MANIFEST_FILE,
        {
            "shared_doc_id": shared_doc_id,
            "ingested_doc_ids": sorted(ingested_doc_ids, key=lambda x: int(x)),
        },
    )


def _build_shared_settings() -> LocalRagSettings:
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


async def _cleanup_rag_instance(service: LocalRagService, rag_doc_id: str) -> None:
    rag_instances = getattr(service, "_rag_instances", None)
    if not isinstance(rag_instances, dict):
        return
    rag_instance = rag_instances.get(rag_doc_id)
    if rag_instance is None:
        return
    try:
        await rag_instance.finalize_storages()
        rag_instances.pop(rag_doc_id, None)
    except Exception as exc:
        logger.warning("Failed to cleanup RAG instance %s: %s", rag_doc_id, exc)


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Failed to clear GPU cache: %s", exc)


def _clear_local_model_cache() -> None:
    try:
        import raganything.services.local_rag as local_rag_module

        model_cache = getattr(local_rag_module, "_MODEL_CACHE", None)
        if isinstance(model_cache, dict):
            model_cache.clear()
    except Exception as exc:
        logger.warning("Failed to clear local model cache: %s", exc)


async def _recycle_local_rag_service(
    service: LocalRagService,
    settings: LocalRagSettings,
    shared_doc_id: str,
    *,
    clear_model_cache: bool = True,
) -> LocalRagService:
    await _cleanup_rag_instance(service, shared_doc_id)
    if clear_model_cache:
        _clear_local_model_cache()
    del service
    gc.collect()
    _clear_cuda_cache()
    return LocalRagService(settings)


def _save_generation_config(
    *,
    profile_name: str,
    eval_prompt_filename: str,
    one_sentence: bool,
    max_async_generate: int,
    ingest_flush_every: int,
    shared_doc_id: str,
    disable_locator_cleanup: bool,
    start_id: int,
    end_id: int,
    query_params: dict[str, Any],
) -> None:
    _save_json(
        GENERATION_CONFIG_FILE,
        {
            "profile_name": profile_name,
            "eval_prompt_filename": eval_prompt_filename,
            "one_sentence": bool(one_sentence),
            "max_async_generate": int(max_async_generate),
            "ingest_flush_every": int(ingest_flush_every),
            "shared_doc_id": shared_doc_id,
            "disable_locator_cleanup": bool(disable_locator_cleanup),
            "start_id": int(start_id),
            "end_id": int(end_id),
            "effective_query_params": dict(query_params),
        },
    )


async def generate_answers_shared(
    *,
    start_id: int,
    end_id: int,
    resume: bool,
    max_async_generate: int,
    ingest_flush_every: int,
    one_sentence: bool,
    profile_name: str,
    eval_prompt_filename: str,
    shared_doc_id: str,
    disable_locator_cleanup: bool,
) -> None:
    max_async_generate = _normalize_max_async(max_async_generate, default=1)
    ingest_flush_every = _normalize_flush_every(ingest_flush_every, default=6)
    query_params = _build_query_params(one_sentence=one_sentence)
    _save_generation_config(
        profile_name=profile_name,
        eval_prompt_filename=eval_prompt_filename,
        one_sentence=one_sentence,
        max_async_generate=max_async_generate,
        ingest_flush_every=ingest_flush_every,
        shared_doc_id=shared_doc_id,
        disable_locator_cleanup=disable_locator_cleanup,
        start_id=start_id,
        end_id=end_id,
        query_params=query_params,
    )

    os.environ[DISABLE_LOCATOR_CLEANUP_ENV] = "1" if disable_locator_cleanup else "0"
    logger.info(
        "Shared extract cleanup flag: %s=%s",
        DISABLE_LOCATOR_CLEANUP_ENV,
        os.environ[DISABLE_LOCATOR_CLEANUP_ENV],
    )

    settings = _build_shared_settings()
    service = LocalRagService(settings)
    _bridge_lightrag_logs_to_run_file()

    processed_keys: set[str] = set()
    if resume and SYSTEM_ANSWERS_FILE.exists():
        with open(SYSTEM_ANSWERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                key = f"{item['doc_id']}_{item['question']}"
                processed_keys.add(key)
        logger.info("Resume: %d answers already generated.", len(processed_keys))

    manifest = _load_ingest_manifest() if resume else {"shared_doc_id": "", "ingested_doc_ids": []}
    ingested_doc_ids = set(str(x) for x in manifest.get("ingested_doc_ids", []))
    if manifest.get("shared_doc_id") != shared_doc_id and ingested_doc_ids:
        logger.warning(
            "Manifest shared_doc_id mismatch (%s != %s). Reset ingest manifest.",
            manifest.get("shared_doc_id"),
            shared_doc_id,
        )
        ingested_doc_ids = set()

    logger.info("Shared doc_id: %s", shared_doc_id)
    logger.info("Generate range: %d-%d", start_id, end_id - 1)
    logger.info("Ingest flush every: %d (0 = disabled)", ingest_flush_every)

    # Phase 1: build/update shared storage
    ingested_since_flush = 0
    for doc_id in range(start_id, end_id):
        doc_name = str(doc_id)
        folder_path = DATA_ROOT / doc_name
        if not folder_path.exists():
            logger.warning("[%s] Folder not found: %s", doc_name, folder_path)
            continue
        pdf_file, qa_file = _find_doc_files(folder_path)
        if not pdf_file or not qa_file:
            logger.warning("[%s] Missing PDF or QA file", doc_name)
            continue
        if resume and doc_name in ingested_doc_ids:
            logger.info("[%s] Shared ingest already done, skip", doc_name)
            continue

        logger.info("[%s] Ingest into shared storage: %s", doc_name, pdf_file.name)
        doc_output_dir = str(OUTPUT_MD_DIR / f"docbench_{doc_name}")
        await service.ingest(
            file_path=str(pdf_file),
            output_dir=doc_output_dir,
            doc_id=shared_doc_id,
        )
        ingested_doc_ids.add(doc_name)
        ingested_since_flush += 1
        _save_ingest_manifest(shared_doc_id, ingested_doc_ids)
        if ingest_flush_every > 0 and ingested_since_flush >= ingest_flush_every:
            logger.info(
                "Recycling LocalRagService after %d ingested docs to control GPU cache.",
                ingested_since_flush,
            )
            service = await _recycle_local_rag_service(
                service,
                settings,
                shared_doc_id,
                clear_model_cache=False,
            )
            _bridge_lightrag_logs_to_run_file()
            ingested_since_flush = 0

    # Ensure ingest-phase temporary memory is released before query phase.
    service = await _recycle_local_rag_service(
        service,
        settings,
        shared_doc_id,
        clear_model_cache=False,
    )
    _bridge_lightrag_logs_to_run_file()

    # Phase 2: question answering against shared storage (global pending pool)
    pending_questions: list[dict[str, Any]] = []
    order_idx = 0
    for doc_id in range(start_id, end_id):
        doc_name = str(doc_id)
        folder_path = DATA_ROOT / doc_name
        if not folder_path.exists():
            continue
        _, qa_file = _find_doc_files(folder_path)
        if not qa_file:
            continue

        with open(qa_file, "r", encoding="utf-8") as f_qa:
            qa_list = [json.loads(line) for line in f_qa]

        doc_pending = 0
        for qa_idx, qa_item in enumerate(qa_list):
            key = f"{doc_name}_{qa_item['question']}"
            if resume and key in processed_keys:
                continue
            pending_questions.append(
                {
                    "order_idx": order_idx,
                    "doc_name": doc_name,
                    "qa_idx": qa_idx,
                    "qa_item": qa_item,
                }
            )
            order_idx += 1
            doc_pending += 1

        if doc_pending == 0:
            logger.info("[%s] All questions already generated, skip", doc_name)
        else:
            logger.info("[%s] Pending questions on shared storage: %d", doc_name, doc_pending)

    if pending_questions:
        logger.info(
            "Answering %d questions from shared pool (max_async_generate=%d)",
            len(pending_questions),
            max_async_generate,
        )
        sem = asyncio.Semaphore(max_async_generate)
        progress_lock = asyncio.Lock()
        done_count = 0
        total_pending = len(pending_questions)

        async def _answer_one(entry: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            nonlocal done_count
            doc_name = entry["doc_name"]
            qa_idx = entry["qa_idx"]
            qa_item = entry["qa_item"]
            question = qa_item["question"]
            logger.info("[%s][Q%d] Question: %s", doc_name, qa_idx + 1, question[:80])
            async with sem:
                try:
                    answer = await service.query(
                        doc_id=shared_doc_id,
                        query=question,
                        **query_params,
                    )
                    logger.info("[%s][Q%d] Answer: %s", doc_name, qa_idx + 1, answer[:80])
                except Exception as exc:
                    logger.error("[%s][Q%d] query failed: %s", doc_name, qa_idx + 1, exc)
                    answer = ""

            result = {
                "doc_id": doc_name,
                "question": question,
                "sys_ans": answer,
                "ref_ans": qa_item["answer"],
                "type": qa_item["type"],
                "evidence": qa_item["evidence"],
            }
            async with progress_lock:
                done_count += 1
                if done_count == total_pending or done_count % max(1, total_pending // 10) == 0:
                    logger.info("Generate progress: %d/%d", done_count, total_pending)
            return entry["order_idx"], result

        tasks = [asyncio.create_task(_answer_one(item)) for item in pending_questions]
        results = await asyncio.gather(*tasks)
        with open(SYSTEM_ANSWERS_FILE, "a", encoding="utf-8") as f_out:
            for _, payload in sorted(results, key=lambda x: x[0]):
                _append_jsonl_record(f_out, payload)
    else:
        logger.info("No pending questions to answer in shared pool.")

    service = await _recycle_local_rag_service(
        service,
        settings,
        shared_doc_id,
        clear_model_cache=False,
    )
    del service
    gc.collect()
    _clear_cuda_cache()
    logger.info("Shared generate complete. Output: %s", SYSTEM_ANSWERS_FILE)


async def evaluate_answers(*, resume: bool, max_async_judge: int, eval_prompt_filename: str) -> None:
    if not SYSTEM_ANSWERS_FILE.exists():
        logger.error("Input file not found: %s", SYSTEM_ANSWERS_FILE)
        return

    try:
        eval_prompt = _load_eval_prompt(eval_prompt_filename)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return

    with open(SYSTEM_ANSWERS_FILE, "r", encoding="utf-8") as f:
        answers = [json.loads(line) for line in f]

    evaluated_keys = set()
    if resume and EVAL_RESULTS_FILE.exists():
        with open(EVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                evaluated_keys.add(f"{data['doc_id']}_{data['question']}")

    pending = []
    skipped = 0
    for i, item in enumerate(answers, 1):
        key = f"{item['doc_id']}_{item['question']}"
        if resume and key in evaluated_keys:
            skipped += 1
            continue
        pending.append((i, item))

    if not pending:
        logger.info("No pending answers to evaluate.")
        return

    logger.info(
        "Evaluating %d/%d answers using %s (max_async_judge=%d, prompt=%s)",
        len(pending),
        len(answers),
        JUDGE_MODEL_NAME,
        max_async_judge,
        eval_prompt_filename,
    )
    if skipped:
        logger.info("Skipped %d already-evaluated answers.", skipped)

    judge_client = AsyncOpenAI(api_key="EMPTY", base_url=JUDGE_API_BASE)
    sem = asyncio.Semaphore(_normalize_max_async(max_async_judge))
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    done_count = 0
    total_pending = len(pending)

    with open(EVAL_RESULTS_FILE, "a", encoding="utf-8") as f_out:

        async def _eval_one(index: int, item: dict[str, Any]) -> None:
            nonlocal done_count
            logger.info("[%d/%d] Doc %s", index, len(answers), item["doc_id"])
            logger.info("  Q: %s", item["question"][:60])
            logger.info("  A: %s", item["sys_ans"][:60])
            prompt = _build_eval_prompt(eval_prompt, item)
            async with sem:
                try:
                    response = await judge_client.chat.completions.create(
                        model=JUDGE_MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a helpful evaluator."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=50,
                    )
                    eval_result = response.choices[0].message.content.strip()
                    score = _parse_eval_score(eval_result)
                    logger.info("  Score: %d | %s", score, eval_result[:80])
                except Exception as exc:
                    logger.error("[%d] evaluate failed: %s", index, exc)
                    eval_result = f"[ERROR: {exc}]"
                    score = 0

            payload = {**item, "eval": eval_result, "score": score}
            async with write_lock:
                _append_jsonl_record(f_out, payload)
            async with progress_lock:
                done_count += 1
                if done_count == total_pending or done_count % max(1, total_pending // 10) == 0:
                    logger.info("Evaluate progress: %d/%d", done_count, total_pending)

        tasks = [asyncio.create_task(_eval_one(i, item)) for i, item in pending]
        await asyncio.gather(*tasks)

    logger.info("Shared evaluate complete. Output: %s", EVAL_RESULTS_FILE)


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


def calculate_statistics() -> None:
    if not EVAL_RESULTS_FILE.exists():
        logger.error("Result file not found: %s", EVAL_RESULTS_FILE)
        return

    with open(EVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    total = len(results)
    correct = sum(1 for r in results if r.get("score", 0) == 1)
    overall_acc = correct / total * 100 if total else 0.0
    logger.info("Overall Accuracy: %.2f%% (%d/%d)", overall_acc, correct, total)

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        qtype = str(r.get("type", ""))
        by_type.setdefault(qtype, {"correct": 0, "total": 0})
        by_type[qtype]["total"] += 1
        if r.get("score", 0) == 1:
            by_type[qtype]["correct"] += 1

    by_group = {k: {"correct": 0, "total": 0} for k in TYPE_GROUP_ORDER}
    unknown_groups: dict[str, int] = {}
    for r in results:
        group = _map_type_group(r.get("type"))
        if group is None:
            key = str(r.get("type", ""))
            unknown_groups[key] = unknown_groups.get(key, 0) + 1
            continue
        by_group[group]["total"] += 1
        if r.get("score", 0) == 1:
            by_group[group]["correct"] += 1

    stats_payload = {
        "overall": {"accuracy": overall_acc, "correct": correct, "total": total},
        "by_type": {
            qtype: {
                "accuracy": v["correct"] / v["total"] * 100 if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for qtype, v in by_type.items()
        },
        "by_type_group": {
            group: {
                "accuracy": v["correct"] / v["total"] * 100 if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for group, v in by_group.items()
        },
        "unknown_type_labels": unknown_groups,
        "generation_config": _load_json(GENERATION_CONFIG_FILE),
    }
    _save_json(STATS_FILE, stats_payload)
    logger.info("Shared stats saved: %s", STATS_FILE)


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DocBench shared-storage evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["generate", "evaluate", "stats"])
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=49)
    parser.add_argument("--shared_doc_id", type=str, default=DEFAULT_SHARED_DOC_ID)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--raganything_eval_setup", action="store_true")
    parser.add_argument("--max_async_generate", type=int, default=1)
    parser.add_argument("--max_async_judge", type=int, default=4)
    parser.add_argument(
        "--ingest_flush_every",
        type=int,
        default=6,
        help="Recycle LocalRagService every N ingested docs to limit GPU cache growth; 0 disables.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--disable_locator_cleanup",
        dest="disable_locator_cleanup",
        action="store_true",
        help="Disable locator cleanup rules (figure/table/equation labels) during extraction.",
    )
    group.add_argument(
        "--keep_locator_cleanup",
        dest="disable_locator_cleanup",
        action="store_false",
        help="Keep default locator cleanup rules.",
    )
    parser.set_defaults(disable_locator_cleanup=True)
    args = parser.parse_args()

    profile_name, one_sentence, eval_prompt_filename = _resolve_eval_setup(
        args.raganything_eval_setup
    )
    resume = not args.no_resume

    logger.info(
        "Mode=%s Range=%d-%d Resume=%s SharedDocID=%s Profile=%s OneSentence=%s "
        "EvalPrompt=%s MaxAsyncGen=%d MaxAsyncJudge=%d IngestFlushEvery=%d DisableLocatorCleanup=%s",
        args.mode,
        args.start_id,
        args.end_id - 1,
        resume,
        args.shared_doc_id,
        profile_name,
        one_sentence,
        eval_prompt_filename,
        args.max_async_generate,
        args.max_async_judge,
        args.ingest_flush_every,
        args.disable_locator_cleanup,
    )

    if args.mode == "generate":
        await generate_answers_shared(
            start_id=args.start_id,
            end_id=args.end_id,
            resume=resume,
            max_async_generate=args.max_async_generate,
            ingest_flush_every=args.ingest_flush_every,
            one_sentence=one_sentence,
            profile_name=profile_name,
            eval_prompt_filename=eval_prompt_filename,
            shared_doc_id=args.shared_doc_id,
            disable_locator_cleanup=args.disable_locator_cleanup,
        )
    elif args.mode == "evaluate":
        await evaluate_answers(
            resume=resume,
            max_async_judge=args.max_async_judge,
            eval_prompt_filename=eval_prompt_filename,
        )
    else:
        calculate_statistics()


if __name__ == "__main__":
    asyncio.run(main())

