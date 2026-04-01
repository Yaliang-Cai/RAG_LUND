#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocBench Shared-Storage Evaluation Script
=========================================

Use one shared RAG workspace/workspace_id for a document range (default: 0-48),
then evaluate generated answers.
"""

import os
import sys
import json
import hashlib
import re
import asyncio
import logging
import gc
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from openai import AsyncOpenAI

from evaluate_local.ablation_flags import (
    AblationFlags,
    add_ablation_arguments,
    apply_ablation_flags_to_settings,
    build_index_profile,
    ensure_workspace_index_profile,
    validate_ablation_flags,
    validate_workspace_env_isolation,
)

# Keep MinerU memory usage aligned with evaluate.py
os.environ.setdefault("MINERU_VLLM_GPU_MEMORY_UTILIZATION", "0.1")

# Add project root to import local_rag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raganything.services.local_rag import LocalRagService, LocalRagSettings
from raganything.constants import DEFAULT_EVAL_RETRY_FAILED_ONLY


DEFAULT_SCRIPT_DIR = "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/DocBench"
DEFAULT_DATA_ROOT = "/data/y50056788/Yaliang/datasets_for_eval/data_for_DocBench"
DEFAULT_OUTPUT_DIR_NAME = "docbench_shared_results"

SCRIPT_DIR = Path(os.getenv("DOCBENCH_SHARED_SCRIPT_DIR", DEFAULT_SCRIPT_DIR))
DATA_ROOT = Path(os.getenv("DOCBENCH_SHARED_DATA_ROOT", DEFAULT_DATA_ROOT))
_output_dir_override = str(os.getenv("DOCBENCH_SHARED_OUTPUT_DIR", "")).strip()
if _output_dir_override:
    OUTPUT_DIR = Path(_output_dir_override)
else:
    OUTPUT_DIR = SCRIPT_DIR / DEFAULT_OUTPUT_DIR_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORKING_DIR_ROOT = OUTPUT_DIR / "rag_workspaces"
WORKING_DIR_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_MD_DIR = OUTPUT_DIR / "mineru_outputs"
OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_ANSWERS_FILE = OUTPUT_DIR / "system_answers.jsonl"
EVAL_RESULTS_FILE = OUTPUT_DIR / "eval_results.jsonl"
STATS_FILE = OUTPUT_DIR / "statistics.json"
RERANK_CHUNK_STATS_FILE = OUTPUT_DIR / "rerank_chunk_stats.jsonl"
RERANK_CHUNK_SUMMARY_FILE = OUTPUT_DIR / "rerank_chunk_summary.json"
GENERATION_CONFIG_FILE = OUTPUT_DIR / "generation_config.json"
INGEST_MANIFEST_FILE = OUTPUT_DIR / "shared_ingest_manifest.json"
INGEST_FAILURES_FILE = OUTPUT_DIR / "shared_ingest_failures.jsonl"

RAG_API_BASE = "http://localhost:8001/v1"
JUDGE_API_BASE = "http://localhost:8002/v1"
RAG_API_KEY = "EMPTY"
RAG_VISION_MODEL_PATH = "/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8"
RAG_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

DOCBENCH_EVAL_PROMPT_FILENAME = "evaluation_prompt.txt"
RAGANYTHING_EVAL_PROMPT_FILENAME = "evaluation_prompt_RAG-Anything.txt"

DEFAULT_SHARED_WORKSPACE_ID = "docbench_shared_0_48"
DEFAULT_INGEST_FLUSH_EVERY = 6

DOCBENCH_QUERY_PARAMS = {
    "mode": "hybrid",
    "top_k": 40,
    "chunk_top_k": 20,
    "enable_rerank": True,
    "rerank_score_scope": "all",
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

_MASTER_LOG_PATH: Path | None = None


def _set_component_log_levels() -> None:
    """
    Keep component log verbosity aligned with evaluate.py so extraction/query
    details are visible in the consolidated log.
    """
    logging.getLogger("raganything").setLevel(logging.INFO)
    logging.getLogger("raganything.processor").setLevel(logging.INFO)
    logging.getLogger("raganything.parser").setLevel(logging.INFO)
    logging.getLogger("lightrag").setLevel(logging.INFO)


def _ensure_master_log_handler() -> None:
    """
    Keep one consolidated log file for the whole evaluate_shared run.
    """
    global _MASTER_LOG_PATH
    if _MASTER_LOG_PATH is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _MASTER_LOG_PATH = OUTPUT_DIR / "logs" / f"evaluate_generate_{ts}.log"
        _MASTER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(getattr(handler, "baseFilename", "")) == _MASTER_LOG_PATH:
                return

    file_handler = logging.FileHandler(_MASTER_LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    logger.info("Master log file: %s", _MASTER_LOG_PATH)


def _prune_non_master_file_handlers() -> None:
    """
    Remove non-master file handlers created by LocalRagService run_*.log setup.
    Keep console handler and the single master file handler only.
    """
    if _MASTER_LOG_PATH is None:
        return

    removed = 0
    deleted = 0

    def _prune_logger_file_handlers(target_logger: logging.Logger) -> tuple[int, int]:
        local_removed = 0
        local_deleted = 0
        for handler in list(target_logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            file_path = Path(getattr(handler, "baseFilename", ""))
            if file_path == _MASTER_LOG_PATH:
                continue
            target_logger.removeHandler(handler)
            handler.close()
            local_removed += 1
            if file_path.name.startswith("run_") and file_path.exists():
                try:
                    file_path.unlink()
                    local_deleted += 1
                except Exception:
                    # Non-fatal: stale run logs should not block evaluation.
                    pass
        return local_removed, local_deleted

    root_logger = logging.getLogger()
    rm, dl = _prune_logger_file_handlers(root_logger)
    removed += rm
    deleted += dl

    # Also prune non-master file handlers from named loggers.
    for logger_obj in list(logging.root.manager.loggerDict.values()):
        if isinstance(logger_obj, logging.Logger):
            rm, dl = _prune_logger_file_handlers(logger_obj)
            removed += rm
            deleted += dl

    # Extra safety: delete leftover run_*.log files in output log dir.
    log_dir = OUTPUT_DIR / "logs"
    if log_dir.exists():
        for run_log in log_dir.glob("run_*.log"):
            try:
                run_log.unlink()
                deleted += 1
            except Exception:
                pass

    if removed > 0:
        logger.info(
            "Pruned non-master file handlers: removed=%d, deleted_run_logs=%d",
            removed,
            deleted,
        )


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


def _refresh_master_logging() -> None:
    _set_component_log_levels()
    _ensure_master_log_handler()
    _prune_non_master_file_handlers()
    _bridge_lightrag_logs_to_run_file()


def _normalize_max_async(max_async: int, default: int = 4) -> int:
    try:
        return max(1, int(max_async))
    except Exception:
        return default


def _resolve_eval_setup(use_raganything_eval_setup: bool) -> tuple[str, bool, str]:
    if use_raganything_eval_setup:
        return ("rag_anything", True, RAGANYTHING_EVAL_PROMPT_FILENAME)
    return ("docbench_official", False, DOCBENCH_EVAL_PROMPT_FILENAME)


def _build_query_params(
    one_sentence: bool = False,
    *,
    ablation_flags: AblationFlags | None = None,
) -> dict[str, Any]:
    flags = ablation_flags or AblationFlags()
    params = dict(DOCBENCH_QUERY_PARAMS)
    params.update(flags.to_query_kwargs())
    if one_sentence:
        params["user_prompt"] = ONE_SENTENCE_USER_PROMPT
        params["response_type"] = "Single Sentence"
    return params


def _build_experiment_signature(
    *,
    shared_workspace_id: str,
    profile_name: str,
    eval_prompt_filename: str,
    one_sentence: bool,
    start_id: int,
    end_id: int,
    ablation_flags: AblationFlags,
    query_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "shared_workspace_id": str(shared_workspace_id),
        "setup": {
            "profile_name": str(profile_name),
            "eval_prompt_filename": str(eval_prompt_filename),
            "one_sentence": bool(one_sentence),
        },
        "ablation_flags": ablation_flags.to_dict(),
        "query_params": dict(query_params),
        "range": {
            "start_id": int(start_id),
            "end_id": int(end_id),
        },
    }


def _build_experiment_id(
    *,
    shared_workspace_id: str,
    profile_name: str,
    eval_prompt_filename: str,
    one_sentence: bool,
    start_id: int,
    end_id: int,
    ablation_flags: AblationFlags,
    query_params: dict[str, Any],
) -> str:
    signature = _build_experiment_signature(
        shared_workspace_id=shared_workspace_id,
        profile_name=profile_name,
        eval_prompt_filename=eval_prompt_filename,
        one_sentence=one_sentence,
        start_id=start_id,
        end_id=end_id,
        ablation_flags=ablation_flags,
        query_params=query_params,
    )
    canonical = json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"docbench_shared_{digest}"


def _record_matches_experiment(payload: Any, experiment_id: str) -> bool:
    if not isinstance(payload, dict):
        return False
    return str(payload.get("experiment_id", "")).strip() == experiment_id


def _answer_record_key(payload: dict[str, Any]) -> str:
    doc_id = str(payload.get("doc_id", "")).strip()
    qa_idx = payload.get("qa_idx")
    if qa_idx is not None:
        return f"{doc_id}::{qa_idx}"
    question = str(payload.get("question", "")).strip()
    return f"{doc_id}::{question}"


def _append_jsonl_record(file_obj: TextIO, payload: dict[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file_obj.flush()


def _append_ingest_failure_record(payload: dict[str, Any]) -> None:
    with open(INGEST_FAILURES_FILE, "a", encoding="utf-8") as f:
        _append_jsonl_record(f, payload)


def _to_float_scores(values: Any) -> list[float]:
    scores: list[float] = []
    if not isinstance(values, list):
        return scores
    for value in values:
        try:
            scores.append(round(float(value), 6))
        except (TypeError, ValueError):
            continue
    return scores


def _percentile(sorted_scores: list[float], q: float) -> float | None:
    if not sorted_scores:
        return None
    if q <= 0:
        return sorted_scores[0]
    if q >= 1:
        return sorted_scores[-1]

    position = (len(sorted_scores) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_scores[lower]
    weight = position - lower
    return sorted_scores[lower] * (1 - weight) + sorted_scores[upper] * weight


def _build_score_distribution(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }

    sorted_scores = sorted(scores)
    count = len(sorted_scores)
    mean = sum(sorted_scores) / count
    variance = sum((value - mean) ** 2 for value in sorted_scores) / count
    std = math.sqrt(variance)

    return {
        "count": count,
        "min": round(sorted_scores[0], 6),
        "max": round(sorted_scores[-1], 6),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "p10": round(_percentile(sorted_scores, 0.10), 6),
        "p25": round(_percentile(sorted_scores, 0.25), 6),
        "p50": round(_percentile(sorted_scores, 0.50), 6),
        "p75": round(_percentile(sorted_scores, 0.75), 6),
        "p90": round(_percentile(sorted_scores, 0.90), 6),
    }


def _build_threshold_retention(scores: list[float]) -> list[dict[str, Any]]:
    thresholds = [round(step * 0.05, 2) for step in range(21)]
    total = len(scores)
    retention: list[dict[str, Any]] = []
    for threshold in thresholds:
        kept = sum(1 for score in scores if score >= threshold)
        ratio = (kept / total) if total else 0.0
        retention.append(
            {
                "threshold": threshold,
                "kept": kept,
                "ratio": round(ratio, 6),
            }
        )
    return retention


def _build_macro_score_distribution(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metric_names = ("count", "min", "max", "mean", "std", "p10", "p25", "p50", "p75", "p90")
    buckets: dict[str, list[float]] = {name: [] for name in metric_names}

    for record in records:
        if not isinstance(record, dict):
            continue
        dist = record.get("distribution", {})
        if not isinstance(dist, dict):
            continue
        group = dist.get(key, {})
        if not isinstance(group, dict):
            continue
        for name in metric_names:
            value = group.get(name)
            if isinstance(value, (int, float)):
                buckets[name].append(float(value))

    out: dict[str, Any] = {}
    out["avg_count"] = round(sum(buckets["count"]) / len(buckets["count"]), 6) if buckets["count"] else 0.0
    for name in metric_names[1:]:
        values = buckets[name]
        out[name] = round(sum(values) / len(values), 6) if values else None
    return out


def _extract_rerank_chunk_payload(
    trace: dict[str, Any],
    *,
    query_params: dict[str, Any],
) -> dict[str, Any]:
    metadata = trace.get("metadata") if isinstance(trace, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    rerank_debug = metadata.get("rerank_chunk_debug")
    if not isinstance(rerank_debug, dict):
        rerank_debug = {}
    has_rerank_debug = bool(rerank_debug)

    scores_all = _to_float_scores(rerank_debug.get("scores_all"))
    scores_after_threshold = _to_float_scores(rerank_debug.get("scores_after_threshold"))
    scores_final = _to_float_scores(rerank_debug.get("scores_final"))

    selected_chunk_count = 0
    selected_missing_rerank_score = 0
    if not scores_final:
        data_section = trace.get("data") if isinstance(trace, dict) else {}
        chunks = data_section.get("chunks", []) if isinstance(data_section, dict) else []
        if isinstance(chunks, list):
            selected_chunk_count = len([chunk for chunk in chunks if isinstance(chunk, dict)])
            selected_missing_rerank_score = len(
                [
                    chunk
                    for chunk in chunks
                    if isinstance(chunk, dict)
                    and not isinstance(chunk.get("rerank_score"), (int, float))
                ]
            )
            scores_final = _to_float_scores(
                [chunk.get("rerank_score") for chunk in chunks if isinstance(chunk, dict)]
            )
    else:
        data_section = trace.get("data") if isinstance(trace, dict) else {}
        chunks = data_section.get("chunks", []) if isinstance(data_section, dict) else []
        if isinstance(chunks, list):
            selected_chunk_count = len([chunk for chunk in chunks if isinstance(chunk, dict)])
            selected_missing_rerank_score = len(
                [
                    chunk
                    for chunk in chunks
                    if isinstance(chunk, dict)
                    and not isinstance(chunk.get("rerank_score"), (int, float))
                ]
            )

    if not scores_all and scores_final:
        scores_all = list(scores_final)
    if not scores_after_threshold and scores_final:
        scores_after_threshold = list(scores_final)

    count_input = rerank_debug.get("count_input")
    count_after_rerank = rerank_debug.get("count_after_rerank")
    count_after_threshold = rerank_debug.get("count_after_threshold")
    count_final = rerank_debug.get("count_final")

    def _safe_count(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    return {
        "rerank_scope": str(
            rerank_debug.get("scope", query_params.get("rerank_score_scope", "top_k"))
        ),
        "min_rerank_score": rerank_debug.get("min_rerank_score"),
        "counts": {
            "input": _safe_count(count_input, len(scores_all)),
            "all": _safe_count(count_after_rerank, len(scores_all)),
            "after_threshold": _safe_count(count_after_threshold, len(scores_after_threshold)),
            "final": _safe_count(count_final, len(scores_final)),
        },
        "distribution": {
            "all": _build_score_distribution(scores_all),
            "after_threshold": _build_score_distribution(scores_after_threshold),
            "final": _build_score_distribution(scores_final),
        },
        "scores": {
            "all": scores_all,
            "after_threshold": scores_after_threshold,
            "final": scores_final,
        },
        "threshold_retention": _build_threshold_retention(scores_all),
        "has_rerank_debug": has_rerank_debug,
        "selected_chunk_count": selected_chunk_count,
        "selected_missing_rerank_score": selected_missing_rerank_score,
    }


def _refresh_rerank_chunk_summary(*, experiment_id: str | None = None) -> None:
    if not RERANK_CHUNK_STATS_FILE.exists():
        logger.info("Rerank stats file not found, skip summary: %s", RERANK_CHUNK_STATS_FILE)
        return

    records: list[dict[str, Any]] = []
    ignored_records = 0
    with open(RERANK_CHUNK_STATS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            if experiment_id is not None and not _record_matches_experiment(item, experiment_id):
                ignored_records += 1
                continue
            records.append(item)

    if ignored_records:
        logger.info(
            "Rerank summary: ignored %d records from other experiments.",
            ignored_records,
        )

    all_scores_all: list[float] = []
    all_scores_after_threshold: list[float] = []
    all_scores_final: list[float] = []
    by_type_scores: dict[str, dict[str, list[float]]] = {}
    threshold_bucket: dict[float, dict[str, float]] = {}

    for record in records:
        qtype = str(record.get("type", ""))
        by_type_scores.setdefault(
            qtype,
            {"all": [], "after_threshold": [], "final": []},
        )

        score_section = record.get("scores", {}) if isinstance(record, dict) else {}
        scores_all = _to_float_scores(score_section.get("all"))
        scores_after_threshold = _to_float_scores(score_section.get("after_threshold"))
        scores_final = _to_float_scores(score_section.get("final"))

        all_scores_all.extend(scores_all)
        all_scores_after_threshold.extend(scores_after_threshold)
        all_scores_final.extend(scores_final)
        by_type_scores[qtype]["all"].extend(scores_all)
        by_type_scores[qtype]["after_threshold"].extend(scores_after_threshold)
        by_type_scores[qtype]["final"].extend(scores_final)

        for row in record.get("threshold_retention", []):
            if not isinstance(row, dict):
                continue
            try:
                threshold = round(float(row.get("threshold")), 2)
            except (TypeError, ValueError):
                continue
            bucket = threshold_bucket.setdefault(
                threshold,
                {"kept_sum": 0.0, "ratio_sum": 0.0, "count": 0},
            )
            try:
                kept_value = float(row.get("kept", 0))
            except (TypeError, ValueError):
                kept_value = 0.0
            try:
                ratio_value = float(row.get("ratio", 0.0))
            except (TypeError, ValueError):
                ratio_value = 0.0
            bucket["kept_sum"] += kept_value
            bucket["ratio_sum"] += ratio_value
            bucket["count"] += 1

    by_type_distribution: dict[str, Any] = {}
    for qtype, score_groups in by_type_scores.items():
        by_type_distribution[qtype] = {
            "all": _build_score_distribution(score_groups["all"]),
            "after_threshold": _build_score_distribution(score_groups["after_threshold"]),
            "final": _build_score_distribution(score_groups["final"]),
        }

    threshold_retention_overall = []
    for threshold in sorted(threshold_bucket):
        bucket = threshold_bucket[threshold]
        if bucket["count"] <= 0:
            continue
        threshold_retention_overall.append(
            {
                "threshold": threshold,
                "avg_kept": round(bucket["kept_sum"] / bucket["count"], 6),
                "avg_ratio": round(bucket["ratio_sum"] / bucket["count"], 6),
            }
        )

    questions_with_trace = sum(
        1
        for record in records
        if int((record.get("counts", {}) or {}).get("all", 0)) > 0
        or int((record.get("counts", {}) or {}).get("final", 0)) > 0
    )

    summary_payload = {
        "total_questions": len(records),
        "questions_with_rerank_trace": questions_with_trace,
        "overall_distribution": {
            "all": _build_score_distribution(all_scores_all),
            "after_threshold": _build_score_distribution(all_scores_after_threshold),
            "final": _build_score_distribution(all_scores_final),
        },
        "macro_distribution_over_questions": {
            "all": _build_macro_score_distribution(records, "all"),
            "after_threshold": _build_macro_score_distribution(records, "after_threshold"),
            "final": _build_macro_score_distribution(records, "final"),
        },
        "by_type_distribution": by_type_distribution,
        "threshold_retention_overall": threshold_retention_overall,
        "generation_config": _load_json(GENERATION_CONFIG_FILE),
    }
    _save_json(RERANK_CHUNK_SUMMARY_FILE, summary_payload)
    logger.info("Rerank chunk summary saved: %s", RERANK_CHUNK_SUMMARY_FILE)


def _assert_rerank_contract(
    *,
    rerank_stats: dict[str, Any],
    query_params: dict[str, Any],
    doc_id: str,
    qa_idx: int,
) -> None:
    if not bool(query_params.get("enable_rerank", True)):
        return
    violations: list[str] = []
    if str(rerank_stats.get("rerank_scope", "")).strip().lower() != "all":
        violations.append(
            f"rerank_scope={rerank_stats.get('rerank_scope')!r} (expected 'all')"
        )
    selected_chunk_count = int(rerank_stats.get("selected_chunk_count", 0) or 0)
    selected_missing_rerank_score = int(
        rerank_stats.get("selected_missing_rerank_score", 0) or 0
    )
    if selected_chunk_count > 0 and selected_missing_rerank_score > 0:
        violations.append(
            "selected chunks missing rerank_score "
            f"({selected_missing_rerank_score}/{selected_chunk_count})"
        )
    if violations:
        detail = "; ".join(violations)
        raise RuntimeError(
            "Strict rerank contract violated for evaluate_shared: "
            f"doc_id={doc_id}, qa_idx={qa_idx}. {detail}"
        )


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
    return {
        "shared_workspace_id": "",
        "ingested_doc_ids": [],
        "ablation_group": "",
        "ablation_flags": None,
        "index_profile": None,
    }


def _save_ingest_manifest(
    shared_workspace_id: str,
    ingested_doc_ids: set[str],
    ablation_flags: AblationFlags,
) -> None:
    _save_json(
        INGEST_MANIFEST_FILE,
        {
            "shared_workspace_id": shared_workspace_id,
            "ingested_doc_ids": sorted(ingested_doc_ids, key=lambda x: int(x)),
            "ablation_group": ablation_flags.ablation_group(),
            "ablation_flags": ablation_flags.to_dict(),
            "index_profile": ablation_flags.to_index_profile(),
        },
    )


def _resolve_manifest_ingested_doc_ids(
    manifest: dict[str, Any],
    *,
    shared_workspace_id: str,
    ablation_flags: AblationFlags,
) -> set[str]:
    ingested_doc_ids = set(str(x) for x in manifest.get("ingested_doc_ids", []))
    manifest_workspace_id = str(manifest.get("shared_workspace_id") or "")
    if manifest_workspace_id != shared_workspace_id and ingested_doc_ids:
        logger.warning(
            "Manifest shared_workspace_id mismatch (%s != %s). Reset ingest manifest.",
            manifest_workspace_id,
            shared_workspace_id,
        )
        return set()
    if manifest_workspace_id == shared_workspace_id and ingested_doc_ids:
        existing_flags = AblationFlags.from_mapping(manifest.get("ablation_flags"))
        if existing_flags is None:
            raise RuntimeError(
                "Existing shared ingest manifest lacks ablation_flags metadata. "
                "To guarantee strict ablation isolation, use a new shared_workspace_id or clear this workspace and manifest."
            )
        if not existing_flags.is_index_compatible_with(ablation_flags):
            raise RuntimeError(
                "Shared workspace index profile mismatch detected: "
                f"existing={existing_flags.to_index_profile()} current={ablation_flags.to_index_profile()}. "
                "Use an independent shared_workspace_id for each DB/V1/V2 group."
            )
    return ingested_doc_ids


def _load_ingest_failures() -> dict[str, dict[str, Any]]:
    failures: dict[str, dict[str, Any]] = {}
    if not INGEST_FAILURES_FILE.exists():
        return failures
    with open(INGEST_FAILURES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            doc_id = str(payload.get("doc_id", "")).strip()
            if not doc_id:
                continue
            failures[doc_id] = payload
    return failures


def _rewrite_ingest_failures(failures: dict[str, dict[str, Any]]) -> None:
    if not failures:
        if INGEST_FAILURES_FILE.exists():
            INGEST_FAILURES_FILE.unlink()
        return
    with open(INGEST_FAILURES_FILE, "w", encoding="utf-8") as f:
        for doc_id in sorted(failures.keys(), key=lambda x: int(x) if x.isdigit() else x):
            _append_jsonl_record(f, failures[doc_id])


def _build_shared_settings(
    *,
    ablation_flags: AblationFlags,
) -> LocalRagSettings:
    settings = LocalRagSettings.from_env()
    settings.working_dir_root = str(WORKING_DIR_ROOT)
    settings.output_dir = str(OUTPUT_MD_DIR)
    settings.log_dir = str(OUTPUT_DIR / "logs")
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
    apply_ablation_flags_to_settings(settings, ablation_flags)
    return settings


async def _cleanup_rag_instance(service: LocalRagService, rag_workspace_id: str) -> None:
    rag_instances = getattr(service, "_rag_instances", None)
    if not isinstance(rag_instances, dict):
        return
    rag_instance = rag_instances.get(rag_workspace_id)
    if rag_instance is None:
        return
    try:
        await rag_instance.finalize_storages()
        rag_instances.pop(rag_workspace_id, None)
    except Exception as exc:
        logger.warning("Failed to cleanup RAG instance %s: %s", rag_workspace_id, exc)


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
    shared_workspace_id: str,
    *,
    clear_model_cache: bool = True,
) -> LocalRagService:
    await _cleanup_rag_instance(service, shared_workspace_id)
    if clear_model_cache:
        _clear_local_model_cache()
    del service
    gc.collect()
    _clear_cuda_cache()
    new_service = LocalRagService(settings)
    _refresh_master_logging()
    return new_service


def _save_generation_config(
    *,
    profile_name: str,
    eval_prompt_filename: str,
    one_sentence: bool,
    max_async_ingest: int,
    max_async_generate: int,
    shared_workspace_id: str,
    start_id: int,
    end_id: int,
    query_params: dict[str, Any],
    ablation_flags: AblationFlags,
    experiment_id: str,
    index_profile: dict[str, Any],
) -> None:
    experiment_signature = _build_experiment_signature(
        shared_workspace_id=shared_workspace_id,
        profile_name=profile_name,
        eval_prompt_filename=eval_prompt_filename,
        one_sentence=one_sentence,
        start_id=start_id,
        end_id=end_id,
        ablation_flags=ablation_flags,
        query_params=query_params,
    )
    _save_json(
        GENERATION_CONFIG_FILE,
        {
            "experiment_id": experiment_id,
            "experiment_signature": experiment_signature,
            "profile_name": profile_name,
            "eval_prompt_filename": eval_prompt_filename,
            "one_sentence": bool(one_sentence),
            "max_async_ingest": int(max_async_ingest),
            "max_async_generate": int(max_async_generate),
            "shared_workspace_id": shared_workspace_id,
            "start_id": int(start_id),
            "end_id": int(end_id),
            "ablation_group": ablation_flags.ablation_group(),
            "ablation_flags": ablation_flags.to_dict(),
            "index_profile": dict(index_profile),
            "effective_query_params": dict(query_params),
        },
    )


async def generate_answers_shared(
    *,
    start_id: int,
    end_id: int,
    resume: bool,
    max_async_ingest: int,
    max_async_generate: int,
    one_sentence: bool,
    profile_name: str,
    eval_prompt_filename: str,
    shared_workspace_id: str,
    retry_failed_only: bool,
    clear_failures_on_success: bool,
    ablation_flags: AblationFlags,
    query_params: dict[str, Any],
    experiment_id: str,
    allow_legacy_index_profile_adoption: bool,
) -> None:
    max_async_ingest = _normalize_max_async(max_async_ingest, default=4)
    max_async_generate = _normalize_max_async(max_async_generate, default=1)
    ingest_flush_every = DEFAULT_INGEST_FLUSH_EVERY
    settings = _build_shared_settings(
        ablation_flags=ablation_flags,
    )
    current_index_profile = build_index_profile(ablation_flags, settings=settings)
    ensured_index_profile = ensure_workspace_index_profile(
        working_dir_root=settings.working_dir_root,
        workspace_id=shared_workspace_id,
        index_profile=current_index_profile,
        allow_legacy_adoption=allow_legacy_index_profile_adoption,
    )
    _save_generation_config(
        profile_name=profile_name,
        eval_prompt_filename=eval_prompt_filename,
        one_sentence=one_sentence,
        max_async_ingest=max_async_ingest,
        max_async_generate=max_async_generate,
        shared_workspace_id=shared_workspace_id,
        start_id=start_id,
        end_id=end_id,
        query_params=query_params,
        ablation_flags=ablation_flags,
        experiment_id=experiment_id,
        index_profile=ensured_index_profile,
    )

    service = LocalRagService(settings)
    _refresh_master_logging()

    if not resume:
        for stale_output in (
            SYSTEM_ANSWERS_FILE,
            RERANK_CHUNK_STATS_FILE,
            RERANK_CHUNK_SUMMARY_FILE,
        ):
            if stale_output.exists():
                stale_output.unlink()
        # Keep failure manifest when retry_failed_only is requested, even with no_resume.
        if INGEST_FAILURES_FILE.exists() and not retry_failed_only:
            INGEST_FAILURES_FILE.unlink()

    processed_keys: set[str] = set()
    if resume and SYSTEM_ANSWERS_FILE.exists():
        ignored_answer_records = 0
        with open(SYSTEM_ANSWERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not _record_matches_experiment(item, experiment_id):
                    ignored_answer_records += 1
                    continue
                key = _answer_record_key(item)
                processed_keys.add(key)
        logger.info(
            "Resume: %d answers already generated for experiment_id=%s.",
            len(processed_keys),
            experiment_id,
        )
        if ignored_answer_records:
            logger.info(
                "Resume: ignored %d answer records from other experiments.",
                ignored_answer_records,
            )

    processed_rerank_keys: set[str] = set()
    if resume and RERANK_CHUNK_STATS_FILE.exists():
        ignored_rerank_records = 0
        with open(RERANK_CHUNK_STATS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not _record_matches_experiment(item, experiment_id):
                    ignored_rerank_records += 1
                    continue
                key = _answer_record_key(item)
                processed_rerank_keys.add(key)
        logger.info(
            "Resume: %d rerank stat records already generated for experiment_id=%s.",
            len(processed_rerank_keys),
            experiment_id,
        )
        if ignored_rerank_records:
            logger.info(
                "Resume: ignored %d rerank stat records from other experiments.",
                ignored_rerank_records,
            )

    existing_failures = _load_ingest_failures() if (resume or retry_failed_only) else {}
    if existing_failures:
        logger.info("Resume: %d ingest failures loaded.", len(existing_failures))
    elif retry_failed_only:
        logger.warning(
            "retry_failed_only is enabled but no failure records found: %s",
            INGEST_FAILURES_FILE,
        )

    manifest = _load_ingest_manifest()
    ingested_doc_ids = _resolve_manifest_ingested_doc_ids(
        manifest,
        shared_workspace_id=shared_workspace_id,
        ablation_flags=ablation_flags,
    )

    failed_ingest_docs: dict[str, dict[str, Any]] = dict(existing_failures)
    resolved_failure_docs: set[str] = set()
    logger.info("Shared workspace_id: %s", shared_workspace_id)
    logger.info("Experiment ID: %s", experiment_id)
    logger.info("Generate range: %d-%d", start_id, end_id - 1)
    logger.info("Max async ingest: %d", max_async_ingest)
    logger.info("Ingest flush every: %d (0 = disabled)", ingest_flush_every)
    logger.info("Retry failed only: %s", retry_failed_only)
    logger.info("Clear failures on success: %s", clear_failures_on_success)
    logger.info("Workspace index profile: %s", ensured_index_profile)

    # Phase 1: build/update shared storage
    ingest_jobs: list[tuple[str, Path]] = []
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
        if retry_failed_only:
            if doc_name not in failed_ingest_docs:
                continue
            if doc_name in ingested_doc_ids:
                logger.info(
                    "[%s] Already ingested in manifest, skip retry_failed_only replay.",
                    doc_name,
                )
                if clear_failures_on_success:
                    failed_ingest_docs.pop(doc_name, None)
                    resolved_failure_docs.add(doc_name)
                continue
        elif doc_name in ingested_doc_ids:
            logger.info("[%s] Shared ingest already done (manifest), skip", doc_name)
            continue
        ingest_jobs.append((doc_name, pdf_file))

    if ingest_jobs:
        logger.info(
            "Pending shared-ingest docs: %d (max_async_ingest=%d)",
            len(ingest_jobs),
            max_async_ingest,
        )
    else:
        logger.info("No pending shared-ingest docs.")

    ingested_since_flush = 0
    for batch_start in range(0, len(ingest_jobs), max_async_ingest):
        batch = ingest_jobs[batch_start : batch_start + max_async_ingest]
        batch_label = f"{batch_start + 1}-{batch_start + len(batch)}"
        logger.info("Shared ingest batch [%s/%d]", batch_label, len(ingest_jobs))

        async def _ingest_one(doc_name: str, pdf_file: Path) -> str:
            logger.info("[%s] Ingest into shared storage: %s", doc_name, pdf_file.name)
            doc_output_dir = str(OUTPUT_MD_DIR / f"docbench_{doc_name}")
            await service.ingest(
                file_path=str(pdf_file),
                output_dir=doc_output_dir,
                workspace_id=shared_workspace_id,
                serialize_by_workspace_id=False,
            )
            return doc_name

        batch_results = await asyncio.gather(
            *[_ingest_one(doc_name, pdf_file) for doc_name, pdf_file in batch],
            return_exceptions=True,
        )

        batch_errors: list[tuple[str, Exception]] = []
        success_count = 0
        for (doc_name, pdf_file), result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error("[%s] Shared ingest failed: %s", doc_name, result)
                error_trace = "".join(
                    traceback.format_exception(
                        type(result),
                        result,
                        result.__traceback__,
                    )[-3:]
                ).strip()
                failure_payload = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "doc_id": doc_name,
                    "workspace_id": shared_workspace_id,
                    "file_name": pdf_file.name,
                    "error": str(result),
                    "traceback_tail": error_trace,
                }
                failed_ingest_docs[doc_name] = failure_payload
                _append_ingest_failure_record(failure_payload)
                batch_errors.append((doc_name, result))
                continue
            ingested_doc_ids.add(result)
            success_count += 1
            if clear_failures_on_success:
                if result in failed_ingest_docs:
                    failed_ingest_docs.pop(result, None)
                resolved_failure_docs.add(result)

        if success_count > 0:
            ingested_since_flush += success_count
            _save_ingest_manifest(shared_workspace_id, ingested_doc_ids, ablation_flags)

        if batch_errors:
            logger.warning(
                "Shared ingest batch finished with %d failures (continuing): %s",
                len(batch_errors),
                ", ".join(doc_name for doc_name, _ in batch_errors),
            )

        if ingest_flush_every > 0 and ingested_since_flush >= ingest_flush_every:
            logger.info(
                "Recycling LocalRagService after %d ingested docs to control GPU cache.",
                ingested_since_flush,
            )
            service = await _recycle_local_rag_service(
                service,
                settings,
                shared_workspace_id,
                clear_model_cache=False,
            )
            ingested_since_flush = 0

    logger.info(
        "Shared ingest summary: success=%d failed=%d total=%d",
        len(ingested_doc_ids),
        len(failed_ingest_docs),
        len(ingest_jobs),
    )
    if clear_failures_on_success:
        _rewrite_ingest_failures(failed_ingest_docs)
        if resolved_failure_docs:
            logger.info(
                "Cleared %d resolved ingest failure entries.",
                len(resolved_failure_docs),
            )

    # Ensure ingest-phase temporary memory is released before query phase.
    service = await _recycle_local_rag_service(
        service,
        settings,
        shared_workspace_id,
        clear_model_cache=False,
    )

    # Phase 2: question answering against shared storage (global pending pool)
    pending_questions: list[dict[str, Any]] = []
    order_idx = 0
    for doc_id in range(start_id, end_id):
        doc_name = str(doc_id)
        if doc_name not in ingested_doc_ids:
            if doc_name in failed_ingest_docs:
                logger.warning("[%s] Skip query due to ingest failure", doc_name)
            continue
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
            key = _answer_record_key(
                {"doc_id": doc_name, "qa_idx": qa_idx, "question": qa_item["question"]}
            )
            legacy_key = _answer_record_key(
                {"doc_id": doc_name, "question": qa_item["question"]}
            )
            write_answer = True
            if resume and (key in processed_keys or legacy_key in processed_keys):
                if key in processed_rerank_keys or legacy_key in processed_rerank_keys:
                    continue
                write_answer = False
            pending_questions.append(
                {
                    "order_idx": order_idx,
                    "doc_name": doc_name,
                    "qa_idx": qa_idx,
                    "qa_item": qa_item,
                    "write_answer": write_answer,
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
        write_lock = asyncio.Lock()
        done_count = 0
        total_pending = len(pending_questions)

        async def _answer_one(
            entry: dict[str, Any],
        ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
            nonlocal done_count
            doc_name = entry["doc_name"]
            qa_idx = entry["qa_idx"]
            qa_item = entry["qa_item"]
            question = qa_item["question"]
            write_answer = bool(entry.get("write_answer", True))
            logger.info("[%s][Q%d] Question: %s", doc_name, qa_idx + 1, question[:80])
            async with sem:
                try:
                    response = await service.query_with_trace(
                        workspace_id=shared_workspace_id,
                        query=question,
                        **query_params,
                    )
                    answer = str(response.get("answer", ""))
                    trace = response.get("trace", {})
                    logger.info("[%s][Q%d] Answer: %s", doc_name, qa_idx + 1, answer[:80])
                except Exception as exc:
                    logger.error("[%s][Q%d] query failed: %s", doc_name, qa_idx + 1, exc)
                    answer = ""
                    trace = {}

            result = None
            if write_answer:
                result = {
                    "experiment_id": experiment_id,
                    "doc_id": doc_name,
                    "qa_idx": qa_idx,
                    "question": question,
                    "sys_ans": answer,
                    "ref_ans": qa_item["answer"],
                    "type": qa_item["type"],
                    "evidence": qa_item["evidence"],
                }
            rerank_stats = _extract_rerank_chunk_payload(
                trace if isinstance(trace, dict) else {},
                query_params=query_params,
            )
            _assert_rerank_contract(
                rerank_stats=rerank_stats,
                query_params=query_params,
                doc_id=doc_name,
                qa_idx=qa_idx,
            )
            rerank_payload = {
                "experiment_id": experiment_id,
                "doc_id": doc_name,
                "qa_idx": qa_idx,
                "question": question,
                "rerank_scope": rerank_stats["rerank_scope"],
                "min_rerank_score": rerank_stats["min_rerank_score"],
                "counts": rerank_stats["counts"],
                "distribution": rerank_stats["distribution"],
                "scores": rerank_stats["scores"],
                "threshold_retention": rerank_stats["threshold_retention"],
                "type": qa_item["type"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            async with progress_lock:
                done_count += 1
                if done_count == total_pending or done_count % max(1, total_pending // 10) == 0:
                    logger.info("Generate progress: %d/%d", done_count, total_pending)
            return entry["order_idx"], result, rerank_payload

        with open(SYSTEM_ANSWERS_FILE, "a", encoding="utf-8") as f_out, open(
            RERANK_CHUNK_STATS_FILE, "a", encoding="utf-8"
        ) as f_rerank:
            total_batches = max(
                1, (len(pending_questions) + max_async_generate - 1) // max_async_generate
            )
            persisted = 0
            persisted_rerank = 0
            for batch_idx, batch_start in enumerate(
                range(0, len(pending_questions), max_async_generate), start=1
            ):
                question_batch = pending_questions[
                    batch_start : batch_start + max_async_generate
                ]
                tasks = [asyncio.create_task(_answer_one(item)) for item in question_batch]
                results = await asyncio.gather(*tasks)
                ordered_results = sorted(results, key=lambda x: x[0])

                async with write_lock:
                    written_answers_batch = 0
                    for _, payload, rerank_payload in ordered_results:
                        if payload is not None:
                            _append_jsonl_record(f_out, payload)
                            written_answers_batch += 1
                        if rerank_payload is not None:
                            rerank_key = _answer_record_key(rerank_payload)
                            legacy_rerank_key = _answer_record_key(
                                {
                                    "doc_id": rerank_payload.get("doc_id"),
                                    "question": rerank_payload.get("question"),
                                }
                            )
                            if (
                                rerank_key not in processed_rerank_keys
                                and legacy_rerank_key not in processed_rerank_keys
                            ):
                                _append_jsonl_record(f_rerank, rerank_payload)
                                processed_rerank_keys.add(rerank_key)
                                persisted_rerank += 1

                persisted += written_answers_batch
                logger.info(
                    "Persisted shared question batch %d/%d (%d answers, total=%d, rerank_stats_total=%d)",
                    batch_idx,
                    total_batches,
                    written_answers_batch,
                    persisted,
                    persisted_rerank,
                )

                gc.collect()
                _clear_cuda_cache()
    else:
        logger.info("No pending questions to answer in shared pool.")

    service = await _recycle_local_rag_service(
        service,
        settings,
        shared_workspace_id,
        clear_model_cache=False,
    )
    del service
    gc.collect()
    _clear_cuda_cache()
    logger.info("Shared generate complete. Output: %s", SYSTEM_ANSWERS_FILE)
    _refresh_rerank_chunk_summary(experiment_id=experiment_id)
    if failed_ingest_docs:
        logger.info(
            "Shared ingest failures recorded: %d (file: %s)",
            len(failed_ingest_docs),
            INGEST_FAILURES_FILE,
        )


async def evaluate_answers(
    *,
    resume: bool,
    max_async_judge: int,
    eval_prompt_filename: str,
    experiment_id: str,
) -> None:
    if not SYSTEM_ANSWERS_FILE.exists():
        logger.error("Input file not found: %s", SYSTEM_ANSWERS_FILE)
        return

    try:
        eval_prompt = _load_eval_prompt(eval_prompt_filename)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return

    if not resume and EVAL_RESULTS_FILE.exists():
        EVAL_RESULTS_FILE.unlink()

    with open(SYSTEM_ANSWERS_FILE, "r", encoding="utf-8") as f:
        all_answers = [json.loads(line) for line in f]

    answers = [item for item in all_answers if _record_matches_experiment(item, experiment_id)]
    ignored_answer_records = len(all_answers) - len(answers)
    if ignored_answer_records:
        logger.info(
            "Evaluate: ignored %d answer records from other experiments.",
            ignored_answer_records,
        )
    if not answers:
        logger.warning(
            "No answers found for experiment_id=%s in %s",
            experiment_id,
            SYSTEM_ANSWERS_FILE,
        )
        return

    evaluated_keys = set()
    if resume and EVAL_RESULTS_FILE.exists():
        ignored_eval_records = 0
        with open(EVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not _record_matches_experiment(data, experiment_id):
                    ignored_eval_records += 1
                    continue
                evaluated_keys.add(_answer_record_key(data))
        logger.info(
            "Resume: %d eval records already exist for experiment_id=%s.",
            len(evaluated_keys),
            experiment_id,
        )
        if ignored_eval_records:
            logger.info(
                "Resume: ignored %d eval records from other experiments.",
                ignored_eval_records,
            )

    pending = []
    skipped = 0
    for i, item in enumerate(answers, 1):
        key = _answer_record_key(item)
        legacy_key = _answer_record_key(
            {"doc_id": item.get("doc_id"), "question": item.get("question")}
        )
        if resume and (key in evaluated_keys or legacy_key in evaluated_keys):
            skipped += 1
            continue
        pending.append((i, item))

    if not pending:
        logger.info("No pending answers to evaluate.")
        return

    logger.info(
        "Evaluating %d/%d answers using %s (max_async_judge=%d, prompt=%s, experiment_id=%s)",
        len(pending),
        len(answers),
        JUDGE_MODEL_NAME,
        max_async_judge,
        eval_prompt_filename,
        experiment_id,
    )
    if skipped:
        logger.info("Skipped %d already-evaluated answers.", skipped)

    judge_client = AsyncOpenAI(api_key="EMPTY", base_url=JUDGE_API_BASE)
    sem = asyncio.Semaphore(_normalize_max_async(max_async_judge))
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    done_count = 0
    total_pending = len(pending)

    write_mode = "a" if resume else "w"
    with open(EVAL_RESULTS_FILE, write_mode, encoding="utf-8") as f_out:

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

            payload = {
                **item,
                "experiment_id": experiment_id,
                "eval": eval_result,
                "score": score,
            }
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


def calculate_statistics(*, experiment_id: str) -> None:
    if not EVAL_RESULTS_FILE.exists():
        logger.error("Result file not found: %s", EVAL_RESULTS_FILE)
        return

    with open(EVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        all_results = [json.loads(line) for line in f]

    tagged_results = [
        item
        for item in all_results
        if isinstance(item, dict) and "experiment_id" in item
    ]
    if tagged_results:
        results = [item for item in tagged_results if _record_matches_experiment(item, experiment_id)]
        ignored_results = len(all_results) - len(results)
        if ignored_results:
            logger.info(
                "Stats: ignored %d eval records from other experiments.",
                ignored_results,
            )
    else:
        results = all_results
        logger.warning(
            "Stats input has no experiment_id metadata. Using all eval records (legacy mode)."
        )

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
    _refresh_rerank_chunk_summary(experiment_id=experiment_id)


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DocBench shared-storage evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["generate", "evaluate", "stats"])
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=49)
    parser.add_argument(
        "--shared_workspace_id", type=str, default=DEFAULT_SHARED_WORKSPACE_ID
    )
    parser.add_argument("--no_resume", action="store_true")
    eval_setup_group = parser.add_mutually_exclusive_group()
    eval_setup_group.add_argument(
        "--raganything_eval_setup",
        dest="raganything_eval_setup",
        action="store_true",
        help="Use RAG-Anything evaluation setup (default).",
    )
    eval_setup_group.add_argument(
        "--docbench_eval_setup",
        dest="raganything_eval_setup",
        action="store_false",
        help="Use official DocBench evaluation setup.",
    )
    parser.set_defaults(raganything_eval_setup=True)
    parser.add_argument("--max_async_ingest", type=int, default=4)
    parser.add_argument("--max_async_generate", type=int, default=1)
    parser.add_argument("--max_async_judge", type=int, default=4)
    add_ablation_arguments(parser)
    parser.add_argument(
        "--allow_legacy_index_profile_adoption",
        action="store_true",
        help=(
            "Allow adopting the current index profile when an existing workspace has "
            "artifacts but no .ablation_index_profile.json."
        ),
    )
    parser.add_argument(
        "--retry_failed_only",
        action="store_true",
        default=DEFAULT_EVAL_RETRY_FAILED_ONLY,
        help="Only ingest docs listed in shared_ingest_failures.jsonl within [start_id, end_id).",
    )
    parser.add_argument(
        "--clear_failures_on_success",
        dest="clear_failures_on_success",
        action="store_true",
        default=True,
        help="Remove resolved docs from shared_ingest_failures.jsonl after successful ingest.",
    )
    parser.add_argument(
        "--no_clear_failures_on_success",
        dest="clear_failures_on_success",
        action="store_false",
        help="Keep historical failure records even when docs succeed later.",
    )
    args = parser.parse_args()
    ablation_flags = validate_ablation_flags(args, naming_style="underscore")
    validate_workspace_env_isolation(workspace_id=args.shared_workspace_id)

    _ensure_master_log_handler()

    profile_name, one_sentence, eval_prompt_filename = _resolve_eval_setup(
        args.raganything_eval_setup
    )
    resume = not args.no_resume
    query_params = _build_query_params(
        one_sentence=one_sentence,
        ablation_flags=ablation_flags,
    )
    experiment_id = _build_experiment_id(
        shared_workspace_id=args.shared_workspace_id,
        profile_name=profile_name,
        eval_prompt_filename=eval_prompt_filename,
        one_sentence=one_sentence,
        start_id=args.start_id,
        end_id=args.end_id,
        ablation_flags=ablation_flags,
        query_params=query_params,
    )

    logger.info(
        "Mode=%s Range=%d-%d Resume=%s SharedWorkspaceID=%s Profile=%s OneSentence=%s "
        "EvalPrompt=%s MaxAsyncIngest=%d MaxAsyncGen=%d MaxAsyncJudge=%d "
        "IngestFlushEvery=%d RetryFailedOnly=%s ClearFailuresOnSuccess=%s "
        "AllowLegacyIndexProfileAdoption=%s ExperimentID=%s",
        args.mode,
        args.start_id,
        args.end_id - 1,
        resume,
        args.shared_workspace_id,
        profile_name,
        one_sentence,
        eval_prompt_filename,
        args.max_async_ingest,
        args.max_async_generate,
        args.max_async_judge,
        DEFAULT_INGEST_FLUSH_EVERY,
        args.retry_failed_only,
        args.clear_failures_on_success,
        args.allow_legacy_index_profile_adoption,
        experiment_id,
    )

    if args.mode == "generate":
        await generate_answers_shared(
            start_id=args.start_id,
            end_id=args.end_id,
            resume=resume,
            max_async_ingest=args.max_async_ingest,
            max_async_generate=args.max_async_generate,
            one_sentence=one_sentence,
            profile_name=profile_name,
            eval_prompt_filename=eval_prompt_filename,
            shared_workspace_id=args.shared_workspace_id,
            retry_failed_only=args.retry_failed_only,
            clear_failures_on_success=args.clear_failures_on_success,
            ablation_flags=ablation_flags,
            query_params=query_params,
            experiment_id=experiment_id,
            allow_legacy_index_profile_adoption=args.allow_legacy_index_profile_adoption,
        )
    elif args.mode == "evaluate":
        await evaluate_answers(
            resume=resume,
            max_async_judge=args.max_async_judge,
            eval_prompt_filename=eval_prompt_filename,
            experiment_id=experiment_id,
        )
    else:
        calculate_statistics(experiment_id=experiment_id)


if __name__ == "__main__":
    asyncio.run(main())

