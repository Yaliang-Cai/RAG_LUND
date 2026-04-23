#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import math
import os
import re
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MINERU_VLLM_GPU_MEMORY_UTILIZATION", "0.1")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
_projects_root = Path(__file__).resolve().parents[3]
_local_lightrag_root = _projects_root / "lightrag"
if _local_lightrag_root.exists():
    sys.path.insert(0, str(_local_lightrag_root))

from evaluate_local.ablation_flags import (
    AblationFlags,
    add_ablation_arguments,
    apply_ablation_flags_to_settings,
    build_index_profile,
    ensure_workspace_index_profile,
    as_bool,
    validate_ablation_flags,
    validate_workspace_env_isolation,
)
from raganything.constants import (
    DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
    DEFAULT_KG_CHUNK_SELECTION_SOURCE,
    DEFAULT_RECOGNITION_TOP_K,
    DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS,
    DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
    DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
_output_dir_override = str(os.getenv("SURGE_FAST_OUTPUT_DIR", "")).strip()
if _output_dir_override:
    OUTPUT_ROOT_DIR = Path(_output_dir_override)
else:
    OUTPUT_ROOT_DIR = SCRIPT_DIR
RETRIEVAL_DIR = OUTPUT_ROOT_DIR / "retrieval_results_fast"
SURVEY_DIR = OUTPUT_ROOT_DIR / "survey_results_fast"
LOG_DIR = OUTPUT_ROOT_DIR / "logs"
_rag_storage_override = str(os.getenv("SURGE_FAST_RAG_STORAGE_DIR", "")).strip()
if _rag_storage_override:
    RAG_STORAGE_DIR = Path(_rag_storage_override)
else:
    RAG_STORAGE_DIR = OUTPUT_ROOT_DIR / "rag_storage"
_rag_output_override = str(os.getenv("SURGE_FAST_RAG_OUTPUT_DIR", "")).strip()
if _rag_output_override:
    RAG_OUTPUT_DIR = Path(_rag_output_override)
else:
    RAG_OUTPUT_DIR = OUTPUT_ROOT_DIR / "rag_outputs"

DEFAULT_DATA_ROOT = "/data/y50056788/Yaliang/datasets_for_eval/data_for_SurGE"
DEFAULT_SUBSET_DIR = "subset_output"
DEFAULT_QUERIES = "subset_queries.json"
DEFAULT_SURVEYS = "subset_surveys.json"
DEFAULT_CHUNKS = "subset_chunks.jsonl"
DEFAULT_CORPUS = "subset_corpus.json"
DEFAULT_WORKSPACE = "surge_subset_fast_shared"
SURGE_NEVER_SPLIT_DELIMITER = "__SURGE_NEVER_SPLIT__"

PER_QUERY_FILE = RETRIEVAL_DIR / "retrieval_per_query.jsonl"
SUMMARY_FILE = RETRIEVAL_DIR / "retrieval_summary.json"
RERANK_STATS_FILE = RETRIEVAL_DIR / "rerank_chunk_stats.jsonl"
RERANK_SUMMARY_FILE = RETRIEVAL_DIR / "rerank_chunk_summary.json"
WARNINGS_FILE = RETRIEVAL_DIR / "mapping_warnings.jsonl"
INGEST_MANIFEST = RETRIEVAL_DIR / "shared_ingest_manifest_fast.json"
INGEST_FAILURES = RETRIEVAL_DIR / "shared_ingest_failures_fast.jsonl"
CHUNK_SOURCE_MAP_FILE = RETRIEVAL_DIR / "chunk_source_map.json"
SURVEY_STATUS = SURVEY_DIR / "survey_mode_status.json"
SURVEY_PER_FILE = SURVEY_DIR / "survey_retrieval_per_survey.jsonl"
SURVEY_SUMMARY_FILE = SURVEY_DIR / "survey_retrieval_summary.json"
SURVEY_RERANK_STATS_FILE = SURVEY_DIR / "survey_rerank_chunk_stats.jsonl"
SURVEY_RERANK_SUMMARY_FILE = SURVEY_DIR / "survey_rerank_chunk_summary.json"
SURVEY_WARNINGS_FILE = SURVEY_DIR / "survey_mapping_warnings.jsonl"

INT_RE = re.compile(r"^-?\d+$")
MASTER_LOG: Path | None = None

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def import_rag_dependencies():
    try:
        from lightrag import QueryParam
        from raganything.services.local_rag import LocalRagService, LocalRagSettings
        return QueryParam, LocalRagService, LocalRagSettings
    except Exception as exc:
        raise RuntimeError(
            "Cannot import RAG dependencies. Please run under the rag-anything runtime environment."
        ) from exc


def get_ablation_flags(args: argparse.Namespace) -> AblationFlags:
    flags = getattr(args, "ablation_flags", None)
    if isinstance(flags, AblationFlags):
        return flags
    flags = AblationFlags.from_namespace(args)
    args.ablation_flags = flags
    return flags


def parse_k_list(raw: str) -> list[int]:
    ks = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise ValueError(f"k must > 0, got {k}")
        ks.append(k)
    if not ks:
        raise ValueError("k-list is empty")
    return sorted(dict.fromkeys(ks))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_line(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and INT_RE.match(v.strip()):
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


def parse_non_negative_int(v: Any) -> int | None:
    iv = parse_int(v)
    if iv is None or iv < 0:
        return None
    return iv


def ensure_dirs() -> None:
    # For current SurGE retrieval pipeline we ingest prebuilt chunks directly,
    # so parser output dir is not required.
    for p in [RETRIEVAL_DIR, SURVEY_DIR, LOG_DIR, RAG_STORAGE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def prune_non_master_file_handlers(master_log: Path) -> None:
    def prune_logger(target_logger: logging.Logger) -> None:
        for handler in list(target_logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            file_path = Path(getattr(handler, "baseFilename", ""))
            if file_path == master_log:
                continue
            target_logger.removeHandler(handler)
            handler.close()

    prune_logger(logging.getLogger())
    for logger_obj in list(logging.root.manager.loggerDict.values()):
        if isinstance(logger_obj, logging.Logger):
            prune_logger(logger_obj)
    for run_log in LOG_DIR.glob("run_*.log"):
        try:
            run_log.unlink()
        except Exception:
            pass


def bridge_lightrag_file_handlers(root: logging.Logger) -> None:
    try:
        from lightrag.utils import logger as lightrag_logger
    except Exception:
        return

    for handler in root.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        if all(
            getattr(existing, "baseFilename", None) != getattr(handler, "baseFilename", None)
            for existing in lightrag_logger.handlers
        ):
            lightrag_logger.addHandler(handler)
    lightrag_logger.setLevel(logging.INFO)


def sync_master_logging_handlers() -> None:
    if MASTER_LOG is None:
        return
    root = logging.getLogger()
    prune_non_master_file_handlers(MASTER_LOG)
    bridge_lightrag_file_handlers(root)


def refresh_logging(mode: str) -> None:
    global MASTER_LOG
    logging.getLogger("raganything").setLevel(logging.INFO)
    logging.getLogger("raganything.processor").setLevel(logging.INFO)
    logging.getLogger("raganything.parser").setLevel(logging.INFO)
    logging.getLogger("lightrag").setLevel(logging.INFO)
    if MASTER_LOG is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        MASTER_LOG = LOG_DIR / f"evaluate_surge_fast_{mode}_{ts}.log"
    root = logging.getLogger()
    has_master = False
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            p = Path(getattr(h, "baseFilename", ""))
            if p == MASTER_LOG:
                has_master = True
                break
    if not has_master:
        fh = logging.FileHandler(MASTER_LOG, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root.addHandler(fh)
        logger.info("Master log file: %s", MASTER_LOG)
    sync_master_logging_handlers()


def settings_for_surge(args: argparse.Namespace) -> LocalRagSettings:
    _, _, LocalRagSettings = import_rag_dependencies()
    # Pin RAGAnything context extraction switches to avoid env-drift across runs.
    os.environ["ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE"] = "true"
    os.environ["CONTEXT_ZERO_WINDOW_CONTENT_TYPES"] = str(
        DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES
    )

    s = LocalRagSettings.from_env()
    s.working_dir_root = str(RAG_STORAGE_DIR)
    s.output_dir = str(RAG_OUTPUT_DIR)
    s.log_dir = str(LOG_DIR)
    # Keep SurGE ingest behavior aligned with user preference for workspace-level ingest serialization.
    s.serialize_ingest_by_workspace_id = False
    # Keep non-ablation switches stable and enabled across runs.
    s.enable_entity_surface_normalization = True
    s.enable_keyword_case_normalization = True
    s.strict_relation_endpoint_entity_match = True
    s.recognition_top_k = DEFAULT_RECOGNITION_TOP_K
    s.recognition_prompt_max_tokens = DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS
    s.recognition_prompt_output_max_tokens = DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS
    s.recognition_prompt_reserved_tokens = DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS
    apply_ablation_flags_to_settings(s, get_ablation_flags(args))
    return s


def build_query_params(args: argparse.Namespace, *, chunk_top_k: int) -> dict[str, Any]:
    query_params = {
        "mode": args.query_mode,
        "top_k": args.top_k,
        "chunk_top_k": chunk_top_k,
        "max_total_tokens": int(getattr(args, "max_total_tokens", 45000)),
        "enable_rerank": True,
        "rerank_score_scope": "all",
        "keyword_fanout_mode": str(getattr(args, "keyword_fanout_mode", "joined")).strip(),
        "entity_qdrant_retrieval_mode": str(
            getattr(args, "entity_retrieval_mode", "dense")
        ).strip(),
        "chunk_qdrant_retrieval_mode": str(
            getattr(args, "chunk_retrieval_mode", "dense")
        ).strip(),
        "kg_chunk_selection_source": str(
            getattr(
                args,
                "kg_chunk_selection_source",
                DEFAULT_KG_CHUNK_SELECTION_SOURCE,
            )
        ).strip(),
        "bypass_query_cache": bool(getattr(args, "bypass_query_cache", False)),
        "bypass_keywords_cache": bool(
            getattr(args, "bypass_keywords_cache", False)
        ),
    }
    exclude_synonym_edges = getattr(args, "exclude_synonym_edges", None)
    if exclude_synonym_edges is not None:
        query_params["exclude_synonym_edges"] = bool(exclude_synonym_edges)
    query_params.update(get_ablation_flags(args).to_query_kwargs())
    query_params["enable_rerank"] = True
    query_params["rerank_score_scope"] = "all"
    return query_params


async def _cleanup_workspace_service(
    service: Any | None,
    workspace_id: str,
    *,
    stage: str,
) -> None:
    try:
        if service is not None:
            await service.cleanup_workspace_instance(workspace_id)
    except Exception as exc:
        logger.warning(
            "Workspace cleanup failed after %s for %s: %s",
            stage,
            workspace_id,
            exc,
        )
    finally:
        gc.collect()
        clear_cuda_cache()


@asynccontextmanager
async def prepared_workspace_service(
    args: argparse.Namespace,
    source_records: list[dict[str, Any]],
    *,
    stage: str,
):
    _, LocalRagService, _ = import_rag_dependencies()
    ablation_flags = get_ablation_flags(args)
    settings = settings_for_surge(args)
    current_index_profile = build_index_profile(ablation_flags, settings=settings)
    ensured_index_profile = ensure_workspace_index_profile(
        working_dir_root=settings.working_dir_root,
        workspace_id=args.workspace_id,
        index_profile=current_index_profile,
        allow_legacy_adoption=bool(args.allow_legacy_index_profile_adoption),
    )
    service = None
    try:
        service = LocalRagService(settings)
        ingest_summary = await ensure_workspace_index(
            service,
            args.workspace_id,
            source_records,
            ablation_flags,
            args.max_retries,
            args.ingest_batch_size,
            args.batch_doc_concurrency,
            args.llm_model_max_async,
        )
        sync_master_logging_handlers()
        blockers = collect_ingest_blockers(ingest_summary)
        if blockers:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(blockers.items()))
            logger.error("Ingest integrity check failed before %s: %s", stage, detail)
            raise RuntimeError(
                f"Workspace ingest incomplete; abort {stage}. Details: {detail}"
            )
        rag = await service.get_rag(args.workspace_id)
        await ensure_rag_runtime_ready(rag, args.workspace_id)
        yield service, ablation_flags, ensured_index_profile, ingest_summary
    finally:
        await _cleanup_workspace_service(
            service,
            args.workspace_id,
            stage=stage,
        )


def load_chunks(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    by_doc: dict[str, dict[str, Any]] = {}
    dup = 0
    bad = 0
    empty = 0
    for row in rows:
        doc = parse_int(row.get("doc_id"))
        if doc is None:
            bad += 1
            continue
        key = str(doc)
        text = str(row.get("text") or row.get("abstract") or "").strip()
        if not text:
            empty += 1
        if key in by_doc:
            dup += 1
            continue
        by_doc[key] = row
    return by_doc, {
        "input_rows": len(rows),
        "unique_doc_ids": len(by_doc),
        "duplicate_doc_ids": dup,
        "invalid_doc_id_rows": bad,
        "rows_with_empty_text": empty,
    }


def prepare_source_records(
    chunks_by_doc: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    source_records: dict[str, dict[str, Any]] = {}
    chunk_source_map: dict[str, dict[str, Any]] = {}
    source_chunk_id_set: set[str] = set()

    sorted_doc_ids = sorted(
        chunks_by_doc.keys(),
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )
    for doc_id in sorted_doc_ids:
        row = chunks_by_doc.get(doc_id) or {}
        raw_text = str(row.get("text") or row.get("abstract") or "")
        text = sanitize_text_for_encoding(raw_text).strip()
        if not text:
            raise ValueError(f"empty text after sanitize for doc_id={doc_id}")

        source_doc_id = parse_int(doc_id)
        if source_doc_id is None:
            raise ValueError(f"invalid source doc_id={doc_id}")

        source_chunk_id = str(row.get("chunk_id") or f"{doc_id}#0").strip()
        if not source_chunk_id:
            raise ValueError(f"invalid source chunk_id for doc_id={doc_id}")
        if source_chunk_id in source_chunk_id_set:
            raise ValueError(f"duplicate source_chunk_id={source_chunk_id}")
        source_chunk_id_set.add(source_chunk_id)

        lightrag_chunk_id = compute_mdhash_id(text, prefix="chunk-")
        existing = chunk_source_map.get(lightrag_chunk_id)
        if existing is not None:
            raise ValueError(
                f"chunk hash collision: {lightrag_chunk_id} maps to multiple source docs "
                f"({existing.get('source_doc_id')} vs {source_doc_id})"
            )

        source_records[doc_id] = {
            "source_doc_id": source_doc_id,
            "source_chunk_id": source_chunk_id,
            "text": text,
            "lightrag_chunk_id": lightrag_chunk_id,
        }
        chunk_source_map[lightrag_chunk_id] = {
            "source_doc_id": source_doc_id,
            "source_chunk_id": source_chunk_id,
        }

    if len(chunk_source_map) != len(source_records):
        raise ValueError(
            "chunk_source_map size mismatch; mapping is not one-to-one with source records"
        )

    stats = {
        "source_doc_count": len(source_records),
        "source_chunk_count": len(chunk_source_map),
    }
    return source_records, chunk_source_map, stats


def persist_chunk_source_map(
    chunk_source_map: dict[str, dict[str, Any]],
    source_stats: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "surge_chunk_source_map_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_stats": source_stats,
        "map_size": len(chunk_source_map),
        "map": chunk_source_map,
    }
    save_json(CHUNK_SOURCE_MAP_FILE, payload)


def build_virtual_batches(
    source_records: dict[str, dict[str, Any]],
    ingest_batch_size: int,
) -> list[dict[str, Any]]:
    sorted_doc_ids = sorted(
        source_records.keys(),
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )
    batch_size = max(1, int(ingest_batch_size))
    batches: list[dict[str, Any]] = []
    for batch_idx, batch_doc_ids in enumerate(iter_batches(sorted_doc_ids, batch_size), start=1):
        rows = [source_records[doc_id] for doc_id in batch_doc_ids]
        texts = [str(row["text"]) for row in rows]
        delimiter = resolve_safe_split_delimiter(texts)
        batch_doc_id = f"surge_batch_{batch_idx:05d}"
        batches.append(
            {
                "batch_doc_id": batch_doc_id,
                "file_path": f"{batch_doc_id}.txt",
                "delimiter": delimiter,
                "content": delimiter.join(texts),
                "source_doc_ids": [int(row["source_doc_id"]) for row in rows],
                "source_chunk_ids": [str(row["source_chunk_id"]) for row in rows],
                "expected_chunk_ids": [str(row["lightrag_chunk_id"]) for row in rows],
                "expected_chunk_count": len(rows),
            }
        )
    return batches


def load_queries(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    q = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(q, list):
        raise ValueError(f"queries must be list: {path}")
    return q[:limit] if limit > 0 else q


def load_surveys(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"surveys must be list: {path}")
    return data[:limit] if limit > 0 else data


def iter_batches(values: list[Any], batch_size: int):
    size = max(1, int(batch_size))
    for i in range(0, len(values), size):
        yield values[i : i + size]


def parse_chunk_store_row_id(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    rid = str(
        row.get("chunk_id")
        or row.get("id")
        or row.get("_id")
        or row.get("__id__")
        or row.get("key")
        or ""
    ).strip()
    return rid or None


def parse_store_row_id(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    rid = str(
        row.get("id")
        or row.get("_id")
        or row.get("__id__")
        or row.get("doc_id")
        or row.get("full_doc_id")
        or row.get("key")
        or ""
    ).strip()
    return rid or None


def resolve_safe_split_delimiter(texts: list[str]) -> str:
    preferred = SURGE_NEVER_SPLIT_DELIMITER
    if all(preferred not in text for text in texts):
        return preferred
    while True:
        candidate = f"{preferred}_{uuid.uuid4().hex}"
        if all(candidate not in text for text in texts):
            return candidate


async def fetch_existing_chunk_ids(store: Any, expected_ids: list[str], batch_size: int = 2000) -> set[str]:
    found: set[str] = set()
    for batch in iter_batches(expected_ids, batch_size):
        rows = await store.get_by_ids(batch)
        if not isinstance(rows, list):
            continue
        for row in rows:
            rid = parse_chunk_store_row_id(row)
            if rid:
                found.add(rid)
    return found


def normalize_status_value(v: Any) -> str:
    if hasattr(v, "value"):
        v = getattr(v, "value")
    return str(v or "").strip().lower()


def normalize_chunk_id_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        cid = str(item or "").strip()
        if cid:
            out.append(cid)
    return out


async def inspect_workspace_index_state(
    rag: Any,
    target_docs: list[str],
    expected_chunk_count_by_doc: dict[str, int],
    expected_chunk_ids_by_doc: dict[str, list[str]],
) -> dict[str, Any]:
    processed_statuses = {"processed", "preprocessed"}
    target_set = set(target_docs)
    missing_full_set = set(await rag.lightrag.full_docs.filter_keys(target_set))

    status_rows = await rag.lightrag.doc_status.get_by_ids(target_docs)
    if not isinstance(status_rows, list):
        status_rows = []
    status_by_doc: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(status_rows):
        if not isinstance(row, dict):
            continue
        rid = parse_store_row_id(row)
        if rid:
            status_by_doc[rid] = row
            continue
        if idx < len(target_docs):
            status_by_doc[target_docs[idx]] = row

    missing_doc_status_set: set[str] = set()
    status_not_processed_set: set[str] = set()
    missing_chunk_set: set[str] = set()
    missing_vdb_set: set[str] = set()
    chunk_count_mismatch_set: set[str] = set()
    expected_chunk_ids_present_by_doc: dict[str, list[str]] = {}

    for doc_id in target_docs:
        row = status_by_doc.get(doc_id)
        expected_chunk_ids = expected_chunk_ids_by_doc.get(doc_id, [])
        expected_count = max(0, int(expected_chunk_count_by_doc.get(doc_id, len(expected_chunk_ids))))
        if not isinstance(row, dict):
            missing_doc_status_set.add(doc_id)
            continue
        status_value = normalize_status_value(row.get("status"))
        if status_value not in processed_statuses:
            status_not_processed_set.add(doc_id)
            continue
        chunk_ids = normalize_chunk_id_list(row.get("chunks_list"))
        if not chunk_ids and expected_count > 0:
            missing_chunk_set.add(doc_id)
            missing_vdb_set.add(doc_id)
            chunk_count_mismatch_set.add(doc_id)
            continue
        if len(chunk_ids) != expected_count:
            chunk_count_mismatch_set.add(doc_id)
        if expected_chunk_ids:
            if set(chunk_ids) != set(expected_chunk_ids):
                chunk_count_mismatch_set.add(doc_id)
            expected_chunk_ids_present_by_doc[doc_id] = expected_chunk_ids
        else:
            expected_chunk_ids_present_by_doc[doc_id] = chunk_ids

    all_expected_chunk_ids = sorted(
        {cid for ids in expected_chunk_ids_present_by_doc.values() for cid in ids}
    )
    if all_expected_chunk_ids:
        existing_text_chunk_ids = await fetch_existing_chunk_ids(
            rag.lightrag.text_chunks, all_expected_chunk_ids
        )
        existing_vdb_chunk_ids = await fetch_existing_chunk_ids(
            rag.lightrag.chunks_vdb, all_expected_chunk_ids
        )
        for doc_id, chunk_ids in expected_chunk_ids_present_by_doc.items():
            if any(cid not in existing_text_chunk_ids for cid in chunk_ids):
                missing_chunk_set.add(doc_id)
            if any(cid not in existing_vdb_chunk_ids for cid in chunk_ids):
                missing_vdb_set.add(doc_id)

    return {
        "missing_full_set": missing_full_set,
        "missing_doc_status_set": missing_doc_status_set,
        "status_not_processed_set": status_not_processed_set,
        "missing_chunk_set": missing_chunk_set,
        "missing_vdb_set": missing_vdb_set,
        "chunk_count_mismatch_set": chunk_count_mismatch_set,
        "multi_chunk_set": chunk_count_mismatch_set,
    }


async def ensure_rag_runtime_ready(rag: Any, workspace_id: str) -> None:
    init_result = await rag._ensure_lightrag_initialized()
    if isinstance(init_result, dict) and not init_result.get("success", True):
        error_msg = str(init_result.get("error") or "unknown initialization error")
        raise RuntimeError(
            f"LightRAG initialization failed for workspace '{workspace_id}': {error_msg}"
        )
    if getattr(rag, "lightrag", None) is None:
        raise RuntimeError(
            f"LightRAG runtime is not available for workspace '{workspace_id}'"
        )


def apply_llm_model_max_async_override(rag: Any, llm_model_max_async: int) -> None:
    target = max(1, int(llm_model_max_async))
    kwargs = getattr(rag, "lightrag_kwargs", None)
    if isinstance(kwargs, dict):
        kwargs["llm_model_max_async"] = target
    runtime = getattr(rag, "lightrag", None)
    if runtime is not None:
        setattr(runtime, "llm_model_max_async", target)
    logger.info(
        "Applied evaluate-fast override: llm_model_max_async=%d",
        target,
    )


async def with_retries(coro_factory, label: str, retries: int) -> Any:
    retries = max(0, retries)
    last: Exception | None = None
    for i in range(retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: PERF203
            last = exc
            if i >= retries:
                break
            await asyncio.sleep(min(0.5 * (2 ** i), 3.0))
            logger.warning("%s retry %d/%d: %s", label, i + 1, retries + 1, exc)
    raise RuntimeError(f"{label} failed after {retries + 1} attempts: {last}") from last


def clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def score_percentile(sorted_scores: list[float], q: float) -> float | None:
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


def score_distribution(scores: list[float]) -> dict[str, Any]:
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
    s = sorted(scores)
    n = len(s)
    mean = sum(s) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / n)
    return {
        "count": n,
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "p10": round(score_percentile(s, 0.10), 6),
        "p25": round(score_percentile(s, 0.25), 6),
        "p50": round(score_percentile(s, 0.50), 6),
        "p75": round(score_percentile(s, 0.75), 6),
        "p90": round(score_percentile(s, 0.90), 6),
    }


def build_threshold_retention(scores: list[float]) -> list[dict[str, Any]]:
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


def summarize_threshold_retention(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bucket: dict[float, dict[str, float]] = {}
    for row in rows:
        for item in row.get("threshold_retention", []) if isinstance(row, dict) else []:
            if not isinstance(item, dict):
                continue
            try:
                threshold = round(float(item.get("threshold")), 2)
            except (TypeError, ValueError):
                continue
            b = bucket.setdefault(threshold, {"kept_sum": 0.0, "ratio_sum": 0.0, "count": 0.0})
            try:
                kept = float(item.get("kept", 0))
            except (TypeError, ValueError):
                kept = 0.0
            try:
                ratio = float(item.get("ratio", 0.0))
            except (TypeError, ValueError):
                ratio = 0.0
            b["kept_sum"] += kept
            b["ratio_sum"] += ratio
            b["count"] += 1.0

    out: list[dict[str, Any]] = []
    for threshold in sorted(bucket):
        b = bucket[threshold]
        if b["count"] <= 0:
            continue
        out.append(
            {
                "threshold": threshold,
                "avg_kept": round(b["kept_sum"] / b["count"], 6),
                "avg_ratio": round(b["ratio_sum"] / b["count"], 6),
            }
        )
    return out


def summarize_macro_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean_or_none(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    metric_keys = ("count", "min", "max", "mean", "std", "p10", "p25", "p50", "p75", "p90")
    groups = {"all": {}, "after_threshold": {}, "final": {}}
    for metric in metric_keys:
        groups["all"][metric] = []
        groups["after_threshold"][metric] = []
        groups["final"][metric] = []

    for row in rows:
        dist = row.get("distribution", {}) if isinstance(row, dict) else {}
        if not isinstance(dist, dict):
            continue
        for scope in ("all", "after_threshold", "final"):
            scope_dist = dist.get(scope, {})
            if not isinstance(scope_dist, dict):
                continue
            for metric in metric_keys:
                value = scope_dist.get(metric)
                if isinstance(value, (int, float)):
                    groups[scope][metric].append(float(value))

    out: dict[str, Any] = {}
    for scope in ("all", "after_threshold", "final"):
        out[scope] = {
            "avg_count": _mean_or_none(groups[scope]["count"]),
            "min": _mean_or_none(groups[scope]["min"]),
            "max": _mean_or_none(groups[scope]["max"]),
            "mean": _mean_or_none(groups[scope]["mean"]),
            "std": _mean_or_none(groups[scope]["std"]),
            "p10": _mean_or_none(groups[scope]["p10"]),
            "p25": _mean_or_none(groups[scope]["p25"]),
            "p50": _mean_or_none(groups[scope]["p50"]),
            "p75": _mean_or_none(groups[scope]["p75"]),
            "p90": _mean_or_none(groups[scope]["p90"]),
        }
    return out


def count_rows_with_rerank_trace(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        counts = row.get("counts", {})
        if not isinstance(counts, dict):
            continue
        all_count = parse_non_negative_int(counts.get("all")) or 0
        final_count = parse_non_negative_int(counts.get("final")) or 0
        if all_count > 0 or final_count > 0:
            count += 1
    return count


def extract_rerank_payload(retrieval: dict[str, Any], query_params: dict[str, Any]) -> dict[str, Any]:
    md = retrieval.get("metadata", {}) if isinstance(retrieval, dict) else {}
    if not isinstance(md, dict):
        md = {}
    dbg = md.get("rerank_chunk_debug", {})
    if not isinstance(dbg, dict):
        dbg = {}
    scores_all = [float(x) for x in dbg.get("scores_all", []) if isinstance(x, (int, float))]
    scores_thr = [float(x) for x in dbg.get("scores_after_threshold", []) if isinstance(x, (int, float))]
    scores_final = [float(x) for x in dbg.get("scores_final", []) if isinstance(x, (int, float))]
    if not scores_final:
        chunks = retrieval.get("data", {}).get("chunks", []) if isinstance(retrieval, dict) else []
        if isinstance(chunks, list):
            scores_final = [float(c.get("rerank_score")) for c in chunks if isinstance(c, dict) and isinstance(c.get("rerank_score"), (int, float))]
    if not scores_all and scores_final:
        scores_all = list(scores_final)
    if not scores_thr and scores_final:
        scores_thr = list(scores_final)
    input_count = parse_non_negative_int(dbg.get("count_input"))
    all_count = parse_non_negative_int(dbg.get("count_after_rerank"))
    thr_count = parse_non_negative_int(dbg.get("count_after_threshold"))
    top_k_count = parse_non_negative_int(dbg.get("count_after_chunk_top_k"))
    final_count = parse_non_negative_int(dbg.get("count_final"))
    chunk_ids = extract_chunk_ids_by_stage(retrieval)
    selected_chunks = (
        retrieval.get("data", {}).get("chunks", []) if isinstance(retrieval, dict) else []
    )
    if not isinstance(selected_chunks, list):
        selected_chunks = []
    selected_chunk_count = len(selected_chunks)
    selected_missing_rerank_score = sum(
        1
        for c in selected_chunks
        if isinstance(c, dict) and not isinstance(c.get("rerank_score"), (int, float))
    )
    counts = {
        "input": input_count if input_count is not None else len(scores_all),
        "all": all_count if all_count is not None else len(scores_all),
        "after_threshold": thr_count if thr_count is not None else len(scores_thr),
        "after_chunk_top_k": top_k_count if top_k_count is not None else len(chunk_ids.get("after_threshold", [])),
        "final": final_count if final_count is not None else len(scores_final),
    }
    return {
        "rerank_scope": str(dbg.get("scope", query_params.get("rerank_score_scope", "all"))),
        "min_rerank_score": dbg.get("min_rerank_score"),
        "counts": counts,
        "distribution": {
            "all": score_distribution(scores_all),
            "after_threshold": score_distribution(scores_thr),
            "final": score_distribution(scores_final),
        },
        "scores": {"all": scores_all, "after_threshold": scores_thr, "final": scores_final},
        "chunk_ids": chunk_ids,
        "selected_chunk_count": selected_chunk_count,
        "selected_missing_rerank_score": selected_missing_rerank_score,
        "threshold_retention": build_threshold_retention(scores_all),
    }


def assert_rerank_contract(
    *,
    rerank_payload: dict[str, Any],
    query_params: dict[str, Any],
    record_key: str,
    record_id: Any,
) -> None:
    if not bool(query_params.get("enable_rerank", True)):
        return
    violations: list[str] = []
    if str(rerank_payload.get("rerank_scope", "")).strip().lower() != "all":
        violations.append(
            f"rerank_scope={rerank_payload.get('rerank_scope')!r} (expected 'all')"
        )
    selected_chunk_count = int(rerank_payload.get("selected_chunk_count", 0) or 0)
    selected_missing = int(rerank_payload.get("selected_missing_rerank_score", 0) or 0)
    if selected_chunk_count > 0 and selected_missing > 0:
        violations.append(
            "selected chunks missing rerank_score "
            f"({selected_missing}/{selected_chunk_count})"
        )
    if violations:
        detail = "; ".join(violations)
        raise RuntimeError(
            "Strict rerank contract violated for evaluate_surge_fast: "
            f"{record_key}={record_id}. {detail}"
        )


async def ensure_workspace_index(
    service: LocalRagService,
    workspace_id: str,
    source_records: dict[str, dict[str, Any]],
    ablation_flags: AblationFlags,
    retries: int,
    ingest_batch_size: int,
    batch_doc_concurrency: int,
    llm_model_max_async: int,
) -> dict[str, Any]:
    rag = await service.get_rag(workspace_id)
    # Ensure llm_model_max_async is applied before first LightRAG initialization,
    # so the internal LLM worker queue is created with the requested concurrency.
    kwargs = getattr(rag, "lightrag_kwargs", None)
    if isinstance(kwargs, dict):
        kwargs["llm_model_max_async"] = max(1, int(llm_model_max_async))
    await ensure_rag_runtime_ready(rag, workspace_id)
    apply_llm_model_max_async_override(rag, llm_model_max_async)
    batch_workers = max(1, int(batch_doc_concurrency))
    batch_size = max(1, int(ingest_batch_size))
    batches = build_virtual_batches(source_records, batch_size)
    target_docs = [batch["batch_doc_id"] for batch in batches]
    expected_chunk_count_by_doc = {
        batch["batch_doc_id"]: int(batch["expected_chunk_count"]) for batch in batches
    }
    expected_chunk_ids_by_doc = {
        batch["batch_doc_id"]: list(batch["expected_chunk_ids"]) for batch in batches
    }
    batch_by_id = {batch["batch_doc_id"]: batch for batch in batches}
    target = set(target_docs)
    sort_key = lambda x: x
    before_state = await inspect_workspace_index_state(
        rag=rag,
        target_docs=target_docs,
        expected_chunk_count_by_doc=expected_chunk_count_by_doc,
        expected_chunk_ids_by_doc=expected_chunk_ids_by_doc,
    )
    missing_before_full_set = set(before_state["missing_full_set"])
    missing_before_doc_status_set = set(before_state["missing_doc_status_set"])
    missing_before_status_set = set(before_state["status_not_processed_set"])
    missing_before_chunk_set = set(before_state["missing_chunk_set"])
    missing_before_vdb_set = set(before_state["missing_vdb_set"])
    missing_before_chunk_mismatch_set = set(before_state["chunk_count_mismatch_set"])
    to_ingest = sorted(
        missing_before_full_set
        | missing_before_doc_status_set
        | missing_before_status_set
        | missing_before_chunk_set
        | missing_before_vdb_set
        | missing_before_chunk_mismatch_set,
        key=sort_key,
    )
    logger.info(
        (
            "Workspace %s [fast ingest] missing full_docs: %d/%d, missing doc_status: %d/%d, "
            "status not processed: %d/%d, missing text_chunks: %d/%d, missing vdb_chunks: %d/%d, "
            "chunk_count_mismatch: %d/%d"
        ),
        workspace_id,
        len(missing_before_full_set),
        len(target),
        len(missing_before_doc_status_set),
        len(target),
        len(missing_before_status_set),
        len(target),
        len(missing_before_chunk_set),
        len(target),
        len(missing_before_vdb_set),
        len(target),
        len(missing_before_chunk_mismatch_set),
        len(target),
    )
    failures: list[dict[str, Any]] = []
    ingested_batch_count = 0
    ingested_source_doc_count = 0
    ingest_batch_failure_count = 0
    stale_cleanup_ok = 0
    stale_cleanup_failed = 0
    ingest_done = 0
    ingest_total = len(to_ingest)
    next_progress_log = 10
    with open(INGEST_FAILURES, "w", encoding="utf-8") as _:
        pass

    def record_failure(batch_id: str, error: str) -> None:
        batch = batch_by_id.get(batch_id, {})
        failure = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_doc_id": batch_id,
            "error": error,
            "source_doc_ids": list(batch.get("source_doc_ids", []))[:20],
        }
        failures.append(failure)
        append_jsonl_line(INGEST_FAILURES, failure)

    def mark_ingest_progress(step: int = 1) -> None:
        nonlocal ingest_done, next_progress_log
        ingest_done += step
        should_log = False
        while ingest_done >= next_progress_log:
            should_log = True
            next_progress_log += 10
        if should_log or ingest_done == ingest_total:
            logger.info("Ingest progress: %d/%d", ingest_done, ingest_total)
            gc.collect()
            clear_cuda_cache()

    stale_docs = sorted((set(to_ingest) - missing_before_full_set), key=sort_key)
    for batch_doc_id in stale_docs:
        try:
            await with_retries(
                lambda: service.lightrag_adelete_by_doc_id(
                    workspace_id,
                    batch_doc_id,
                    delete_llm_cache=False,
                ),
                label=f"cleanup stale doc_id={batch_doc_id}",
                retries=retries,
            )
            stale_cleanup_ok += 1
        except Exception as exc:
            stale_cleanup_failed += 1
            record_failure(batch_doc_id, f"stale cleanup failed: {exc}")

    async def ainsert_batch(batch: dict[str, Any]) -> None:
        await service.lightrag_ainsert(
            workspace_id,
            input=str(batch["content"]),
            ids=str(batch["batch_doc_id"]),
            file_paths=str(batch["file_path"]),
            split_by_character=str(batch["delimiter"]),
            split_by_character_only=True,
        )

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(batch_workers)
    to_ingest_batches = [batch_by_id[batch_doc_id] for batch_doc_id in to_ingest if batch_doc_id in batch_by_id]

    async def process_batch(batch: dict[str, Any]) -> None:
        nonlocal ingested_batch_count, ingested_source_doc_count, ingest_batch_failure_count
        batch_doc_id = str(batch["batch_doc_id"])
        expected_count = int(batch["expected_chunk_count"])
        try:
            await with_retries(
                lambda: ainsert_batch(batch),
                label=f"ingest fast batch {batch_doc_id} (chunks={expected_count})",
                retries=retries,
            )
            async with lock:
                ingested_batch_count += 1
                ingested_source_doc_count += expected_count
        except Exception as exc:
            ingest_batch_failure_count += 1
            record_failure(batch_doc_id, f"ingest failed: {exc}")
        finally:
            async with lock:
                mark_ingest_progress(step=1)

    async def process_batch_with_sem(batch: dict[str, Any]) -> None:
        async with sem:
            await process_batch(batch)

    if to_ingest_batches:
        await asyncio.gather(
            *[asyncio.create_task(process_batch_with_sem(batch)) for batch in to_ingest_batches]
        )

    after_state = await inspect_workspace_index_state(
        rag=rag,
        target_docs=target_docs,
        expected_chunk_count_by_doc=expected_chunk_count_by_doc,
        expected_chunk_ids_by_doc=expected_chunk_ids_by_doc,
    )
    missing_after_full_set = set(after_state["missing_full_set"])
    missing_after_doc_status = sorted(list(after_state["missing_doc_status_set"]), key=sort_key)
    missing_after_status = sorted(list(after_state["status_not_processed_set"]), key=sort_key)
    missing_after_chunks = sorted(list(after_state["missing_chunk_set"]), key=sort_key)
    missing_after_vdb = sorted(list(after_state["missing_vdb_set"]), key=sort_key)
    missing_after_chunk_mismatch = sorted(list(after_state["chunk_count_mismatch_set"]), key=sort_key)
    ingest_attempt_source_doc_count = sum(
        int(batch_by_id[batch_doc_id]["expected_chunk_count"])
        for batch_doc_id in to_ingest
        if batch_doc_id in batch_by_id
    )
    expected_chunk_total = sum(int(batch["expected_chunk_count"]) for batch in batches)
    summary = {
        "workspace_id": workspace_id,
        "ablation_group": ablation_flags.ablation_group(),
        "ablation_flags": ablation_flags.to_dict(),
        "index_profile": ablation_flags.to_index_profile(),
        "ingest_mode": "virtual_batch",
        # compatibility: keep evaluate_surge field semantics (doc-level = source_doc)
        "target_doc_count": len(source_records),
        "target_batch_count": len(target),
        "target_source_doc_count": len(source_records),
        "expected_chunk_total": expected_chunk_total,
        "missing_before_full_doc_count": len(missing_before_full_set),
        "missing_before_doc_status_count": len(missing_before_doc_status_set),
        "missing_before_status_not_processed_count": len(missing_before_status_set),
        "missing_before_chunk_doc_count": len(missing_before_chunk_set),
        "missing_before_vdb_doc_count": len(missing_before_vdb_set),
        "chunk_count_mismatch_batch_count_before": len(missing_before_chunk_mismatch_set),
        "missing_before_multi_chunk_doc_count": len(missing_before_chunk_mismatch_set),
        "stale_doc_count": len(stale_docs),
        "stale_batch_count": len(stale_docs),
        "stale_cleanup_success_count": stale_cleanup_ok,
        "stale_cleanup_failure_count": stale_cleanup_failed,
        "ingest_attempt_count": ingest_attempt_source_doc_count,
        "ingest_attempt_batch_count": len(to_ingest),
        "ingest_attempt_source_doc_count": ingest_attempt_source_doc_count,
        "ingested_now_count": ingested_source_doc_count,
        "ingested_now_batch_count": ingested_batch_count,
        "ingested_now_source_doc_count": ingested_source_doc_count,
        "ingest_concurrency": batch_workers,
        "batch_doc_concurrency": batch_workers,
        "ingest_effective_max_parallel_insert": batch_workers,
        "ingest_batch_failure_count": ingest_batch_failure_count,
        "ingest_batch_fallback_doc_count": 0,
        "llm_model_max_async": int(llm_model_max_async),
        "ingest_batch_size": batch_size,
        "ingest_batch_count": len(batches),
        "ingest_split_by_character_only": True,
        "ingest_split_delimiter_preferred": SURGE_NEVER_SPLIT_DELIMITER,
        "ingest_split_delimiter_strategy": "dynamic_safe_per_call",
        "ingest_failure_count": len(failures),
        "missing_after_full_doc_count": len(missing_after_full_set),
        "missing_after_doc_status_count": len(missing_after_doc_status),
        "missing_after_status_not_processed_count": len(missing_after_status),
        "missing_after_chunk_doc_count": len(missing_after_chunks),
        "missing_after_vdb_doc_count": len(missing_after_vdb),
        "chunk_count_mismatch_batch_count": len(missing_after_chunk_mismatch),
        "missing_after_multi_chunk_doc_count": len(missing_after_chunk_mismatch),
        "missing_before_full_doc_sample": sorted(list(missing_before_full_set), key=sort_key)[:20],
        "missing_before_doc_status_sample": sorted(list(missing_before_doc_status_set), key=sort_key)[:20],
        "missing_before_status_not_processed_sample": sorted(list(missing_before_status_set), key=sort_key)[:20],
        "chunk_count_mismatch_batch_sample_before": sorted(
            list(missing_before_chunk_mismatch_set), key=sort_key
        )[:20],
        "missing_before_multi_chunk_doc_sample": sorted(
            list(missing_before_chunk_mismatch_set), key=sort_key
        )[:20],
        "missing_after_full_doc_sample": sorted(list(missing_after_full_set), key=sort_key)[:20],
        "missing_after_doc_status_sample": missing_after_doc_status[:20],
        "missing_after_status_not_processed_sample": missing_after_status[:20],
        "missing_after_chunk_doc_sample": missing_after_chunks[:20],
        "missing_after_vdb_doc_sample": missing_after_vdb[:20],
        "chunk_count_mismatch_batch_sample": missing_after_chunk_mismatch[:20],
        "missing_after_multi_chunk_doc_sample": missing_after_chunk_mismatch[:20],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(INGEST_MANIFEST, summary)
    return summary


def recall_at_k(gt: set[int], retrieved: list[int], ks: list[int]) -> dict[str, float | None]:
    if not gt:
        return {str(k): None for k in ks}
    out: dict[str, float | None] = {}
    for k in ks:
        hit = len(gt & set(retrieved[:k]))
        out[str(k)] = round(hit / len(gt), 6)
    return out


def hit_at_k(gt: set[int], retrieved: list[int], ks: list[int]) -> dict[str, int]:
    return {str(k): len(gt & set(retrieved[:k])) for k in ks}


def collect_ingest_blockers(summary: dict[str, Any]) -> dict[str, str]:
    blocker_keys = [
        "ingest_failure_count",
        "missing_after_full_doc_count",
        "missing_after_doc_status_count",
        "missing_after_status_not_processed_count",
        "missing_after_chunk_doc_count",
        "missing_after_vdb_doc_count",
        "chunk_count_mismatch_batch_count",
        "missing_after_multi_chunk_doc_count",
    ]
    blockers: dict[str, str] = {}
    for key in blocker_keys:
        if key not in summary:
            blockers[key] = "missing"
            continue
        iv = parse_int(summary.get(key))
        if iv is None:
            blockers[key] = "invalid"
            continue
        if iv > 0:
            blockers[key] = str(iv)
    return blockers


def resolve_chunk_top_k(raw_chunk_top_k: int, ks: list[int]) -> int:
    if raw_chunk_top_k > 0:
        return raw_chunk_top_k
    return 0


def _normalize_chunk_ids(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        cid = str(item or "").strip()
        if cid:
            out.append(cid)
    return out


def extract_chunk_ids_by_stage(retrieval: dict[str, Any]) -> dict[str, list[str]]:
    md = retrieval.get("metadata", {}) if isinstance(retrieval, dict) else {}
    if not isinstance(md, dict):
        md = {}
    dbg = md.get("rerank_chunk_debug", {})
    if not isinstance(dbg, dict):
        dbg = {}

    final_from_data: list[str] = []
    chunks = retrieval.get("data", {}).get("chunks", []) if isinstance(retrieval, dict) else []
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            cid = str(chunk.get("chunk_id") or "").strip()
            if cid:
                final_from_data.append(cid)

    chunk_ids_all = _normalize_chunk_ids(dbg.get("chunk_ids_all"))
    chunk_ids_threshold = _normalize_chunk_ids(dbg.get("chunk_ids_after_threshold"))
    chunk_ids_final = _normalize_chunk_ids(dbg.get("chunk_ids_final"))
    if not chunk_ids_final:
        chunk_ids_final = final_from_data
    if not chunk_ids_threshold and chunk_ids_final:
        chunk_ids_threshold = list(chunk_ids_final)
    if not chunk_ids_all and chunk_ids_threshold:
        chunk_ids_all = list(chunk_ids_threshold)

    return {
        "all": chunk_ids_all,
        "after_threshold": chunk_ids_threshold,
        "final": chunk_ids_final,
    }


def map_chunk_ids_to_doc_ids(
    chunk_source_map: dict[str, dict[str, Any]],
    chunk_ids: list[str],
    record_key: str,
    record_id: Any,
    stage: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    warns: list[dict[str, Any]] = []
    seen: set[int] = set()
    retrieved: list[int] = []

    for raw_cid in chunk_ids:
        cid = str(raw_cid or "").strip()
        if not cid:
            warns.append({record_key: record_id, "stage": stage, "chunk_id": cid, "reason": "missing chunk_id"})
            continue
        mapped = chunk_source_map.get(cid)
        if not isinstance(mapped, dict):
            warns.append({record_key: record_id, "stage": stage, "chunk_id": cid, "reason": "chunk_id not in source map"})
            continue
        doc_id = parse_int(mapped.get("source_doc_id"))
        if doc_id is None:
            warns.append({record_key: record_id, "stage": stage, "chunk_id": cid, "reason": "invalid source_doc_id in map"})
            continue
        if doc_id not in seen:
            seen.add(doc_id)
            retrieved.append(doc_id)
    return retrieved, warns


async def map_chunks_to_doc_ids(
    chunk_source_map: dict[str, dict[str, Any]],
    retrieval: dict[str, Any],
    record_key: str,
    record_id: Any,
) -> tuple[list[int], list[dict[str, Any]]]:
    chunks = retrieval.get("data", {}).get("chunks", []) if isinstance(retrieval, dict) else []
    if not isinstance(chunks, list):
        chunks = []
    chunk_ids: list[str] = []
    for c in chunks:
        if isinstance(c, dict):
            chunk_ids.append(str(c.get("chunk_id") or "").strip())
    return map_chunk_ids_to_doc_ids(
        chunk_source_map=chunk_source_map,
        chunk_ids=chunk_ids,
        record_key=record_key,
        record_id=record_id,
        stage="final",
    )


def compute_macro_recall(
    rows: list[dict[str, Any]],
    ks: list[int],
    recall_key: str = "recall_at_k",
) -> dict[str, float | None]:
    macro: dict[str, float | None] = {}
    for k in ks:
        key = str(k)
        vals = [
            float(r.get(recall_key, {}).get(key))
            for r in rows
            if isinstance(r.get(recall_key, {}).get(key), (int, float))
        ]
        macro[key] = round(sum(vals) / len(vals), 6) if vals else None
    return macro


def compute_micro_recall(
    rows: list[dict[str, Any]],
    ks: list[int],
    hit_key: str = "hit_at_k",
) -> dict[str, float | None]:
    denom = sum(int(r.get("gt_count", 0)) for r in rows if int(r.get("gt_count", 0)) > 0)
    micro: dict[str, float | None] = {}
    for k in ks:
        key = str(k)
        if denom <= 0:
            micro[key] = None
            continue
        hits = sum(
            int(r.get(hit_key, {}).get(key, 0))
            for r in rows
            if isinstance(r.get(hit_key, {}).get(key), int)
        )
        micro[key] = round(hits / denom, 6)
    return micro


def compute_scope_macro_micro_recall(
    rows: list[dict[str, Any]],
    ks: list[int],
    recall_scopes_key: str,
    hit_scopes_key: str,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, float | None]]]:
    scopes = ("all", "threshold", "final")
    macro: dict[str, dict[str, float | None]] = {}
    micro: dict[str, dict[str, float | None]] = {}
    for scope in scopes:
        scoped_rows: list[dict[str, Any]] = []
        for row in rows:
            scoped_recall = (row.get(recall_scopes_key, {}) or {}).get(scope, {})
            scoped_hit = (row.get(hit_scopes_key, {}) or {}).get(scope, {})
            scoped_rows.append(
                {
                    "gt_count": row.get("gt_count", 0),
                    "recall_at_k": scoped_recall if isinstance(scoped_recall, dict) else {},
                    "hit_at_k": scoped_hit if isinstance(scoped_hit, dict) else {},
                }
            )
        macro[scope] = compute_macro_recall(scoped_rows, ks, recall_key="recall_at_k")
        micro[scope] = compute_micro_recall(scoped_rows, ks, hit_key="hit_at_k")
    return macro, micro


def summarize_final_cut_reason(
    rerank_rows: list[dict[str, Any]],
    *,
    chunk_top_k: int,
) -> dict[str, Any]:
    drop_threshold = 0
    drop_chunk_top_k = 0
    drop_final = 0
    for row in rerank_rows:
        counts = row.get("counts", {}) if isinstance(row, dict) else {}
        if not isinstance(counts, dict):
            counts = {}
        c_all = parse_non_negative_int(counts.get("all")) or 0
        c_thr = parse_non_negative_int(counts.get("after_threshold"))
        c_thr = c_all if c_thr is None else c_thr
        c_after_top_k = parse_non_negative_int(counts.get("after_chunk_top_k"))
        c_after_top_k = c_thr if c_after_top_k is None else c_after_top_k
        c_final = parse_non_negative_int(counts.get("final"))
        c_final = c_after_top_k if c_final is None else c_final
        drop_threshold += max(0, c_all - c_thr)
        drop_chunk_top_k += max(0, c_thr - c_after_top_k)
        drop_final += max(0, c_after_top_k - c_final)

    reasons: list[str] = []
    if drop_threshold > 0:
        reasons.append("min_rerank_score")
    if drop_chunk_top_k > 0:
        reasons.append("chunk_top_k")
    if drop_final > 0:
        reasons.append("token_or_context_budget")
    if not reasons:
        reasons.append("none")
    if chunk_top_k == 0 and "chunk_top_k" in reasons:
        reasons = [r for r in reasons if r != "chunk_top_k"] or ["none"]

    return {
        "final_cut_reason": "+".join(reasons),
        "final_cut_breakdown": {
            "dropped_by_threshold": drop_threshold,
            "dropped_by_chunk_top_k": drop_chunk_top_k,
            "dropped_after_top_k_to_final": drop_final,
            "chunk_top_k_limit_enabled": chunk_top_k > 0,
        },
    }


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        numeric = float(v)
        if numeric in (0.0, 1.0):
            return bool(int(numeric))
        raise ValueError(f"invalid bool numeric value: {v!r}")
    if isinstance(v, str):
        return as_bool(v)
    raise ValueError(f"invalid bool value: {v!r}")


def has_matching_survey_retrieval(
    summary: dict[str, Any],
    args: argparse.Namespace,
    ks: list[int],
    expected_survey_count: int | None = None,
) -> bool:
    expected_flags = get_ablation_flags(args)
    expected_chunk_top_k = resolve_chunk_top_k(args.chunk_top_k, ks)
    expected_subset = Path(args.data_root) / args.subset_dir
    summary_checks = {
        "mode": "survey",
        "survey_stage": "retrieval",
    }
    for key, value in summary_checks.items():
        if str(summary.get(key)) != value:
            return False
    if parse_int(summary.get("top_k")) != args.top_k:
        return False
    if parse_int(summary.get("chunk_top_k")) != expected_chunk_top_k:
        return False
    if list(summary.get("k_list") or []) != ks:
        return False

    summary_checks_extra = {
        "workspace_id": str(args.workspace_id),
        "data_root": str(Path(args.data_root)),
        "subset_dir": str(expected_subset),
        "surveys_file": str(expected_subset / args.surveys_file),
        "chunks_file": str(expected_subset / args.chunks_file),
        "corpus_file": str(expected_subset / args.corpus_file),
    }
    for key, value in summary_checks_extra.items():
        if str(summary.get(key)) != value:
            return False
    params = summary.get("effective_query_params", {})
    if not isinstance(params, dict):
        return False
    if str(params.get("mode")) != str(args.query_mode):
        return False
    if parse_int(params.get("top_k")) != args.top_k:
        return False
    if parse_int(params.get("chunk_top_k")) != expected_chunk_top_k:
        return False
    try:
        enable_rerank = to_bool(params.get("enable_rerank"))
    except Exception:
        return False
    if enable_rerank != bool(args.enable_rerank):
        return False
    if str(params.get("rerank_score_scope", "all")) != "all":
        return False
    if str(
        params.get("kg_chunk_selection_source", DEFAULT_KG_CHUNK_SELECTION_SOURCE)
    ) != str(
        getattr(args, "kg_chunk_selection_source", DEFAULT_KG_CHUNK_SELECTION_SOURCE)
    ):
        return False
    stored_flags = AblationFlags.from_mapping(params.get("ablation_flags"))
    if stored_flags is None or stored_flags != expected_flags:
        return False
    if list(params.get("k_list") or []) != ks:
        return False
    if expected_survey_count is not None:
        if parse_int(summary.get("survey_count")) != expected_survey_count:
            return False
    if parse_int(summary.get("failed_count")) != 0:
        return False
    return True


def has_matching_query_retrieval(
    summary: dict[str, Any],
    args: argparse.Namespace,
    ks: list[int],
    expected_query_count: int | None = None,
) -> bool:
    expected_flags = get_ablation_flags(args)
    expected_chunk_top_k = resolve_chunk_top_k(args.chunk_top_k, ks)
    expected_subset = Path(args.data_root) / args.subset_dir
    if str(summary.get("mode")) != "retrieval":
        return False
    if parse_int(summary.get("top_k")) != args.top_k:
        return False
    if parse_int(summary.get("chunk_top_k")) != expected_chunk_top_k:
        return False
    if list(summary.get("k_list") or []) != ks:
        return False

    summary_checks_extra = {
        "workspace_id": str(args.workspace_id),
        "data_root": str(Path(args.data_root)),
        "subset_dir": str(expected_subset),
        "queries_file": str(expected_subset / args.queries_file),
        "chunks_file": str(expected_subset / args.chunks_file),
        "corpus_file": str(expected_subset / args.corpus_file),
    }
    for key, value in summary_checks_extra.items():
        if str(summary.get(key)) != value:
            return False

    params = summary.get("effective_query_params", {})
    if not isinstance(params, dict):
        return False
    if str(params.get("mode")) != str(args.query_mode):
        return False
    if parse_int(params.get("top_k")) != args.top_k:
        return False
    if parse_int(params.get("chunk_top_k")) != expected_chunk_top_k:
        return False
    try:
        enable_rerank = to_bool(params.get("enable_rerank"))
    except Exception:
        return False
    if enable_rerank != bool(args.enable_rerank):
        return False
    if str(params.get("rerank_score_scope", "all")) != "all":
        return False
    if str(
        params.get("kg_chunk_selection_source", DEFAULT_KG_CHUNK_SELECTION_SOURCE)
    ) != str(
        getattr(args, "kg_chunk_selection_source", DEFAULT_KG_CHUNK_SELECTION_SOURCE)
    ):
        return False
    stored_flags = AblationFlags.from_mapping(params.get("ablation_flags"))
    if stored_flags is None or stored_flags != expected_flags:
        return False
    if list(params.get("k_list") or []) != ks:
        return False
    if expected_query_count is not None:
        if parse_int(summary.get("query_count")) != expected_query_count:
            return False
    if parse_int(summary.get("failed_count")) != 0:
        return False
    return True


async def ensure_query_retrieval_for_survey(args: argparse.Namespace) -> None:
    ks = parse_k_list(args.k_list)
    data_root = Path(args.data_root)
    subset = data_root / args.subset_dir
    expected_query_count = len(load_queries(subset / args.queries_file, args.limit))

    need_retrieval = True
    if SUMMARY_FILE.exists() and PER_QUERY_FILE.exists() and RERANK_STATS_FILE.exists():
        try:
            summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                need_retrieval = not has_matching_query_retrieval(
                    summary=summary,
                    args=args,
                    ks=ks,
                    expected_query_count=expected_query_count,
                )
        except Exception:
            need_retrieval = True

    if need_retrieval:
        logger.info(
            "Survey mode requires query-level retrieval results; auto-running retrieval mode."
        )
        await run_retrieval(args)
    else:
        logger.info(
            "Query-level retrieval results exist and match current parameters; skip auto-retrieval."
        )


async def run_retrieval(args: argparse.Namespace) -> int:
    ks = parse_k_list(args.k_list)
    chunk_top_k = resolve_chunk_top_k(args.chunk_top_k, ks)
    QueryParam, _, _ = import_rag_dependencies()
    data_root = Path(args.data_root)
    subset = data_root / args.subset_dir
    chunks_by_doc, chunk_stats = load_chunks(subset / args.chunks_file)
    source_records, chunk_source_map, source_map_stats = prepare_source_records(chunks_by_doc)
    persist_chunk_source_map(chunk_source_map, source_map_stats)
    queries = load_queries(subset / args.queries_file, args.limit)
    async with prepared_workspace_service(
        args,
        source_records,
        stage="retrieval evaluation",
    ) as (service, ablation_flags, ensured_index_profile, ingest_summary):
        query_params = build_query_params(args, chunk_top_k=chunk_top_k)
        sem = asyncio.Semaphore(max(1, args.max_concurrency))
        done = 0
        lock = asyncio.Lock()
        total = len(queries)

        async def one(i: int, item: dict[str, Any]):
            nonlocal done
            qid = item.get("query_id", i + 1)
            q = str(item.get("prefix_titles_query") or "").strip()
            gt = {int(x) for x in item.get("cites", []) if parse_int(x) is not None}
            t0 = time.perf_counter()
            warns = []
            error = None
            retrieved: list[int] = []
            retrieval = {}
            async with sem:
                try:
                    if not q:
                        raise ValueError("empty prefix_titles_query")
                    param = QueryParam(**query_params)
                    retrieval = await with_retries(
                        lambda: service.lightrag_aquery_data(
                            args.workspace_id,
                            q,
                            param=param,
                        ),
                        label=f"query {qid}",
                        retries=args.max_retries,
                    )
                    rerank_for_contract = extract_rerank_payload(retrieval, query_params)
                    assert_rerank_contract(
                        rerank_payload=rerank_for_contract,
                        query_params=query_params,
                        record_key="query_id",
                        record_id=qid,
                    )
                    retrieved, warns = await map_chunks_to_doc_ids(
                        chunk_source_map=chunk_source_map,
                        retrieval=retrieval,
                        record_key="query_id",
                        record_id=qid,
                    )
                except Exception as exc:
                    error = {
                        "query_id": qid,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": "".join(
                            traceback.format_exception(
                                type(exc),
                                exc,
                                exc.__traceback__,
                            )[-3:]
                        ).strip(),
                    }
            row = {
                "query_id": qid,
                "question": q,
                "category": item.get("category"),
                "gt_count": len(gt),
                "retrieved_count": len(retrieved),
                "gt_doc_ids": sorted(gt),
                "retrieved_doc_ids": retrieved,
                "hit_at_k": {str(k): len(gt & set(retrieved[:k])) for k in ks},
                "recall_at_k": recall_at_k(gt, retrieved, ks),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error": error,
            }
            rerank = extract_rerank_payload(retrieval, query_params)
            rerank_row = {
                "query_id": qid,
                "question": q,
                "rerank_scope": rerank.get("rerank_scope"),
                "min_rerank_score": rerank.get("min_rerank_score"),
                "counts": rerank.get("counts", {}),
                "distribution": rerank.get("distribution", {}),
                "scores": rerank.get("scores", {}),
                "threshold_retention": rerank.get("threshold_retention", []),
                "category": item.get("category"),
                "top_k": args.top_k,
                "chunk_top_k": chunk_top_k,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            async with lock:
                done += 1
                if done == total or done % max(1, total // 10) == 0:
                    logger.info("Retrieval progress: %d/%d", done, total)
            return row, rerank_row, warns, error

        results = await asyncio.gather(
            *[asyncio.create_task(one(i, q)) for i, q in enumerate(queries)]
        )
        per_rows, rerank_rows, warnings, errors = [], [], [], []
        for pr, rr, ws, err in results:
            per_rows.append(pr)
            rerank_rows.append(rr)
            warnings.extend(ws)
            if err:
                errors.append(err)
        append_jsonl(PER_QUERY_FILE, per_rows)
        append_jsonl(RERANK_STATS_FILE, rerank_rows)
        append_jsonl(WARNINGS_FILE, warnings)
        avg = compute_macro_recall(per_rows, ks)
        save_json(SUMMARY_FILE, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "retrieval",
            "top_k": args.top_k,
            "chunk_top_k": chunk_top_k,
            "k_list": ks,
            "query_count": len(per_rows),
            "success_count": sum(1 for r in per_rows if r.get("error") is None),
            "failed_count": len(errors),
            "mapping_warning_count": len(warnings),
            "non_empty_retrieval_count": sum(
                1 for r in per_rows if int(r.get("retrieved_count", 0)) > 0
            ),
            "avg_recall_at_k": avg,
            "workspace_id": args.workspace_id,
            "data_root": str(data_root),
            "subset_dir": str(subset),
            "queries_file": str(subset / args.queries_file),
            "chunks_file": str(subset / args.chunks_file),
            "corpus_file": str(subset / args.corpus_file),
            "effective_query_params": query_params | {
                "ablation_group": ablation_flags.ablation_group(),
                "ablation_flags": ablation_flags.to_dict(),
                "index_profile": ensured_index_profile,
                "max_concurrency": args.max_concurrency,
                "max_retries": args.max_retries,
                "k_list": ks,
            },
            "ingest_summary": ingest_summary,
            "chunks_source_stats": chunk_stats,
        })
        all_scores = [
            s
            for r in rerank_rows
            for s in r.get("scores", {}).get("all", [])
            if isinstance(s, (int, float))
        ]
        thr_scores = [
            s
            for r in rerank_rows
            for s in r.get("scores", {}).get("after_threshold", [])
            if isinstance(s, (int, float))
        ]
        fin_scores = [
            s
            for r in rerank_rows
            for s in r.get("scores", {}).get("final", [])
            if isinstance(s, (int, float))
        ]
        save_json(RERANK_SUMMARY_FILE, {
            "total_queries": len(rerank_rows),
            "questions_with_rerank_trace": count_rows_with_rerank_trace(rerank_rows),
            "overall_distribution": {
                "all": score_distribution([float(x) for x in all_scores]),
                "after_threshold": score_distribution([float(x) for x in thr_scores]),
                "final": score_distribution([float(x) for x in fin_scores]),
            },
            "macro_distribution_over_queries": summarize_macro_distribution(rerank_rows),
            "threshold_retention_overall": summarize_threshold_retention(rerank_rows),
        })
        logger.info("Retrieval complete: %s", SUMMARY_FILE)
        return 0


async def run_survey_retrieval(args: argparse.Namespace) -> int:
    survey_ks = parse_k_list(args.survey_k_list)
    chunk_top_k = resolve_chunk_top_k(args.chunk_top_k, survey_ks)
    QueryParam, _, _ = import_rag_dependencies()
    data_root = Path(args.data_root)
    subset = data_root / args.subset_dir
    chunks_by_doc, chunk_stats = load_chunks(subset / args.chunks_file)
    source_records, chunk_source_map, source_map_stats = prepare_source_records(chunks_by_doc)
    persist_chunk_source_map(chunk_source_map, source_map_stats)
    surveys = load_surveys(subset / args.surveys_file, args.limit)
    async with prepared_workspace_service(
        args,
        source_records,
        stage="survey retrieval evaluation",
    ) as (service, ablation_flags, ensured_index_profile, ingest_summary):
        query_params = build_query_params(args, chunk_top_k=chunk_top_k)

        sem = asyncio.Semaphore(max(1, args.max_concurrency))
        done = 0
        lock = asyncio.Lock()
        total = len(surveys)

        async def one(i: int, item: dict[str, Any]):
            nonlocal done
            survey_id = item.get("survey_id", i + 1)
            survey_title = str(item.get("survey_title") or "").strip()
            gt = {int(x) for x in item.get("all_cites", []) if parse_int(x) is not None}
            t0 = time.perf_counter()
            warns = []
            error = None
            retrieved: list[int] = []
            retrieved_by_scope: dict[str, list[int]] = {
                "all": [],
                "threshold": [],
                "final": [],
            }
            retrieval = {}
            rerank: dict[str, Any] = {}

            async with sem:
                try:
                    if not survey_title:
                        raise ValueError("empty survey_title")
                    param = QueryParam(**query_params)
                    retrieval = await with_retries(
                        lambda: service.lightrag_aquery_data(
                            args.workspace_id,
                            survey_title,
                            param=param,
                        ),
                        label=f"survey {survey_id}",
                        retries=args.max_retries,
                    )
                    rerank = extract_rerank_payload(retrieval, query_params)
                    assert_rerank_contract(
                        rerank_payload=rerank,
                        query_params=query_params,
                        record_key="survey_id",
                        record_id=survey_id,
                    )
                    chunk_ids = (
                        rerank.get("chunk_ids", {}) if isinstance(rerank, dict) else {}
                    )
                    if not isinstance(chunk_ids, dict):
                        chunk_ids = {}

                    all_chunk_ids = chunk_ids.get("all", [])
                    thr_chunk_ids = chunk_ids.get("after_threshold", [])
                    fin_chunk_ids = chunk_ids.get("final", [])
                    if not isinstance(all_chunk_ids, list):
                        all_chunk_ids = []
                    if not isinstance(thr_chunk_ids, list):
                        thr_chunk_ids = []
                    if not isinstance(fin_chunk_ids, list):
                        fin_chunk_ids = []

                    all_docs, all_warns = map_chunk_ids_to_doc_ids(
                        chunk_source_map=chunk_source_map,
                        chunk_ids=all_chunk_ids,
                        record_key="survey_id",
                        record_id=survey_id,
                        stage="all",
                    )
                    thr_docs, thr_warns = map_chunk_ids_to_doc_ids(
                        chunk_source_map=chunk_source_map,
                        chunk_ids=thr_chunk_ids,
                        record_key="survey_id",
                        record_id=survey_id,
                        stage="threshold",
                    )
                    fin_docs, fin_warns = map_chunk_ids_to_doc_ids(
                        chunk_source_map=chunk_source_map,
                        chunk_ids=fin_chunk_ids,
                        record_key="survey_id",
                        record_id=survey_id,
                        stage="final",
                    )
                    warns = all_warns + thr_warns + fin_warns
                    retrieved_by_scope["all"] = all_docs
                    retrieved_by_scope["threshold"] = thr_docs
                    retrieved_by_scope["final"] = fin_docs
                    retrieved = list(fin_docs)
                except Exception as exc:
                    error = {
                        "survey_id": survey_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": "".join(
                            traceback.format_exception(
                                type(exc),
                                exc,
                                exc.__traceback__,
                            )[-3:]
                        ).strip(),
                    }

            hit_final = hit_at_k(gt, retrieved_by_scope["final"], survey_ks)
            recall_final = recall_at_k(gt, retrieved_by_scope["final"], survey_ks)
            hit_by_scope = {
                "all": hit_at_k(gt, retrieved_by_scope["all"], survey_ks),
                "threshold": hit_at_k(gt, retrieved_by_scope["threshold"], survey_ks),
                "final": hit_final,
            }
            recall_by_scope = {
                "all": recall_at_k(gt, retrieved_by_scope["all"], survey_ks),
                "threshold": recall_at_k(gt, retrieved_by_scope["threshold"], survey_ks),
                "final": recall_final,
            }
            row = {
                "survey_id": survey_id,
                "survey_title": survey_title,
                "gt_count": len(gt),
                "retrieved_count": len(retrieved),
                "retrieved_count_by_scope": {
                    "all": len(retrieved_by_scope["all"]),
                    "threshold": len(retrieved_by_scope["threshold"]),
                    "final": len(retrieved_by_scope["final"]),
                },
                "gt_doc_ids": sorted(gt),
                "retrieved_doc_ids": retrieved,
                "retrieved_doc_ids_by_scope": retrieved_by_scope,
                "hit_at_k": hit_final,
                "recall_at_k": recall_final,
                "hit_at_k_by_scope": hit_by_scope,
                "recall_at_k_by_scope": recall_by_scope,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error": error,
            }
            if not rerank:
                rerank = extract_rerank_payload(retrieval, query_params)
            rerank_row = {
                "survey_id": survey_id,
                "survey_title": survey_title,
                "rerank_scope": rerank.get("rerank_scope"),
                "min_rerank_score": rerank.get("min_rerank_score"),
                "counts": rerank.get("counts", {}),
                "distribution": rerank.get("distribution", {}),
                "scores": rerank.get("scores", {}),
                "chunk_ids": rerank.get("chunk_ids", {}),
                "threshold_retention": rerank.get("threshold_retention", []),
                "top_k": args.top_k,
                "chunk_top_k": chunk_top_k,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            async with lock:
                done += 1
                if done == total or done % max(1, total // 5) == 0:
                    logger.info("Survey retrieval progress: %d/%d", done, total)
            return row, rerank_row, warns, error

        results = await asyncio.gather(
            *[asyncio.create_task(one(i, s)) for i, s in enumerate(surveys)]
        )
        per_rows: list[dict[str, Any]] = []
        rerank_rows: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for pr, rr, ws, err in results:
            per_rows.append(pr)
            rerank_rows.append(rr)
            warnings.extend(ws)
            if err:
                errors.append(err)

        append_jsonl(SURVEY_PER_FILE, per_rows)
        append_jsonl(SURVEY_RERANK_STATS_FILE, rerank_rows)
        append_jsonl(SURVEY_WARNINGS_FILE, warnings)

        macro = compute_macro_recall(per_rows, survey_ks)
        micro = compute_micro_recall(per_rows, survey_ks)
        macro_by_scope, micro_by_scope = compute_scope_macro_micro_recall(
            per_rows,
            survey_ks,
            recall_scopes_key="recall_at_k_by_scope",
            hit_scopes_key="hit_at_k_by_scope",
        )
        cut_summary = summarize_final_cut_reason(
            rerank_rows,
            chunk_top_k=chunk_top_k,
        )

        save_json(SURVEY_SUMMARY_FILE, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "survey",
            "survey_stage": "retrieval",
            "top_k": args.top_k,
            "chunk_top_k": chunk_top_k,
            "k_list": survey_ks,
            "survey_count": len(per_rows),
            "success_count": sum(1 for r in per_rows if r.get("error") is None),
            "failed_count": len(errors),
            "mapping_warning_count": len(warnings),
            "non_empty_retrieval_count": sum(
                1 for r in per_rows if int(r.get("retrieved_count", 0)) > 0
            ),
            "macro_recall_at_k": macro,
            "micro_recall_at_k": micro,
            "macro_recall_at_k_by_scope": macro_by_scope,
            "micro_recall_at_k_by_scope": micro_by_scope,
            "final_cut_reason": cut_summary.get("final_cut_reason"),
            "final_cut_breakdown": cut_summary.get("final_cut_breakdown", {}),
            "workspace_id": args.workspace_id,
            "data_root": str(data_root),
            "subset_dir": str(subset),
            "surveys_file": str(subset / args.surveys_file),
            "chunks_file": str(subset / args.chunks_file),
            "corpus_file": str(subset / args.corpus_file),
            "effective_query_params": query_params | {
                "ablation_group": ablation_flags.ablation_group(),
                "ablation_flags": ablation_flags.to_dict(),
                "index_profile": ensured_index_profile,
                "max_concurrency": args.max_concurrency,
                "max_retries": args.max_retries,
                "k_list": survey_ks,
            },
            "ingest_summary": ingest_summary,
            "chunks_source_stats": chunk_stats,
        })

        all_scores = [
            s
            for r in rerank_rows
            for s in r.get("scores", {}).get("all", [])
            if isinstance(s, (int, float))
        ]
        thr_scores = [
            s
            for r in rerank_rows
            for s in r.get("scores", {}).get("after_threshold", [])
            if isinstance(s, (int, float))
        ]
        fin_scores = [
            s
            for r in rerank_rows
            for s in r.get("scores", {}).get("final", [])
            if isinstance(s, (int, float))
        ]
        save_json(SURVEY_RERANK_SUMMARY_FILE, {
            "total_surveys": len(rerank_rows),
            "questions_with_rerank_trace": count_rows_with_rerank_trace(rerank_rows),
            "overall_distribution": {
                "all": score_distribution([float(x) for x in all_scores]),
                "after_threshold": score_distribution([float(x) for x in thr_scores]),
                "final": score_distribution([float(x) for x in fin_scores]),
            },
            "macro_distribution_over_surveys": summarize_macro_distribution(
                rerank_rows
            ),
            "threshold_retention_overall": summarize_threshold_retention(rerank_rows),
        })
        logger.info("Survey retrieval complete: %s", SURVEY_SUMMARY_FILE)
        return 0


async def run_survey_generate_placeholder(args: argparse.Namespace) -> int:
    survey_ks = parse_k_list(args.survey_k_list)
    data_root = Path(args.data_root)
    subset = data_root / args.subset_dir
    expected_survey_count = len(load_surveys(subset / args.surveys_file, args.limit))

    need_retrieval = True
    if (
        SURVEY_SUMMARY_FILE.exists()
        and SURVEY_PER_FILE.exists()
        and SURVEY_RERANK_STATS_FILE.exists()
    ):
        try:
            summary = json.loads(SURVEY_SUMMARY_FILE.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                need_retrieval = not has_matching_survey_retrieval(
                    summary=summary,
                    args=args,
                    ks=survey_ks,
                    expected_survey_count=expected_survey_count,
                )
        except Exception:
            need_retrieval = True

    if need_retrieval:
        logger.info("Survey generate stage requires survey retrieval results; auto-running retrieval stage.")
        await run_survey_retrieval(args)
    else:
        logger.info("Survey retrieval results exist and match current parameters; skip auto-retrieval.")

    save_json(SURVEY_STATUS, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "survey",
        "survey_stage": "generate",
        "status": "not_implemented",
        "message": "Survey generation/evaluation is not implemented yet in evaluate_surge_fast.py",
        "workspace_id": args.workspace_id,
        "depends_on_survey_retrieval": str(SURVEY_SUMMARY_FILE),
    })
    logger.info("Survey generate placeholder written: %s", SURVEY_STATUS)
    return 0


async def run_survey(args: argparse.Namespace) -> int:
    await ensure_query_retrieval_for_survey(args)
    if args.survey_stage == "retrieval":
        return await run_survey_retrieval(args)
    if args.survey_stage == "generate":
        return await run_survey_generate_placeholder(args)
    raise ValueError(f"Unsupported survey-stage: {args.survey_stage}")


def resolve_log_mode(args: argparse.Namespace) -> str:
    if args.mode == "survey":
        return f"survey_{args.survey_stage}"
    return "retrieval"


def resolve_log_mode_from_argv(argv: list[str]) -> str:
    mode = "retrieval"
    survey_stage = "retrieval"
    for i, token in enumerate(argv):
        if token.startswith("--mode="):
            mode = token.split("=", 1)[1].strip() or mode
        elif token == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1].strip() or mode
        elif token.startswith("--survey-stage="):
            survey_stage = token.split("=", 1)[1].strip() or survey_stage
        elif token == "--survey-stage" and i + 1 < len(argv):
            survey_stage = argv[i + 1].strip() or survey_stage
    if mode == "survey":
        return f"survey_{survey_stage}"
    return "retrieval"


def parse_args_and_bootstrap(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    ensure_dirs()
    refresh_logging(resolve_log_mode_from_argv(argv))
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        if code != 0:
            logger.error("Argument parsing failed with exit code %s", code)
        raise
    refresh_logging(resolve_log_mode(args))
    logger.info("Args: %s", {k: getattr(args, k) for k in sorted(vars(args).keys())})
    return args


def validate_args(args: argparse.Namespace) -> None:
    parse_k_list(args.k_list)
    parse_k_list(args.survey_k_list)
    if args.ingest_concurrency is not None:
        if args.ingest_concurrency <= 0:
            raise ValueError(
                f"--ingest-concurrency must be > 0, got {args.ingest_concurrency}"
            )
        if args.batch_doc_concurrency == 1 and args.ingest_concurrency != 1:
            args.batch_doc_concurrency = int(args.ingest_concurrency)
            logger.warning(
                "--ingest-concurrency is deprecated in evaluate_surge_fast.py; "
                "mapped to --batch-doc-concurrency=%d",
                args.batch_doc_concurrency,
            )
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be > 0, got {args.top_k}")
    if args.chunk_top_k == 0:
        pass
    elif args.chunk_top_k < 0:
        raise ValueError(
            f"--chunk-top-k must be > 0 or 0(disabled), got {args.chunk_top_k}"
        )
    if not bool(args.enable_rerank):
        raise ValueError(
            "--enable-rerank must be true for evaluation; selected chunks must be reranked."
        )
    if args.max_concurrency <= 0:
        raise ValueError(
            f"--max-concurrency must be > 0, got {args.max_concurrency}"
        )
    if args.batch_doc_concurrency <= 0:
        raise ValueError(
            f"--batch-doc-concurrency must be > 0, got {args.batch_doc_concurrency}"
        )
    if args.ingest_batch_size <= 0:
        raise ValueError(
            f"--ingest-batch-size must be > 0, got {args.ingest_batch_size}"
        )
    if args.llm_model_max_async <= 0:
        raise ValueError(
            f"--llm-model-max-async must be > 0, got {args.llm_model_max_async}"
        )
    if args.max_retries < 0:
        raise ValueError(f"--max-retries must be >= 0, got {args.max_retries}")
    args.ablation_flags = validate_ablation_flags(args, naming_style="hyphen")
    validate_workspace_env_isolation(workspace_id=args.workspace_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SurGE evaluation for RAGAnything")
    p.add_argument("--mode", choices=["retrieval", "survey"], default="retrieval")
    p.add_argument("--survey-stage", choices=["retrieval", "generate"], default="retrieval")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--subset-dir", default=DEFAULT_SUBSET_DIR)
    p.add_argument("--queries-file", default=DEFAULT_QUERIES)
    p.add_argument("--surveys-file", default=DEFAULT_SURVEYS)
    p.add_argument("--chunks-file", default=DEFAULT_CHUNKS)
    p.add_argument("--corpus-file", default=DEFAULT_CORPUS)
    p.add_argument("--workspace-id", default=DEFAULT_WORKSPACE)
    p.add_argument(
        "--query-mode",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass", "ppr_local", "ppr"],
        default="hybrid",
    )
    p.add_argument(
        "--keyword_fanout_mode",
        choices=["joined", "per_keyword_rrf"],
        default="joined",
    )
    p.add_argument(
        "--entity_retrieval_mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
    )
    p.add_argument(
        "--chunk_retrieval_mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
    )
    p.add_argument(
        "--kg-chunk-selection-source",
        choices=["truncated", "untruncated"],
        default=DEFAULT_KG_CHUNK_SELECTION_SOURCE,
        help=(
            "KG source set for entity/relation-related chunk selection. "
            "Default keeps prompt-truncated KG results."
        ),
    )
    p.add_argument(
        "--exclude_synonym_edges",
        type=as_bool,
        default=None,
        help="Override query-time synonym-edge filtering. Omit to keep auto/default behavior.",
    )
    p.add_argument("--bypass_query_cache", action="store_true")
    p.add_argument("--bypass_keywords_cache", action="store_true")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument(
        "--chunk-top-k",
        type=int,
        default=0,
        help="0 disables chunk_top_k truncation; >0 keeps only top-k chunks before final token budgeting.",
    )
    p.add_argument("--max-total-tokens", type=int, default=45000)
    p.add_argument("--k-list", default="5,10,20,30,50")
    p.add_argument("--survey-k-list", default="50,100,200,500")
    p.add_argument("--enable-rerank", type=as_bool, default=True)
    p.add_argument(
        "--batch-doc-concurrency",
        type=int,
        default=2,
        help="Concurrent virtual batch-doc ingest workers (default 2).",
    )
    p.add_argument(
        "--ingest-concurrency",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ingest-batch-size",
        type=int,
        default=384,
        help="Number of source chunks packed into one virtual batch-doc.",
    )
    p.add_argument(
        "--llm-model-max-async",
        type=int,
        default=48,
        help="Evaluate-fast override for LLM extraction worker concurrency during ingest.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Evaluation query concurrency (query-level / survey-retrieval).",
    )
    p.add_argument("--max-retries", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 means all queries/surveys")
    p.add_argument(
        "--allow-legacy-index-profile-adoption",
        type=as_bool,
        default=False,
        help=(
            "When true, allow adopting current V1/V2 index profile for an existing "
            "workspace without profile metadata. Keep false for strict ablation isolation."
        ),
    )
    add_ablation_arguments(p)
    return p


async def amain(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.mode == "retrieval":
        return await run_retrieval(args)
    return await run_survey(args)


def main() -> int:
    try:
        args = parse_args_and_bootstrap()
        return asyncio.run(amain(args))
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        return code
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        logger.debug("Fatal traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
