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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MINERU_VLLM_GPU_MEMORY_UTILIZATION", "0.1")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SCRIPT_DIR = Path(__file__).resolve().parent
RETRIEVAL_DIR = SCRIPT_DIR / "retrieval_results"
SURVEY_DIR = SCRIPT_DIR / "survey_results"
LOG_DIR = SCRIPT_DIR / "logs"
RAG_STORAGE_DIR = SCRIPT_DIR / "rag_storage"
RAG_OUTPUT_DIR = SCRIPT_DIR / "rag_outputs"

DEFAULT_DATA_ROOT = "/data/y50056788/Yaliang/datasets_for_eval/data_for_SurGE"
DEFAULT_SUBSET_DIR = "subset_output"
DEFAULT_QUERIES = "subset_queries.json"
DEFAULT_CHUNKS = "subset_chunks.jsonl"
DEFAULT_CORPUS = "subset_corpus.json"
DEFAULT_WORKSPACE = "surge_subset_shared"

PER_QUERY_FILE = RETRIEVAL_DIR / "retrieval_per_query.jsonl"
SUMMARY_FILE = RETRIEVAL_DIR / "retrieval_summary.json"
RERANK_STATS_FILE = RETRIEVAL_DIR / "rerank_chunk_stats.jsonl"
RERANK_SUMMARY_FILE = RETRIEVAL_DIR / "rerank_chunk_summary.json"
WARNINGS_FILE = RETRIEVAL_DIR / "mapping_warnings.jsonl"
RUN_MANIFEST = RETRIEVAL_DIR / "run_manifest.json"
INGEST_MANIFEST = RETRIEVAL_DIR / "shared_ingest_manifest.json"
INGEST_FAILURES = RETRIEVAL_DIR / "shared_ingest_failures.jsonl"
SURVEY_STATUS = SURVEY_DIR / "survey_mode_status.json"

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


def as_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    t = str(v).strip().lower()
    if t in {"1", "true", "yes", "y", "on"}:
        return True
    if t in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


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


def parse_doc_id_from_file_path(v: Any) -> int | None:
    if not isinstance(v, str) or not v.strip():
        return None
    base = os.path.basename(v.strip())
    if base.startswith("surge_doc_"):
        tail = base[len("surge_doc_") :]
        m = re.match(r"(\d+)", tail)
        if m:
            return int(m.group(1))
    m = re.search(r"^(\d+)(?:\.[^.]+)?$", base)
    return int(m.group(1)) if m else None


def slugify_title(v: Any, max_len: int = 80) -> str:
    if not isinstance(v, str):
        return "untitled"
    text = v.strip()
    if not text:
        return "untitled"
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"\s+", "_", ascii_text)
    ascii_text = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_text).strip("_.-")
    if not ascii_text:
        return "untitled"
    return ascii_text[:max_len]


def build_virtual_file_path(doc_id: str, row: dict[str, Any]) -> str:
    title = row.get("title") or row.get("Title") or ""
    slug = slugify_title(title)
    return f"surge_doc_{doc_id}__{slug}.txt"


def ensure_dirs() -> None:
    # For current SurGE retrieval pipeline we ingest prebuilt chunks directly,
    # so parser output dir is not required.
    for p in [RETRIEVAL_DIR, SURVEY_DIR, LOG_DIR, RAG_STORAGE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def refresh_logging(mode: str) -> None:
    global MASTER_LOG
    logging.getLogger("raganything").setLevel(logging.INFO)
    logging.getLogger("raganything.processor").setLevel(logging.INFO)
    logging.getLogger("raganything.parser").setLevel(logging.INFO)
    logging.getLogger("lightrag").setLevel(logging.INFO)
    if MASTER_LOG is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        MASTER_LOG = LOG_DIR / f"evaluate_surge_{mode}_{ts}.log"
    root = logging.getLogger()
    has_master = False
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            p = Path(getattr(h, "baseFilename", ""))
            if p == MASTER_LOG:
                has_master = True
                continue
            root.removeHandler(h)
            h.close()
    if not has_master:
        fh = logging.FileHandler(MASTER_LOG, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root.addHandler(fh)
        logger.info("Master log file: %s", MASTER_LOG)
    for run_log in LOG_DIR.glob("run_*.log"):
        try:
            run_log.unlink()
        except Exception:
            pass
    try:
        from lightrag.utils import logger as lightrag_logger
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                if all(getattr(eh, "baseFilename", None) != getattr(h, "baseFilename", None) for eh in lightrag_logger.handlers):
                    lightrag_logger.addHandler(h)
        lightrag_logger.setLevel(logging.INFO)
    except Exception:
        pass


def settings_for_surge() -> LocalRagSettings:
    _, _, LocalRagSettings = import_rag_dependencies()
    s = LocalRagSettings.from_env()
    s.working_dir_root = str(RAG_STORAGE_DIR)
    s.output_dir = str(RAG_OUTPUT_DIR)
    s.log_dir = str(LOG_DIR)
    return s


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


def load_queries(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    q = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(q, list):
        raise ValueError(f"queries must be list: {path}")
    return q[:limit] if limit > 0 else q


def load_abstract_index(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    idx: dict[str, list[int]] = {}
    for row in data:
        doc_id = parse_int(row.get("doc_id"))
        abs_text = str(row.get("Abstract") or "").strip()
        if doc_id is None or not abs_text:
            continue
        idx.setdefault(abs_text, []).append(doc_id)
    return idx


def build_chunk_row_lookup(rows: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_chunk_id: dict[str, dict[str, Any]] = {}
    ordered_rows: list[dict[str, Any]] = []
    if isinstance(rows, dict):
        iterable = rows.values()
    elif isinstance(rows, list):
        iterable = rows
    else:
        iterable = []
    for row in iterable:
        if not isinstance(row, dict):
            continue
        ordered_rows.append(row)
        rid = str(
            row.get("chunk_id")
            or row.get("id")
            or row.get("_id")
            or row.get("__id__")
            or row.get("key")
            or ""
        ).strip()
        if rid:
            by_chunk_id[rid] = row
    return by_chunk_id, ordered_rows


def chunk_id_for_doc(doc_id: str) -> str:
    return f"surge-chunk-{doc_id}"


def iter_batches(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


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


def score_distribution(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    s = sorted(scores)
    n = len(s)
    mean = sum(s) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / n)
    return {"count": n, "min": round(s[0], 6), "max": round(s[-1], 6), "mean": round(mean, 6), "std": round(std, 6)}


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
    return {
        "rerank_scope": str(dbg.get("scope", query_params.get("rerank_score_scope", "top_k"))),
        "min_rerank_score": dbg.get("min_rerank_score"),
        "counts": {
            "input": int(dbg.get("count_input", len(scores_all))) if str(dbg.get("count_input", "")).isdigit() else len(scores_all),
            "all": int(dbg.get("count_after_rerank", len(scores_all))) if str(dbg.get("count_after_rerank", "")).isdigit() else len(scores_all),
            "after_threshold": int(dbg.get("count_after_threshold", len(scores_thr))) if str(dbg.get("count_after_threshold", "")).isdigit() else len(scores_thr),
            "final": int(dbg.get("count_final", len(scores_final))) if str(dbg.get("count_final", "")).isdigit() else len(scores_final),
        },
        "scores": {"all": scores_all, "after_threshold": scores_thr, "final": scores_final},
        "distribution": {
            "all": score_distribution(scores_all),
            "after_threshold": score_distribution(scores_thr),
            "final": score_distribution(scores_final),
        },
    }


async def ensure_workspace_index(service: LocalRagService, workspace_id: str, chunks_by_doc: dict[str, dict[str, Any]], retries: int) -> dict[str, Any]:
    rag = await service.get_rag(workspace_id)
    await ensure_rag_runtime_ready(rag, workspace_id)
    target = set(chunks_by_doc.keys())
    sort_key = lambda x: (0, int(x)) if x.isdigit() else (1, x)
    missing_before_full = await rag.lightrag.full_docs.filter_keys(target)
    missing_before_full_set = set(missing_before_full)
    chunk_id_map = {doc_id: chunk_id_for_doc(doc_id) for doc_id in target}
    expected_chunk_ids = list(chunk_id_map.values())
    existing_text_chunk_ids = await fetch_existing_chunk_ids(
        rag.lightrag.text_chunks, expected_chunk_ids
    )
    existing_vdb_chunk_ids = await fetch_existing_chunk_ids(
        rag.lightrag.chunks_vdb, expected_chunk_ids
    )
    missing_before_chunk_set = {
        doc_id for doc_id, chunk_id in chunk_id_map.items() if chunk_id not in existing_text_chunk_ids
    }
    missing_before_vdb_set = {
        doc_id for doc_id, chunk_id in chunk_id_map.items() if chunk_id not in existing_vdb_chunk_ids
    }
    to_ingest = sorted(
        missing_before_full_set | missing_before_chunk_set | missing_before_vdb_set,
        key=sort_key,
    )
    logger.info(
        "Workspace %s missing full_docs: %d/%d, missing text_chunks: %d/%d, missing vdb_chunks: %d/%d",
        workspace_id,
        len(missing_before_full_set),
        len(target),
        len(missing_before_chunk_set),
        len(target),
        len(missing_before_vdb_set),
        len(target),
    )
    failures: list[dict[str, Any]] = []
    ingested = 0
    stale_cleanup_ok = 0
    stale_cleanup_failed = 0
    with open(INGEST_FAILURES, "w", encoding="utf-8") as _:
        pass

    def record_failure(doc_id: str, error: str) -> None:
        failure = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_id": doc_id,
            "error": error,
        }
        failures.append(failure)
        append_jsonl_line(INGEST_FAILURES, failure)

    stale_docs = sorted(
        (missing_before_chunk_set | missing_before_vdb_set) - missing_before_full_set,
        key=sort_key,
    )
    for doc_id in stale_docs:
        try:
            await with_retries(
                lambda: rag.lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=False),
                label=f"cleanup stale doc_id={doc_id}",
                retries=retries,
            )
            stale_cleanup_ok += 1
        except Exception as exc:
            stale_cleanup_failed += 1
            record_failure(doc_id, f"stale cleanup failed: {exc}")

    for idx, doc_id in enumerate(to_ingest, start=1):
        row = chunks_by_doc.get(doc_id)
        text = str((row or {}).get("text") or (row or {}).get("abstract") or "").strip()
        if not row or not text:
            record_failure(doc_id, "missing row or empty text")
            continue
        content_list = [{"type": "text", "text": text, "page_idx": 0}]
        file_path = build_virtual_file_path(doc_id, row)
        try:
            await with_retries(
                lambda: rag.insert_content_list(
                    content_list=content_list,
                    file_path=file_path,
                    doc_id=doc_id,
                    display_stats=False,
                ),
                label=f"ingest doc_id={doc_id}",
                retries=retries,
            )
            ingested += 1
        except Exception as exc:
            record_failure(doc_id, f"ingest failed: {exc}")
        if idx % 100 == 0:
            logger.info("Ingest progress: %d/%d", idx, len(to_ingest))
            gc.collect()
            clear_cuda_cache()
    missing_after_full = await rag.lightrag.full_docs.filter_keys(target)
    missing_after_full_set = set(missing_after_full)
    existing_after_text_chunk_ids = await fetch_existing_chunk_ids(
        rag.lightrag.text_chunks, expected_chunk_ids
    )
    existing_after_vdb_chunk_ids = await fetch_existing_chunk_ids(
        rag.lightrag.chunks_vdb, expected_chunk_ids
    )
    missing_after_chunks = sorted(
        [doc_id for doc_id, chunk_id in chunk_id_map.items() if chunk_id not in existing_after_text_chunk_ids],
        key=sort_key,
    )
    missing_after_vdb = sorted(
        [doc_id for doc_id, chunk_id in chunk_id_map.items() if chunk_id not in existing_after_vdb_chunk_ids],
        key=sort_key,
    )
    summary = {
        "workspace_id": workspace_id,
        "target_doc_count": len(target),
        "missing_before_full_doc_count": len(missing_before_full_set),
        "missing_before_chunk_doc_count": len(missing_before_chunk_set),
        "missing_before_vdb_doc_count": len(missing_before_vdb_set),
        "stale_doc_count": len(stale_docs),
        "stale_cleanup_success_count": stale_cleanup_ok,
        "stale_cleanup_failure_count": stale_cleanup_failed,
        "ingest_attempt_count": len(to_ingest),
        "ingested_now_count": ingested,
        "ingest_failure_count": len(failures),
        "missing_after_full_doc_count": len(missing_after_full_set),
        "missing_after_chunk_doc_count": len(missing_after_chunks),
        "missing_after_vdb_doc_count": len(missing_after_vdb),
        "missing_before_full_doc_sample": sorted(list(missing_before_full_set), key=sort_key)[:20],
        "missing_after_full_doc_sample": sorted(list(missing_after_full_set), key=sort_key)[:20],
        "missing_after_chunk_doc_sample": missing_after_chunks[:20],
        "missing_after_vdb_doc_sample": missing_after_vdb[:20],
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


async def run_retrieval(args: argparse.Namespace) -> int:
    QueryParam, LocalRagService, _ = import_rag_dependencies()
    refresh_logging("retrieval")
    data_root = Path(args.data_root)
    subset = data_root / args.subset_dir
    chunks_by_doc, chunk_stats = load_chunks(subset / args.chunks_file)
    queries = load_queries(subset / args.queries_file, args.limit)
    abstract_idx = load_abstract_index(subset / args.corpus_file)
    ks = parse_k_list(args.k_list)
    if args.chunk_top_k <= 0:
        args.chunk_top_k = max(ks)
    settings = settings_for_surge()
    service = LocalRagService(settings)
    refresh_logging("retrieval")
    ingest_summary = await ensure_workspace_index(service, args.workspace_id, chunks_by_doc, args.max_retries)
    rag = await service.get_rag(args.workspace_id)
    await ensure_rag_runtime_ready(rag, args.workspace_id)
    query_params = {"mode": args.query_mode, "top_k": args.top_k, "chunk_top_k": args.chunk_top_k, "enable_rerank": args.enable_rerank, "rerank_score_scope": "top_k"}
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
                param = QueryParam(mode=args.query_mode, top_k=args.top_k, chunk_top_k=args.chunk_top_k, enable_rerank=args.enable_rerank)
                retrieval = await with_retries(lambda: rag.lightrag.aquery_data(q, param=param), label=f"query {qid}", retries=args.max_retries)
                chunks = retrieval.get("data", {}).get("chunks", [])
                chunk_ids = [str(c.get("chunk_id", "")).strip() for c in chunks if isinstance(c, dict) and str(c.get("chunk_id", "")).strip()]
                rows = await rag.lightrag.text_chunks.get_by_ids(chunk_ids) if chunk_ids else []
                row_by_chunk_id, row_list = build_chunk_row_lookup(rows)
                seen = set()
                for idx, c in enumerate(chunks):
                    if not isinstance(c, dict):
                        continue
                    cid = str(c.get("chunk_id", "")).strip()
                    row = row_by_chunk_id.get(cid) if cid else None
                    if row is None and idx < len(row_list):
                        row = row_list[idx]
                    doc_id = parse_int(c.get("full_doc_id") or c.get("doc_id"))
                    if doc_id is None and isinstance(row, dict):
                        doc_id = parse_int((row or {}).get("full_doc_id") or (row or {}).get("doc_id"))
                    if doc_id is None:
                        fp = (row or {}).get("file_path") if isinstance(row, dict) else c.get("file_path")
                        doc_id = parse_doc_id_from_file_path(fp)
                    if doc_id is None:
                        txt = str(c.get("content") or c.get("text") or "").strip()
                        if not txt and isinstance(row, dict):
                            txt = str(row.get("content") or row.get("text") or "").strip()
                        cand = abstract_idx.get(txt, [])
                        if len(cand) == 1:
                            doc_id = cand[0]
                    if doc_id is None:
                        warns.append({"query_id": qid, "chunk_id": cid, "reason": "cannot map doc_id"})
                        continue
                    if doc_id not in seen:
                        seen.add(doc_id)
                        retrieved.append(doc_id)
            except Exception as exc:
                error = {
                    "query_id": qid,
                    "error": str(exc),
                    "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip(),
                }
        row = {
            "query_id": qid,
            "question": q,
            "category": item.get("category"),
            "cite_extract_rate": item.get("cite_extract_rate"),
            "gt_count": len(gt),
            "retrieved_count": len(retrieved),
            "gt_doc_ids": sorted(gt),
            "retrieved_doc_ids": retrieved,
            "recall_at_k": recall_at_k(gt, retrieved, ks),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "error": error,
        }
        rerank = extract_rerank_payload(retrieval, query_params)
        rerank_row = {
            "query_id": qid,
            "question": q,
            "category": item.get("category"),
            "workspace_id": args.workspace_id,
            "query_mode": args.query_mode,
            "top_k": args.top_k,
            "chunk_top_k": args.chunk_top_k,
            "enable_rerank": args.enable_rerank,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **rerank,
        }
        async with lock:
            done += 1
            if done == total or done % max(1, total // 10) == 0:
                logger.info("Retrieval progress: %d/%d", done, total)
        return row, rerank_row, warns, error

    results = await asyncio.gather(*[asyncio.create_task(one(i, q)) for i, q in enumerate(queries)])
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
    avg = {}
    for k in ks:
        key = str(k)
        vals = [float(r["recall_at_k"][key]) for r in per_rows if isinstance(r.get("recall_at_k", {}).get(key), (int, float))]
        avg[key] = round(sum(vals) / len(vals), 6) if vals else None
    save_json(SUMMARY_FILE, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "retrieval",
        "workspace_id": args.workspace_id,
        "query_mode": args.query_mode,
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "enable_rerank": args.enable_rerank,
        "k_list": ks,
        "query_count": len(per_rows),
        "success_count": sum(1 for r in per_rows if r.get("error") is None),
        "failed_count": len(errors),
        "mapping_warning_count": len(warnings),
        "non_empty_retrieval_count": sum(1 for r in per_rows if int(r.get("retrieved_count", 0)) > 0),
        "avg_recall_at_k": avg,
        "ingest_summary": ingest_summary,
        "chunks_source_stats": chunk_stats,
    })
    all_scores = [s for r in rerank_rows for s in r.get("scores", {}).get("all", []) if isinstance(s, (int, float))]
    thr_scores = [s for r in rerank_rows for s in r.get("scores", {}).get("after_threshold", []) if isinstance(s, (int, float))]
    fin_scores = [s for r in rerank_rows for s in r.get("scores", {}).get("final", []) if isinstance(s, (int, float))]
    save_json(RERANK_SUMMARY_FILE, {
        "total_queries": len(rerank_rows),
        "overall_distribution": {
            "all": score_distribution([float(x) for x in all_scores]),
            "after_threshold": score_distribution([float(x) for x in thr_scores]),
            "final": score_distribution([float(x) for x in fin_scores]),
        },
    })
    save_json(RUN_MANIFEST, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "retrieval",
        "data_root": str(data_root),
        "subset_dir": str(subset),
        "queries_file": str(subset / args.queries_file),
        "chunks_file": str(subset / args.chunks_file),
        "corpus_file": str(subset / args.corpus_file),
        "workspace_id": args.workspace_id,
        "effective_query_params": query_params | {"max_concurrency": args.max_concurrency, "max_retries": args.max_retries, "k_list": ks},
        "result_files": {
            "per_query": str(PER_QUERY_FILE),
            "summary": str(SUMMARY_FILE),
            "rerank_stats": str(RERANK_STATS_FILE),
            "rerank_summary": str(RERANK_SUMMARY_FILE),
            "mapping_warnings": str(WARNINGS_FILE),
            "ingest_manifest": str(INGEST_MANIFEST),
            "ingest_failures": str(INGEST_FAILURES),
        },
    })
    logger.info("Retrieval complete: %s", SUMMARY_FILE)
    gc.collect()
    clear_cuda_cache()
    return 0


async def run_survey_placeholder(args: argparse.Namespace) -> int:
    refresh_logging("survey")
    save_json(SURVEY_STATUS, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "survey",
        "status": "not_implemented",
        "message": "Survey generation/evaluation is not implemented yet in evaluate_surge.py",
        "workspace_id": args.workspace_id,
    })
    logger.info("Survey placeholder written: %s", SURVEY_STATUS)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SurGE evaluation for RAGAnything")
    p.add_argument("--mode", choices=["retrieval", "survey"], default="retrieval")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--subset-dir", default=DEFAULT_SUBSET_DIR)
    p.add_argument("--queries-file", default=DEFAULT_QUERIES)
    p.add_argument("--chunks-file", default=DEFAULT_CHUNKS)
    p.add_argument("--corpus-file", default=DEFAULT_CORPUS)
    p.add_argument("--workspace-id", default=DEFAULT_WORKSPACE)
    p.add_argument("--query-mode", choices=["local", "global", "hybrid", "naive", "mix", "bypass"], default="hybrid")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--chunk-top-k", type=int, default=0, help="<=0 means auto use max(k-list)")
    p.add_argument("--k-list", default="20,30,100,200,500,1000")
    p.add_argument("--enable-rerank", type=as_bool, default=True)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 means all queries")
    return p


async def amain(args: argparse.Namespace) -> int:
    ensure_dirs()
    if args.mode == "retrieval":
        return await run_retrieval(args)
    return await run_survey_placeholder(args)


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        logger.debug("Fatal traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
