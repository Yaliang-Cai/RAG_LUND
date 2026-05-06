#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1 of the MultiHopQA evaluation pipeline: build a LightRAG workspace
from the subset corpus that corresponds to the sampled evaluation questions.

Run this BEFORE evaluate_multihop.py. Use the same --n-samples and --seed
in both scripts so the evaluation queries match the indexed corpus.

The indexer uses a SurGE-style fast ingest path:
  - sampled-question paragraphs are converted to stable source records
  - source records are packed into virtual batch documents
  - each virtual batch is split back into one paragraph per LightRAG chunk
  - virtual batch documents are inserted concurrently
  - source map + manifest files are written next to the workspace
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

_projects_root = Path(__file__).resolve().parents[3]
_raganything_root = Path(__file__).resolve().parents[2]
_lightrag_root = _projects_root / "lightrag"
for p in (_raganything_root, _lightrag_root, _projects_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv

load_dotenv()

VALID_DATASETS = ("hotpotqa", "musique", "2wiki")
MULTIHOPQA_NEVER_SPLIT_DELIMITER = "__MULTIHOPQA_NEVER_SPLIT__"
SOURCE_MAP_FILENAME = "multihopqa_chunk_source_map.json"
SOURCE_RECORDS_FILENAME = "multihopqa_source_records.jsonl"
INGEST_MANIFEST_FILENAME = "multihopqa_ingest_manifest.json"
INGEST_PROGRESS_FILENAME = "multihopqa_ingest_progress.jsonl"
INGEST_FAILURES_FILENAME = "multihopqa_ingest_failures.jsonl"
BUILD_LOG_FILENAME = "multihopqa_build_index.log"
INDEX_PROFILE_FILENAME = "multihopqa_index_profile.json"
MULTIHOPQA_INDEX_PROFILE_KEY = "v0"
VALID_INDEX_PROFILES = ("v0",)


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())


class _TeeOutput:
    def __init__(self, log_file: Path) -> None:
        self._log_file = log_file
        self._file: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None

    def __enter__(self) -> Path:
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_file.open("w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _TeeStream(sys.stdout, self._file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, self._file)  # type: ignore[assignment]
        return self._log_file

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._file is not None:
            self._file.flush()
            self._file.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_log_file(working_dir: str | Path, log_file: str | None) -> Path:
    working_dir_path = Path(working_dir)
    if log_file:
        candidate = Path(log_file).expanduser()
        return candidate if candidate.is_absolute() else working_dir_path / candidate
    return working_dir_path / BUILD_LOG_FILENAME


def apply_multihopqa_index_profile(
    settings: Any,
    profile: str = MULTIHOPQA_INDEX_PROFILE_KEY,
) -> dict[str, Any]:
    """Apply a MultiHopQA index-materialization profile.

    V3/PPR knobs are query-time only, so multi-hop construction stays disabled.
    """
    profile_key = str(profile or MULTIHOPQA_INDEX_PROFILE_KEY).strip().lower()
    if profile_key not in VALID_INDEX_PROFILES:
        raise ValueError(
            f"Unsupported MultiHopQA index profile: {profile!r}. "
            f"Valid: {', '.join(VALID_INDEX_PROFILES)}"
        )

    enable_disambiguation = False
    enable_synonyms = False

    settings.enable_entity_disambiguation = enable_disambiguation
    settings.enable_synonym_linking = enable_synonyms
    settings.enable_multi_hop = False
    settings.enable_entity_surface_normalization = True
    settings.enable_keyword_case_normalization = True
    settings.strict_relation_endpoint_entity_match = True

    ablation_flags = {
        "enable_entity_disambiguation": enable_disambiguation,
        "enable_synonym_linking": enable_synonyms,
        "enable_multi_hop": False,
        "multi_hop_depth": int(getattr(settings, "multi_hop_depth", 2)),
        "ppr_damping": float(getattr(settings, "ppr_damping", 0.85)),
        "ppr_top_k": int(getattr(settings, "ppr_top_k", 60)),
        "ppr_qa_top_k": int(getattr(settings, "ppr_qa_top_k", 5)),
        "passage_node_weight": float(getattr(settings, "passage_node_weight", 0.05)),
    }
    index_profile = {
        "profile_version": 2,
        "profile_key": profile_key,
        "chunk_token_size": int(getattr(settings, "chunk_token_size", 0) or 0),
        "enable_entity_disambiguation": enable_disambiguation,
        "enable_synonym_linking": enable_synonyms,
        "enable_entity_surface_normalization": True,
        "enable_keyword_case_normalization": True,
        "strict_relation_endpoint_entity_match": True,
    }
    if enable_synonyms and hasattr(settings, "synonymy_threshold"):
        index_profile["synonymy_threshold"] = float(getattr(settings, "synonymy_threshold"))
    if enable_synonyms and hasattr(settings, "synonymy_topk"):
        index_profile["synonymy_topk"] = int(getattr(settings, "synonymy_topk"))
    if enable_synonyms and hasattr(settings, "synonymy_min_entity_len"):
        index_profile["synonymy_min_entity_len"] = int(
            getattr(settings, "synonymy_min_entity_len")
        )

    return {
        "ablation_profile": profile_key,
        "ablation_group": "DB-only",
        "ablation_flags": ablation_flags,
        "index_profile": index_profile,
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl_line(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    size = max(1, int(batch_size))
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _format_paragraph(title: Any, text: Any) -> str:
    title_text = str(title or "").strip()
    body_text = str(text or "").strip()
    if title_text and body_text:
        return f"{title_text}\n{body_text}".strip()
    return (title_text or body_text).strip()


def prepare_source_records(
    *,
    dataset: str,
    corpus: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Convert extracted paragraphs into stable source records and chunk map.

    LightRAG chunk IDs are content hashes. Because virtual-batch ingest uses
    split_by_character_only=True, each formatted paragraph becomes exactly one
    chunk, so the source paragraph can be mapped to the future chunk ID before
    insertion.
    """
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding
    from evaluate_local.MultiHopQA.dataset_adapters import paragraph_source_key

    source_records: dict[str, dict[str, Any]] = {}
    chunk_source_map: dict[str, dict[str, Any]] = {}
    empty_count = 0

    for idx, row in enumerate(corpus, start=1):
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        content = sanitize_text_for_encoding(_format_paragraph(title, text)).strip()
        if not content:
            empty_count += 1
            continue

        source_paragraph_id = f"{dataset}_{idx:06d}"
        source_key = str(row.get("source_key") or paragraph_source_key(dataset, title, text))
        lightrag_chunk_id = compute_mdhash_id(content, prefix="chunk-")
        if lightrag_chunk_id in chunk_source_map:
            existing = chunk_source_map[lightrag_chunk_id]
            raise ValueError(
                "duplicate LightRAG chunk id "
                f"{lightrag_chunk_id}: {existing.get('source_paragraph_id')} "
                f"and {source_paragraph_id}"
            )

        record = {
            "source_paragraph_id": source_paragraph_id,
            "source_key": source_key,
            "dataset": dataset,
            "title": title,
            "text": text,
            "content": content,
            "lightrag_chunk_id": lightrag_chunk_id,
        }
        source_records[source_paragraph_id] = record
        chunk_source_map[lightrag_chunk_id] = {
            "source_paragraph_id": source_paragraph_id,
            "source_key": source_key,
            "dataset": dataset,
            "title": title,
            "text": text,
            "content": content,
        }

    stats = {
        "source_paragraph_count": len(source_records),
        "source_chunk_count": len(chunk_source_map),
        "empty_paragraph_count": empty_count,
    }
    return source_records, chunk_source_map, stats


def resolve_safe_split_delimiter(texts: list[str]) -> str:
    """Return a delimiter absent from every source text."""
    preferred = MULTIHOPQA_NEVER_SPLIT_DELIMITER
    if all(preferred not in text for text in texts):
        return preferred
    while True:
        candidate = f"{preferred}_{uuid.uuid4().hex}"
        if all(candidate not in text for text in texts):
            return candidate


def build_virtual_batches(
    *,
    source_records: dict[str, dict[str, Any]],
    ingest_batch_size: int,
) -> list[dict[str, Any]]:
    source_ids = sorted(source_records)
    batches: list[dict[str, Any]] = []
    for batch_idx, batch_source_ids in enumerate(
        _iter_batches(source_ids, max(1, int(ingest_batch_size))),
        start=1,
    ):
        rows = [source_records[source_id] for source_id in batch_source_ids]
        texts = [str(row["content"]) for row in rows]
        delimiter = resolve_safe_split_delimiter(texts)
        batch_doc_id = f"multihopqa_batch_{batch_idx:06d}"
        batches.append(
            {
                "batch_doc_id": batch_doc_id,
                "file_path": f"{batch_doc_id}.txt",
                "delimiter": delimiter,
                "content": delimiter.join(texts),
                "source_paragraph_ids": [str(row["source_paragraph_id"]) for row in rows],
                "expected_chunk_ids": [str(row["lightrag_chunk_id"]) for row in rows],
                "expected_chunk_count": len(rows),
            }
        )
    return batches


def _load_successful_batch_ids(progress_path: Path) -> set[str]:
    successful: set[str] = set()
    if not progress_path.exists():
        return successful
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("batch_doc_id"):
                successful.add(str(row["batch_doc_id"]))
    return successful


def validate_existing_manifest_for_resume(
    *,
    manifest_path: Path,
    workspace: str,
    dataset: str,
    n_samples: int,
    seed: int,
    expected_index_profile: dict[str, Any] | None = None,
) -> None:
    """Refuse resume when an existing manifest belongs to another corpus."""
    if not manifest_path.exists():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid existing manifest JSON: {manifest_path}") from exc

    expected = {
        "workspace_id": workspace,
        "dataset": dataset,
        "n_samples": int(n_samples),
        "seed": int(seed),
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = payload.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: existing={actual_value!r} current={expected_value!r}")
    if expected_index_profile is not None:
        actual_profile = payload.get("index_profile")
        if actual_profile != expected_index_profile:
            mismatches.append(
                "index_profile: "
                f"existing={actual_profile!r} current={expected_index_profile!r}"
            )
    if mismatches:
        raise ValueError(
            "Existing MultiHopQA ingest manifest does not match this --resume run. "
            "Use the original workspace/dataset/n-samples/seed, or choose a new "
            f"working-dir/workspace. Details: {'; '.join(mismatches)}"
        )


def validate_or_write_index_profile(
    *,
    working_dir: Path,
    workspace: str,
    dataset: str,
    n_samples: int,
    seed: int,
    index_profile_metadata: dict[str, Any],
) -> None:
    """Persist index materialization settings before ingest starts.

    This guards interrupted workspaces too: the final manifest is only written
    after successful ingest, while partial LightRAG artifacts can already exist.
    """
    profile_path = working_dir / INDEX_PROFILE_FILENAME
    expected_profile = index_profile_metadata["index_profile"]
    expected_identity = {
        "workspace_id": workspace,
        "dataset": dataset,
        "n_samples": int(n_samples),
        "seed": int(seed),
    }
    if profile_path.exists():
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid existing index profile JSON: {profile_path}") from exc
        mismatches: list[str] = []
        for key, expected_value in expected_identity.items():
            actual_value = payload.get(key)
            if actual_value != expected_value:
                mismatches.append(
                    f"{key}: existing={actual_value!r} current={expected_value!r}"
                )
        actual_profile = payload.get("index_profile")
        if actual_profile != expected_profile:
            mismatches.append(
                "index_profile: "
                f"existing={actual_profile!r} current={expected_profile!r}"
            )
        if mismatches:
            raise ValueError(
                "Existing MultiHopQA index profile does not match this run. "
                "Use a new working-dir/workspace or rebuild cleanly. "
                f"Details: {'; '.join(mismatches)}"
            )
        return

    existing_state = [
        path.name
        for path in working_dir.iterdir()
        if path.name != BUILD_LOG_FILENAME and path.suffix != ".log"
    ]
    if existing_state:
        raise ValueError(
            "Refusing to use a MultiHopQA workspace with existing artifacts "
            "but no multihopqa_index_profile.json. Use a new clean workspace, "
            "or rebuild this one from scratch. "
            f"Existing entries: {existing_state[:10]}"
        )

    _save_json(
        profile_path,
        {
            "schema_version": "multihopqa_index_profile_v1",
            "generated_at": _utc_now(),
            **expected_identity,
            **index_profile_metadata,
        },
    )


async def _with_retries(label: str, retries: int, coro_factory):
    attempt = 0
    while True:
        try:
            return await coro_factory()
        except Exception:
            if attempt >= retries:
                raise
            attempt += 1
            wait_seconds = min(30, 2**attempt)
            print(f"[build_index] WARN: {label} failed; retry {attempt}/{retries} in {wait_seconds}s")
            await asyncio.sleep(wait_seconds)


def _doc_status_field(status_doc: Any, key: str, default: Any = None) -> Any:
    if isinstance(status_doc, dict):
        return status_doc.get(key, default)
    return getattr(status_doc, key, default)


def _doc_status_value(status_doc: Any) -> str:
    raw_status = _doc_status_field(status_doc, "status", "")
    if hasattr(raw_status, "value"):
        raw_status = raw_status.value
    return str(raw_status).lower()


def _validate_batch_doc_status(status_doc: Any, batch: dict[str, Any]) -> None:
    batch_doc_id = str(batch["batch_doc_id"])
    if not status_doc:
        raise RuntimeError(f"LightRAG doc_status missing for {batch_doc_id}")

    status = _doc_status_value(status_doc)
    if "failed" in status:
        error_msg = str(_doc_status_field(status_doc, "error_msg", "") or "")
        raise RuntimeError(f"LightRAG marked {batch_doc_id} failed: {error_msg}")
    if "processed" not in status:
        raise RuntimeError(f"LightRAG did not finish {batch_doc_id}: status={status!r}")

    expected_chunk_ids = {str(chunk_id) for chunk_id in batch["expected_chunk_ids"]}
    actual_chunk_ids = {
        str(chunk_id)
        for chunk_id in (_doc_status_field(status_doc, "chunks_list", []) or [])
    }
    missing = sorted(expected_chunk_ids - actual_chunk_ids)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"LightRAG doc_status for {batch_doc_id} is missing expected chunks: {preview}"
        )


async def _get_batch_doc_status(
    *,
    service: Any,
    workspace: str,
    working_dir: str,
    batch_doc_id: str,
) -> Any:
    rag = await service.get_rag(workspace, working_dir=working_dir)
    return await rag.lightrag.doc_status.get_by_id(batch_doc_id)


async def _wait_for_batch_doc_status(
    *,
    service: Any,
    workspace: str,
    working_dir: str,
    batch: dict[str, Any],
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> Any:
    """Wait until LightRAG's queue has durably processed this virtual batch."""
    batch_doc_id = str(batch["batch_doc_id"])
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, float(timeout_seconds))
    last_error: RuntimeError | None = None

    while True:
        status_doc = await _get_batch_doc_status(
            service=service,
            workspace=workspace,
            working_dir=working_dir,
            batch_doc_id=batch_doc_id,
        )
        try:
            _validate_batch_doc_status(status_doc, batch)
            return status_doc
        except RuntimeError as exc:
            last_error = exc
            status = _doc_status_value(status_doc) if status_doc else ""
            if "failed" in status or "processed" in status:
                raise
            if loop.time() >= deadline:
                raise RuntimeError(
                    "LightRAG did not finish "
                    f"{batch_doc_id} within {timeout_seconds:g}s: "
                    f"last_status={status!r}"
                ) from last_error
            await asyncio.sleep(max(0.1, float(poll_interval_seconds)))


def _persist_source_artifacts(
    *,
    working_dir: Path,
    workspace: str,
    dataset: str,
    n_samples: int,
    seed: int,
    source_records: dict[str, dict[str, Any]],
    chunk_source_map: dict[str, dict[str, Any]],
    source_stats: dict[str, Any],
) -> None:
    source_records_path = working_dir / SOURCE_RECORDS_FILENAME
    source_map_path = working_dir / SOURCE_MAP_FILENAME

    _write_jsonl(source_records_path, source_records.values())
    _save_json(
        source_map_path,
        {
            "schema_version": "multihopqa_chunk_source_map_v1",
            "generated_at": _utc_now(),
            "workspace_id": workspace,
            "dataset": dataset,
            "n_samples": n_samples,
            "seed": seed,
            "source_stats": source_stats,
            "map_size": len(chunk_source_map),
            "map": chunk_source_map,
        },
    )


async def _ingest_batches(
    *,
    service: Any,
    workspace: str,
    working_dir: str,
    batches: list[dict[str, Any]],
    progress_path: Path,
    failures_path: Path,
    resume: bool,
    batch_doc_concurrency: int,
    max_retries: int,
    doc_status_timeout: float,
    doc_status_poll_interval: float,
) -> dict[str, Any]:
    successful_before = _load_successful_batch_ids(progress_path) if resume else set()
    failures_path.unlink(missing_ok=True)
    if not resume:
        progress_path.unlink(missing_ok=True)

    pending_batches = [
        batch for batch in batches if str(batch["batch_doc_id"]) not in successful_before
    ]
    print(
        "[build_index] Virtual batches: "
        f"{len(batches)} total, {len(successful_before)} already ok, {len(pending_batches)} pending"
    )
    print(f"[build_index] Concurrent virtual batch workers: {batch_doc_concurrency}")

    sem = asyncio.Semaphore(max(1, int(batch_doc_concurrency)))
    lock = asyncio.Lock()
    done = 0
    success_count = 0
    failure_count = 0
    ingested_source_count = 0

    async def _insert_one(batch: dict[str, Any]) -> None:
        await service.lightrag_ainsert(
            workspace_id=workspace,
            input=str(batch["content"]),
            ids=str(batch["batch_doc_id"]),
            file_paths=str(batch["file_path"]),
            split_by_character=str(batch["delimiter"]),
            split_by_character_only=True,
            working_dir=working_dir,
        )

    async def _process_one(batch: dict[str, Any]) -> None:
        nonlocal done, success_count, failure_count, ingested_source_count
        batch_doc_id = str(batch["batch_doc_id"])
        expected_count = int(batch["expected_chunk_count"])
        async with sem:
            try:
                await _with_retries(
                    label=f"ingest {batch_doc_id}",
                    retries=max(0, int(max_retries)),
                    coro_factory=lambda: _insert_one(batch),
                )
                status_doc = await _wait_for_batch_doc_status(
                    service=service,
                    workspace=workspace,
                    working_dir=working_dir,
                    batch=batch,
                    timeout_seconds=doc_status_timeout,
                    poll_interval_seconds=doc_status_poll_interval,
                )
                progress_row = {
                    "timestamp": _utc_now(),
                    "status": "ok",
                    "batch_doc_id": batch_doc_id,
                    "expected_chunk_count": expected_count,
                    "source_paragraph_ids": batch["source_paragraph_ids"],
                    "expected_chunk_ids": batch["expected_chunk_ids"],
                }
                _append_jsonl_line(progress_path, progress_row)
                async with lock:
                    success_count += 1
                    ingested_source_count += expected_count
            except Exception as exc:
                failure_row = {
                    "timestamp": _utc_now(),
                    "status": "failed",
                    "batch_doc_id": batch_doc_id,
                    "error": str(exc),
                    "expected_chunk_count": expected_count,
                    "source_paragraph_ids": list(batch["source_paragraph_ids"])[:20],
                }
                _append_jsonl_line(failures_path, failure_row)
                async with lock:
                    failure_count += 1
            finally:
                async with lock:
                    done += 1
                    if done % 10 == 0 or done == len(pending_batches):
                        print(f"[build_index]   {done}/{len(pending_batches)} virtual batches processed")

    if pending_batches:
        await asyncio.gather(*[asyncio.create_task(_process_one(batch)) for batch in pending_batches])

    return {
        "successful_before_batch_count": len(successful_before),
        "pending_batch_count": len(pending_batches),
        "successful_now_batch_count": success_count,
        "failed_now_batch_count": failure_count,
        "ingested_now_source_paragraph_count": ingested_source_count,
    }


async def main(args: argparse.Namespace) -> None:
    from raganything.services.local_rag import LocalRagService, LocalRagSettings
    from evaluate_local.MultiHopQA.dataset_adapters import (
        extract_corpus_hotpotqa,
        extract_corpus_musique,
        extract_corpus_2wiki,
        extract_corpus_hotpotqa_hipporag2,
        extract_corpus_musique_hipporag2,
        extract_corpus_2wiki_hipporag2,
    )

    hipporag2_dir = Path(args.hipporag2_data_dir).resolve() if args.hipporag2_data_dir else None

    if hipporag2_dir:
        extractors = {
            "hotpotqa": lambda **_: extract_corpus_hotpotqa_hipporag2(hipporag2_dir),
            "musique":  lambda **_: extract_corpus_musique_hipporag2(hipporag2_dir),
            "2wiki":    lambda **_: extract_corpus_2wiki_hipporag2(hipporag2_dir),
        }
    else:
        extractors = {
            "hotpotqa": extract_corpus_hotpotqa,
            "musique":  extract_corpus_musique,
            "2wiki":    extract_corpus_2wiki,
        }

    # In hipporag2 mode the corpus is fixed; use sentinel n_samples=0 so the
    # manifest identity key still stores a meaningful value.
    effective_n_samples = 0 if hipporag2_dir else args.n_samples
    effective_seed      = 0 if hipporag2_dir else args.seed

    working_dir_path = Path(args.working_dir).resolve()
    working_dir_path.mkdir(parents=True, exist_ok=True)
    progress_path = working_dir_path / INGEST_PROGRESS_FILENAME
    failures_path = working_dir_path / INGEST_FAILURES_FILENAME
    manifest_path = working_dir_path / INGEST_MANIFEST_FILENAME

    settings = LocalRagSettings.from_env()
    index_profile_metadata = apply_multihopqa_index_profile(
        settings,
        profile=args.index_profile,
    )
    validate_or_write_index_profile(
        working_dir=working_dir_path,
        workspace=args.workspace,
        dataset=args.dataset,
        n_samples=effective_n_samples,
        seed=effective_seed,
        index_profile_metadata=index_profile_metadata,
    )
    if args.resume:
        validate_existing_manifest_for_resume(
            manifest_path=manifest_path,
            workspace=args.workspace,
            dataset=args.dataset,
            n_samples=effective_n_samples,
            seed=effective_seed,
            expected_index_profile=index_profile_metadata["index_profile"],
        )

    if hipporag2_dir:
        print(f"[build_index] HippoRAG2 mode: loading corpus from {hipporag2_dir}")
        corpus = extractors[args.dataset]()
    else:
        print(f"[build_index] Extracting corpus: {args.dataset} n={args.n_samples} seed={args.seed}")
        corpus = extractors[args.dataset](n=args.n_samples, seed=args.seed)
    print(f"[build_index] Corpus size: {len(corpus)} unique paragraphs")

    source_records, chunk_source_map, source_stats = prepare_source_records(
        dataset=args.dataset,
        corpus=corpus,
    )
    print(f"[build_index] Source records: {source_stats}")
    _persist_source_artifacts(
        working_dir=working_dir_path,
        workspace=args.workspace,
        dataset=args.dataset,
        n_samples=effective_n_samples,
        seed=effective_seed,
        source_records=source_records,
        chunk_source_map=chunk_source_map,
        source_stats=source_stats,
    )

    batches = build_virtual_batches(
        source_records=source_records,
        ingest_batch_size=args.ingest_batch_size,
    )

    # Match SurGE fast-ingest behavior: the virtual batches are independent and
    # LightRAG's own LLM/embedding semaphores remain the real resource guard.
    settings.serialize_ingest_by_workspace_id = False
    service = LocalRagService(settings)

    try:
        rag = await service.get_rag(args.workspace, working_dir=str(working_dir_path))
        kwargs = getattr(rag, "lightrag_kwargs", None)
        if isinstance(kwargs, dict):
            kwargs["llm_model_max_async"] = max(1, int(args.llm_model_max_async))

        print(f"[build_index] Target workspace: {args.workspace!r}")
        print(f"[build_index] Working dir:      {working_dir_path}")
        print(f"[build_index] Log file:         {args.resolved_log_file}")
        print(f"[build_index] Ingest batch size: {args.ingest_batch_size} paragraphs/virtual-doc")
        print(f"[build_index] LLM max async:     {args.llm_model_max_async}")
        print(f"[build_index] Chunk token size:  {settings.chunk_token_size}")
        print(
            "[build_index] Index profile:     "
            f"{index_profile_metadata['ablation_profile']} "
            f"({index_profile_metadata['ablation_group']})"
        )
        print(f"[build_index] Resume:            {args.resume}")

        ingest_stats = await _ingest_batches(
            service=service,
            workspace=args.workspace,
            working_dir=str(working_dir_path),
            batches=batches,
            progress_path=progress_path,
            failures_path=failures_path,
            resume=args.resume,
            batch_doc_concurrency=args.batch_doc_concurrency,
            max_retries=args.max_retries,
            doc_status_timeout=args.doc_status_timeout,
            doc_status_poll_interval=args.doc_status_poll_interval,
        )
        if settings.enable_synonym_linking:
            synonym_result = await service.finalize_workspace_synonyms(
                args.workspace,
                force=False,
                reset_existing=True,
            )
            print(f"[build_index] Synonym finalize: {synonym_result}")

        manifest = {
            "schema_version": "multihopqa_ingest_manifest_v1",
            "generated_at": _utc_now(),
            "workspace_id": args.workspace,
            "dataset": args.dataset,
            "corpus_source": "hipporag2" if hipporag2_dir else "huggingface",
            "hipporag2_data_dir": str(hipporag2_dir) if hipporag2_dir else None,
            "n_samples": effective_n_samples,
            "seed": effective_seed,
            "working_dir": str(working_dir_path),
            "ingest_mode": "virtual_batch",
            "ingest_batch_size": args.ingest_batch_size,
            "batch_doc_concurrency": args.batch_doc_concurrency,
            "llm_model_max_async": args.llm_model_max_async,
            "chunk_token_size": settings.chunk_token_size,
            "doc_status_timeout": args.doc_status_timeout,
            "doc_status_poll_interval": args.doc_status_poll_interval,
            "ablation_profile": index_profile_metadata["ablation_profile"],
            "ablation_group": index_profile_metadata["ablation_group"],
            "ablation_flags": index_profile_metadata["ablation_flags"],
            "index_profile": index_profile_metadata["index_profile"],
            "batch_count": len(batches),
            "expected_chunk_total": len(chunk_source_map),
            "source_stats": source_stats,
            "ingest_stats": ingest_stats,
            "artifacts": {
                "source_records": str(working_dir_path / SOURCE_RECORDS_FILENAME),
                "chunk_source_map": str(working_dir_path / SOURCE_MAP_FILENAME),
                "progress": str(progress_path),
                "failures": str(failures_path),
                "log_file": str(args.resolved_log_file),
                "index_profile": str(working_dir_path / INDEX_PROFILE_FILENAME),
            },
        }
        _save_json(manifest_path, manifest)

        if ingest_stats["failed_now_batch_count"]:
            raise RuntimeError(
                f"{ingest_stats['failed_now_batch_count']} virtual batch(es) failed. "
                f"See {failures_path}"
            )

        print("\n[build_index] Indexing complete.")
        print(f"[build_index] Manifest:         {manifest_path}")
        print(f"[build_index] Chunk source map: {working_dir_path / SOURCE_MAP_FILENAME}")
        print("[build_index] Run evaluation with:")
        print("  python evaluate_local/MultiHopQA/evaluate_multihop.py \\")
        print(f"    --dataset {args.dataset} \\")
        print(f"    --workspace {args.workspace} \\")
        print(f"    --working-dir {working_dir_path} \\")
        print(f"    --n-samples {args.n_samples} \\")
        print(f"    --seed {args.seed} \\")
        print("    --output-dir ./multihop_results \\")
        print("    --modes naive hybrid ppr")
    finally:
        await service.cleanup_workspace_instance(args.workspace)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Index a multi-hop QA corpus subset into a LightRAG workspace."
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=VALID_DATASETS,
        help="Which dataset's context paragraphs to index",
    )
    p.add_argument("--workspace", required=True, help="Workspace ID to create or resume")
    p.add_argument(
        "--working-dir",
        required=True,
        dest="working_dir",
        help="Directory where LightRAG stores this workspace index and source-map files",
    )
    p.add_argument(
        "--hipporag2-data-dir",
        default=None,
        dest="hipporag2_data_dir",
        help=(
            "Path to HippoRAG2 dataset directory containing *_corpus.json files. "
            "When set, uses the exact HippoRAG2 corpus (9811/11656/6119 paragraphs) "
            "instead of sampling from HuggingFace. Download with "
            "download_hipporag2_datasets.py. Overrides --n-samples and --seed."
        ),
    )
    p.add_argument(
        "--index-profile",
        default=MULTIHOPQA_INDEX_PROFILE_KEY,
        choices=VALID_INDEX_PROFILES,
        dest="index_profile",
        help=(
            "Index materialization profile. v0 disables entity disambiguation "
            "and synonym linking."
        ),
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        dest="n_samples",
        help="Number of questions to sample when NOT using --hipporag2-data-dir",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when NOT using --hipporag2-data-dir",
    )
    p.add_argument(
        "--ingest-batch-size",
        "--batch-size",
        type=int,
        default=256,
        dest="ingest_batch_size",
        help="Source paragraphs packed into one virtual batch document",
    )
    p.add_argument(
        "--batch-doc-concurrency",
        type=int,
        default=2,
        help="Concurrent virtual batch document inserts",
    )
    p.add_argument(
        "--llm-model-max-async",
        type=int,
        default=48,
        help="LightRAG LLM extraction worker concurrency during ingest",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retries per failed virtual batch insert",
    )
    p.add_argument(
        "--doc-status-timeout",
        type=float,
        default=7200.0,
        dest="doc_status_timeout",
        help=(
            "Seconds to wait for a queued LightRAG virtual batch to reach "
            "processed status before marking it failed"
        ),
    )
    p.add_argument(
        "--doc-status-poll-interval",
        type=float,
        default=5.0,
        dest="doc_status_poll_interval",
        help="Seconds between LightRAG doc_status checks",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip virtual batches already marked ok in the progress JSONL",
    )
    p.add_argument(
        "--log-file",
        default=None,
        dest="log_file",
        help=(
            "Build log path. Defaults to multihopqa_build_index.log inside "
            "--working-dir. Relative paths are resolved under --working-dir."
        ),
    )
    args = p.parse_args()
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be > 0")
    if args.ingest_batch_size <= 0:
        raise SystemExit("--ingest-batch-size/--batch-size must be > 0")
    if args.batch_doc_concurrency <= 0:
        raise SystemExit("--batch-doc-concurrency must be > 0")
    if args.llm_model_max_async <= 0:
        raise SystemExit("--llm-model-max-async must be > 0")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.doc_status_timeout <= 0:
        raise SystemExit("--doc-status-timeout must be > 0")
    if args.doc_status_poll_interval <= 0:
        raise SystemExit("--doc-status-poll-interval must be > 0")
    working_dir_path = Path(args.working_dir).resolve()
    args.working_dir = str(working_dir_path)
    args.resolved_log_file = resolve_log_file(working_dir_path, args.log_file)
    return args


if __name__ == "__main__":
    parsed_args = _parse_args()
    with _TeeOutput(parsed_args.resolved_log_file) as log_path:
        try:
            print(f"[build_index] Writing log to: {log_path}")
            asyncio.run(main(parsed_args))
        except Exception:
            traceback.print_exc()
            raise
