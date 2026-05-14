#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import hashlib
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTRAG_ROOT = REPO_ROOT.parent / "lightrag"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if LIGHTRAG_ROOT.exists() and str(LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTRAG_ROOT))

from evaluate_local.ablation_flags import (
    AblationFlags,
    build_index_profile,
    ensure_workspace_index_profile,
)
from raganything.constants import (
    DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_LLM_MODEL_MAX_ASYNC,
    DEFAULT_MAX_ASYNC_INGEST,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_SUPPORTED_FILE_EXTENSIONS,
)

TEST_RAW_DIR = Path("/data/y50056788/Yaliang/datasets_raw_test")
TEST_STORAGE_ROOT = Path("/data/y50056788/Yaliang/internal_test")
PROD_RAW_DIR = Path("/data/y50056788/Yaliang/datasets_raw")
PROD_STORAGE_ROOT = Path("/data/y50056788/Yaliang/internal")

DEFAULT_INGEST_TIMEOUT_SECONDS = 7200.0
DEFAULT_MAX_FILE_ATTEMPTS = 2


@dataclass(frozen=True)
class BuildProfile:
    name: str
    raw_dir: Path
    storage_root: Path
    workspace_id: str

    @property
    def uploads_dir(self) -> Path:
        return self.storage_root / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.storage_root / "output"

    @property
    def working_dir_root(self) -> Path:
        return self.storage_root / "rag_workspace"

    @property
    def log_dir(self) -> Path:
        return self.storage_root / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.storage_root / "reports"


def _path_env(path: Path) -> str:
    return path.as_posix()


def resolve_profile(
    profile: str,
    *,
    raw_dir: Path | None = None,
    storage_root: Path | None = None,
    workspace_id: str | None = None,
) -> BuildProfile:
    profile_name = str(profile).strip().lower()
    if profile_name == "test":
        default_raw_dir = TEST_RAW_DIR
        default_storage_root = TEST_STORAGE_ROOT
        default_workspace_id = "internal_test"
    elif profile_name == "prod":
        default_raw_dir = PROD_RAW_DIR
        default_storage_root = PROD_STORAGE_ROOT
        default_workspace_id = "internal"
    else:
        raise ValueError("profile must be 'test' or 'prod'")

    return BuildProfile(
        name=profile_name,
        raw_dir=(raw_dir or default_raw_dir).expanduser(),
        storage_root=(storage_root or default_storage_root).expanduser(),
        workspace_id=workspace_id or default_workspace_id,
    )


def build_local_env(
    profile: BuildProfile,
    *,
    base_env: dict[str, str] | None = None,
    max_async_ingest: int = DEFAULT_MAX_ASYNC_INGEST,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    max_async_ingest = max(1, int(max_async_ingest))
    env.update(
        {
            "RAGANYTHING_WORKDIR_ROOT": _path_env(profile.working_dir_root),
            "RAGANYTHING_OUTPUT_DIR": _path_env(profile.output_dir),
            "RAGANYTHING_UPLOADS_DIR": _path_env(profile.uploads_dir),
            "RAGANYTHING_LOG_DIR": _path_env(profile.log_dir),
            "RAGANYTHING_ENABLE_ENTITY_DISAMBIGUATION": "false",
            "RAGANYTHING_ENABLE_SYNONYM_LINKING": "true",
            "RAGANYTHING_ENABLE_RESILIENCE": "true",
            "RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION": "true",
            "RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION": "true",
            "RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH": "true",
            "ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE": "true",
            "CONTEXT_ZERO_WINDOW_CONTENT_TYPES": DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
            "RAGANYTHING_SERIALIZE_MINERU": "true",
            "MAX_CONCURRENT_FILES": str(max_async_ingest),
        }
    )

    pythonpath_entries = [str(REPO_ROOT)]
    if LIGHTRAG_ROOT.exists():
        pythonpath_entries.append(str(LIGHTRAG_ROOT))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


@contextmanager
def _temporary_env(env: dict[str, str]):
    old_values: dict[str, str | None] = {}
    for key, value in env.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _internal_index_flags() -> AblationFlags:
    return AblationFlags(
        enable_entity_disambiguation=False,
        enable_synonym_linking=True,
        enable_multi_hop=False,
    )


def _settings_summary_from_env(env: dict[str, str]) -> dict[str, Any]:
    return {
        "enable_entity_disambiguation": env["RAGANYTHING_ENABLE_ENTITY_DISAMBIGUATION"]
        == "true",
        "enable_synonym_linking": env["RAGANYTHING_ENABLE_SYNONYM_LINKING"] == "true",
        "enable_resilience": env["RAGANYTHING_ENABLE_RESILIENCE"] == "true",
        "enable_entity_surface_normalization": env[
            "RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION"
        ]
        == "true",
        "enable_keyword_case_normalization": env[
            "RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION"
        ]
        == "true",
        "strict_relation_endpoint_entity_match": env[
            "RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH"
        ]
        == "true",
    }


def build_local_settings(profile: BuildProfile):
    from raganything.services.local_rag import LocalRagSettings

    settings = LocalRagSettings.from_env()
    settings.working_dir_root = str(profile.working_dir_root)
    settings.output_dir = str(profile.output_dir)
    settings.uploads_dir = str(profile.uploads_dir)
    settings.log_dir = str(profile.log_dir)
    settings.enable_entity_disambiguation = False
    settings.enable_synonym_linking = True
    settings.enable_resilience = True
    settings.enable_entity_surface_normalization = True
    settings.enable_keyword_case_normalization = True
    settings.strict_relation_endpoint_entity_match = True
    return settings


def create_local_rag_service(settings):
    from raganything.services.local_rag import LocalRagService

    return LocalRagService(settings)


def collect_supported_files(raw_dir: Path) -> list[Path]:
    supported = {
        ext.strip().lower()
        for ext in DEFAULT_SUPPORTED_FILE_EXTENSIONS.split(",")
        if ext.strip()
    }
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    return sorted(
        (
            path
            for path in raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported
        ),
        key=lambda path: path.name.lower(),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": _path_env(path),
        "size": stat.st_size,
        "sha256": _file_sha256(path),
    }


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem) or "file"


def _iter_batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return _path_env(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _json_safe(vars(value))
    return value


def _get_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _ensure_rag_initialized(rag: Any) -> Any:
    ensure = getattr(rag, "_ensure_lightrag_initialized", None)
    if callable(ensure):
        await _maybe_await(ensure())
    return getattr(rag, "lightrag", rag)


async def _get_lightrag(service: Any, workspace_id: str) -> Any:
    rag = await _maybe_await(service.get_rag(workspace_id))
    return await _ensure_rag_initialized(rag)


async def _storage_get_by_id(storage: Any, key: str) -> Any:
    if storage is None or not key:
        return None
    getter = getattr(storage, "get_by_id", None)
    if callable(getter):
        return await _maybe_await(getter(key))
    getter_many = getattr(storage, "get_by_ids", None)
    if callable(getter_many):
        rows = await _maybe_await(getter_many([key]))
        return rows[0] if rows else None
    return None


async def _storage_get_by_ids(storage: Any, keys: list[str]) -> list[Any]:
    if storage is None or not keys:
        return []
    getter = getattr(storage, "get_by_ids", None)
    if callable(getter):
        return list(await _maybe_await(getter(keys)))
    rows = []
    for key in keys:
        rows.append(await _storage_get_by_id(storage, key))
    return rows


def _document_row(doc_id: str, payload: Any) -> dict[str, Any]:
    path_value = _get_value(payload, "file_path", "")
    chunks_list = _as_list(_get_value(payload, "chunks_list", []))
    chunk_count = _get_value(payload, "chunks_count", None)
    if chunk_count is None:
        chunk_count = len(chunks_list)
    file_name = Path(str(path_value)).name if path_value else ""
    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "file_path": path_value,
        "status": _get_value(payload, "status", ""),
        "chunks_count": chunk_count,
        "chunks_list": chunks_list,
        "multimodal_processed": _get_value(payload, "multimodal_processed", ""),
        "multimodal_stage": _get_value(payload, "multimodal_stage", ""),
        "multimodal_failed_items": _json_safe(
            _get_value(payload, "multimodal_failed_items", [])
        ),
        "multimodal_chunk_ids": _as_list(
            _get_value(payload, "multimodal_chunk_ids", [])
        ),
        "created_at": _get_value(payload, "created_at", ""),
        "updated_at": _get_value(payload, "updated_at", ""),
        "raw_status": _json_safe(payload),
    }


async def collect_documents(service: Any, workspace_id: str) -> dict[str, Any]:
    lightrag = await _get_lightrag(service, workspace_id)
    doc_status = getattr(lightrag, "doc_status", None)
    if doc_status is None:
        return {"workspace_id": workspace_id, "count": 0, "documents": []}

    documents: list[dict[str, Any]] = []
    page = 1
    page_size = 500
    while True:
        rows, total = await _maybe_await(
            doc_status.get_docs_paginated(
                page=page,
                page_size=page_size,
                sort_field="updated_at",
                sort_direction="desc",
                status_filter=None,
            )
        )
        for item in rows:
            if isinstance(item, tuple) and len(item) == 2:
                doc_id, payload = item
            else:
                payload = item
                doc_id = _get_value(payload, "id", _get_value(payload, "doc_id", ""))
            documents.append(_document_row(str(doc_id), payload))
        if len(documents) >= int(total or 0) or not rows:
            break
        page += 1
    return {
        "workspace_id": workspace_id,
        "count": len(documents),
        "documents": documents,
    }


def _source_chunks(value: Any) -> list[str]:
    parts: list[str] = []
    for item in _as_list(value):
        if item is None:
            continue
        for token in str(item).replace("|", "<SEP>").split("<SEP>"):
            token = token.strip()
            if token:
                parts.append(token)
    return sorted(set(parts))


async def collect_entities(service: Any, workspace_id: str) -> dict[str, Any]:
    lightrag = await _get_lightrag(service, workspace_id)
    graph = getattr(lightrag, "chunk_entity_relation_graph", None)
    if graph is None:
        return {"workspace_id": workspace_id, "count": 0, "entities": []}

    labels = list(await _maybe_await(graph.get_all_labels()))
    node_payload = await _maybe_await(graph.get_nodes_batch(labels)) if labels else {}
    if isinstance(node_payload, dict):
        nodes = node_payload
    else:
        nodes = {
            str(_get_value(node, "entity_id", _get_value(node, "entity_name", index))): node
            for index, node in enumerate(node_payload or [])
        }

    entities: list[dict[str, Any]] = []
    for label in labels:
        node = nodes.get(label) if isinstance(nodes, dict) else None
        if node is None:
            node = {}
        source_id = _get_value(node, "source_id", "")
        entities.append(
            {
                "entity_id": _get_value(node, "entity_id", label),
                "entity_name": _get_value(node, "entity_name", label),
                "entity_type": _get_value(node, "entity_type", _get_value(node, "type", "")),
                "description": _get_value(node, "description", ""),
                "source_id": source_id,
                "chunk_ids": _source_chunks(source_id),
                "file_path": _get_value(node, "file_path", ""),
                "metadata": _json_safe(node),
            }
        )
    return {"workspace_id": workspace_id, "count": len(entities), "entities": entities}


async def collect_relations(service: Any, workspace_id: str) -> dict[str, Any]:
    lightrag = await _get_lightrag(service, workspace_id)
    graph = getattr(lightrag, "chunk_entity_relation_graph", None)
    if graph is None or not hasattr(graph, "get_all_edges"):
        return {"workspace_id": workspace_id, "count": 0, "relations": []}

    edges = list(await _maybe_await(graph.get_all_edges()))
    relations: list[dict[str, Any]] = []
    for edge in edges:
        source_id = _get_value(edge, "source_id", "")
        relations.append(
            {
                "source": _get_value(edge, "source", ""),
                "target": _get_value(edge, "target", ""),
                "relation_type": _get_value(edge, "relation_type", _get_value(edge, "type", "")),
                "description": _get_value(edge, "description", ""),
                "keywords": _get_value(edge, "keywords", ""),
                "weight": _get_value(edge, "weight", ""),
                "source_id": source_id,
                "chunk_ids": _source_chunks(source_id),
                "file_path": _get_value(edge, "file_path", ""),
                "metadata": _json_safe(edge),
            }
        )
    return {"workspace_id": workspace_id, "count": len(relations), "relations": relations}


async def collect_graph_stats(
    service: Any,
    workspace_id: str,
    profile: BuildProfile,
) -> dict[str, Any]:
    entities = await collect_entities(service, workspace_id)
    relations = await collect_relations(service, workspace_id)
    graphml_path = (
        profile.working_dir_root
        / workspace_id
        / "graph_chunk_entity_relation.graphml"
    )
    return {
        "workspace_id": workspace_id,
        "source": "graph_storage",
        "entity_count": entities["count"],
        "relation_count": relations["count"],
        "graphml_size": graphml_path.stat().st_size if graphml_path.exists() else 0,
    }


async def collect_doc_storage_presence(
    service: Any,
    workspace_id: str,
    doc_id: str,
) -> dict[str, Any]:
    lightrag = await _get_lightrag(service, workspace_id)
    doc_status_payload = await _storage_get_by_id(
        getattr(lightrag, "doc_status", None), doc_id
    )
    chunks = _as_list(_get_value(doc_status_payload, "chunks_list", []))
    full_entities = await _storage_get_by_id(getattr(lightrag, "full_entities", None), doc_id)
    full_relations = await _storage_get_by_id(getattr(lightrag, "full_relations", None), doc_id)
    text_chunk_rows = await _storage_get_by_ids(getattr(lightrag, "text_chunks", None), chunks)
    chunks_vdb_rows = await _storage_get_by_ids(getattr(lightrag, "chunks_vdb", None), chunks)
    return {
        "doc_id": doc_id,
        "doc_status_present": doc_status_payload is not None,
        "full_doc_present": await _storage_get_by_id(getattr(lightrag, "full_docs", None), doc_id)
        is not None,
        "full_entities_present": full_entities is not None,
        "full_relations_present": full_relations is not None,
        "chunk_ids": chunks,
        "text_chunks_present_count": sum(1 for row in text_chunk_rows if row),
        "chunks_vdb_present_count": sum(1 for row in chunks_vdb_rows if row),
        "full_entities": _json_safe(full_entities),
        "full_relations": _json_safe(full_relations),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _write_audit_reports(
    *,
    report_dir: Path,
    documents_payload: dict[str, Any],
    entities_payload: dict[str, Any],
    relations_payload: dict[str, Any],
    graph_stats_payload: dict[str, Any],
) -> None:
    documents = list(documents_payload.get("documents", []))
    entities = list(entities_payload.get("entities", []))
    relations = list(relations_payload.get("relations", []))
    _write_json(report_dir / "documents.json", documents_payload)
    _write_json(report_dir / "entities.json", entities_payload)
    _write_json(report_dir / "relations.json", relations_payload)
    _write_json(report_dir / "graph_stats.json", graph_stats_payload)
    _write_csv(report_dir / "documents.csv", documents)
    _write_csv(report_dir / "entities.csv", entities)
    _write_csv(report_dir / "relations.csv", relations)


def _settings_summary(settings: Any, env: dict[str, str]) -> dict[str, Any]:
    summary = _settings_summary_from_env(env)
    for key in (
        "working_dir_root",
        "output_dir",
        "log_dir",
        "uploads_dir",
        "enable_entity_disambiguation",
        "enable_synonym_linking",
        "enable_resilience",
        "enable_entity_surface_normalization",
        "enable_keyword_case_normalization",
        "strict_relation_endpoint_entity_match",
    ):
        if hasattr(settings, key):
            summary[key] = _json_safe(getattr(settings, key))
    return summary


def _summary_base(
    *,
    profile: BuildProfile,
    report_dir: Path,
    env: dict[str, str],
    max_async_ingest: int,
    files: list[Path] | None = None,
    settings: Any | None = None,
) -> dict[str, Any]:
    return {
        "execution_mode": "local_service",
        "profile": profile.name,
        "workspace_id": profile.workspace_id,
        "raw_dir": _path_env(profile.raw_dir),
        "storage_root": _path_env(profile.storage_root),
        "report_dir": _path_env(report_dir),
        "file_count": len(files or []),
        "files": [_file_record(path) for path in (files or [])],
        "storage_roots": {
            "output": _path_env(profile.output_dir),
            "workspace": _path_env(profile.working_dir_root),
            "logs": _path_env(profile.log_dir),
        },
        "settings": _settings_summary(settings, env)
        if settings is not None
        else _settings_summary_from_env(env),
        "concurrency": {
            "max_async_ingest": max_async_ingest,
            "MAX_CONCURRENT_FILES": env["MAX_CONCURRENT_FILES"],
            "serialize_mineru": env["RAGANYTHING_SERIALIZE_MINERU"],
            "serialize_by_workspace_id": False,
            "lightrag_llm_model_max_async": DEFAULT_LLM_MODEL_MAX_ASYNC,
            "lightrag_max_parallel_insert": DEFAULT_MAX_PARALLEL_INSERT,
            "embedding_batch_num": DEFAULT_EMBEDDING_BATCH_NUM,
            "embedding_func_max_async": DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
        },
        "retries": [],
    }


async def _ingest_one(
    service: Any,
    *,
    profile: BuildProfile,
    file_path: Path,
    timeout_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    output_dir = profile.output_dir / profile.workspace_id / _safe_stem(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        start = time.time()
        try:
            result = await asyncio.wait_for(
                service.ingest(
                    file_path=str(file_path),
                    output_dir=str(output_dir),
                    workspace_id=profile.workspace_id,
                    serialize_by_workspace_id=False,
                ),
                timeout=float(timeout_seconds),
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "elapsed_seconds": time.time() - start,
                    "status": "success",
                }
            )
            return {
                "file": file_path.name,
                "path": _path_env(file_path),
                "status": "success",
                "attempts": attempts,
                "result": _json_safe(result),
            }
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "elapsed_seconds": time.time() - start,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if attempt >= max(1, int(max_attempts)):
                return {
                    "file": file_path.name,
                    "path": _path_env(file_path),
                    "status": "failed",
                    "attempts": attempts,
                    "error": str(exc),
                }
    raise AssertionError("unreachable")


async def _collect_and_write_reports(
    service: Any,
    profile: BuildProfile,
    report_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    documents_payload = await collect_documents(service, profile.workspace_id)
    entities_payload = await collect_entities(service, profile.workspace_id)
    relations_payload = await collect_relations(service, profile.workspace_id)
    graph_stats_payload = await collect_graph_stats(
        service, profile.workspace_id, profile
    )
    _write_audit_reports(
        report_dir=report_dir,
        documents_payload=documents_payload,
        entities_payload=entities_payload,
        relations_payload=relations_payload,
        graph_stats_payload=graph_stats_payload,
    )
    return documents_payload, entities_payload, relations_payload, graph_stats_payload


async def delete_document(
    service: Any,
    *,
    profile: BuildProfile,
    report_dir: Path,
    doc_id: str,
    delete_llm_cache: bool = False,
) -> dict[str, Any]:
    before_presence = await collect_doc_storage_presence(
        service, profile.workspace_id, doc_id
    )
    before_graph_stats = await collect_graph_stats(service, profile.workspace_id, profile)
    result = await _maybe_await(
        service.lightrag_adelete_by_doc_id(
            profile.workspace_id,
            doc_id,
            delete_llm_cache=delete_llm_cache,
        )
    )
    delete_result = _json_safe(result)
    status = str(_get_value(delete_result, "status", "")).lower()
    synonym_rebuild = None
    if status == "success" and getattr(service.settings, "enable_synonym_linking", False):
        synonym_rebuild = await _maybe_await(
            service.finalize_workspace_synonyms(
                profile.workspace_id,
                force=True,
                reset_existing=True,
            )
        )

    after_presence = await collect_doc_storage_presence(
        service, profile.workspace_id, doc_id
    )
    after_graph_stats = await collect_graph_stats(service, profile.workspace_id, profile)
    delete_check = {
        "target_doc_id": doc_id,
        "before": {
            "storage_presence": before_presence,
            "graph_stats": before_graph_stats,
        },
        "delete_result": delete_result,
        "synonym_rebuild": _json_safe(synonym_rebuild),
        "after": {
            "storage_presence": after_presence,
            "graph_stats": after_graph_stats,
        },
    }
    _write_json(report_dir / "delete_check.json", delete_check)
    return delete_check


async def _run_build_async(
    *,
    args: argparse.Namespace,
    profile: BuildProfile,
    report_dir: Path,
    env: dict[str, str],
    files: list[Path],
    max_async_ingest: int,
) -> int:
    settings = build_local_settings(profile)
    index_profile = build_index_profile(_internal_index_flags(), settings=settings)
    ensured_profile = ensure_workspace_index_profile(
        working_dir_root=profile.working_dir_root,
        workspace_id=profile.workspace_id,
        index_profile=index_profile,
        allow_legacy_adoption=bool(args.allow_legacy_index_profile_adoption),
    )
    service = create_local_rag_service(settings)
    start_time = time.time()
    summary = _summary_base(
        profile=profile,
        report_dir=report_dir,
        env=env,
        max_async_ingest=max_async_ingest,
        files=files,
        settings=settings,
    )
    summary["index_profile"] = _json_safe(ensured_profile)
    try:
        batch_results: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(
            _iter_batches(files, max_async_ingest),
            start=1,
        ):
            results = await asyncio.gather(
                *[
                    _ingest_one(
                        service,
                        profile=profile,
                        file_path=file_path,
                        timeout_seconds=float(args.ingest_timeout_seconds),
                        max_attempts=int(args.max_file_attempts),
                    )
                    for file_path in batch
                ]
            )
            batch_results.append(
                {
                    "batch_index": batch_index,
                    "files": [path.name for path in batch],
                    "results": results,
                }
            )
            for result in results:
                for attempt in result.get("attempts", []):
                    if attempt.get("status") == "failed":
                        retry_record = {
                            "batch_index": batch_index,
                            "file": result["file"],
                            **attempt,
                        }
                        summary["retries"].append(retry_record)

        synonym_result = None
        if getattr(settings, "enable_synonym_linking", False):
            synonym_result = await _maybe_await(
                service.finalize_workspace_synonyms(
                    profile.workspace_id,
                    force=False,
                    reset_existing=True,
                )
            )

        (
            documents_payload,
            entities_payload,
            relations_payload,
            graph_stats_payload,
        ) = await _collect_and_write_reports(service, profile, report_dir)

        delete_check = None
        if args.delete_check:
            target_doc_id = args.delete_check_doc_id
            if not target_doc_id:
                documents = documents_payload.get("documents", [])
                if documents:
                    target_doc_id = str(documents[0].get("doc_id", ""))
            if not target_doc_id:
                raise RuntimeError("--delete-check requested but no document is available")
            delete_check = await delete_document(
                service,
                profile=profile,
                report_dir=report_dir,
                doc_id=target_doc_id,
                delete_llm_cache=bool(args.delete_llm_cache),
            )

        failed_results = [
            result
            for batch in batch_results
            for result in batch["results"]
            if result.get("status") != "success"
        ]
        summary.update(
            {
                "elapsed_seconds": time.time() - start_time,
                "batch_results": batch_results,
                "succeeded_count": len(files) - len(failed_results),
                "failed_count": len(failed_results),
                "failed_files": failed_results,
                "synonym_result": _json_safe(synonym_result),
                "documents_count": documents_payload.get("count", 0),
                "entities_count": entities_payload.get("count", 0),
                "relations_count": relations_payload.get("count", 0),
                "graph_stats": graph_stats_payload,
                "delete_check": delete_check,
            }
        )
        _write_json(report_dir / "summary.json", summary)
        print(f"Reports written to {report_dir}")
        return 1 if failed_results else 0
    finally:
        cleanup = getattr(service, "cleanup_workspace_instance", None)
        if callable(cleanup):
            await _maybe_await(cleanup(profile.workspace_id))


async def _run_report_async(
    *,
    args: argparse.Namespace,
    profile: BuildProfile,
    report_dir: Path,
    env: dict[str, str],
    max_async_ingest: int,
) -> int:
    settings = build_local_settings(profile)
    service = create_local_rag_service(settings)
    start_time = time.time()
    try:
        (
            documents_payload,
            entities_payload,
            relations_payload,
            graph_stats_payload,
        ) = await _collect_and_write_reports(service, profile, report_dir)
        summary = _summary_base(
            profile=profile,
            report_dir=report_dir,
            env=env,
            max_async_ingest=max_async_ingest,
            files=[],
            settings=settings,
        )
        summary.update(
            {
                "command": "report",
                "elapsed_seconds": time.time() - start_time,
                "documents_count": documents_payload.get("count", 0),
                "entities_count": entities_payload.get("count", 0),
                "relations_count": relations_payload.get("count", 0),
                "graph_stats": graph_stats_payload,
            }
        )
        _write_json(report_dir / "summary.json", summary)
        print(f"Reports written to {report_dir}")
        return 0
    finally:
        cleanup = getattr(service, "cleanup_workspace_instance", None)
        if callable(cleanup):
            await _maybe_await(cleanup(profile.workspace_id))


async def _run_delete_async(
    *,
    args: argparse.Namespace,
    profile: BuildProfile,
    report_dir: Path,
    env: dict[str, str],
    max_async_ingest: int,
) -> int:
    settings = build_local_settings(profile)
    service = create_local_rag_service(settings)
    start_time = time.time()
    try:
        delete_check = await delete_document(
            service,
            profile=profile,
            report_dir=report_dir,
            doc_id=str(args.doc_id),
            delete_llm_cache=bool(args.delete_llm_cache),
        )
        (
            documents_payload,
            entities_payload,
            relations_payload,
            graph_stats_payload,
        ) = await _collect_and_write_reports(service, profile, report_dir)
        summary = _summary_base(
            profile=profile,
            report_dir=report_dir,
            env=env,
            max_async_ingest=max_async_ingest,
            files=[],
            settings=settings,
        )
        summary.update(
            {
                "command": "delete-doc",
                "elapsed_seconds": time.time() - start_time,
                "documents_count": documents_payload.get("count", 0),
                "entities_count": entities_payload.get("count", 0),
                "relations_count": relations_payload.get("count", 0),
                "graph_stats": graph_stats_payload,
                "delete_check": delete_check,
            }
        )
        _write_json(report_dir / "summary.json", summary)
        print(f"Reports written to {report_dir}")
        return 0
    finally:
        cleanup = getattr(service, "cleanup_workspace_instance", None)
        if callable(cleanup):
            await _maybe_await(cleanup(profile.workspace_id))


def _prepare_profile_and_dirs(args: argparse.Namespace) -> tuple[BuildProfile, Path, int, dict[str, str]]:
    profile = resolve_profile(
        args.profile,
        raw_dir=Path(args.raw_dir).expanduser() if args.raw_dir else None,
        storage_root=Path(args.storage_root).expanduser()
        if args.storage_root
        else None,
        workspace_id=args.workspace_id,
    )
    max_async_ingest = (
        int(args.max_async_ingest)
        if args.max_async_ingest is not None
        else DEFAULT_MAX_ASYNC_INGEST
    )
    if max_async_ingest < 1:
        raise ValueError("--max-async-ingest must be >= 1")
    env = build_local_env(profile, max_async_ingest=max_async_ingest)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = profile.reports_dir / run_id
    for directory in (
        profile.output_dir,
        profile.working_dir_root,
        profile.log_dir,
        report_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return profile, report_dir, max_async_ingest, env


def run_build(args: argparse.Namespace) -> int:
    profile, report_dir, max_async_ingest, env = _prepare_profile_and_dirs(args)
    files = collect_supported_files(profile.raw_dir)
    if not files:
        raise RuntimeError(f"No supported files found in {profile.raw_dir}")
    if args.dry_run:
        summary = _summary_base(
            profile=profile,
            report_dir=report_dir,
            env=env,
            max_async_ingest=max_async_ingest,
            files=files,
        )
        summary["dry_run"] = True
        _write_json(report_dir / "summary.json", summary)
        print(f"Dry run wrote summary to {report_dir / 'summary.json'}")
        return 0
    with _temporary_env(env):
        return asyncio.run(
            _run_build_async(
                args=args,
                profile=profile,
                report_dir=report_dir,
                env=env,
                files=files,
                max_async_ingest=max_async_ingest,
            )
        )


def run_report(args: argparse.Namespace) -> int:
    profile, report_dir, max_async_ingest, env = _prepare_profile_and_dirs(args)
    with _temporary_env(env):
        return asyncio.run(
            _run_report_async(
                args=args,
                profile=profile,
                report_dir=report_dir,
                env=env,
                max_async_ingest=max_async_ingest,
            )
        )


def run_delete_doc(args: argparse.Namespace) -> int:
    if not args.doc_id:
        raise ValueError("delete-doc requires --doc-id")
    profile, report_dir, max_async_ingest, env = _prepare_profile_and_dirs(args)
    with _temporary_env(env):
        return asyncio.run(
            _run_delete_async(
                args=args,
                profile=profile,
                report_dir=report_dir,
                env=env,
                max_async_ingest=max_async_ingest,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or inspect an internal RAG workspace through LocalRagService."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("build", "report", "delete-doc"),
        default="build",
    )
    parser.add_argument("--profile", choices=("test", "prod"), default="test")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument(
        "--max-async-ingest",
        "--file-batch-size",
        dest="max_async_ingest",
        type=int,
        default=None,
        help="Number of files ingested concurrently. Defaults to constants.py.",
    )
    parser.add_argument(
        "--ingest-timeout-seconds",
        type=float,
        default=DEFAULT_INGEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-file-attempts",
        "--max-batch-attempts",
        dest="max_file_attempts",
        type=int,
        default=DEFAULT_MAX_FILE_ATTEMPTS,
    )
    parser.add_argument("--delete-check", action="store_true")
    parser.add_argument("--delete-check-doc-id", default=None)
    parser.add_argument("--delete-llm-cache", action="store_true")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-legacy-index-profile-adoption",
        action="store_true",
        help="Allow adopting an existing workspace without an index_profile.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        return run_report(args)
    if args.command == "delete-doc":
        return run_delete_doc(args)
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
