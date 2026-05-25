#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import gc
import hashlib
import inspect
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
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
    DEFAULT_LLM_CONTEXT_MAX_TOKENS,
    DEFAULT_LLM_CONTEXT_RESERVED_TOKENS,
    DEFAULT_LLM_MODEL_MAX_ASYNC,
    DEFAULT_MAX_ASYNC_INGEST,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_MINERU_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_MULTIMODAL_ITEM_PARALLELISM,
    DEFAULT_SUPPORTED_FILE_EXTENSIONS,
)

TEST_RAW_DIR = Path("/data/y50056788/Yaliang/datasets_raw_test")
TEST_STORAGE_ROOT = Path("/data/y50056788/Yaliang/internal_test")
PROD_RAW_DIR = Path("/data/y50056788/Yaliang/datasets_raw")
PROD_STORAGE_ROOT = Path("/data/y50056788/Yaliang/internal")
RAN2_133_BIS_RAW_DIR = Path("/data/y50056788/Yaliang/datasets_raw_RAN2_133_BIS")
RAN2_133_BIS_STORAGE_ROOT = Path("/data/y50056788/Yaliang/internal_RAN2_133_BIS")

DEFAULT_INGEST_TIMEOUT_SECONDS = 7200.0
DEFAULT_MAX_FILE_ATTEMPTS = 2
DEFAULT_BUILD_LIBREOFFICE_CONVERT_TIMEOUT_SECONDS = 900
DEFAULT_MANUAL_LIBREOFFICE_CONVERT_TIMEOUT_SECONDS = 1800
MINERU_DIRECT_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".webp",
}
MINERU_OFFICE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
}
MINERU_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
}
MINERU_PREPARSE_EXTENSIONS = (
    MINERU_DIRECT_EXTENSIONS | MINERU_OFFICE_EXTENSIONS | MINERU_TEXT_EXTENSIONS
)
INTERNAL_SOURCE_IDS_LIMIT = 99999

LOGGER = logging.getLogger("internal_build")
_ACTIVE_BUILD_LOG_CONTEXT: "BuildLogContext | None" = None


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


class _CountingFileHandler(logging.FileHandler):
    def __init__(self, filename: Path) -> None:
        super().__init__(filename, mode="w", encoding="utf-8")
        self.warning_count = 0
        self.error_count = 0
        self._internal_build_handler = True

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1
        super().emit(record)


class _InternalConsoleHandler(logging.StreamHandler):
    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._internal_build_handler = True


class _TeeTextStream(io.TextIOBase):
    def __init__(self, original: Any, mirror: Any) -> None:
        self._original = original
        self._mirror = mirror
        self._internal_build_stream = True

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", None) or "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())

    def write(self, text: str) -> int:
        written = self._original.write(text)
        self._mirror.write(text)
        return written

    def flush(self) -> None:
        self._original.flush()
        self._mirror.flush()


@dataclass
class BuildLogContext:
    log_file: Path
    handler: _CountingFileHandler
    console_handler: _InternalConsoleHandler
    root_logger: logging.Logger
    old_root_level: int
    old_logger_levels: dict[str, int]
    old_stdout: Any
    old_stderr: Any
    stream_mirror: Any
    bridged_loggers: list[logging.Logger] = dataclasses.field(default_factory=list)

    @property
    def warning_count(self) -> int:
        return int(self.handler.warning_count)

    @property
    def error_count(self) -> int:
        return int(self.handler.error_count)


def _path_env(path: Path) -> str:
    return path.as_posix()


def setup_build_logging(report_dir: Path) -> BuildLogContext:
    global _ACTIVE_BUILD_LOG_CONTEXT

    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = report_dir / "internal_build.log"
    root_logger = logging.getLogger()

    for handler in list(root_logger.handlers):
        if getattr(handler, "_internal_build_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    old_root_level = root_logger.level
    logger_names = ("internal_build", "raganything", "lightrag")
    old_logger_levels = {
        logger_name: logging.getLogger(logger_name).level
        for logger_name in logger_names
    }
    handler = _CountingFileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    root_logger.addHandler(handler)

    console_handler = _InternalConsoleHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    root_logger.addHandler(console_handler)

    if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)

    for logger_name in logger_names:
        logging.getLogger(logger_name).setLevel(logging.INFO)

    LOGGER.info("Internal workspace log initialized: %s", log_file)
    stream_mirror = log_file.open("a", encoding="utf-8", buffering=1)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _TeeTextStream(old_stdout, stream_mirror)
    sys.stderr = _TeeTextStream(old_stderr, stream_mirror)
    context = BuildLogContext(
        log_file=log_file,
        handler=handler,
        console_handler=console_handler,
        root_logger=root_logger,
        old_root_level=old_root_level,
        old_logger_levels=old_logger_levels,
        old_stdout=old_stdout,
        old_stderr=old_stderr,
        stream_mirror=stream_mirror,
    )
    _ACTIVE_BUILD_LOG_CONTEXT = context
    _bridge_lightrag_logs_to_internal_build_log(context)
    return context


def close_build_logging(context: BuildLogContext) -> None:
    global _ACTIVE_BUILD_LOG_CONTEXT

    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = context.old_stdout
    sys.stderr = context.old_stderr
    context.handler.flush()
    context.console_handler.flush()
    context.root_logger.removeHandler(context.handler)
    context.root_logger.removeHandler(context.console_handler)
    for logger in list(context.bridged_loggers):
        if context.handler in logger.handlers:
            logger.removeHandler(context.handler)
    context.handler.close()
    context.console_handler.close()
    context.stream_mirror.flush()
    context.stream_mirror.close()
    context.root_logger.setLevel(context.old_root_level)
    for logger_name, level in context.old_logger_levels.items():
        logging.getLogger(logger_name).setLevel(level)
    if _ACTIVE_BUILD_LOG_CONTEXT is context:
        _ACTIVE_BUILD_LOG_CONTEXT = None


def _bridge_logger_to_internal_build_log(
    context: BuildLogContext,
    logger: logging.Logger,
) -> bool:
    if context.handler in logger.handlers:
        return False
    logger.addHandler(context.handler)
    logger.setLevel(logging.INFO)
    context.bridged_loggers.append(logger)
    return True


def _bridge_lightrag_logs_to_internal_build_log(
    context: BuildLogContext | None = None,
) -> None:
    context = context or _ACTIVE_BUILD_LOG_CONTEXT
    if context is None:
        return

    for logger_name in (
        "raganything",
        "raganything.processor",
        "raganything.parser",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = True

    try:
        from lightrag.utils import logger as lightrag_logger
    except Exception as exc:
        LOGGER.debug("LightRAG logger bridge not ready: %s", exc)
        return

    attached = _bridge_logger_to_internal_build_log(context, lightrag_logger)
    lightrag_logger.setLevel(logging.INFO)
    lightrag_logger.propagate = False
    for handler in lightrag_logger.handlers:
        stream = getattr(handler, "stream", None)
        if getattr(stream, "_internal_build_stream", False):
            try:
                handler.setStream(context.old_stderr)
            except Exception:
                pass
    LOGGER.info(
        "LightRAG log bridge ready (attached internal file handler: %s)",
        int(attached),
    )


def _count_log_markers(log_file: Path) -> tuple[int, int]:
    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0, 0
    warning_count = 0
    error_count = 0
    for line in text.splitlines():
        upper = line.upper()
        if "ERROR" in upper or "TRACEBACK" in upper:
            error_count += 1
        elif "WARNING" in upper:
            warning_count += 1
    return warning_count, error_count


def _attach_log_summary(
    summary: dict[str, Any],
    log_context: BuildLogContext,
) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    log_context.handler.flush()
    log_context.stream_mirror.flush()
    marker_warnings, marker_errors = _count_log_markers(log_context.log_file)
    summary["log_file"] = _path_env(log_context.log_file)
    summary["warning_count"] = max(log_context.warning_count, marker_warnings)
    summary["error_count"] = max(log_context.error_count, marker_errors)


def _write_partial_summary(
    report_dir: Path,
    summary: dict[str, Any],
    log_context: BuildLogContext,
    *,
    status: str,
) -> None:
    partial = dict(summary)
    partial["status"] = status
    partial["is_partial"] = True
    _attach_log_summary(partial, log_context)
    _write_json(report_dir / "summary.partial.json", partial)


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
    elif profile_name == "ran2_133_bis":
        default_raw_dir = RAN2_133_BIS_RAW_DIR
        default_storage_root = RAN2_133_BIS_STORAGE_ROOT
        default_workspace_id = "internal_RAN2_133_BIS"
    else:
        raise ValueError("profile must be 'test', 'prod', or 'ran2_133_bis'")

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
    mineru_gpu_memory_utilization = env.get(
        "MINERU_VLLM_GPU_MEMORY_UTILIZATION",
        str(DEFAULT_MINERU_VLLM_GPU_MEMORY_UTILIZATION),
    )
    libreoffice_timeout = env.get(
        "LIBREOFFICE_CONVERT_TIMEOUT_SECONDS",
        str(DEFAULT_BUILD_LIBREOFFICE_CONVERT_TIMEOUT_SECONDS),
    )
    multimodal_item_parallelism = env.get(
        "RAGANYTHING_MULTIMODAL_ITEM_PARALLELISM",
        str(DEFAULT_MULTIMODAL_ITEM_PARALLELISM or 3),
    )
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
            "RAGANYTHING_PRELOAD_RERANKER_MODEL": "false",
            "RAGANYTHING_PRESERVE_EXISTING_LOGGING": "true",
            "RAGANYTHING_DISABLE_LOCAL_RUN_LOG": "true",
            "RAGANYTHING_SERIALIZE_MINERU": "true",
            "RAGANYTHING_LLM_CONTEXT_MAX_TOKENS": env.get(
                "RAGANYTHING_LLM_CONTEXT_MAX_TOKENS",
                str(DEFAULT_LLM_CONTEXT_MAX_TOKENS),
            ),
            "RAGANYTHING_LLM_CONTEXT_RESERVED_TOKENS": env.get(
                "RAGANYTHING_LLM_CONTEXT_RESERVED_TOKENS",
                str(DEFAULT_LLM_CONTEXT_RESERVED_TOKENS),
            ),
            "RAGANYTHING_MULTIMODAL_ITEM_PARALLELISM": str(
                multimodal_item_parallelism
            ),
            "MINERU_VLLM_GPU_MEMORY_UTILIZATION": str(
                mineru_gpu_memory_utilization
            ),
            "LIBREOFFICE_CONVERT_TIMEOUT_SECONDS": str(libreoffice_timeout),
            "MAX_SOURCE_IDS_PER_ENTITY": str(INTERNAL_SOURCE_IDS_LIMIT),
            "MAX_SOURCE_IDS_PER_RELATION": str(INTERNAL_SOURCE_IDS_LIMIT),
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

    service = LocalRagService(settings)
    _bridge_lightrag_logs_to_internal_build_log()
    return service


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _clear_local_model_cache() -> None:
    try:
        import raganything.services.local_rag as local_rag_module

        model_cache = getattr(local_rag_module, "_MODEL_CACHE", None)
        if isinstance(model_cache, dict):
            model_cache.clear()
    except Exception:
        return


async def recycle_local_rag_service(
    service: Any,
    settings: Any,
    workspace_id: str,
    *,
    clear_model_cache: bool = False,
) -> Any:
    LOGGER.info(
        "Recycling LocalRagService for workspace=%s clear_model_cache=%s",
        workspace_id,
        clear_model_cache,
    )
    cleanup = getattr(service, "cleanup_workspace_instance", None)
    if callable(cleanup):
        await _maybe_await(cleanup(workspace_id))
    if clear_model_cache:
        _clear_local_model_cache()
    del service
    gc.collect()
    _clear_cuda_cache()
    new_service = create_local_rag_service(settings)
    LOGGER.info("LocalRagService recycled for workspace=%s", workspace_id)
    return new_service


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


def _mineru_workspace_output_dir(profile: BuildProfile) -> Path:
    return profile.output_dir / profile.workspace_id


def _primary_mineru_content_json_candidates(
    output_root: Path, source_path: Path
) -> tuple[list[Path], list[Path]]:
    stem = source_path.stem
    safe_stem = _safe_stem(source_path)
    candidates: list[Path] = []
    searched_roots: list[Path] = [output_root]
    direct_json = output_root / f"{stem}_content_list.json"
    if direct_json.exists():
        candidates.append(direct_json)

    for subdir_name in dict.fromkeys((stem, safe_stem)):
        subdir = output_root / subdir_name
        searched_roots.append(subdir)
        if subdir.is_dir():
            candidates.extend(subdir.rglob(f"{stem}_content_list.json"))
    return candidates, searched_roots


def _mineru_content_json_candidates(
    output_root: Path, source_path: Path
) -> tuple[list[Path], list[Path], set[str]]:
    stem = source_path.stem
    primary_candidates, searched_roots = _primary_mineru_content_json_candidates(
        output_root, source_path
    )
    candidates: list[Path] = list(primary_candidates)
    fallback_keys: set[str] = set()
    if output_root.is_dir():
        searched_roots.append(output_root / "**")
        primary_keys = {str(path) for path in primary_candidates}
        for candidate in output_root.rglob(f"{stem}_content_list.json"):
            key = str(candidate)
            if key not in primary_keys:
                fallback_keys.add(key)
            candidates.append(candidate)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped, searched_roots, fallback_keys


def _content_list_reject_reason(json_path: Path) -> str | None:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"unreadable_json:{type(exc).__name__}:{exc}"
    if isinstance(payload, list) and payload:
        return None
    if isinstance(payload, list):
        return "empty_list"
    return f"not_list:{type(payload).__name__}"


def _read_nonempty_content_list(json_path: Path) -> list[Any] | None:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list) and payload:
        return payload
    return None


def _inspect_mineru_artifacts(output_root: Path, source_path: Path) -> dict[str, Any]:
    source_mtime = source_path.stat().st_mtime
    candidates, searched_roots, fallback_keys = _mineru_content_json_candidates(
        output_root, source_path
    )
    valid_candidates: list[Path] = []
    reject_reasons: list[dict[str, str]] = []
    recovered_by_fallback = False
    for candidate in candidates:
        reason = _content_list_reject_reason(candidate)
        if reason is not None:
            reject_reasons.append({"path": _path_env(candidate), "reason": reason})
            continue
        try:
            if candidate.stat().st_mtime < source_mtime:
                LOGGER.warning(
                    "Reusing MinerU artifact older than source file=%s artifact=%s. "
                    "Delete the artifact to force reparsing after source content changes.",
                    source_path.name,
                    candidate,
                )
        except OSError:
            pass
        valid_candidates.append(candidate)
    artifact = None
    if valid_candidates:
        try:
            artifact = max(valid_candidates, key=lambda path: path.stat().st_mtime)
        except OSError:
            artifact = valid_candidates[0]
    if artifact is not None and str(artifact) in fallback_keys:
        recovered_by_fallback = True
    return {
        "artifact": artifact,
        "recovered_by_fallback": bool(artifact and recovered_by_fallback),
        "searched_roots": [_path_env(path) for path in searched_roots],
        "candidate_paths": [_path_env(path) for path in candidates],
        "candidate_reject_reasons": reject_reasons,
    }


def _find_valid_mineru_artifact(output_root: Path, source_path: Path) -> Path | None:
    return _inspect_mineru_artifacts(output_root, source_path).get("artifact")


def _should_preparse_with_mineru(file_path: Path) -> bool:
    return file_path.suffix.lower() in MINERU_PREPARSE_EXTENSIONS


def _converted_pdf_paths(file_path: Path, output_root: Path) -> tuple[Path, ...]:
    preferred_dir = output_root / _safe_stem(file_path)
    preferred_pdf = preferred_dir / f"{file_path.stem}.pdf"
    legacy_root_pdf = output_root / f"{file_path.stem}.pdf"
    if preferred_pdf == legacy_root_pdf:
        return (preferred_pdf,)
    return (preferred_pdf, legacy_root_pdf)


def _valid_converted_pdf(pdf_path: Path, source_path: Path) -> bool:
    try:
        if not pdf_path.exists() or pdf_path.stat().st_size <= 0:
            return False
        if pdf_path.stat().st_mtime < source_path.stat().st_mtime:
            LOGGER.warning(
                "Reusing converted PDF older than source file=%s pdf=%s. "
                "Delete the PDF to force reconversion after source content changes.",
                source_path.name,
                pdf_path,
            )
        return True
    except OSError:
        return False


def _find_valid_converted_pdf(file_path: Path, output_root: Path) -> Path | None:
    for pdf_path in _converted_pdf_paths(file_path, output_root):
        if _valid_converted_pdf(pdf_path, file_path):
            return pdf_path
    return None


def _load_historical_preparse_conversion_failures(
    profile: BuildProfile, current_report_dir: Path
) -> dict[Path, dict[str, str]]:
    failures: dict[Path, dict[str, str]] = {}
    reports_dir = profile.reports_dir
    if not reports_dir.exists():
        return failures
    try:
        current_report_resolved = current_report_dir.expanduser().resolve()
    except OSError:
        current_report_resolved = current_report_dir.expanduser()

    failure_files = list(reports_dir.glob("*/mineru_preparse_failures.json"))

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    for failure_file in sorted(failure_files, key=_mtime):
        try:
            if failure_file.parent.expanduser().resolve() == current_report_resolved:
                continue
        except OSError:
            pass
        try:
            payload = json.loads(failure_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for record in payload:
            if not isinstance(record, dict):
                continue
            source_path = record.get("source_path")
            if not source_path:
                continue
            try:
                key = Path(str(source_path)).expanduser().resolve()
            except OSError:
                key = Path(str(source_path)).expanduser()
            normalized_record = {
                str(k): str(v) for k, v in record.items() if v is not None
            }
            normalized_record["previous_report"] = _path_env(failure_file)
            failures[key] = normalized_record
    return failures


def _historical_conversion_failure_record(
    file_path: Path,
    output_root: Path,
    previous_failure: dict[str, str],
) -> dict[str, str]:
    message = (
        "Previous MinerU preparse conversion failed and no manual PDF exists; "
        "skipping LibreOffice retry. Create the recommended PDF and rerun the "
        "same build command."
    )
    record = _conversion_failure_record(file_path, output_root, RuntimeError(message))
    previous_error = previous_failure.get("error")
    previous_report = previous_failure.get("previous_report")
    if previous_error:
        record["previous_error"] = previous_error
    if previous_report:
        record["previous_report"] = previous_report
    return record


def _shell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _manual_convert_command(source_path: Path, pdf_path: Path) -> str:
    output_dir = pdf_path.parent
    return "\n".join(
        [
            f"src={_shell_quote(source_path)}",
            f"dst={_shell_quote(pdf_path)}",
            'out="$(dirname "$dst")"',
            'safe_name="$(basename "$src" | tr \'/ []()#:&;\' \'_________\')"',
            'log_file="$manual_convert_log_dir/${safe_name}.log"',
            'mkdir -p "$out" "$manual_convert_log_dir"',
            ': > "$log_file"',
            'echo "SOURCE: $src" >> "$log_file"',
            'echo "TARGET: $dst" >> "$log_file"',
            'echo "TIMEOUT_SEC: $TIMEOUT_SEC" >> "$log_file"',
            'echo "LO_COMMANDS: ${LO_COMMANDS[*]}" >> "$log_file"',
            'if [ -s "$dst" ]; then',
            '  echo "OK: $src already converted at $dst"',
            '  echo "ALREADY_EXISTS" >> "$log_file"',
            'else',
            'ext="${src##*.}"',
            'ext="$(echo "$ext" | tr \'[:upper:]\' \'[:lower:]\')"',
            'case "$ext" in',
            '  doc|docx)',
            '    filter="pdf:writer_pdf_Export"',
            '    ;;',
            '  ppt|pptx)',
            '    filter="pdf:impress_pdf_Export"',
            '    ;;',
            '  xls|xlsx)',
            '    filter="pdf:calc_pdf_Export"',
            '    ;;',
            '  *)',
            '    filter="pdf"',
            '    ;;',
            'esac',
            'echo "FILTER: $filter" >> "$log_file"',
            'job_dir="$(mktemp -d /tmp/lo_job.XXXXXX)"',
            'lo_out="$job_dir/out"',
            'profile="$job_dir/profile"',
            'mkdir -p "$lo_out" "$profile"',
            'status=0',
            'pdf_found=""',
            '# Kill stale LibreOffice workers from previous failed conversions for this user.',
            'pkill -9 -u "$USER" -f \'soffice|libreoffice|soffice.bin\' 2>/dev/null || true',
            'sleep 1',
            'for lo_cmd in "${LO_COMMANDS[@]}"; do',
            '  rm -rf "$profile"',
            '  profile="$(mktemp -d /tmp/lo_profile_XXXXXX)"',
            '  echo "Trying command: $lo_cmd" | tee -a "$log_file"',
            '  (',
            '    unset LOCPATH',
            '    export LANG="${LIBREOFFICE_LANG:-C.utf8}"',
            '    export LC_ALL="${LIBREOFFICE_LC_ALL:-C.utf8}"',
            "    export GTK_MODULES=''",
            '    export SAL_USE_VCLPLUGIN="${SAL_USE_VCLPLUGIN:-svp}"',
            '    timeout -k 20s "${TIMEOUT_SEC}s" \\',
            '    "$lo_cmd" "-env:UserInstallation=file://${profile}" \\',
            '      --headless --invisible --nologo --nodefault --nofirststartwizard \\',
            '      --nolockcheck --norestore --convert-to "$filter" \\',
            '      --outdir "$lo_out" "$src"',
            '  ) >> "$log_file" 2>&1',
            '  status=$?',
            '  pdf_found="$(find "$lo_out" -maxdepth 1 -type f -iname "*.pdf" -print -quit)"',
            '  echo "Command status: $status" >> "$log_file"',
            '  echo "PDF_FOUND: ${pdf_found:-}" >> "$log_file"',
            '  if [ "$status" -eq 0 ] && [ -n "$pdf_found" ] && [ -s "$pdf_found" ]; then',
            '    break',
            '  fi',
            '  echo "FAILED_COMMAND: $lo_cmd status=$status" >> "$log_file"',
            '  pkill -9 -u "$USER" -f \'soffice|libreoffice|soffice.bin\' 2>/dev/null || true',
            '  sleep 1',
            'done',
            'if [ "$status" -eq 0 ] && [ -n "$pdf_found" ] && [ -s "$pdf_found" ]; then',
            '  mv -f "$pdf_found" "$dst"',
            '  touch "$dst"',
            "  echo \"OK: $src\"",
            'else',
            '  echo "FAILED: $src status=$status (see $log_file)" >&2',
            '  manual_convert_failed=1',
            'fi',
            'pkill -9 -u "$USER" -f \'soffice|libreoffice|soffice.bin\' 2>/dev/null || true',
            'rm -rf "$job_dir"',
            'fi',
        ]
    )


def _conversion_failure_record(
    file_path: Path, output_root: Path, exc: BaseException
) -> dict[str, str]:
    pdf_paths = _converted_pdf_paths(file_path, output_root)
    recommended_pdf = pdf_paths[0]
    record = {
        "file": file_path.name,
        "source_path": _path_env(file_path),
        "recommended_pdf": _path_env(recommended_pdf),
        "error": str(exc),
        "manual_command": _manual_convert_command(file_path, recommended_pdf),
    }
    if len(pdf_paths) > 1:
        record["legacy_root_pdf"] = _path_env(pdf_paths[1])
    return record


def _write_mineru_preparse_failure_reports(
    report_dir: Path, failures: list[dict[str, str]]
) -> None:
    _write_json(report_dir / "mineru_preparse_failures.json", failures)
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "",
        "# Generated by build_internal_workspace.py.",
        "# Converts all failed files it can, reports failures, then exits non-zero if any remain.",
        "manual_convert_failed=0",
        f'TIMEOUT_SEC="${{LIBREOFFICE_CONVERT_TIMEOUT_SECONDS:-{DEFAULT_MANUAL_LIBREOFFICE_CONVERT_TIMEOUT_SECONDS}}}"',
        'manual_convert_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'manual_convert_log_dir="${manual_convert_log_dir:-${manual_convert_script_dir}/manual_convert_logs}"',
        'mkdir -p "$manual_convert_log_dir"',
        'echo "Manual conversion logs: $manual_convert_log_dir"',
        'echo "Manual conversion timeout: $TIMEOUT_SEC seconds"',
        'LO_COMMANDS=()',
        'if [ -n "${LIBREOFFICE_CONVERT_COMMANDS:-}" ]; then',
        '  IFS=" " read -r -a LO_COMMANDS <<< "$LIBREOFFICE_CONVERT_COMMANDS"',
        'else',
        '  command -v libreoffice >/dev/null 2>&1 && LO_COMMANDS+=("$(command -v libreoffice)")',
        '  command -v soffice >/dev/null 2>&1 && LO_COMMANDS+=("$(command -v soffice)")',
        'fi',
        'if [ "${#LO_COMMANDS[@]}" -eq 0 ]; then',
        '  echo "ERROR: libreoffice/soffice not found in PATH" >&2',
        '  exit 1',
        'fi',
        'echo "Using LibreOffice commands: ${LO_COMMANDS[*]}"',
        "",
    ]
    for index, failure in enumerate(failures, start=1):
        lines.extend(
            [
                f"# {index}. {failure['file']}",
                failure["manual_command"],
                "",
            ]
        )
    lines.extend(
        [
            "if [ \"$manual_convert_failed\" -ne 0 ]; then",
            "  echo \"One or more manual conversions failed.\" >&2",
            "  exit 1",
            "fi",
            "echo \"Manual conversions completed.\"",
            "",
        ]
    )
    (report_dir / "manual_convert_commands.sh").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _write_mineru_preparse_reports(
    report_dir: Path,
    summary: dict[str, Any],
) -> None:
    conversion_failures = list(summary.get("conversion_failed", []) or [])
    missing_artifacts = list(summary.get("missing_after_parse", []) or [])
    if conversion_failures:
        _write_mineru_preparse_failure_reports(report_dir, conversion_failures)
    _write_json(
        report_dir / "mineru_preparse_missing_artifacts.json",
        missing_artifacts,
    )
    _write_json(
        report_dir / "mineru_preparse_failures_all.json",
        _preparse_failure_results(summary),
    )
    _write_json(report_dir / "mineru_preparse_summary.json", summary)


def _prepare_mineru_preparse_input(file_path: Path, output_root: Path) -> Path:
    suffix = file_path.suffix.lower()
    if suffix in MINERU_OFFICE_EXTENSIONS:
        existing_pdf = _find_valid_converted_pdf(file_path, output_root)
        if existing_pdf is not None:
            LOGGER.info(
                "Reusing LibreOffice PDF for MinerU preparse file=%s pdf=%s",
                file_path.name,
                existing_pdf,
            )
            return existing_pdf
        from raganything.parser import MineruParser

        preferred_pdf = _converted_pdf_paths(file_path, output_root)[0]
        preferred_pdf.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Converting Office document before MinerU preparse file=%s output_dir=%s",
            file_path.name,
            preferred_pdf.parent,
        )
        return MineruParser.convert_office_to_pdf(file_path, preferred_pdf.parent)
    if suffix in MINERU_TEXT_EXTENSIONS:
        existing_pdf = _find_valid_converted_pdf(file_path, output_root)
        if existing_pdf is not None:
            LOGGER.info(
                "Reusing text PDF for MinerU preparse file=%s pdf=%s",
                file_path.name,
                existing_pdf,
            )
            return existing_pdf
        from raganything.parser import MineruParser

        preferred_pdf = _converted_pdf_paths(file_path, output_root)[0]
        preferred_pdf.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Converting text document before MinerU preparse file=%s output_dir=%s",
            file_path.name,
            preferred_pdf.parent,
        )
        return MineruParser.convert_text_to_pdf(file_path, preferred_pdf.parent)
    return file_path


def _stage_mineru_input(input_path: Path, staging_dir: Path, used_names: set[str]) -> Path:
    staged_path = staging_dir / input_path.name
    if staged_path.name in used_names:
        raise RuntimeError(
            "MinerU preparse input name collision for "
            f"{input_path.name}. Use unique source file stems."
        )
    used_names.add(staged_path.name)
    try:
        os.symlink(input_path, staged_path)
    except Exception:
        try:
            os.link(input_path, staged_path)
        except Exception:
            shutil.copy2(input_path, staged_path)
    return staged_path


def _run_mineru_preparse_command(input_dir: Path, output_root: Path) -> None:
    from raganything.parser import MineruParser

    MineruParser._run_mineru_command(
        input_path=input_dir,
        output_dir=output_root,
        method="auto",
    )


def preparse_mineru_files(
    profile: BuildProfile,
    files: list[Path],
    report_dir: Path,
) -> dict[str, Any]:
    output_root = _mineru_workspace_output_dir(profile)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = [path for path in files if _should_preparse_with_mineru(path)]
    summary: dict[str, Any] = {
        "enabled": True,
        "output_dir": _path_env(output_root),
        "candidate_count": len(candidates),
        "skipped_count": 0,
        "parsed_count": 0,
        "pending_count": 0,
        "skipped": [],
        "parsed": [],
        "missing_after_parse": [],
        "recovered_by_fallback": [],
        "conversion_failed": [],
        "historical_failure_skipped": [],
        "historical_failure_skipped_count": 0,
        "single_retry_count": 0,
        "single_retry_succeeded_count": 0,
        "single_retry_failed_count": 0,
        "conversion_failed_count": 0,
        "missing_artifact_count": 0,
        "recovered_by_fallback_count": 0,
        "failed_count": 0,
    }

    pending: list[Path] = []
    historical_failures = _load_historical_preparse_conversion_failures(
        profile, report_dir
    )
    for file_path in candidates:
        artifact_info = _inspect_mineru_artifacts(output_root, file_path)
        existing = artifact_info.get("artifact")
        if existing is not None:
            skipped_record = {
                "file": file_path.name,
                "artifact": _path_env(existing),
            }
            if artifact_info.get("recovered_by_fallback"):
                skipped_record["recovered_by_fallback"] = True
                summary["recovered_by_fallback"].append(skipped_record)
                LOGGER.info(
                    "Recovered MinerU artifact by fallback search file=%s artifact=%s",
                    file_path.name,
                    existing,
                )
            summary["skipped"].append(skipped_record)
            continue
        try:
            resolved_file_path = file_path.expanduser().resolve()
        except OSError:
            resolved_file_path = file_path.expanduser()
        previous_failure = historical_failures.get(resolved_file_path)
        if (
            previous_failure is not None
            and file_path.suffix.lower()
            in (MINERU_OFFICE_EXTENSIONS | MINERU_TEXT_EXTENSIONS)
            and _find_valid_converted_pdf(file_path, output_root) is None
        ):
            failure = _historical_conversion_failure_record(
                file_path,
                output_root,
                previous_failure,
            )
            summary["conversion_failed"].append(failure)
            summary["historical_failure_skipped"].append(failure)
            LOGGER.warning(
                "Skipping previously failed MinerU preparse conversion until manual PDF exists file=%s",
                file_path.name,
            )
            continue
        pending.append(file_path)

    summary["skipped_count"] = len(summary["skipped"])
    summary["pending_count"] = len(pending)
    summary["historical_failure_skipped_count"] = len(
        summary["historical_failure_skipped"]
    )
    summary["conversion_failed_count"] = len(summary["conversion_failed"])
    summary["recovered_by_fallback_count"] = len(summary["recovered_by_fallback"])
    if not pending:
        summary["missing_artifact_count"] = 0
        summary["failed_count"] = summary["conversion_failed_count"]
        _write_mineru_preparse_reports(report_dir, summary)
        LOGGER.info(
            "MinerU preparse skipped for all stageable files candidate_count=%d conversion_failed=%d missing_artifacts=%d recovered_by_fallback=%d output_dir=%s",
            len(candidates),
            summary["conversion_failed_count"],
            summary["missing_artifact_count"],
            summary["recovered_by_fallback_count"],
            output_root,
        )
        return summary

    LOGGER.info(
        "MinerU preparse start pending=%d skipped=%d output_dir=%s",
        len(pending),
        summary["skipped_count"],
        output_root,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="mineru_preparse_", dir=report_dir
    ) as temp_dir:
        staging_dir = Path(temp_dir) / "inputs"
        staging_dir.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        staged_records: list[dict[str, str]] = []
        staged_source_paths: list[Path] = []
        for file_path in pending:
            try:
                mineru_input = _prepare_mineru_preparse_input(file_path, output_root)
            except Exception as exc:
                failure = _conversion_failure_record(file_path, output_root, exc)
                summary["conversion_failed"].append(failure)
                LOGGER.exception(
                    "MinerU preparse input preparation failed file=%s",
                    file_path.name,
                )
                continue
            staged_input = _stage_mineru_input(mineru_input, staging_dir, used_names)
            staged_records.append(
                {
                    "file": file_path.name,
                    "mineru_input": _path_env(mineru_input),
                    "staged_input": _path_env(staged_input),
                }
            )
            staged_source_paths.append(file_path)

        if staged_records:
            LOGGER.info(
                "Executing MinerU preparse directory input_dir=%s files=%d",
                staging_dir,
                len(staged_records),
            )
            try:
                _run_mineru_preparse_command(staging_dir, output_root)
            except Exception as exc:
                summary["mineru_command_error"] = str(exc)
                LOGGER.exception(
                    "MinerU preparse directory command failed; build will continue "
                    "with successfully materialized artifacts"
                )
        else:
            LOGGER.warning("MinerU preparse has no stageable inputs after conversion")

    retry_errors: dict[Path, str] = {}
    retry_inputs: list[tuple[Path, Path]] = []
    for record, file_path in zip(staged_records, staged_source_paths):
        artifact_info = _inspect_mineru_artifacts(output_root, file_path)
        if artifact_info.get("artifact") is None:
            retry_inputs.append((file_path, Path(record["mineru_input"])))

    if retry_inputs:
        summary["single_retry_count"] = len(retry_inputs)
        LOGGER.warning(
            "MinerU preparse missing artifacts after directory parse; retrying files individually count=%d files=%s",
            len(retry_inputs),
            ", ".join(file_path.name for file_path, _ in retry_inputs),
        )
    for file_path, mineru_input in retry_inputs:
        try:
            LOGGER.info(
                "Retrying MinerU preparse as single file file=%s input=%s",
                file_path.name,
                mineru_input,
            )
            _run_mineru_preparse_command(mineru_input, output_root)
        except Exception as exc:
            retry_errors[file_path] = str(exc)
            LOGGER.exception(
                "Single-file MinerU preparse retry failed file=%s input=%s",
                file_path.name,
                mineru_input,
            )

    parsed_records: list[dict[str, str]] = []
    missing_records: list[dict[str, Any]] = []
    for file_path in staged_source_paths:
        artifact_info = _inspect_mineru_artifacts(output_root, file_path)
        artifact = artifact_info.get("artifact")
        if artifact is None:
            error = retry_errors.get(
                file_path, "content_list artifact missing after MinerU preparse"
            )
            missing_records.append(
                {
                    "file": file_path.name,
                    "stem": file_path.stem,
                    "safe_stem": _safe_stem(file_path),
                    "source_path": _path_env(file_path),
                    "output_root": _path_env(output_root),
                    "searched_roots": artifact_info.get("searched_roots", []),
                    "candidate_paths": artifact_info.get("candidate_paths", []),
                    "candidate_reject_reasons": artifact_info.get(
                        "candidate_reject_reasons", []
                    ),
                    "error": error,
                }
            )
            continue
        parsed_record = {
            "file": file_path.name,
            "artifact": _path_env(artifact),
        }
        if artifact_info.get("recovered_by_fallback"):
            parsed_record["recovered_by_fallback"] = True
            summary["recovered_by_fallback"].append(parsed_record)
            LOGGER.info(
                "Recovered MinerU artifact by fallback search file=%s artifact=%s",
                file_path.name,
                artifact,
            )
        parsed_records.append(parsed_record)

    summary["parsed"] = parsed_records
    summary["parsed_count"] = len(parsed_records)
    summary["missing_after_parse"] = missing_records
    summary["single_retry_succeeded_count"] = (
        summary["single_retry_count"] - len(missing_records)
    )
    summary["single_retry_failed_count"] = len(missing_records)
    summary["conversion_failed_count"] = len(summary["conversion_failed"])
    summary["missing_artifact_count"] = len(missing_records)
    summary["recovered_by_fallback_count"] = len(summary["recovered_by_fallback"])
    summary["failed_count"] = (
        summary["conversion_failed_count"] + summary["missing_artifact_count"]
    )
    _write_mineru_preparse_reports(report_dir, summary)
    if missing_records:
        LOGGER.error(
            "MinerU preparse completed with missing artifacts: %s",
            ", ".join(record["file"] for record in missing_records),
        )

    LOGGER.info(
        "MinerU preparse complete parsed=%d skipped=%d conversion_failed=%d missing_artifacts=%d recovered_by_fallback=%d failed=%d output_dir=%s",
        summary["parsed_count"],
        summary["skipped_count"],
        summary["conversion_failed_count"],
        summary["missing_artifact_count"],
        summary["recovered_by_fallback_count"],
        summary["failed_count"],
        output_root,
    )
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, type):
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", str(value)))
        return f"{module}.{qualname}" if module else str(qualname)
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
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
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


def _preparse_failure_results(preparse_summary: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in preparse_summary.get("conversion_failed", []) or []:
        results.append(
            {
                "file": record.get("file"),
                "status": "failed",
                "stage": "mineru_preparse_conversion",
                "error": record.get("error"),
                "source_path": record.get("source_path"),
                "recommended_pdf": record.get("recommended_pdf"),
            }
        )
    for record in preparse_summary.get("missing_after_parse", []) or []:
        results.append(
            {
                "file": record.get("file"),
                "status": "failed",
                "stage": "mineru_preparse_missing_artifact",
                "error": record.get("error"),
                "source_path": record.get("source_path"),
            }
        )
    return results


def _preparse_failed_source_paths(preparse_summary: dict[str, Any]) -> set[Path]:
    failed_paths: set[Path] = set()
    for result in _preparse_failure_results(preparse_summary):
        source_path = result.get("source_path")
        if not source_path:
            continue
        try:
            failed_paths.add(Path(str(source_path)).expanduser().resolve())
        except OSError:
            failed_paths.add(Path(str(source_path)).expanduser())
    return failed_paths


def _write_audit_reports(
    *,
    report_dir: Path,
    documents_payload: dict[str, Any],
    entities_payload: dict[str, Any],
    relations_payload: dict[str, Any],
    graph_stats_payload: dict[str, Any],
) -> None:
    documents = list(documents_payload.get("documents", []))
    _ = entities_payload, relations_payload
    _write_json(report_dir / "documents.json", documents_payload)
    _write_json(report_dir / "graph_stats.json", graph_stats_payload)
    _write_csv(report_dir / "documents.csv", documents)


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
        "preload_reranker_model",
        "llm_context_max_tokens",
        "llm_context_reserved_tokens",
        "multimodal_item_parallelism",
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
    recycle_service_every: int,
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
            "multimodal_item_parallelism": env.get(
                "RAGANYTHING_MULTIMODAL_ITEM_PARALLELISM",
                str(DEFAULT_MULTIMODAL_ITEM_PARALLELISM or DEFAULT_MAX_PARALLEL_INSERT),
            ),
            "recycle_service_every": recycle_service_every,
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
    output_dir = _mineru_workspace_output_dir(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        start = time.time()
        LOGGER.info(
            "Ingest start file=%s attempt=%d workspace=%s output_dir=%s",
            file_path.name,
            attempt,
            profile.workspace_id,
            output_dir,
        )
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
            LOGGER.info(
                "Ingest success file=%s attempt=%d elapsed=%.3fs",
                file_path.name,
                attempt,
                time.time() - start,
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
            LOGGER.exception(
                "Ingest failed file=%s attempt=%d elapsed=%.3fs",
                file_path.name,
                attempt,
                time.time() - start,
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
    LOGGER.info("Collecting audit reports for workspace=%s", profile.workspace_id)
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
    LOGGER.info(
        "Audit reports written documents=%s entities=%s relations=%s",
        documents_payload.get("count", 0),
        entities_payload.get("count", 0),
        relations_payload.get("count", 0),
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
    LOGGER.info("Delete-doc start workspace=%s doc_id=%s", profile.workspace_id, doc_id)
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
        LOGGER.info("Delete-doc succeeded; rebuilding synonym edges doc_id=%s", doc_id)
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
    LOGGER.info("Delete-doc complete doc_id=%s status=%s", doc_id, status)
    return delete_check


async def _run_build_async(
    *,
    args: argparse.Namespace,
    profile: BuildProfile,
    report_dir: Path,
    env: dict[str, str],
    files: list[Path],
    max_async_ingest: int,
    recycle_service_every: int,
    log_context: BuildLogContext,
) -> int:
    settings = build_local_settings(profile)
    index_profile = build_index_profile(_internal_index_flags(), settings=settings)
    ensured_profile = ensure_workspace_index_profile(
        working_dir_root=profile.working_dir_root,
        workspace_id=profile.workspace_id,
        index_profile=index_profile,
        allow_legacy_adoption=bool(args.allow_legacy_index_profile_adoption),
    )
    service = None
    start_time = time.time()
    summary = _summary_base(
        profile=profile,
        report_dir=report_dir,
        env=env,
        max_async_ingest=max_async_ingest,
        recycle_service_every=recycle_service_every,
        files=files,
        settings=settings,
    )
    summary["index_profile"] = _json_safe(ensured_profile)
    try:
        LOGGER.info(
            (
                "Internal build start profile=%s workspace=%s raw_dir=%s "
                "storage_root=%s files=%d max_async_ingest=%d "
                "recycle_service_every=%d"
            ),
            profile.name,
            profile.workspace_id,
            profile.raw_dir,
            profile.storage_root,
            len(files),
            max_async_ingest,
            recycle_service_every,
        )
        _write_partial_summary(
            report_dir,
            summary,
            log_context,
            status="build_start",
        )
        if getattr(args, "skip_mineru_preparse", False):
            summary["mineru_preparse"] = {
                "enabled": False,
                "reason": "disabled_by_cli",
                "output_dir": _path_env(_mineru_workspace_output_dir(profile)),
                "failed_count": 0,
            }
            LOGGER.info("MinerU preparse disabled by CLI")
        else:
            summary["mineru_preparse"] = preparse_mineru_files(
                profile,
                files,
                report_dir,
            )

        preparse_failed_paths = _preparse_failed_source_paths(summary["mineru_preparse"])
        ingest_files = [
            path
            for path in files
            if path.expanduser().resolve() not in preparse_failed_paths
        ]
        if preparse_failed_paths:
            LOGGER.warning(
                "Skipping %d files with MinerU preparse failures before ingest",
                len(preparse_failed_paths),
            )
        summary["ingest_planned_count"] = len(ingest_files)
        _write_partial_summary(
            report_dir,
            summary,
            log_context,
            status="preparse_complete",
        )

        service = create_local_rag_service(settings)
        batch_results: list[dict[str, Any]] = []
        attempted_since_recycle = 0
        for batch_index, batch in enumerate(
            _iter_batches(ingest_files, max_async_ingest),
            start=1,
        ):
            LOGGER.info(
                "Build batch start batch_index=%d files=%s",
                batch_index,
                ", ".join(path.name for path in batch),
            )
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
            failed_in_batch = [
                result for result in results if result.get("status") != "success"
            ]
            LOGGER.info(
                "Build batch complete batch_index=%d succeeded=%d failed=%d",
                batch_index,
                len(results) - len(failed_in_batch),
                len(failed_in_batch),
            )
            summary["batch_results"] = batch_results
            summary["elapsed_seconds"] = time.time() - start_time
            _write_partial_summary(
                report_dir,
                summary,
                log_context,
                status=f"batch_{batch_index}_complete",
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
            attempted_since_recycle += len(batch)
            if (
                recycle_service_every > 0
                and attempted_since_recycle >= recycle_service_every
            ):
                service = await recycle_local_rag_service(
                    service,
                    settings,
                    profile.workspace_id,
                    clear_model_cache=False,
                )
                attempted_since_recycle = 0

        synonym_result = None
        if getattr(settings, "enable_synonym_linking", False):
            LOGGER.info(
                "Finalizing workspace synonyms workspace=%s",
                profile.workspace_id,
            )
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
        preparse_failed_results = _preparse_failure_results(summary["mineru_preparse"])
        failed_results = preparse_failed_results + failed_results
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
        _attach_log_summary(summary, log_context)
        _write_json(report_dir / "summary.json", summary)
        LOGGER.info(
            "Internal build complete workspace=%s succeeded=%d failed=%d reports=%s",
            profile.workspace_id,
            summary["succeeded_count"],
            summary["failed_count"],
            report_dir,
        )
        print(f"Reports written to {report_dir}")
        return 1 if failed_results else 0
    finally:
        if service is not None:
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
    recycle_service_every: int,
    log_context: BuildLogContext,
) -> int:
    settings = build_local_settings(profile)
    service = create_local_rag_service(settings)
    start_time = time.time()
    try:
        LOGGER.info(
            "Report command start profile=%s workspace=%s",
            profile.name,
            profile.workspace_id,
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
            recycle_service_every=recycle_service_every,
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
        _attach_log_summary(summary, log_context)
        _write_json(report_dir / "summary.json", summary)
        LOGGER.info(
            "Report command complete workspace=%s reports=%s",
            profile.workspace_id,
            report_dir,
        )
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
    recycle_service_every: int,
    log_context: BuildLogContext,
) -> int:
    settings = build_local_settings(profile)
    service = create_local_rag_service(settings)
    start_time = time.time()
    try:
        LOGGER.info(
            "Delete-doc command start profile=%s workspace=%s doc_id=%s",
            profile.name,
            profile.workspace_id,
            args.doc_id,
        )
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
            recycle_service_every=recycle_service_every,
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
        _attach_log_summary(summary, log_context)
        _write_json(report_dir / "summary.json", summary)
        LOGGER.info(
            "Delete-doc command complete workspace=%s doc_id=%s reports=%s",
            profile.workspace_id,
            args.doc_id,
            report_dir,
        )
        print(f"Reports written to {report_dir}")
        return 0
    finally:
        cleanup = getattr(service, "cleanup_workspace_instance", None)
        if callable(cleanup):
            await _maybe_await(cleanup(profile.workspace_id))


def _prepare_profile_and_dirs(
    args: argparse.Namespace,
) -> tuple[BuildProfile, Path, int, int, dict[str, str]]:
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
    recycle_service_every = (
        int(args.recycle_service_every)
        if args.recycle_service_every is not None
        else max_async_ingest
    )
    if recycle_service_every < 0:
        raise ValueError("--recycle-service-every must be >= 0")
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
    return profile, report_dir, max_async_ingest, recycle_service_every, env


def run_build(args: argparse.Namespace) -> int:
    profile, report_dir, max_async_ingest, recycle_service_every, env = (
        _prepare_profile_and_dirs(args)
    )
    log_context = setup_build_logging(report_dir)
    try:
        files = collect_supported_files(profile.raw_dir)
        if not files:
            raise RuntimeError(f"No supported files found in {profile.raw_dir}")
        if args.dry_run:
            LOGGER.info(
                "Dry-run build start profile=%s workspace=%s files=%d",
                profile.name,
                profile.workspace_id,
                len(files),
            )
            summary = _summary_base(
                profile=profile,
                report_dir=report_dir,
                env=env,
                max_async_ingest=max_async_ingest,
                recycle_service_every=recycle_service_every,
                files=files,
            )
            summary["dry_run"] = True
            summary["mineru_preparse"] = {
                "enabled": not bool(args.skip_mineru_preparse),
                "dry_run": True,
                "output_dir": _path_env(_mineru_workspace_output_dir(profile)),
            }
            _attach_log_summary(summary, log_context)
            _write_json(report_dir / "summary.json", summary)
            LOGGER.info("Dry-run build complete reports=%s", report_dir)
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
                    recycle_service_every=recycle_service_every,
                    log_context=log_context,
                )
            )
    except Exception:
        LOGGER.exception("Internal build command failed")
        raise
    finally:
        close_build_logging(log_context)


def run_report(args: argparse.Namespace) -> int:
    profile, report_dir, max_async_ingest, recycle_service_every, env = (
        _prepare_profile_and_dirs(args)
    )
    log_context = setup_build_logging(report_dir)
    try:
        with _temporary_env(env):
            return asyncio.run(
                _run_report_async(
                    args=args,
                    profile=profile,
                    report_dir=report_dir,
                    env=env,
                    max_async_ingest=max_async_ingest,
                    recycle_service_every=recycle_service_every,
                    log_context=log_context,
                )
            )
    except Exception:
        LOGGER.exception("Internal report command failed")
        raise
    finally:
        close_build_logging(log_context)


def run_delete_doc(args: argparse.Namespace) -> int:
    if not args.doc_id:
        raise ValueError("delete-doc requires --doc-id")
    profile, report_dir, max_async_ingest, recycle_service_every, env = (
        _prepare_profile_and_dirs(args)
    )
    log_context = setup_build_logging(report_dir)
    try:
        with _temporary_env(env):
            return asyncio.run(
                _run_delete_async(
                    args=args,
                    profile=profile,
                    report_dir=report_dir,
                    env=env,
                    max_async_ingest=max_async_ingest,
                    recycle_service_every=recycle_service_every,
                    log_context=log_context,
                )
            )
    except Exception:
        LOGGER.exception("Internal delete-doc command failed")
        raise
    finally:
        close_build_logging(log_context)


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
    parser.add_argument(
        "--profile",
        choices=("test", "prod", "ran2_133_bis"),
        default="test",
    )
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
        "--recycle-service-every",
        type=int,
        default=None,
        help=(
            "Attempted files between LocalRagService cleanup/recreation. "
            "Defaults to --max-async-ingest; use 0 to disable."
        ),
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
        "--skip-mineru-preparse",
        action="store_true",
        help=(
            "Disable the default MinerU preparse stage and let each ingest parse "
            "through the normal LocalRagService path."
        ),
    )
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
