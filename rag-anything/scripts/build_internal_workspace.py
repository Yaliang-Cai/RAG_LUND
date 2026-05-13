#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTRAG_ROOT = REPO_ROOT.parent / "lightrag"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if LIGHTRAG_ROOT.exists() and str(LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTRAG_ROOT))

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

SERVER_HOST = "127.0.0.1"
DEFAULT_PORT = 0
DEFAULT_INGEST_TIMEOUT_SECONDS = 7200.0
DEFAULT_BATCH_SIZE = DEFAULT_MAX_ASYNC_INGEST


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
        raw_dir=raw_dir or default_raw_dir,
        storage_root=storage_root or default_storage_root,
        workspace_id=workspace_id or default_workspace_id,
    )


def build_server_env(
    profile: BuildProfile,
    *,
    base_env: dict[str, str] | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "RAGANYTHING_WORKDIR_ROOT": _path_env(profile.working_dir_root),
            "RAGANYTHING_OUTPUT_DIR": _path_env(profile.output_dir),
            "RAGANYTHING_UPLOADS_DIR": _path_env(profile.uploads_dir),
            "RAGANYTHING_LOG_DIR": _path_env(profile.log_dir),
            "RAGANYTHING_ENABLE_SYNONYM_LINKING": "true",
            "RAGANYTHING_ENABLE_RESILIENCE": "true",
            "RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION": "true",
            "RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION": "true",
            "RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH": "true",
            "ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE": "true",
            "CONTEXT_ZERO_WINDOW_CONTENT_TYPES": DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
            "MAX_CONCURRENT_FILES": str(DEFAULT_BATCH_SIZE),
        }
    )
    if api_key:
        env["RAGANYTHING_API_KEY"] = api_key

    pythonpath_entries = [str(REPO_ROOT)]
    if LIGHTRAG_ROOT.exists():
        pythonpath_entries.append(str(LIGHTRAG_ROOT))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


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


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((SERVER_HOST, 0))
        return int(sock.getsockname()[1])


class ManagedServer:
    def __init__(
        self,
        *,
        env: dict[str, str],
        port: int,
        report_dir: Path,
    ) -> None:
        self.env = env
        self.port = port
        self.report_dir = report_dir
        self.process: subprocess.Popen[str] | None = None
        self._stdout = None
        self._stderr = None

    @property
    def base_url(self) -> str:
        return f"http://{SERVER_HOST}:{self.port}"

    def start(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._stdout = (self.report_dir / "server_stdout.log").open("a", encoding="utf-8")
        self._stderr = (self.report_dir / "server_stderr.log").open("a", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            SERVER_HOST,
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=self.env,
            stdout=self._stdout,
            stderr=self._stderr,
            text=True,
        )
        time.sleep(0.5)
        if self.process.poll() is not None:
            raise RuntimeError(
                "Managed server exited during startup. "
                f"Check {self.report_dir / 'server_stderr.log'}"
            )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        if self._stdout is not None:
            self._stdout.close()
            self._stdout = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def restart(self) -> None:
        self.stop()
        self.start()


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-Api-Key": api_key} if api_key else {}


def wait_for_server(
    base_url: str,
    *,
    api_key: str | None,
    timeout_seconds: float = 120.0,
) -> None:
    import httpx

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(
                f"{base_url}/config",
                headers=_headers(api_key),
                timeout=5.0,
            )
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Server did not become ready: {last_error}")


def _get_json(base_url: str, path: str, *, api_key: str | None, timeout: float = 60.0) -> dict[str, Any]:
    import httpx

    response = httpx.get(
        f"{base_url}{path}",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"result": payload}


def _delete_json(
    base_url: str,
    path: str,
    *,
    api_key: str | None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    import httpx

    response = httpx.delete(
        f"{base_url}{path}",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"result": payload}


def verify_server_roots(base_url: str, profile: BuildProfile, *, api_key: str | None) -> dict[str, Any]:
    payload = _get_json(base_url, "/workspaces", api_key=api_key)
    roots = payload.get("roots", {})
    expected = {
        "uploads": profile.uploads_dir,
        "output": profile.output_dir,
        "workspace": profile.working_dir_root,
    }
    mismatches = []
    for key, expected_path in expected.items():
        actual = roots.get(key)
        if actual is None or Path(actual).resolve() != expected_path.resolve():
            mismatches.append(
                {
                    "key": key,
                    "expected": _path_env(expected_path),
                    "actual": actual,
                }
            )
    if mismatches:
        raise RuntimeError(f"Server roots do not match requested storage root: {mismatches}")
    return payload


def ingest_batch(
    base_url: str,
    files: list[Path],
    *,
    workspace_id: str,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    import httpx

    with ExitStack() as stack:
        multipart_files = [
            (
                "files",
                (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    "application/octet-stream",
                ),
            )
            for path in files
        ]
        response = httpx.post(
            f"{base_url}/ingest/batch",
            headers=_headers(api_key),
            data={"workspace_id": workspace_id},
            files=multipart_files,
            timeout=timeout_seconds,
        )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"result": payload}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
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
                    key: json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _iter_batches(files: list[Path], batch_size: int) -> list[list[Path]]:
    return [files[i : i + batch_size] for i in range(0, len(files), batch_size)]


def _write_audit_reports(
    *,
    report_dir: Path,
    documents_payload: dict[str, Any],
    entities_payload: dict[str, Any],
    graph_stats_payload: dict[str, Any],
) -> None:
    documents = list(documents_payload.get("documents", []))
    entities = list(entities_payload.get("entities", []))
    _write_json(report_dir / "documents.json", documents_payload)
    _write_json(report_dir / "entities.json", entities_payload)
    _write_json(report_dir / "graph_stats.json", graph_stats_payload)
    _write_csv(report_dir / "documents.csv", documents)
    _write_csv(report_dir / "entities.csv", entities)


def run_build(args: argparse.Namespace) -> int:
    profile = resolve_profile(
        args.profile,
        raw_dir=Path(args.raw_dir).expanduser() if args.raw_dir else None,
        storage_root=Path(args.storage_root).expanduser() if args.storage_root else None,
        workspace_id=args.workspace_id,
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = profile.reports_dir / run_id
    files = collect_supported_files(profile.raw_dir)
    if not files:
        raise RuntimeError(f"No supported files found in {profile.raw_dir}")

    for directory in (
        profile.uploads_dir,
        profile.output_dir,
        profile.working_dir_root,
        profile.log_dir,
        report_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    port = int(args.port) if int(args.port) > 0 else find_free_port()
    env = build_server_env(profile, api_key=args.api_key)
    summary: dict[str, Any] = {
        "profile": profile.name,
        "workspace_id": profile.workspace_id,
        "raw_dir": _path_env(profile.raw_dir),
        "storage_root": _path_env(profile.storage_root),
        "report_dir": _path_env(report_dir),
        "file_count": len(files),
        "files": [path.name for path in files],
        "server_host": SERVER_HOST,
        "server_port": port,
        "concurrency": {
            "file_batch_size": DEFAULT_BATCH_SIZE,
            "MAX_CONCURRENT_FILES": env["MAX_CONCURRENT_FILES"],
            "lightrag_llm_model_max_async": DEFAULT_LLM_MODEL_MAX_ASYNC,
            "lightrag_max_parallel_insert": DEFAULT_MAX_PARALLEL_INSERT,
            "embedding_batch_num": DEFAULT_EMBEDDING_BATCH_NUM,
            "embedding_func_max_async": DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
            "vllm_max_num_seqs": "external_model_server_setting",
        },
        "retries": [],
    }

    if args.dry_run:
        _write_json(report_dir / "summary.json", summary)
        print(f"Dry run wrote summary to {report_dir / 'summary.json'}")
        return 0

    server = ManagedServer(env=env, port=port, report_dir=report_dir)
    start_time = time.time()
    try:
        server.start()
        wait_for_server(server.base_url, api_key=args.api_key)
        roots_payload = verify_server_roots(server.base_url, profile, api_key=args.api_key)
        summary["server_roots"] = roots_payload.get("roots", {})

        batch_results = []
        for batch_index, batch in enumerate(_iter_batches(files, DEFAULT_BATCH_SIZE), start=1):
            attempts = 0
            while True:
                attempts += 1
                try:
                    result = ingest_batch(
                        server.base_url,
                        batch,
                        workspace_id=profile.workspace_id,
                        api_key=args.api_key,
                        timeout_seconds=float(args.ingest_timeout_seconds),
                    )
                    batch_results.append(
                        {
                            "batch_index": batch_index,
                            "attempts": attempts,
                            "files": [path.name for path in batch],
                            "response": result,
                        }
                    )
                    break
                except Exception as exc:
                    summary["retries"].append(
                        {
                            "batch_index": batch_index,
                            "attempt": attempts,
                            "files": [path.name for path in batch],
                            "error": str(exc),
                        }
                    )
                    if attempts >= int(args.max_batch_attempts):
                        raise
                    server.restart()
                    wait_for_server(server.base_url, api_key=args.api_key)
                    verify_server_roots(server.base_url, profile, api_key=args.api_key)

        documents_payload = _get_json(
            server.base_url,
            f"/workspace/{profile.workspace_id}/documents",
            api_key=args.api_key,
        )
        entities_payload = _get_json(
            server.base_url,
            f"/graph/{profile.workspace_id}/entities",
            api_key=args.api_key,
        )
        graph_stats_payload = _get_json(
            server.base_url,
            f"/graph/{profile.workspace_id}/stats",
            api_key=args.api_key,
        )
        _write_audit_reports(
            report_dir=report_dir,
            documents_payload=documents_payload,
            entities_payload=entities_payload,
            graph_stats_payload=graph_stats_payload,
        )

        delete_check = None
        if args.delete_check:
            documents = list(documents_payload.get("documents", []))
            target_doc_id = args.delete_check_doc_id
            if not target_doc_id and documents:
                target_doc_id = str(documents[0].get("doc_id", ""))
            if not target_doc_id:
                raise RuntimeError("--delete-check requested but no document is available")
            before_delete = {
                "documents": documents_payload,
                "graph_stats": graph_stats_payload,
            }
            delete_result = _delete_json(
                server.base_url,
                f"/workspace/{profile.workspace_id}/documents/{target_doc_id}",
                api_key=args.api_key,
                timeout=max(300.0, float(args.ingest_timeout_seconds)),
            )
            after_documents = _get_json(
                server.base_url,
                f"/workspace/{profile.workspace_id}/documents",
                api_key=args.api_key,
            )
            after_graph_stats = _get_json(
                server.base_url,
                f"/graph/{profile.workspace_id}/stats",
                api_key=args.api_key,
            )
            delete_check = {
                "target_doc_id": target_doc_id,
                "before": before_delete,
                "delete_result": delete_result,
                "after": {
                    "documents": after_documents,
                    "graph_stats": after_graph_stats,
                },
            }
            _write_json(report_dir / "delete_check.json", delete_check)

        summary.update(
            {
                "elapsed_seconds": time.time() - start_time,
                "batch_results": batch_results,
                "documents_count": documents_payload.get("count", 0),
                "entities_count": entities_payload.get("count", 0),
                "graph_stats": graph_stats_payload,
                "delete_check": delete_check,
            }
        )
        _write_json(report_dir / "summary.json", summary)
        print(f"Reports written to {report_dir}")
        return 0
    finally:
        server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a small internal RAG workspace through server.app."
    )
    parser.add_argument("--profile", choices=("test", "prod"), default="test")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--ingest-timeout-seconds",
        type=float,
        default=DEFAULT_INGEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--max-batch-attempts", type=int, default=2)
    parser.add_argument("--delete-check", action="store_true")
    parser.add_argument("--delete-check-doc-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
