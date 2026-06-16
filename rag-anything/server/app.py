import hashlib
import html
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Set

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from raganything.governance import (
    GovernanceService,
    GovernanceSettings,
    JobRunner,
    WorkspaceFrozenError,
    create_pool,
    mark_orphaned_jobs_crashed,
    run_migrations,
)

from raganything.services.local_rag import LocalRagService, LocalRagSettings
from raganything.agent.session import SessionStore
from server.agent_routes import WorkspaceAgentRunner, build_agent_router
from raganything.chunking import CHUNKING_STRATEGIES
from raganything.constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    MAX_TOP_K as DEFAULT_MAX_TOP_K,
    MAX_CHUNK_TOP_K as DEFAULT_MAX_CHUNK_TOP_K,
    DEFAULT_SUPPORTED_FILE_EXTENSIONS,
    DEFAULT_UPLOADS_DIR,
    DEFAULT_QUERY_MODE,
    DEFAULT_ENABLE_RERANK,
    DEFAULT_VLM_ENHANCED,
    DEFAULT_GRAPH_MAX_DEPTH,
    DEFAULT_GRAPH_MAX_NODES,
    DEFAULT_GRAPH_OVERVIEW_MAX_NODES,
    DEFAULT_GRAPH_HTML_MAX_NODES,
    DEFAULT_GRAPH_SEARCH_SEED_LIMIT,
    DEFAULT_GRAPH_SEARCH_MAX_RESULTS,
    DEFAULT_GRAPH_SEARCH_MAX_SAFE,
    DEFAULT_MULTI_HOP_DEPTH,
    DEFAULT_PPR_DAMPING,
    DEFAULT_PPR_TOP_K,
    DEFAULT_PASSAGE_NODE_WEIGHT,
    DEFAULT_PPR_SYNONYM_WEIGHT_MODE,
    DEFAULT_RECOGNITION_TOP_K,
    DEFAULT_LINKING_TOP_K,
    DEFAULT_PPR_QA_TOP_K,
    DEFAULT_QDRANT_RETRIEVAL_MODE,
    SUPPORTED_IMAGE_EXTENSIONS,
    DEFAULT_MAX_IMAGE_SIZE_MB,
)
from uuid import uuid4

SUPPORTED_IMAGE_EXTS: Set[str] = {e.lower() for e in SUPPORTED_IMAGE_EXTENSIONS}
MAX_IMAGE_BYTES = DEFAULT_MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_IMAGES_PER_QUERY = 4

VALID_CHUNKING_STRATEGIES: Set[str] = set(CHUNKING_STRATEGIES.keys())

logger = logging.getLogger(__name__)

# --- 配置与初始化 ---
APP_ROOT = Path(__file__).resolve().parent

API_KEY_ENV = "RAGANYTHING_API_KEY"
MAX_TOP_K = int(os.getenv("RAGANYTHING_MAX_TOP_K", str(DEFAULT_MAX_TOP_K)))
MAX_CHUNK_TOP_K = int(os.getenv("RAGANYTHING_MAX_CHUNK_TOP_K", str(DEFAULT_MAX_CHUNK_TOP_K)))

SUPPORTED_EXTENSIONS: Set[str] = {
    ext.strip().lower()
    for ext in DEFAULT_SUPPORTED_FILE_EXTENSIONS.split(",")
}

# 三层存储目录
UPLOADS_DIR = Path(os.getenv("RAGANYTHING_UPLOADS_DIR", DEFAULT_UPLOADS_DIR)).resolve()

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_settings = LocalRagSettings.from_env()
    gov_settings = GovernanceSettings.from_env()

    # Phoenix MUST be initialised before any AsyncOpenAI / langgraph object
    # is constructed: phoenix.otel.register(auto_instrument=True) patches
    # those modules at call time, but some OpenAI SDK versions bind the
    # instrumentor in AsyncOpenAI.__init__ — clients created earlier are
    # missed. The CLI script (scripts/query_ppr.py) already follows this
    # order; we mirror it here so web UI traces show up too.
    phoenix_enabled = os.getenv("ENABLE_PHOENIX", "").lower() in ("1", "true", "yes")
    if phoenix_enabled:
        from raganything.observability import setup_phoenix
        setup_phoenix()
        logger.info("lifespan: Phoenix tracing enabled at http://localhost:6006")

    pg_pool = await create_pool(gov_settings)
    await run_migrations(pg_pool)
    crashed = await mark_orphaned_jobs_crashed(pg_pool)
    if crashed:
        logger.warning("lifespan: marked %d orphaned jobs as crashed", crashed)

    rag_service = LocalRagService(rag_settings)
    gov_service = GovernanceService(pg_pool, rag_service)
    job_runner = JobRunner(gov_service, max_concurrent=gov_settings.job_max_concurrent)
    await job_runner.start()

    # Discover existing workspaces on disk and register them.
    working_root = Path(rag_settings.working_dir_root).resolve()
    output_root = Path(rag_settings.output_dir).resolve()
    discovered: set[str] = set()
    for root in (working_root, output_root):
        if root.exists():
            for d in root.iterdir():
                if d.is_dir():
                    discovered.add(d.name)
    if discovered:
        new_count = await gov_service.backfill_legacy_workspaces(sorted(discovered))
        if new_count:
            logger.info("lifespan: backfilled %d legacy workspaces", new_count)

    app.state.pg_pool = pg_pool
    app.state.rag = rag_service
    app.state.gov = gov_service
    app.state.jobs = job_runner
    app.state.gov_settings = gov_settings

    # Agentic RAG v3：会话工作记忆 + /agent/chat、/agent/sessions/{id}/cancel 端点。
    app.state.session_store = SessionStore()
    app.include_router(
        build_agent_router(
            app.state.session_store,
            WorkspaceAgentRunner(rag_service),
        )
    )

    logger.info("lifespan: startup complete")
    try:
        yield
    finally:
        await job_runner.stop(grace_period=gov_settings.job_shutdown_grace)
        await rag_service.aclose()
        await pg_pool.close()
        logger.info("lifespan: shutdown complete")


app = FastAPI(title="RAGAnything Local Service", lifespan=lifespan)


def _compute_workspace_id(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    return cleaned if cleaned else hashlib.md5(name.encode("utf-8")).hexdigest()


def _validate_workspace_id(workspace_id: str):
    if ".." in workspace_id or "/" in workspace_id or "\\" in workspace_id:
        raise HTTPException(status_code=400, detail="Invalid workspace_id")


# --- [核心逻辑 1] 在指定 workspace 的 hybrid_auto 中查找文件 ---
def _safe_filename(filename: str) -> str:
    """Return the basename only; raise 400 if the result is empty or not .md."""
    name = Path(filename).name
    if not name or not name.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Invalid filename: must be a .md file")
    return name


def _find_md_in_hybrid_auto(workspace_id: str, filename: str, output_dir: str) -> Path:
    _validate_workspace_id(workspace_id)
    workspace_dir = Path(output_dir).resolve() / workspace_id
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    safe_name = _safe_filename(filename)

    for hybrid_dir in workspace_dir.rglob("hybrid_auto"):
        if not hybrid_dir.is_dir():
            continue
        target = (hybrid_dir / safe_name).resolve()
        # resolve() + relative_to(): confirm path stays within workspace
        try:
            target.relative_to(workspace_dir)
        except ValueError:
            continue
        if target.exists():
            return target

    all_md = sorted(workspace_dir.rglob("hybrid_auto/*.md"))
    candidates = []
    for p in all_md:
        if safe_name.lower() not in p.name.lower():
            continue
        try:
            p.resolve().relative_to(workspace_dir)
            candidates.append(p)
        except ValueError:
            continue
    if candidates:
        return candidates[0]

    raise HTTPException(
        status_code=404,
        detail=f"File '{filename}' not found in workspace '{workspace_id}'."
    )


# --- 依赖注入 ---
def get_service(request: Request) -> LocalRagService:
    """Backward-compat alias for get_rag(). Prefer Depends(get_rag) in new code."""
    return request.app.state.rag


def get_rag(request: Request) -> LocalRagService:
    return request.app.state.rag


def get_gov(request: Request) -> GovernanceService:
    return request.app.state.gov


def get_optional_gov(request: Request) -> GovernanceService | None:
    return getattr(request.app.state, "gov", None)


def get_jobs(request: Request) -> JobRunner:
    return request.app.state.jobs

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected: return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

def verify_api_key_or_query(
    x_api_key: Optional[str] = Header(default=None),
    key: Optional[str] = Query(default=None),
):
    """Like verify_api_key but also accepts ?key= query param (for <iframe> PDF src URLs)."""
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected: return
    if x_api_key == expected or key == expected: return
    raise HTTPException(status_code=401, detail="Invalid API key")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _doc_status_to_dict(doc_status: Any) -> dict[str, Any]:
    if doc_status is None:
        return {}
    if isinstance(doc_status, dict):
        return dict(doc_status)
    if is_dataclass(doc_status) and not isinstance(doc_status, type):
        return asdict(doc_status)
    if hasattr(doc_status, "__dict__"):
        return {
            key: value
            for key, value in vars(doc_status).items()
            if not key.startswith("_")
        }
    return {}


def _document_row(doc_id: str, doc_status: Any) -> dict[str, Any]:
    data = _doc_status_to_dict(doc_status)
    return _json_safe(
        {
            "doc_id": doc_id,
            "status": data.get("status", ""),
            "file_path": data.get("file_path", ""),
            "chunks_count": data.get("chunks_count", 0),
            "chunks_list": data.get("chunks_list", []),
            "multimodal_processed": data.get("multimodal_processed", False),
            "multimodal_stage": data.get("multimodal_stage", ""),
            "multimodal_failed_items": data.get("multimodal_failed_items", []),
            "multimodal_chunk_ids": data.get("multimodal_chunk_ids", []),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
        }
    )


def _model_dump_json(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json")
    if isinstance(row, dict):
        return _json_safe(row)
    return _json_safe(vars(row))


def _governance_document_row(row: Any) -> dict[str, Any]:
    data = _model_dump_json(row)
    created_at = data.get("created_at") or data.get("ingested_at") or ""
    updated_at = data.get("updated_at") or data.get("finished_at") or created_at
    filename = str(data.get("filename") or "")
    data.update(
        {
            "created_at": created_at,
            "updated_at": updated_at,
            "file_path": data.get("file_path") or filename,
            "chunks_count": data.get("chunks_count", 0),
            "chunks_list": data.get("chunks_list", []),
            "multimodal_processed": data.get("multimodal_processed", False),
            "multimodal_stage": data.get("multimodal_stage", ""),
            "multimodal_failed_items": data.get("multimodal_failed_items", []),
            "multimodal_chunk_ids": data.get("multimodal_chunk_ids", []),
        }
    )
    return data


def _clamp_progress(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, int(round(number))))


def _job_progress_percent(progress: Any, status: str) -> int:
    if isinstance(progress, (int, float)):
        return _clamp_progress(progress)
    if isinstance(progress, dict):
        for key in ("percent", "percentage", "value"):
            if key in progress:
                return _clamp_progress(progress[key])
        total = progress.get("total")
        if total:
            indexed = progress.get("indexed", 0) or 0
            parsed = progress.get("parsed", 0) or 0
            return _clamp_progress((max(indexed, parsed) / float(total)) * 100)
    if status == "done":
        return 100
    return 0


async def _job_api_row(gov: GovernanceService, row: Any) -> dict[str, Any]:
    data = _model_dump_json(row)
    doc_ids = list(getattr(row, "doc_ids", None) or data.get("doc_ids") or [])
    documents: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        doc = await gov.get_document(doc_id)
        if doc is not None:
            documents.append(_governance_document_row(doc))

    filenames = [str(d.get("filename")) for d in documents if d.get("filename")]
    if filenames:
        filename = filenames[0] if len(filenames) == 1 else f"{filenames[0]} +{len(filenames) - 1}"
    elif len(doc_ids) == 1:
        filename = str(doc_ids[0])
    elif doc_ids:
        filename = f"{len(doc_ids)} documents"
    else:
        filename = ""

    status = str(data.get("status") or "")
    progress_detail = data.get("progress") or {}
    created_at = data.get("created_at") or data.get("started_at") or ""
    updated_at = data.get("updated_at") or data.get("finished_at") or created_at
    doc_id_strings = [str(d) for d in doc_ids]
    data.update(
        {
            "doc_ids": doc_id_strings,
            "doc_id": doc_id_strings[0] if doc_id_strings else None,
            "documents": documents,
            "filenames": filenames,
            "filename": filename,
            "progress_detail": progress_detail,
            "progress": _job_progress_percent(progress_detail, status),
            "created_at": created_at,
            "updated_at": updated_at,
        }
    )
    return data


async def _jobs_api_rows(gov: GovernanceService, rows: list[Any]) -> list[dict[str, Any]]:
    return [await _job_api_row(gov, row) for row in rows]


def _audit_api_rows(rows: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        data = _model_dump_json(row)
        timestamp = data.get("timestamp") or data.get("created_at") or ""
        data["created_at"] = timestamp
        data.setdefault("details", data.get("detail") or {})
        data["detail"] = data.get("detail")
        payload.append(data)
    return payload


async def _workspace_documents(
    service: LocalRagService,
    workspace_id: str,
) -> list[dict[str, Any]]:
    rag = await service.get_rag(workspace_id)
    await rag._ensure_lightrag_initialized()
    doc_status = getattr(getattr(rag, "lightrag", None), "doc_status", None)
    if doc_status is None:
        return []

    rows: list[dict[str, Any]] = []
    page = 1
    page_size = 200
    while True:
        docs_page, total = await doc_status.get_docs_paginated(
            page=page,
            page_size=page_size,
            sort_field="file_path",
            sort_direction="asc",
        )
        for doc_id, status in docs_page:
            rows.append(_document_row(str(doc_id), status))
        if len(rows) >= int(total) or not docs_page:
            break
        page += 1
    return rows


def _safe_workspace_child(root: Path, workspace_id: str, *parts: str) -> Path | None:
    workspace_root = (root / workspace_id).resolve()
    candidate = workspace_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        return None
    return candidate


def _delete_document_artifacts(
    service: LocalRagService,
    workspace_id: str,
    file_path: str,
) -> list[str]:
    deleted: list[str] = []
    safe_name = Path(str(file_path or "")).name
    if not safe_name:
        return deleted
    stem = Path(safe_name).stem

    upload_path = _safe_workspace_child(UPLOADS_DIR, workspace_id, safe_name)
    if upload_path and upload_path.exists() and upload_path.is_file():
        upload_path.unlink()
        deleted.append(str(upload_path))

    output_root = Path(service.settings.output_dir).resolve()
    workspace_output = _safe_workspace_child(output_root, workspace_id)
    if not workspace_output or not workspace_output.exists():
        return deleted

    output_doc_dir = _safe_workspace_child(output_root, workspace_id, stem)
    if (
        output_doc_dir
        and output_doc_dir.exists()
        and output_doc_dir.is_dir()
        and output_doc_dir != workspace_output
    ):
        shutil.rmtree(output_doc_dir)
        deleted.append(str(output_doc_dir))

    candidates: set[Path] = set()
    for pattern in (
        f"**/{safe_name}",
        f"**/{stem}.md",
        f"**/{stem}_content_list.json",
        f"**/{stem}_middle.json",
    ):
        for path in workspace_output.glob(pattern):
            if path.is_file():
                try:
                    path.resolve().relative_to(workspace_output)
                except ValueError:
                    continue
                candidates.add(path)

    for path in sorted(candidates):
        path.unlink()
        deleted.append(str(path))
    return deleted

# --- 数据模型 ---
class QueryRequest(BaseModel):
    workspace_id: str
    query: str
    mode: str = DEFAULT_QUERY_MODE
    top_k: int = DEFAULT_TOP_K
    chunk_top_k: int = DEFAULT_CHUNK_TOP_K
    enable_rerank: bool = DEFAULT_ENABLE_RERANK
    vlm_enhanced: bool = DEFAULT_VLM_ENHANCED
    return_graph: bool = False
    graph_max_depth: int = DEFAULT_GRAPH_MAX_DEPTH
    graph_max_nodes: int = DEFAULT_GRAPH_MAX_NODES
    multi_hop_depth: int = DEFAULT_MULTI_HOP_DEPTH
    ppr_damping: float = DEFAULT_PPR_DAMPING
    ppr_top_k: int = DEFAULT_PPR_TOP_K
    passage_node_weight: float = DEFAULT_PASSAGE_NODE_WEIGHT
    recognition_top_k: int = DEFAULT_RECOGNITION_TOP_K
    linking_top_k: int = DEFAULT_LINKING_TOP_K
    ppr_qa_top_k: int = DEFAULT_PPR_QA_TOP_K
    qdrant_retrieval_mode: Literal["dense", "bm25", "hybrid"] = DEFAULT_QDRANT_RETRIEVAL_MODE
    profile: Optional[str] = None  # auto mode only; None = LLM classifier decides
    rerank_candidate_cap: Optional[int] = None  # naive mode only: pre-rerank pool size
    min_rerank_score: Optional[float] = None  # per-query post-rerank filter; None = global default
    conversation_history: list[dict] = []

class EvaluateRequest(BaseModel):
    workspace_id: str
    query: str
    answer: str

# =========================================================================
# 路由
# =========================================================================

# --- 文件列表 (解析产物) ---
@app.get("/files/{workspace_id}")
def list_workspace_files(
    workspace_id: str,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    _validate_workspace_id(workspace_id)
    workspace_dir = Path(service.settings.output_dir).resolve() / workspace_id
    if not workspace_dir.exists():
        return {"files": []}
    file_list = sorted({p.name for p in workspace_dir.rglob("hybrid_auto/*.md")})
    return {"files": file_list}

# --- 文档内容 (Markdown) ---
@app.get("/content/{workspace_id}")
async def get_document_content(
    workspace_id: str,
    filename: Optional[str] = None,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    if not filename or filename == "undefined":
        files_resp = list_workspace_files(workspace_id, service, _auth)
        files = files_resp["files"]
        if not files:
            raise HTTPException(status_code=404, detail="No processed .md files found.")
        filename = files[0]

    try:
        md_path = _find_md_in_hybrid_auto(workspace_id, filename, str(service.settings.output_dir))
        content = md_path.read_text(encoding="utf-8")
        return {"content": content, "filename": md_path.name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")


# =========================================================================
# 上传文件管理 (资料库层)
# =========================================================================

@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(default=None),
    chunking_strategy: Optional[str] = Form(default=None),
    force: bool = Query(default=False),
    _auth: None = Depends(verify_api_key),
    rag_service: LocalRagService = Depends(get_rag),
    gov: GovernanceService = Depends(get_gov),
    jobs: JobRunner = Depends(get_jobs),
):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    if chunking_strategy and chunking_strategy not in VALID_CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunking_strategy '{chunking_strategy}'.",
        )

    file_stem = Path(file.filename).stem
    final_workspace_id = (
        workspace_id.strip() if workspace_id and workspace_id.strip()
        else _compute_workspace_id(file_stem)
    )
    _validate_workspace_id(final_workspace_id)

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")
    file_hash = hashlib.sha256(content).hexdigest()

    # Workspace + frozen check
    await gov.ensure_workspace(final_workspace_id)
    try:
        await gov.ensure_writable(final_workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Idempotency check
    original_filename = Path(file.filename).name
    if not original_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    doc_id, duplicate = await gov.upsert_document(
        final_workspace_id, original_filename, file_hash, len(content), force=force,
    )
    if duplicate:
        return {
            "job_id": None,
            "doc_id": str(doc_id),
            "workspace_id": final_workspace_id,
            "status": "duplicate",
            "duplicate": True,
        }

    # Save the original to uploads/{ws}/{filename}
    upload_dir = UPLOADS_DIR / final_workspace_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / original_filename
    try:
        upload_path.resolve().relative_to(upload_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    upload_path.write_bytes(content)

    # Enqueue the job
    job_id = await gov.create_job(final_workspace_id, [doc_id])

    async def _run():
        await gov.run_ingest(
            job_id, doc_id, final_workspace_id,
            str(upload_path),
            chunking_strategy=chunking_strategy,
        )

    await jobs.submit(job_id, _run)
    return {
        "job_id": str(job_id),
        "doc_id": str(doc_id),
        "workspace_id": final_workspace_id,
        "status": "queued",
        "duplicate": False,
    }


@app.post("/ingest/batch")
async def ingest_batch(
    files: List[UploadFile] = File(...),
    workspace_id: Optional[str] = Form(default=None),
    chunking_strategy: Optional[str] = Form(default=None),
    force: bool = Query(default=False),
    _auth: None = Depends(verify_api_key),
    rag_service: LocalRagService = Depends(get_rag),
    gov: GovernanceService = Depends(get_gov),
    jobs: JobRunner = Depends(get_jobs),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    for f in files:
        if Path(f.filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400,
                                detail=f"Unsupported file type: {f.filename}")
    if chunking_strategy and chunking_strategy not in VALID_CHUNKING_STRATEGIES:
        raise HTTPException(status_code=400, detail="Invalid chunking_strategy")

    first_stem = Path(files[0].filename).stem
    final_workspace_id = (
        workspace_id.strip() if workspace_id and workspace_id.strip()
        else _compute_workspace_id(first_stem)
    )
    _validate_workspace_id(final_workspace_id)

    await gov.ensure_workspace(final_workspace_id)
    try:
        await gov.ensure_writable(final_workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    upload_dir = UPLOADS_DIR / final_workspace_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    new_doc_specs: list[tuple] = []   # (doc_id, upload_path)
    duplicates: list[str] = []

    for f in files:
        content = await f.read()
        file_hash = hashlib.sha256(content).hexdigest()
        name = Path(f.filename).name
        if not name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        upload_path = upload_dir / name
        try:
            upload_path.resolve().relative_to(upload_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")
        doc_id, duplicate = await gov.upsert_document(
            final_workspace_id, name, file_hash, len(content), force=force,
        )
        if duplicate:
            duplicates.append(str(doc_id))
            continue
        upload_path.write_bytes(content)
        new_doc_specs.append((doc_id, upload_path))

    if not new_doc_specs:
        return {
            "job_id": None,
            "workspace_id": final_workspace_id,
            "status": "duplicate",
            "duplicate_doc_ids": duplicates,
        }

    job_id = await gov.create_job(final_workspace_id, [d for d, _ in new_doc_specs])

    async def _run():
        for d, p in new_doc_specs:
            try:
                await gov.run_ingest(job_id, d, final_workspace_id, str(p),
                                     chunking_strategy=chunking_strategy)
            except Exception:
                logger.exception("batch ingest doc %s failed", d)

    await jobs.submit(job_id, _run)
    return {
        "job_id": str(job_id),
        "workspace_id": final_workspace_id,
        "status": "queued",
        "doc_ids": [str(d) for d, _ in new_doc_specs],
        "duplicate_doc_ids": duplicates,
    }


@app.post("/retry/{workspace_id}")
async def retry_ingest(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    rag_service: LocalRagService = Depends(get_rag),
    gov: GovernanceService = Depends(get_gov),
    jobs: JobRunner = Depends(get_jobs),
):
    _validate_workspace_id(workspace_id)
    upload_dir = UPLOADS_DIR / workspace_id
    if not upload_dir.exists():
        raise HTTPException(status_code=404,
                            detail=f"No uploads for workspace '{workspace_id}'")
    files = sorted(p for p in upload_dir.iterdir() if p.is_file())
    if not files:
        raise HTTPException(status_code=404,
                            detail=f"No files in workspace '{workspace_id}'")

    await gov.ensure_workspace(workspace_id)
    try:
        await gov.ensure_writable(workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    doc_specs: list[tuple] = []
    for fp in files:
        content = fp.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        # force=True so repeated retries always re-process; the user explicitly asked to retry
        doc_id, _ = await gov.upsert_document(
            workspace_id, fp.name, file_hash, len(content), force=True,
        )
        doc_specs.append((doc_id, fp))

    job_id = await gov.create_job(workspace_id, [d for d, _ in doc_specs])

    async def _run():
        for d, fp in doc_specs:
            try:
                await gov.run_ingest(job_id, d, workspace_id, str(fp))
            except Exception:
                logger.exception("retry ingest doc %s failed", d)

    await jobs.submit(job_id, _run)
    return {"job_id": str(job_id), "workspace_id": workspace_id, "status": "queued"}


# =========================================================================
# Jobs
# =========================================================================

@app.get("/jobs/{job_id}")
async def get_job_endpoint(
    job_id: str,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    from uuid import UUID
    try:
        jid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    j = await gov.get_job(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return await _job_api_row(gov, j)


@app.get("/jobs")
async def list_jobs_endpoint(
    workspace_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    rows = await gov.list_jobs(
        workspace_id=workspace_id, status=status, limit=max(1, min(limit, 200)),
    )
    return {"jobs": await _jobs_api_rows(gov, rows)}


@app.delete("/jobs/{job_id}")
async def cancel_job_endpoint(
    job_id: str,
    _auth: None = Depends(verify_api_key),
    jobs: JobRunner = Depends(get_jobs),
):
    from uuid import UUID
    try:
        jid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    cancelled = await jobs.cancel(jid)
    return {"job_id": job_id, "cancelled": cancelled}


@app.get("/uploads/{workspace_id}")
def list_uploads(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
):
    _validate_workspace_id(workspace_id)
    upload_dir = UPLOADS_DIR / workspace_id
    if not upload_dir.exists():
        return {"files": []}
    files = []
    for p in sorted(upload_dir.iterdir()):
        if p.is_file():
            stat = p.stat()
            files.append({
                "name": p.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    return {"files": files}


@app.get("/uploads/{workspace_id}/{filename}")
def serve_upload(
    workspace_id: str,
    filename: str,
    _auth: None = Depends(verify_api_key_or_query),
):
    _validate_workspace_id(workspace_id)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = UPLOADS_DIR / workspace_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))


# =========================================================================
# 解析产物图片服务
# =========================================================================

@app.get("/output/{workspace_id}/images/{path:path}")
def serve_output_image(
    workspace_id: str,
    path: str,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    _validate_workspace_id(workspace_id)
    if ".." in path:
        raise HTTPException(status_code=400, detail="Invalid path")
    output_root = Path(service.settings.output_dir).resolve()
    file_path = output_root / workspace_id / path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    # 确保路径在 output_root 下
    try:
        file_path.resolve().relative_to(output_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return FileResponse(str(file_path))


# =========================================================================
# 增强查询 (返回完整结构化数据)
# =========================================================================

@app.post("/query")
async def query_endpoint(
    payload: QueryRequest,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(payload.workspace_id)
    top_k = max(1, min(payload.top_k, MAX_TOP_K))
    chunk_top_k = max(1, min(payload.chunk_top_k, MAX_CHUNK_TOP_K))
    min_rerank_score = (
        None if payload.min_rerank_score is None
        else max(0.0, min(1.0, payload.min_rerank_score))
    )

    result = await service.run_query(
        payload.workspace_id,
        payload.query,
        mode=payload.mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=payload.enable_rerank,
        min_rerank_score=min_rerank_score,
        vlm_enhanced=payload.vlm_enhanced,
        multi_hop_depth=payload.multi_hop_depth,
        ppr_damping=payload.ppr_damping,
        ppr_top_k=payload.ppr_top_k,
        passage_node_weight=payload.passage_node_weight,
        recognition_top_k=payload.recognition_top_k,
        linking_top_k=payload.linking_top_k,
        ppr_qa_top_k=payload.ppr_qa_top_k,
        qdrant_retrieval_mode=payload.qdrant_retrieval_mode,
        profile=payload.profile,
        rerank_candidate_cap=payload.rerank_candidate_cap,
        conversation_history=payload.conversation_history,
    )

    answer = result.get("answer", "")
    data = result.get("data", {}) or {}
    metadata = result.get("metadata", {}) or {}

    graph_data = None
    if payload.return_graph:
        rag = await service.get_rag(payload.workspace_id)
        graph_data = await _get_query_subgraph(
            rag, {"data": data, "metadata": metadata}, payload
        )

    source_nodes = _extract_source_nodes({"data": data, "metadata": metadata})

    return {
        "answer": answer,
        "data": data,
        "metadata": metadata,
        "source_nodes": source_nodes,
        "graph": graph_data,
    }


@app.post("/query/multimodal")
async def query_multimodal_endpoint(
    workspace_id: str = Form(...),
    query: str = Form(...),
    mode: str = Form(DEFAULT_QUERY_MODE),
    top_k: int = Form(DEFAULT_TOP_K),
    chunk_top_k: int = Form(DEFAULT_CHUNK_TOP_K),
    enable_rerank: bool = Form(DEFAULT_ENABLE_RERANK),
    images: List[UploadFile] = File(default=[]),
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    """Multimodal query (image + text), non-streaming.

    Saves uploaded images under uploads/{ws}/_inline/<uuid>.<ext> and
    forwards them as 'image' multimodal_content items to
    LocalRagService.query_with_multimodal, which delegates to
    RAGAnything.aquery_with_multimodal -> VLM describe -> normal aquery().
    """
    _validate_workspace_id(workspace_id)
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not images:
        raise HTTPException(
            status_code=400,
            detail="No images attached. Use POST /query for text-only queries.",
        )
    if len(images) > MAX_IMAGES_PER_QUERY:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images (max {MAX_IMAGES_PER_QUERY})",
        )
    top_k = max(1, min(top_k, MAX_TOP_K))
    chunk_top_k = max(1, min(chunk_top_k, MAX_CHUNK_TOP_K))

    inline_dir = UPLOADS_DIR / workspace_id / "_inline"
    inline_dir.mkdir(parents=True, exist_ok=True)

    multimodal_content: list[dict] = []
    for img in images:
        ext = Path(img.filename or "").suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type '{ext}'. Allowed: {sorted(SUPPORTED_IMAGE_EXTS)}",
            )
        content = await img.read()
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image '{img.filename}' exceeds {DEFAULT_MAX_IMAGE_SIZE_MB} MB",
            )
        saved_path = inline_dir / f"{uuid4().hex}{ext}"
        saved_path.write_bytes(content)
        multimodal_content.append({
            "type": "image",
            "img_path": str(saved_path.resolve()),
        })

    answer = await service.query_with_multimodal(
        workspace_id,
        query,
        multimodal_content,
        mode=mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=enable_rerank,
    )
    return {
        "answer": answer,
        "workspace_id": workspace_id,
        "image_count": len(multimodal_content),
    }


@app.post("/evaluate")
async def evaluate_endpoint(
    payload: EvaluateRequest,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    """Run LLM-based answer quality evaluation. Returns {score: float, gap: str}."""
    _validate_workspace_id(payload.workspace_id)
    return await service.evaluate_answer(payload.workspace_id, payload.query, payload.answer)


def _kg_node_name(node) -> str:
    """LightRAG KnowledgeGraphNode 的可读实体名。

    Neo4j 后端把 ``node.id`` 设为实体名，但 Postgres/AGE 后端把 ``node.id`` 设为
    内部数字 id，真正的名字存在 ``properties['entity_id']`` / ``labels`` 里。
    展示时始终优先用实体名，避免界面显示一串数字。
    """
    name = node.properties.get("entity_id")
    if not name and node.labels:
        name = node.labels[0]
    return name or node.id


async def _get_query_subgraph(rag, retrieval, payload):
    """从查询关键词中提取子图用于可视化"""
    try:
        keywords = retrieval.get("metadata", {}).get("keywords", {})
        ll_kws = keywords.get("low_level", [])
        if not ll_kws:
            return None
        raw_label = ll_kws[0] if isinstance(ll_kws[0], str) else str(ll_kws[0])
        # Use search_labels to find the closest matching entity in the graph
        label = raw_label
        try:
            matches = await rag.lightrag.chunk_entity_relation_graph.search_labels(
                raw_label, limit=1
            )
            if matches:
                label = matches[0]
        except Exception:
            pass  # fallback to raw keyword
        kg = await rag.lightrag.get_knowledge_graph(
            node_label=label,
            max_depth=payload.graph_max_depth,
            max_nodes=payload.graph_max_nodes,
        )
        nodes = []
        for n in kg.nodes:
            nodes.append({
                "id": n.id,
                "label": _kg_node_name(n),
                "type": n.properties.get("entity_type", ""),
                "description": n.properties.get("description", ""),
            })
        edges = []
        for e in kg.edges:
            edges.append({
                "source": e.source,
                "target": e.target,
                "label": e.properties.get("description", ""),
                "weight": e.properties.get("weight", 1.0),
            })
        return {"nodes": nodes, "edges": edges}
    except Exception:
        return None


def _extract_source_nodes(retrieval_data: dict) -> list[dict]:
    """Extract source file references from LightRAG retrieval meta event.

    page_num is None if LightRAG doesn't surface it — frontend degrades gracefully.
    """
    source_nodes: list[dict] = []
    seen: set[str] = set()
    data = retrieval_data.get("data", {})

    for key in ("chunks", "references", "sources", "text_chunks"):
        chunks = data.get(key, [])
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            filename = chunk.get("file_path") or chunk.get("filename") or chunk.get("source")
            if not filename:
                continue
            filename = str(filename).replace("\\", "/").split("/")[-1]
            doc_id = str(chunk.get("doc_id") or chunk.get("document_id") or "")
            excerpt = str(chunk.get("content") or chunk.get("text") or "")[:200]
            page_num = chunk.get("page_num") or chunk.get("page")
            if isinstance(page_num, float):
                page_num = int(page_num)
            elif page_num is not None and not isinstance(page_num, int):
                page_num = None

            key_id = f"{filename}:{page_num}"
            if key_id not in seen:
                seen.add(key_id)
                source_nodes.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "excerpt": excerpt,
                })
    return source_nodes[:5]


# =========================================================================
# 知识图谱 API
# =========================================================================

def _graphml_path(service: LocalRagService, workspace_id: str) -> Path:
    return (
        Path(service.settings.working_dir_root).resolve()
        / workspace_id
        / "graph_chunk_entity_relation.graphml"
    )


def _read_graphml_safe(path: Path):
    """Read a GraphML file with NetworkX; returns None on failure."""
    if not path.exists():
        return None
    try:
        import networkx as nx
        return nx.read_graphml(str(path))
    except Exception:
        return None


@app.get("/graph/{workspace_id}/labels")
async def get_graph_labels(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(workspace_id)
    # 优先尝试 LightRAG API
    try:
        rag = await service.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        labels = await rag.lightrag.get_graph_labels()
        if isinstance(labels, list) and labels:
            return {"labels": labels}
    except Exception as e:
        logger.warning("get_graph_labels LightRAG fallback: %s", e)

    # 兜底：NetworkX 直接读 GraphML
    G = _read_graphml_safe(_graphml_path(service, workspace_id))
    if G is not None and G.number_of_nodes() > 0:
        return {"labels": sorted(G.nodes())[:500]}
    return {"labels": []}


@app.get("/graph/{workspace_id}/entities")
async def get_graph_entities(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(workspace_id)
    entities: list[dict[str, Any]] = []

    try:
        rag = await service.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        graph = getattr(rag.lightrag, "chunk_entity_relation_graph", None)
        if graph is not None:
            labels = await graph.get_all_labels()
            node_rows = await graph.get_nodes_batch(labels)
            for label in labels:
                data = node_rows.get(label) if isinstance(node_rows, dict) else None
                if not isinstance(data, dict):
                    data = {}
                entities.append(
                    _json_safe(
                        {
                            "label": label,
                            "entity_id": data.get("entity_id", label),
                            "entity_name": data.get("entity_name", label),
                            "entity_type": data.get("entity_type", ""),
                            "description": data.get("description", ""),
                            "source_id": data.get("source_id", ""),
                            "file_path": data.get("file_path", ""),
                        }
                    )
                )
            return {
                "workspace_id": workspace_id,
                "count": len(entities),
                "entities": entities,
            }
    except Exception as e:
        logger.warning("get_graph_entities LightRAG fallback: %s", e)

    G = _read_graphml_safe(_graphml_path(service, workspace_id))
    if G is not None:
        for label, data in G.nodes(data=True):
            data = data if isinstance(data, dict) else {}
            entities.append(
                _json_safe(
                    {
                        "label": label,
                        "entity_id": data.get("entity_id", label),
                        "entity_name": data.get("entity_name", label),
                        "entity_type": data.get("entity_type", ""),
                        "description": data.get("description", ""),
                        "source_id": data.get("source_id", ""),
                        "file_path": data.get("file_path", ""),
                    }
                )
            )
    return {"workspace_id": workspace_id, "count": len(entities), "entities": entities}


@app.get("/graph/{workspace_id}/subgraph")
async def get_subgraph(
    workspace_id: str,
    label: str,
    max_depth: int = 2,
    max_nodes: int = 50,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(workspace_id)
    rag = await service.get_rag(workspace_id)
    await rag._ensure_lightrag_initialized()
    try:
        kg = await rag.lightrag.get_knowledge_graph(
            node_label=label, max_depth=max_depth, max_nodes=max_nodes
        )
        nodes = [
            {
                "id": n.id,
                "label": _kg_node_name(n),
                "type": n.properties.get("entity_type", ""),
                "description": n.properties.get("description", ""),
            }
            for n in kg.nodes
        ]
        edges = [
            {
                "source": e.source,
                "target": e.target,
                "label": e.properties.get("description", ""),
                "weight": e.properties.get("weight", 1.0),
            }
            for e in kg.edges
        ]
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get subgraph: {e}")


@app.get("/graph/{workspace_id}/stats")
async def get_graph_stats(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(workspace_id)
    gpath = _graphml_path(service, workspace_id)
    try:
        rag = await service.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        graph = getattr(rag.lightrag, "chunk_entity_relation_graph", None)
        if graph is not None:
            labels = await graph.get_all_labels()
            edges = await graph.get_all_edges()
            return {
                "entity_count": len(labels) if isinstance(labels, list) else 0,
                "relation_count": len(edges) if isinstance(edges, list) else 0,
                "graphml_size": gpath.stat().st_size if gpath.exists() else 0,
                "source": "graph_storage",
            }
    except Exception as e:
        logger.warning("get_graph_stats LightRAG fallback: %s", e)

    G = _read_graphml_safe(gpath)
    if G is not None:
        return {
            "entity_count": G.number_of_nodes(),
            "relation_count": G.number_of_edges(),
            "graphml_size": gpath.stat().st_size,
            "source": "graphml",
        }
    return {
        "entity_count": 0,
        "relation_count": 0,
        "graphml_size": gpath.stat().st_size if gpath.exists() else 0,
        "source": "none",
    }


@app.get("/graph/{workspace_id}/search")
async def search_graph_entities(
    workspace_id: str,
    q: str,
    limit: int = DEFAULT_GRAPH_SEARCH_MAX_RESULTS,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(workspace_id)
    if not q.strip():
        return {"results": []}
    safe_limit = max(1, min(limit, DEFAULT_GRAPH_SEARCH_MAX_SAFE))
    q_stripped = q.strip()

    # 优先尝试 LightRAG API
    try:
        rag = await service.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        results = await rag.lightrag.chunk_entity_relation_graph.search_labels(
            q_stripped, limit=safe_limit
        )
        if isinstance(results, list) and results:
            return {"results": results}
    except Exception as e:
        logger.warning("search_graph_entities LightRAG fallback: %s", e)

    # 兜底：NetworkX 子串匹配
    G = _read_graphml_safe(_graphml_path(service, workspace_id))
    if G is not None:
        q_lower = q_stripped.lower()
        matched = [n for n in G.nodes() if q_lower in n.lower()][:safe_limit]
        return {"results": matched}
    return {"results": []}


@app.get("/graph/{workspace_id}/overview")
async def get_graph_overview(
    workspace_id: str,
    max_nodes: int = DEFAULT_GRAPH_OVERVIEW_MAX_NODES,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    """返回知识图谱概览子图（最高度数节点及其邻域）。

    优先用 LightRAG `get_knowledge_graph(node_label="*")`（Neo4j/Memgraph/Mongo
    后端均原生支持），失败时回退到 GraphML 文件。
    """
    _validate_workspace_id(workspace_id)

    # Try LightRAG API first (works with Neo4j/Memgraph/Mongo backends)
    try:
        rag = await service.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        kg = await rag.lightrag.get_knowledge_graph(
            node_label="*", max_depth=1, max_nodes=max_nodes
        )
        if kg.nodes:
            return {
                "nodes": [
                    {
                        "id": n.id,
                        "label": _kg_node_name(n),
                        "type": n.properties.get("entity_type", ""),
                        "description": n.properties.get("description", ""),
                    }
                    for n in kg.nodes
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "label": e.properties.get("description", ""),
                        "weight": float(e.properties.get("weight", 1.0)),
                    }
                    for e in kg.edges
                ],
            }
    except Exception as e:
        logger.warning("get_graph_overview LightRAG fallback: %s", e)

    # Fallback: NetworkX over GraphML file (only present with NetworkX backend)
    G = _read_graphml_safe(_graphml_path(service, workspace_id))
    if G is None or G.number_of_nodes() == 0:
        return {"nodes": [], "edges": []}

    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
    sub = G.subgraph(top_nodes)
    return {
        "nodes": [
            {
                "id": n,
                "label": n,
                "type": G.nodes[n].get("entity_type", ""),
                "description": G.nodes[n].get("description", ""),
            }
            for n in sub.nodes()
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "label": d.get("description", ""),
                "weight": float(d.get("weight", 1.0)),
            }
            for u, v, d in sub.edges(data=True)
        ],
    }


_ENTITY_COLORS = {
    "PERSON": "#ff6b6b",
    "ORGANIZATION": "#4ecdc4",
    "LOCATION": "#45b7d1",
    "CONCEPT": "#f7b731",
    "TECHNOLOGY": "#a55eea",
    "EVENT": "#26de81",
    "DOCUMENT": "#fd9644",
    "PRODUCT": "#fd79a8",
    "METHOD": "#00cec9",
    "ALGORITHM": "#6c5ce7",
    "DATASET": "#e17055",
    "METRIC": "#fab1a0",
    "TASK": "#55efc4",
    "FIELD": "#74b9ff",
    "INSTITUTION": "#0984e3",
    "PAPER": "#d63031",
    "CATEGORY": "#e84393",
}
_DEFAULT_NODE_COLOR = "#95a5a6"

# Palette for hash-based community cluster coloring (distinct, readable)
_CLUSTER_PALETTE = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    "#8bc34a", "#ff5722", "#607d8b", "#ff9800", "#673ab7",
    "#009688", "#795548", "#f44336", "#4caf50", "#2196f3",
]


def _entity_color_for(etype: str) -> str:
    """Return colour for an entity type. Unknown types get a hash-based colour."""
    c = _ENTITY_COLORS.get(etype.upper())
    if c:
        return c
    if not etype.strip():
        return _DEFAULT_NODE_COLOR
    h = int(hashlib.md5(etype.upper().encode()).hexdigest()[:6], 16)
    r = 100 + (h >> 16 & 0xFF) % 120
    g = 100 + (h >> 8  & 0xFF) % 120
    b = 100 + (h       & 0xFF) % 120
    return f"#{r:02x}{g:02x}{b:02x}"


@app.get("/graph/{workspace_id}/html")
def get_graph_html(
    workspace_id: str,
    q: Optional[str] = None,
    theme: str = "dark",
    max_nodes: int = DEFAULT_GRAPH_HTML_MAX_NODES,
    _auth: None = Depends(verify_api_key_or_query),
    service: LocalRagService = Depends(get_service),
):
    """
    Generate a self-contained pyvis HTML visualisation of the knowledge graph.
    Uses cdn_resources='in_line' so no external CDN is needed at render time.
    Accepts optional ?q= to filter to matching nodes + their 1-hop neighbours.
    Accepts ?key= for iframe-based authentication (via verify_api_key_or_query).
    Accepts ?theme=dark|light to match the parent page's colour scheme.
    """
    import os
    import tempfile

    _validate_workspace_id(workspace_id)
    G = _read_graphml_safe(_graphml_path(service, workspace_id))

    # Theme-aware colours
    if theme == "light":
        _bgcolor    = "#f8f6f1"
        _font_color = "#3a352e"
        _edge_color = "rgba(80,70,60,0.22)"
        _empty_bg   = "#f8f6f1"
        _empty_fg   = "#a09890"
    else:
        _bgcolor    = "#131315"
        _font_color = "#c8c4be"
        _edge_color = "rgba(200,196,190,0.25)"
        _empty_bg   = "#131315"
        _empty_fg   = "#6b6560"

    _NO_GRAPH_HTML = (
        f"<html><body style='margin:0;display:flex;align-items:center;"
        f"justify-content:center;height:100vh;"
        f"background:{_empty_bg};color:{_empty_fg};font-family:sans-serif;font-size:13px'>"
        "<div style='text-align:center'>"
        "<div style='font-size:32px;margin-bottom:12px'>🕸️</div>"
        "<div>No graph data found for this workspace.</div>"
        "<div style='font-size:11px;margin-top:6px;opacity:.6'>"
        "Upload and ingest a document first.</div></div></body></html>"
    )

    if G is None or G.number_of_nodes() == 0:
        from fastapi.responses import HTMLResponse as _HR
        return _HR(content=_NO_GRAPH_HTML)

    # Select nodes
    if q and q.strip():
        q_lower = q.strip().lower()
        seeds = [n for n in G.nodes() if q_lower in n.lower()][:DEFAULT_GRAPH_SEARCH_SEED_LIMIT]
        neighborhood = set(seeds)
        for seed in seeds:
            neighborhood.update(G.neighbors(seed))
        display_nodes = list(neighborhood)[:max_nodes]
    else:
        display_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]

    sub = G.subgraph(display_nodes)

    try:
        from pyvis.network import Network
    except ImportError:
        from fastapi.responses import HTMLResponse as _HR
        return _HR(content="<html><body style='color:red;padding:20px'>pyvis not installed. Run: pip install pyvis</body></html>")

    net = Network(
        height="100%",
        width="100%",
        bgcolor=_bgcolor,
        font_color=_font_color,
        directed=sub.is_directed(),
    )
    try:
        net_kwargs = {"cdn_resources": "in_line"}
        net2 = Network(height="100%", width="100%", bgcolor=_bgcolor, font_color=_font_color,
                       directed=sub.is_directed(), **net_kwargs)
        net = net2
    except TypeError:
        pass  # older pyvis without cdn_resources param

    net.from_nx(sub)

    # Community detection for cluster-based colouring (fallback: entity type)
    node_community: dict = {}
    try:
        from networkx.algorithms import community as _nx_comm
        comms = list(_nx_comm.greedy_modularity_communities(sub.to_undirected()))
        for ci, comm in enumerate(comms):
            for n in comm:
                node_community[n] = ci
    except Exception:
        pass

    for node in net.nodes:
        nid = node["id"]
        attrs = sub.nodes.get(nid, {})
        etype = attrs.get("entity_type", "")
        safe_nid = html.escape(str(nid))
        safe_etype = html.escape(str(etype))
        safe_desc = html.escape(str(attrs.get("description", "")))
        # Colour priority: named entity type → cluster → default
        if etype.upper() in _ENTITY_COLORS:
            color = _ENTITY_COLORS[etype.upper()]
        elif nid in node_community:
            color = _CLUSTER_PALETTE[node_community[nid] % len(_CLUSTER_PALETTE)]
        else:
            color = _entity_color_for(etype)
        node["color"] = color
        node["title"] = f"<b>{safe_nid}</b><br>{safe_etype}<br><br>{safe_desc}"
        node["size"] = max(12, min(36, 12 + G.degree(nid) * 2))
        node["label"] = str(nid)
        node["font"] = {"size": 13}

    for edge in net.edges:
        desc = edge.get("description") or edge.get("label") or ""
        edge["title"] = html.escape(str(desc))
        edge["color"] = _edge_color

    # Physics settings for better layout
    net.set_options("""{
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {"gravitationalConstant": -60, "centralGravity": 0.01,
                             "springLength": 120, "springConstant": 0.08},
        "stabilization": {"iterations": 150}
      },
      "edges": {"smooth": {"type": "continuous"}, "width": 1.5},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }""")

    # Generate HTML
    try:
        html_content = net.generate_html()
    except AttributeError:
        # Fallback for older pyvis
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            tmp_path = f.name
        try:
            net.show(tmp_path, notebook=False)
            with open(tmp_path, encoding="utf-8") as f:
                html_content = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    from fastapi.responses import HTMLResponse as _HR
    return _HR(content=html_content)


# =========================================================================
# 工作空间管理
# =========================================================================

@app.get("/workspace/{workspace_id}/documents")
async def list_workspace_documents(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
    gov: GovernanceService | None = Depends(get_optional_gov),
):
    _validate_workspace_id(workspace_id)
    if gov is not None:
        documents = [_governance_document_row(d) for d in await gov.list_documents(workspace_id)]
    else:
        documents = await _workspace_documents(service, workspace_id)
    if not documents:
        try:
            documents = await _workspace_documents(service, workspace_id)
        except Exception:
            if gov is None:
                raise
            documents = []
    return {
        "workspace_id": workspace_id,
        "count": len(documents),
        "documents": documents,
    }


@app.delete("/workspace/{workspace_id}/documents/{doc_id}")
async def delete_workspace_document(
    workspace_id: str,
    doc_id: str,
    delete_artifacts: bool = True,
    delete_llm_cache: bool = False,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(workspace_id)
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    rag = await service.get_rag(workspace_id)
    await rag._ensure_lightrag_initialized()
    doc_status = getattr(getattr(rag, "lightrag", None), "doc_status", None)
    before_status = (
        await doc_status.get_by_id(doc_id) if doc_status is not None else None
    )
    before_data = _doc_status_to_dict(before_status)
    file_path = str(before_data.get("file_path", "") or "")

    delete_result = await service.lightrag_adelete_by_doc_id(
        workspace_id,
        doc_id,
        delete_llm_cache=delete_llm_cache,
    )
    delete_payload = _json_safe(delete_result)
    if not isinstance(delete_payload, dict) and hasattr(delete_result, "__dict__"):
        delete_payload = _json_safe(vars(delete_result))
    if not isinstance(delete_payload, dict):
        delete_payload = {"result": delete_payload}

    status = str(delete_payload.get("status", "")).lower()
    if status not in {"success", "not_found"}:
        raise HTTPException(
            status_code=int(delete_payload.get("status_code", 500) or 500),
            detail=delete_payload,
        )

    if not file_path:
        file_path = str(delete_payload.get("file_path", "") or "")

    artifacts_deleted: list[str] = []
    if status == "success" and delete_artifacts:
        artifacts_deleted = _delete_document_artifacts(service, workspace_id, file_path)

    synonym_rebuild: dict[str, Any] | None = None
    if status == "success" and service.settings.enable_synonym_linking:
        synonym_rebuild = await service.finalize_workspace_synonyms(
            workspace_id,
            force=True,
            reset_existing=True,
        )

    return {
        "workspace_id": workspace_id,
        "doc_id": doc_id,
        "delete_result": delete_payload,
        "artifacts_deleted": artifacts_deleted,
        "synonym_rebuild": _json_safe(synonym_rebuild),
    }


@app.delete("/workspace/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    request: Request,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
    gov: GovernanceService | None = Depends(get_optional_gov),
):
    import asyncio as _asyncio
    _validate_workspace_id(workspace_id)
    # Agentic RAG v3：删除整个 workspace 时清空所有相关会话（含其 chunk 缓存）。
    session_store = getattr(request.app.state, "session_store", None)
    if session_store is not None:
        session_store.invalidate_workspace(workspace_id)

    if gov is not None:
        try:
            await gov.ensure_writable(workspace_id)
        except WorkspaceFrozenError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # Step 1: Drop all storages (Neo4j, Qdrant, KV) before touching the filesystem
    drop_errors = []
    try:
        rag = await service.get_rag(workspace_id)
        await rag._ensure_lightrag_initialized()
        lg = rag.lightrag
        if lg is not None:
            storages = [s for s in [
                lg.text_chunks, lg.full_docs, lg.full_entities, lg.full_relations,
                lg.entity_chunks, lg.relation_chunks,
                lg.entities_vdb, lg.relationships_vdb, lg.chunks_vdb,
                lg.chunk_entity_relation_graph, lg.doc_status,
            ] if s is not None]
            results = await _asyncio.gather(*[s.drop() for s in storages], return_exceptions=True)
            for storage, result in zip(storages, results):
                if isinstance(result, Exception):
                    drop_errors.append(f"{storage.__class__.__name__}: {result}")
                    logger.warning("[%s] Storage drop error: %s: %s", workspace_id, storage.__class__.__name__, result)
    except Exception as e:
        drop_errors.append(f"storage_init: {e}")
        logger.warning("[%s] Could not initialize storages for drop: %s", workspace_id, e)

    # Step 2: Remove cached rag instance and warmup state
    if hasattr(service, "_rag_instances"):
        service._rag_instances.pop(workspace_id, None)
    if hasattr(service, "_warmed_workspaces"):
        service._warmed_workspaces.discard(workspace_id)

    # Step 3: Delete filesystem directories
    deleted = []
    dirs_to_delete = [
        ("uploads", UPLOADS_DIR / workspace_id),
        ("output", Path(service.settings.output_dir).resolve() / workspace_id),
        ("workspace", Path(service.settings.working_dir_root).resolve() / workspace_id),
    ]
    for name, d in dirs_to_delete:
        if d.exists() and d.is_dir():
            shutil.rmtree(str(d), ignore_errors=True)
            deleted.append(name)

    if gov is not None:
        await gov.record_audit(
            workspace_id,
            "delete_workspace",
            details={"deleted": deleted, "drop_errors": drop_errors},
        )
        await gov.delete_workspace(workspace_id)
    return {"status": "ok", "deleted": deleted, "drop_errors": drop_errors}


@app.get("/workspace/{workspace_id}/stats")
async def get_workspace_stats(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_workspace_id(workspace_id)
    upload_dir = UPLOADS_DIR / workspace_id
    output_dir = Path(service.settings.output_dir).resolve() / workspace_id
    workspace_dir = Path(service.settings.working_dir_root).resolve() / workspace_id

    def _dir_size(d: Path) -> int:
        if not d.exists():
            return 0
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())

    def _file_count(d: Path, pattern: str = "*") -> int:
        if not d.exists():
            return 0
        return sum(1 for f in d.rglob(pattern) if f.is_file())

    # 图谱统计
    graphml_path = workspace_dir / "graph_chunk_entity_relation.graphml"
    entity_count = 0
    relation_count = 0
    if graphml_path.exists():
        try:
            import networkx as nx
            G = nx.read_graphml(str(graphml_path))
            entity_count = G.number_of_nodes()
            relation_count = G.number_of_edges()
        except Exception:
            pass

    return {
        "files": _file_count(upload_dir),
        "entities": entity_count,
        "relations": relation_count,
        "chunks": _file_count(output_dir, "*.md"),
        "graphml_size": graphml_path.stat().st_size if graphml_path.exists() else 0,
        "upload_size_total": _dir_size(upload_dir),
    }


@app.post("/workspace/{workspace_id}/freeze")
@app.patch("/workspace/{workspace_id}/freeze")
async def freeze_workspace(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    await gov.ensure_workspace(workspace_id)
    ws_before = await gov.get_workspace(workspace_id)
    prev = bool(ws_before.frozen) if ws_before else False
    ok = await gov.set_frozen(workspace_id, True)
    if not ok:
        raise HTTPException(status_code=404, detail="Workspace not found")
    await gov.record_audit(workspace_id, "freeze", details={"previous_state": prev})
    return {"workspace_id": workspace_id, "frozen": True}


@app.post("/workspace/{workspace_id}/unfreeze")
@app.patch("/workspace/{workspace_id}/unfreeze")
async def unfreeze_workspace(
    workspace_id: str,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    ws_before = await gov.get_workspace(workspace_id)
    if ws_before is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ok = await gov.set_frozen(workspace_id, False)
    await gov.record_audit(workspace_id, "unfreeze",
                           details={"previous_state": ws_before.frozen})
    return {"workspace_id": workspace_id, "frozen": False}


@app.delete("/workspace/{workspace_id}/document/{doc_id}")
async def delete_document_endpoint(
    workspace_id: str,
    doc_id: str,
    request: Request,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    from uuid import UUID
    try:
        did = UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid doc_id")
    try:
        await gov.ensure_writable(workspace_id)
    except WorkspaceFrozenError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    doc = await gov.get_document(did)
    if doc is None or doc.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Document not found in workspace")
    # Capture chunk provenance before the cascade delete so we can purge the
    # agent session chunk caches (spec §6: cache invalidation on doc delete).
    chunk_ids: list[str] = []
    try:
        prov = await gov.get_provenance_for_doc(did)
        chunk_ids = list(prov.get("chunk", []))
    except Exception:
        logger.warning("delete_document: provenance lookup failed for %s", doc_id, exc_info=True)
    report = await gov.delete_document(workspace_id, did)
    session_store = getattr(request.app.state, "session_store", None)
    if session_store is not None and chunk_ids:
        session_store.drop_chunks(workspace_id, chunk_ids)
    return {"doc_id": doc_id, "workspace_id": workspace_id, "cleanup": report}


# =========================================================================
# 查询默认配置
# =========================================================================

@app.get("/config")
async def get_config(_auth: None = Depends(verify_api_key)):
    """Return server-side query defaults so the WebUI stays in sync with constants.py."""
    return {
        "top_k": DEFAULT_TOP_K,
        "chunk_top_k": DEFAULT_CHUNK_TOP_K,
        "query_mode": DEFAULT_QUERY_MODE,
        "enable_rerank": DEFAULT_ENABLE_RERANK,
        "vlm_enhanced": DEFAULT_VLM_ENHANCED,
        "qdrant_retrieval_mode": DEFAULT_QDRANT_RETRIEVAL_MODE,
    }


# 工作空间列表 (增强)
# =========================================================================

@app.get("/workspaces")
async def list_workspaces(
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
    gov: GovernanceService | None = Depends(get_optional_gov),
):
    working_root = Path(service.settings.working_dir_root).resolve()
    output_root = Path(service.settings.output_dir).resolve()
    workspaces = []

    # 只从有效数据目录收集 workspace_id（不含 uploads），避免已删除的工作空间仍出现
    workspace_ids = set()
    for root_dir in [working_root, output_root]:
        if root_dir.exists():
            for d in root_dir.iterdir():
                if d.is_dir():
                    workspace_ids.add(d.name)

    pg_by_id: dict[str, Any] = {}
    docs_by_workspace: dict[str, list[Any]] = {}
    if gov is not None:
        if workspace_ids:
            await gov.backfill_legacy_workspaces(sorted(workspace_ids))
        pg_rows = await gov.list_workspaces()
        pg_by_id = {row.workspace_id: row for row in pg_rows}
        for row in pg_rows:
            docs = [
                doc
                for doc in await gov.list_documents(row.workspace_id)
                if getattr(doc, "status", None) != "deleted"
            ]
            docs_by_workspace[row.workspace_id] = docs
            if docs:
                workspace_ids.add(row.workspace_id)

    for workspace_id in sorted(workspace_ids):
        workspace_dir = working_root / workspace_id
        upload_dir = UPLOADS_DIR / workspace_id
        ws_row = pg_by_id.get(workspace_id)
        ws_data = _model_dump_json(ws_row) if ws_row is not None else {}
        metadata = ws_data.get("metadata") if isinstance(ws_data.get("metadata"), dict) else {}
        documents = docs_by_workspace.get(workspace_id)

        # 基本信息
        has_files = (workspace_dir / "graph_chunk_entity_relation.graphml").exists()

        # 上传文件列表
        uploaded_files = []
        if upload_dir.exists():
            for p in sorted(upload_dir.iterdir()):
                if p.is_file():
                    uploaded_files.append(p.name)

        workspaces.append({
            "workspace_id": workspace_id,
            "name": metadata.get("name") or workspace_id,
            "frozen": bool(ws_data.get("frozen", False)),
            "document_count": len(documents) if documents is not None else len(uploaded_files),
            "created_at": ws_data.get("created_at"),
            "has_files": has_files,
            "uploaded_files": uploaded_files,
            "metadata": metadata,
        })

    return {
        "workspaces": workspaces,
        "roots": {
            "uploads": str(UPLOADS_DIR),
            "output": str(output_root),
            "workspace": str(working_root),
        },
    }


# =========================================================================
# 审计日志
# =========================================================================

@app.get("/workspace/{workspace_id}/audit")
async def workspace_audit(
    workspace_id: str,
    action: Optional[str] = None,
    limit: int = 100,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    _validate_workspace_id(workspace_id)
    rows = await gov.list_audit(workspace_id=workspace_id, action=action,
                                limit=max(1, min(limit, 500)))
    entries = _audit_api_rows(rows)
    return {"audit": entries, "entries": entries}


@app.get("/admin/audit")
async def admin_audit(
    action: Optional[str] = None,
    limit: int = 100,
    _auth: None = Depends(verify_api_key),
    gov: GovernanceService = Depends(get_gov),
):
    rows = await gov.list_audit(action=action, limit=max(1, min(limit, 500)))
    entries = _audit_api_rows(rows)
    return {"audit": entries, "entries": entries}


# --- SPA fallback (must be last — after all API routes) ---
_DIST_DIR = APP_ROOT / "static" / "dist"
if _DIST_DIR.exists():
    # Serve hashed assets directory directly (no HTML fallback needed for assets)
    _ASSETS_DIR = _DIST_DIR / "assets"
    if _ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        """Serve index.html for all unknown paths (SPA client-side routing)."""
        candidate = _DIST_DIR / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_DIST_DIR / "index.html"))
else:
    import warnings
    warnings.warn(
        "static/dist not found — run `npm run build` in rag-anything/server/frontend/ first",
        stacklevel=1,
    )
