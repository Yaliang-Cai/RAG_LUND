import hashlib
import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Set
import glob

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from raganything.services.local_rag import LocalRagService, LocalRagSettings
from raganything.constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_SUPPORTED_FILE_EXTENSIONS,
)

# --- 配置与初始化 ---
APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))

API_KEY_ENV = "RAGANYTHING_API_KEY"
MAX_TOP_K = int(os.getenv("RAGANYTHING_MAX_TOP_K", str(DEFAULT_TOP_K)))
MAX_CHUNK_TOP_K = int(os.getenv("RAGANYTHING_MAX_CHUNK_TOP_K", str(DEFAULT_CHUNK_TOP_K)))

SUPPORTED_EXTENSIONS: Set[str] = {
    ext.strip().lower()
    for ext in DEFAULT_SUPPORTED_FILE_EXTENSIONS.split(",")
}

app = FastAPI(title="RAGAnything Local Service")
_service: Optional[LocalRagService] = None


def _compute_doc_id(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    return cleaned if cleaned else hashlib.md5(name.encode("utf-8")).hexdigest()


# --- [核心逻辑 1] 在指定 workspace 的 hybrid_auto 中查找文件 ---
def _find_md_in_hybrid_auto(doc_id: str, filename: str, output_dir: str) -> Path:
    """
    在 output/{doc_id}/**/hybrid_auto 下递归查找文件。
    """
    if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    workspace_dir = Path(output_dir).resolve() / doc_id
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail=f"Workspace '{doc_id}' not found")

    # 1. 精确匹配：在所有 hybrid_auto 子目录中查找
    for hybrid_dir in workspace_dir.rglob("hybrid_auto"):
        if not hybrid_dir.is_dir():
            continue
        target = hybrid_dir / filename
        if target.exists() and target.suffix.lower() == ".md":
            return target

    # 2. 模糊匹配（排序后取第一个，保证稳定性）
    all_md = sorted(workspace_dir.rglob("hybrid_auto/*.md"))
    candidates = [
        p for p in all_md
        if filename.lower() in p.name.lower() and p.suffix.lower() == ".md"
    ]
    if candidates:
        return candidates[0]

    raise HTTPException(
        status_code=404,
        detail=f"File '{filename}' not found in workspace '{doc_id}'."
    )


# --- 依赖注入 ---
def get_service() -> LocalRagService:
    global _service
    if _service is None:
        settings = LocalRagSettings.from_env()
        _service = LocalRagService(settings)
    return _service

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected: return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

# --- 数据模型 ---
class QueryRequest(BaseModel):
    doc_id: str
    query: str
    mode: str = "hybrid"
    top_k: int = DEFAULT_TOP_K
    chunk_top_k: int = DEFAULT_CHUNK_TOP_K
    enable_rerank: bool = True
    vlm_enhanced: bool = True

# --- 路由 ---

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

# --- [核心逻辑 2] 列出指定 workspace 的 md 文件 ---
@app.get("/files/{doc_id}")
def list_workspace_files(
    doc_id: str,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    """
    递归扫描 /output/{doc_id}/**/hybrid_auto/*.md 下的文件。
    """
    if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    workspace_dir = Path(service.settings.output_dir).resolve() / doc_id
    if not workspace_dir.exists():
        return {"files": []}

    file_list = sorted({p.name for p in workspace_dir.rglob("hybrid_auto/*.md")})
    return {"files": file_list}

# --- [核心逻辑 3] 获取内容 ---
@app.get("/content/{doc_id}")
async def get_document_content(
    doc_id: str,
    filename: Optional[str] = None,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    # 如果前端没传 filename，自动取列表里的第一个
    if not filename or filename == "undefined":
        files_resp = list_workspace_files(doc_id, service, _auth)
        files = files_resp["files"]
        if not files:
            raise HTTPException(
                status_code=404,
                detail="No processed .md files found in output directory."
            )
        filename = files[0]

    try:
        md_path = _find_md_in_hybrid_auto(
            doc_id, filename, str(service.settings.output_dir)
        )
        content = md_path.read_text(encoding="utf-8")
        return {"content": content, "filename": md_path.name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(default=None),
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    # 1. 扩展名校验
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file_ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # 2. 确定 final_doc_id（在写文件前，确保 output_dir 正确）
    file_stem = Path(file.filename).stem
    final_doc_id = doc_id.strip() if doc_id and doc_id.strip() else _compute_doc_id(file_stem)
    if ".." in final_doc_id or "/" in final_doc_id or "\\" in final_doc_id:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    # 3. 将上传内容写入临时文件（保留原扩展名供 parser 识别格式）
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=file_ext)
    tmp_path = Path(tmp_name)
    try:
        os.write(tmp_fd, content)
    finally:
        os.close(tmp_fd)

    # 4. ingest：传入 workspace 级 output_dir，使 parser 写入 {output_dir}/{doc_id}/{stem}/hybrid_auto/
    workspace_output = str(Path(service.settings.output_dir) / final_doc_id)
    try:
        final_id = await service.ingest(
            str(tmp_path), doc_id=final_doc_id, output_dir=workspace_output
        )
    finally:
        tmp_path.unlink(missing_ok=True)  # 无论成功失败都清理临时文件

    return {"doc_id": final_id}


@app.post("/query")
async def query(
    payload: QueryRequest,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    top_k = max(1, min(payload.top_k, MAX_TOP_K))
    chunk_top_k = max(1, min(payload.chunk_top_k, MAX_CHUNK_TOP_K))
    result = await service.query(
        payload.doc_id,
        payload.query,
        mode=payload.mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=payload.enable_rerank,
        vlm_enhanced=payload.vlm_enhanced,
    )
    return {"answer": result}


@app.get("/workspaces")
def list_workspaces(
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    root = Path(service.settings.working_dir_root).resolve()
    output_root = Path(service.settings.output_dir).resolve()
    items = []
    if root.exists():
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            workspace_output = output_root / entry.name
            has_files = (
                bool(next(workspace_output.rglob("hybrid_auto/*.md"), None))
                if workspace_output.exists()
                else False
            )
            items.append({"doc_id": entry.name, "has_files": has_files})
    return {"workdir_root": str(root), "workspaces": items}
