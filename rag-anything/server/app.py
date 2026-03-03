import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Set

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
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

# 三层存储目录
UPLOADS_DIR = Path(os.getenv("RAGANYTHING_UPLOADS_DIR", "./uploads")).resolve()

app = FastAPI(title="RAGAnything Local Service")
_service: Optional[LocalRagService] = None


def _compute_doc_id(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    return cleaned if cleaned else hashlib.md5(name.encode("utf-8")).hexdigest()


def _validate_doc_id(doc_id: str):
    if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(status_code=400, detail="Invalid doc_id")


# --- [核心逻辑 1] 在指定 workspace 的 hybrid_auto 中查找文件 ---
def _find_md_in_hybrid_auto(doc_id: str, filename: str, output_dir: str) -> Path:
    _validate_doc_id(doc_id)
    workspace_dir = Path(output_dir).resolve() / doc_id
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail=f"Workspace '{doc_id}' not found")

    for hybrid_dir in workspace_dir.rglob("hybrid_auto"):
        if not hybrid_dir.is_dir():
            continue
        target = hybrid_dir / filename
        if target.exists() and target.suffix.lower() == ".md":
            return target

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
    return_graph: bool = False
    graph_max_depth: int = 2
    graph_max_nodes: int = 50

# =========================================================================
# 路由
# =========================================================================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

# --- 文件列表 (解析产物) ---
@app.get("/files/{doc_id}")
def list_workspace_files(
    doc_id: str,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    _validate_doc_id(doc_id)
    workspace_dir = Path(service.settings.output_dir).resolve() / doc_id
    if not workspace_dir.exists():
        return {"files": []}
    file_list = sorted({p.name for p in workspace_dir.rglob("hybrid_auto/*.md")})
    return {"files": file_list}

# --- 文档内容 (Markdown) ---
@app.get("/content/{doc_id}")
async def get_document_content(
    doc_id: str,
    filename: Optional[str] = None,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    if not filename or filename == "undefined":
        files_resp = list_workspace_files(doc_id, service, _auth)
        files = files_resp["files"]
        if not files:
            raise HTTPException(status_code=404, detail="No processed .md files found.")
        filename = files[0]

    try:
        md_path = _find_md_in_hybrid_auto(doc_id, filename, str(service.settings.output_dir))
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
    doc_id: Optional[str] = Form(default=None),
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    file_stem = Path(file.filename).stem
    final_doc_id = doc_id.strip() if doc_id and doc_id.strip() else _compute_doc_id(file_stem)
    _validate_doc_id(final_doc_id)

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")

    # 保存原始文件到 uploads/{doc_id}/
    upload_dir = UPLOADS_DIR / final_doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_filename = file.filename
    upload_path = upload_dir / original_filename
    upload_path.write_bytes(content)

    # 写入临时文件供 parser 使用
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=file_ext)
    tmp_path = Path(tmp_name)
    try:
        os.write(tmp_fd, content)
    finally:
        os.close(tmp_fd)

    workspace_output = str(Path(service.settings.output_dir) / final_doc_id)
    try:
        final_id = await service.ingest(
            str(tmp_path), doc_id=final_doc_id, output_dir=workspace_output
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"doc_id": final_id, "filename": original_filename}


@app.get("/uploads/{doc_id}")
def list_uploads(
    doc_id: str,
    _auth: None = Depends(verify_api_key),
):
    _validate_doc_id(doc_id)
    upload_dir = UPLOADS_DIR / doc_id
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


@app.get("/uploads/{doc_id}/{filename}")
def serve_upload(
    doc_id: str,
    filename: str,
    _auth: None = Depends(verify_api_key),
):
    _validate_doc_id(doc_id)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = UPLOADS_DIR / doc_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))


# =========================================================================
# 解析产物图片服务
# =========================================================================

@app.get("/output/{doc_id}/images/{path:path}")
def serve_output_image(
    doc_id: str,
    path: str,
    service: LocalRagService = Depends(get_service),
    _auth: None = Depends(verify_api_key),
):
    _validate_doc_id(doc_id)
    if ".." in path:
        raise HTTPException(status_code=400, detail="Invalid path")
    output_root = Path(service.settings.output_dir).resolve()
    file_path = output_root / doc_id / path
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
    _validate_doc_id(payload.doc_id)
    top_k = max(1, min(payload.top_k, MAX_TOP_K))
    chunk_top_k = max(1, min(payload.chunk_top_k, MAX_CHUNK_TOP_K))

    rag = await service.get_rag(payload.doc_id)
    await rag._ensure_lightrag_initialized()

    # Step 1: 获取结构化检索数据 (不调用 LLM)
    from lightrag import QueryParam
    data_param = QueryParam(
        mode=payload.mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=payload.enable_rerank,
    )
    retrieval = {}
    try:
        retrieval = await rag.lightrag.aquery_data(payload.query, param=data_param)
    except Exception:
        pass  # 检索数据是增强功能，失败不阻断

    # Step 2: 获取 LLM 答案 (走完整 VLM 增强链路)
    answer = await service.query(
        payload.doc_id,
        payload.query,
        mode=payload.mode,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        enable_rerank=payload.enable_rerank,
        vlm_enhanced=payload.vlm_enhanced,
    )

    # Step 3: 可选获取子图数据
    graph_data = None
    if payload.return_graph:
        graph_data = await _get_query_subgraph(rag, retrieval, payload)

    return {
        "answer": answer,
        "data": retrieval.get("data", {}),
        "metadata": retrieval.get("metadata", {}),
        "graph": graph_data,
    }


async def _get_query_subgraph(rag, retrieval, payload):
    """从查询关键词中提取子图用于可视化"""
    try:
        keywords = retrieval.get("metadata", {}).get("keywords", {})
        ll_kws = keywords.get("low_level", [])
        if not ll_kws:
            return None
        label = ll_kws[0] if isinstance(ll_kws[0], str) else str(ll_kws[0])
        kg = await rag.lightrag.get_knowledge_graph(
            node_label=label,
            max_depth=payload.graph_max_depth,
            max_nodes=payload.graph_max_nodes,
        )
        nodes = []
        for n in kg.nodes:
            nodes.append({
                "id": n.id,
                "label": n.id,
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


# =========================================================================
# 知识图谱 API
# =========================================================================

@app.get("/graph/{doc_id}/labels")
async def get_graph_labels(
    doc_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    rag = await service.get_rag(doc_id)
    await rag._ensure_lightrag_initialized()
    try:
        labels = await rag.lightrag.get_graph_labels()
        return {"labels": labels if isinstance(labels, list) else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get labels: {e}")


@app.get("/graph/{doc_id}/subgraph")
async def get_subgraph(
    doc_id: str,
    label: str,
    max_depth: int = 2,
    max_nodes: int = 50,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    rag = await service.get_rag(doc_id)
    await rag._ensure_lightrag_initialized()
    try:
        kg = await rag.lightrag.get_knowledge_graph(
            node_label=label, max_depth=max_depth, max_nodes=max_nodes
        )
        nodes = [
            {
                "id": n.id,
                "label": n.id,
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


@app.get("/graph/{doc_id}/stats")
async def get_graph_stats(
    doc_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    graphml_path = (
        Path(service.settings.working_dir_root).resolve()
        / doc_id
        / "graph_chunk_entity_relation.graphml"
    )
    if not graphml_path.exists():
        return {"entity_count": 0, "relation_count": 0, "graphml_size": 0}
    try:
        import networkx as nx
        G = nx.read_graphml(str(graphml_path))
        return {
            "entity_count": G.number_of_nodes(),
            "relation_count": G.number_of_edges(),
            "graphml_size": graphml_path.stat().st_size,
        }
    except Exception:
        return {"entity_count": 0, "relation_count": 0, "graphml_size": graphml_path.stat().st_size}


@app.get("/graph/{doc_id}/search")
async def search_graph_entities(
    doc_id: str,
    q: str,
    limit: int = 20,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    if not q.strip():
        return {"results": []}
    rag = await service.get_rag(doc_id)
    await rag._ensure_lightrag_initialized()
    try:
        results = await rag.lightrag.chunk_entity_relation_graph.search_labels(
            q.strip(), limit=min(limit, 100)
        )
        return {"results": results if isinstance(results, list) else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# =========================================================================
# 工作空间管理
# =========================================================================

@app.delete("/workspace/{doc_id}")
async def delete_workspace(
    doc_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    deleted = []
    # 三层目录删除
    dirs_to_delete = [
        ("uploads", UPLOADS_DIR / doc_id),
        ("output", Path(service.settings.output_dir).resolve() / doc_id),
        ("workspace", Path(service.settings.working_dir_root).resolve() / doc_id),
    ]
    for name, d in dirs_to_delete:
        if d.exists() and d.is_dir():
            shutil.rmtree(str(d), ignore_errors=True)
            deleted.append(name)
    # 清除缓存的 rag 实例
    if hasattr(service, "_rag_instances") and doc_id in service._rag_instances:
        del service._rag_instances[doc_id]
    return {"status": "ok", "deleted": deleted}


@app.get("/workspace/{doc_id}/stats")
async def get_workspace_stats(
    doc_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    upload_dir = UPLOADS_DIR / doc_id
    output_dir = Path(service.settings.output_dir).resolve() / doc_id
    workspace_dir = Path(service.settings.working_dir_root).resolve() / doc_id

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


# =========================================================================
# 工作空间列表 (增强)
# =========================================================================

@app.get("/workspaces")
async def list_workspaces(
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    working_root = Path(service.settings.working_dir_root).resolve()
    output_root = Path(service.settings.output_dir).resolve()
    workspaces = []

    # 收集所有可能的 doc_id
    doc_ids = set()
    for root_dir in [working_root, output_root, UPLOADS_DIR]:
        if root_dir.exists():
            for d in root_dir.iterdir():
                if d.is_dir():
                    doc_ids.add(d.name)

    for doc_id in sorted(doc_ids):
        workspace_dir = working_root / doc_id
        upload_dir = UPLOADS_DIR / doc_id

        # 基本信息
        has_files = (workspace_dir / "graph_chunk_entity_relation.graphml").exists()

        # 上传文件列表
        uploaded_files = []
        if upload_dir.exists():
            for p in sorted(upload_dir.iterdir()):
                if p.is_file():
                    uploaded_files.append(p.name)

        workspaces.append({
            "doc_id": doc_id,
            "has_files": has_files,
            "uploaded_files": uploaded_files,
        })

    return {"workspaces": workspaces}