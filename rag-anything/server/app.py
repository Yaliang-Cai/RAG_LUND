import hashlib
import html
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Set

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from raganything.services.local_rag import LocalRagService, LocalRagSettings
from raganything.chunking import CHUNKING_STRATEGIES
from raganything.constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_SUPPORTED_FILE_EXTENSIONS,
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
)

VALID_CHUNKING_STRATEGIES: Set[str] = set(CHUNKING_STRATEGIES.keys())

logger = logging.getLogger(__name__)

# --- 配置与初始化 ---
APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))

# 检测本地静态资源是否已下载（运行 server/download_static.py 后生效）
_STATIC_DIR = APP_ROOT / "static"
_USE_LOCAL_STATIC: bool = all([
    (_STATIC_DIR / "marked.min.js").exists(),
    (_STATIC_DIR / "katex" / "katex.min.js").exists(),
    (_STATIC_DIR / "hljs" / "highlight.min.js").exists(),
])

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
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
_service: Optional[LocalRagService] = None

if _USE_LOCAL_STATIC:
    logger.info("Offline mode: serving JS/CSS from server/static/")
else:
    logger.info("Online mode: loading JS/CSS from CDN")


def _compute_doc_id(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    return cleaned if cleaned else hashlib.md5(name.encode("utf-8")).hexdigest()


def _validate_doc_id(doc_id: str):
    if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(status_code=400, detail="Invalid doc_id")


# --- [核心逻辑 1] 在指定 workspace 的 hybrid_auto 中查找文件 ---
def _safe_filename(filename: str) -> str:
    """Return the basename only; raise 400 if the result is empty or not .md."""
    name = Path(filename).name
    if not name or not name.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Invalid filename: must be a .md file")
    return name


def _find_md_in_hybrid_auto(doc_id: str, filename: str, output_dir: str) -> Path:
    _validate_doc_id(doc_id)
    workspace_dir = Path(output_dir).resolve() / doc_id
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail=f"Workspace '{doc_id}' not found")

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

def verify_api_key_or_query(
    x_api_key: Optional[str] = Header(default=None),
    key: Optional[str] = Query(default=None),
):
    """Like verify_api_key but also accepts ?key= query param (for <iframe> PDF src URLs)."""
    expected = os.getenv(API_KEY_ENV, "").strip()
    if not expected: return
    if x_api_key == expected or key == expected: return
    raise HTTPException(status_code=401, detail="Invalid API key")

# --- 数据模型 ---
class QueryRequest(BaseModel):
    doc_id: str
    query: str
    mode: str = DEFAULT_QUERY_MODE
    top_k: int = DEFAULT_TOP_K
    chunk_top_k: int = DEFAULT_CHUNK_TOP_K
    enable_rerank: bool = DEFAULT_ENABLE_RERANK
    vlm_enhanced: bool = DEFAULT_VLM_ENHANCED
    return_graph: bool = False
    graph_max_depth: int = DEFAULT_GRAPH_MAX_DEPTH
    graph_max_nodes: int = DEFAULT_GRAPH_MAX_NODES

# =========================================================================
# 路由
# =========================================================================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "use_local_static": _USE_LOCAL_STATIC},
    )

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
    chunking_strategy: Optional[str] = Form(default=None),
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
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
            detail=f"Invalid chunking_strategy '{chunking_strategy}'. Valid: {', '.join(sorted(VALID_CHUNKING_STRATEGIES))}",
        )

    file_stem = Path(file.filename).stem
    final_doc_id = doc_id.strip() if doc_id and doc_id.strip() else _compute_doc_id(file_stem)
    _validate_doc_id(final_doc_id)

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")

    # 保存原始文件到 uploads/{doc_id}/
    # basename-only + resolve+relative_to 三重校验，防路径穿越写
    upload_dir = UPLOADS_DIR / final_doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_filename = Path(file.filename).name  # strip any directory components
    if not original_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    upload_path = upload_dir / original_filename
    try:
        upload_path.resolve().relative_to(upload_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    upload_path.write_bytes(content)

    # 写入临时文件供 parser 使用（保留原始文件名以便 MinerU 正确命名输出）
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / original_filename
    tmp_path.write_bytes(content)

    workspace_output = str(Path(service.settings.output_dir) / final_doc_id)
    try:
        final_id = await service.ingest(
            str(tmp_path),
            doc_id=final_doc_id,
            output_dir=workspace_output,
            chunking_strategy=chunking_strategy or None,
        )
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

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
    _auth: None = Depends(verify_api_key_or_query),
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

def _graphml_path(service: LocalRagService, doc_id: str) -> Path:
    return (
        Path(service.settings.working_dir_root).resolve()
        / doc_id
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


@app.get("/graph/{doc_id}/labels")
async def get_graph_labels(
    doc_id: str,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    # 优先尝试 LightRAG API
    try:
        rag = await service.get_rag(doc_id)
        await rag._ensure_lightrag_initialized()
        labels = await rag.lightrag.get_graph_labels()
        if isinstance(labels, list) and labels:
            return {"labels": labels}
    except Exception as e:
        logger.warning("get_graph_labels LightRAG fallback: %s", e)

    # 兜底：NetworkX 直接读 GraphML
    G = _read_graphml_safe(_graphml_path(service, doc_id))
    if G is not None and G.number_of_nodes() > 0:
        return {"labels": sorted(G.nodes())[:500]}
    return {"labels": []}


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
    gpath = _graphml_path(service, doc_id)
    G = _read_graphml_safe(gpath)
    if G is not None:
        return {
            "entity_count": G.number_of_nodes(),
            "relation_count": G.number_of_edges(),
            "graphml_size": gpath.stat().st_size,
        }
    return {"entity_count": 0, "relation_count": 0, "graphml_size": gpath.stat().st_size if gpath.exists() else 0}


@app.get("/graph/{doc_id}/search")
async def search_graph_entities(
    doc_id: str,
    q: str,
    limit: int = DEFAULT_GRAPH_SEARCH_MAX_RESULTS,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    _validate_doc_id(doc_id)
    if not q.strip():
        return {"results": []}
    safe_limit = max(1, min(limit, DEFAULT_GRAPH_SEARCH_MAX_SAFE))
    q_stripped = q.strip()

    # 优先尝试 LightRAG API
    try:
        rag = await service.get_rag(doc_id)
        await rag._ensure_lightrag_initialized()
        results = await rag.lightrag.chunk_entity_relation_graph.search_labels(
            q_stripped, limit=safe_limit
        )
        if isinstance(results, list) and results:
            return {"results": results}
    except Exception as e:
        logger.warning("search_graph_entities LightRAG fallback: %s", e)

    # 兜底：NetworkX 子串匹配
    G = _read_graphml_safe(_graphml_path(service, doc_id))
    if G is not None:
        q_lower = q_stripped.lower()
        matched = [n for n in G.nodes() if q_lower in n.lower()][:safe_limit]
        return {"results": matched}
    return {"results": []}


@app.get("/graph/{doc_id}/overview")
async def get_graph_overview(
    doc_id: str,
    max_nodes: int = DEFAULT_GRAPH_OVERVIEW_MAX_NODES,
    _auth: None = Depends(verify_api_key),
    service: LocalRagService = Depends(get_service),
):
    """返回知识图谱概览子图（最高度数节点及其邻域），不依赖 LightRAG async API。"""
    _validate_doc_id(doc_id)
    G = _read_graphml_safe(_graphml_path(service, doc_id))
    if G is None or G.number_of_nodes() == 0:
        return {"nodes": [], "edges": []}

    # 选取度数最高的 max_nodes 个节点
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
    sub = G.subgraph(top_nodes)

    nodes = [
        {
            "id": n,
            "label": n,
            "type": G.nodes[n].get("entity_type", ""),
            "description": G.nodes[n].get("description", ""),
        }
        for n in sub.nodes()
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "label": d.get("description", ""),
            "weight": float(d.get("weight", 1.0)),
        }
        for u, v, d in sub.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


_ENTITY_COLORS = {
    "PERSON": "#ff6b6b",
    "ORGANIZATION": "#4ecdc4",
    "LOCATION": "#45b7d1",
    "CONCEPT": "#f7b731",
    "TECHNOLOGY": "#a55eea",
    "EVENT": "#26de81",
    "DOCUMENT": "#fd9644",
}
_DEFAULT_NODE_COLOR = "#95a5a6"


@app.get("/graph/{doc_id}/html")
def get_graph_html(
    doc_id: str,
    q: Optional[str] = None,
    max_nodes: int = DEFAULT_GRAPH_HTML_MAX_NODES,
    _auth: None = Depends(verify_api_key_or_query),
    service: LocalRagService = Depends(get_service),
):
    """
    Generate a self-contained pyvis HTML visualisation of the knowledge graph.
    Uses cdn_resources='in_line' so no external CDN is needed at render time.
    Accepts optional ?q= to filter to matching nodes + their 1-hop neighbours.
    Accepts ?key= for iframe-based authentication (via verify_api_key_or_query).
    """
    import os
    import tempfile

    _validate_doc_id(doc_id)
    G = _read_graphml_safe(_graphml_path(service, doc_id))

    _NO_GRAPH_HTML = (
        "<html><body style='margin:0;display:flex;align-items:center;"
        "justify-content:center;height:100vh;"
        "background:#131315;color:#6b6560;font-family:sans-serif;font-size:13px'>"
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
        bgcolor="#131315",
        font_color="#c8c4be",
        directed=sub.is_directed(),
    )
    try:
        net_kwargs = {"cdn_resources": "in_line"}
        net2 = Network(height="100%", width="100%", bgcolor="#131315", font_color="#c8c4be",
                       directed=sub.is_directed(), **net_kwargs)
        net = net2
    except TypeError:
        pass  # older pyvis without cdn_resources param

    net.from_nx(sub)

    for node in net.nodes:
        nid = node["id"]
        attrs = sub.nodes.get(nid, {})
        etype = attrs.get("entity_type", "")
        safe_nid = html.escape(str(nid))
        safe_etype = html.escape(str(etype))
        safe_desc = html.escape(str(attrs.get("description", "")))
        node["color"] = _ENTITY_COLORS.get(etype.upper(), _DEFAULT_NODE_COLOR)
        node["title"] = f"<b>{safe_nid}</b><br>{safe_etype}<br><br>{safe_desc}"
        node["size"] = max(10, min(30, 10 + G.degree(nid) * 2))
        node["label"] = str(nid)

    for edge in net.edges:
        desc = edge.get("description") or edge.get("label") or ""
        edge["title"] = html.escape(str(desc))
        edge["color"] = "rgba(200,196,190,0.25)"

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

    return {
        "workspaces": workspaces,
        "roots": {
            "uploads": str(UPLOADS_DIR),
            "output": str(output_root),
            "workspace": str(working_root),
        },
    }
