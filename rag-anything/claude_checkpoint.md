 Plan(设计全新 RAG WebUI 架构)
  ⎿  Prompt:
       基于以下探索结果和用户选择，设计一个全新的 RAG WebUI 系统的详细实现计划。

       用户选择

       - 前端：Vanilla JS，单 HTML 文件（或少量 JS 模块），CDN 引入库
       - 图谱可视化：Sigma.js (WebGL) + graphology
       - 后端：扩展 server/app.py，新增丰富 API
       - 文档预览：PDF.js 渲染原始 PDF（需保留原始上传文件）

       用户核心需求（四个维度）

       维度 1：溯源即时感 (Side-by-Side)

       - 双栏对照：左侧 PDF 阅读器 + 右侧对话
       - 答案中 [1][2] 引用可点击，PDF 自动跳转到对应页码
       - 引用悬浮卡片（hover 显示原文片段）

       维度 2：思考过程透明化 (Reasoning Trace)

       - 检索链路展示：进度指示器（"正在检索知识图谱..."、"正在重排序 40 个片段..."）
       - 多模态融合反馈：显示识别到的关键图表
       - 搜索词反思：显示 query rewrite 后的关键词

       维度 3：多模态交互流畅性 (Multimodal Fluency)

       - LaTeX 公式渲染（KaTeX）
       - 可交互表格
       - 图文混排答案
       - 图片点击放大

       维度 4：知识库掌控感 (Knowledge Management)

       - 局部图谱可视化（Sigma.js，展示当前问题涉及的实体/关系）
       - 数据健康看板（统计信息）
       - 文件上传/管理（资料库、解析产物、工作空间三层概念）
       - 纠错反馈（编辑实体/关系）
       - 严格模式/创意模式切换

       现有后端架构

       当前 API 端点（rag-anything/server/app.py）

       GET  /              → index.html
       GET  /files/{doc_id}    → {"files": [str]}
       GET  /content/{doc_id}  → {"content": str, "filename": str}
       POST /ingest            → {"doc_id": str}  （上传文件，tempfile 处理后删除）
       POST /query             → {"answer": str}  （仅返回答案文本）
       GET  /workspaces        → {"workspaces": [{doc_id, has_files}]}

       可利用的 LightRAG 内部 API

       LightRAG 内部有丰富的数据但当前 app.py 没暴露：
       - aquery_llm() 返回完整结构：entities[], relationships[], chunks[], references[], metadata{keywords, processing_info}
       - graph_chunk_entity_relation.graphml 存在于每个 workspace 的 working_dir
       - LightRAG API 有 /graphs?label=*&max_depth=3&max_nodes=1000 但需要 LightRAG API server 运行

       目录结构

       output/{doc_id}/{file_stem}/hybrid_auto/  → MinerU 解析的 .md + images/
       rag_workspace/{doc_id}/                   → graphml + 向量数据库

       关键限制

       - 当前 /ingest 用 tempfile，原始 PDF 不保留（需要改为保留）
       - 当前 /query 只返回 answer 字符串（需要扩展）
       - 没有图谱查询 API（需要新增）
       - 没有统计 API（需要新增）
       - 没有删除 workspace API（需要新增）

       CDN 库清单

       - marked.js：Markdown 渲染
       - KaTeX：LaTeX 公式
       - PDF.js：PDF 渲染
       - Sigma.js + graphology：图谱可视化
       - highlight.js：代码高亮

       设计要求

       1. 前端为单个 index.html（所有 CSS/JS 内联或从 CDN 加载）
       2. 后端改动集中在 server/app.py（不改 LightRAG 核心代码，不改 raganything 核心代码）
       3. 保持与现有 LocalRagService 的集成
       4. 保留 API key 认证机制
       5. 需要系统性设计三层存储：
         - 资料库（uploads/）：原始上传文件
         - 解析产物（output/）：MinerU 解析后的 md + images
         - 工作空间（rag_workspace/）：KG + 向量数据库

       请设计：
       1. 后端 API 完整设计（新增/修改的端点、请求/响应格式）
       2. 前端页面整体布局和交互流程
       3. 前端 JS 模块划分
       4. 文件变更清单和实现步骤
       5. 验证方案

       注意：这是一个大型任务，请确保计划足够详细但可执行。优先保证核心功能（查询+溯源+图谱），次要功能（纠错、导出）可标注为
       后续。
  ⎿  Read(D:/HUAWEI/RAG_LUND/server/app.py)

  ⎿  Error: File does not exist. Note: your current working directory is D:\HUAWEI\RAG_LUND.

  ⎿  Search(pattern: "**/*.html", path: "D:/HUAWEI/RAG_LUND")

  ⎿  Error: Sibling tool call errored

  ⎿  Search(pattern: "server/**/*", path: "D:/HUAWEI/RAG_LUND")

  ⎿  Error: Sibling tool call errored

  ⎿  Bash(ls D:/HUAWEI/RAG_LUND/)

  ⎿  Bash(ls D:/HUAWEI/RAG_LUND/rag-anything/server/)

  ⎿  Bash(ls D:/HUAWEI/RAG_LUND/rag-anything/)

  ⎿  Read(D:/HUAWEI/RAG_LUND/rag-anything/server/app.py)

  ⎿  Bash(ls D:/HUAWEI/RAG_LUND/rag-anything/server/templates/)

  ⎿  Read(D:/HUAWEI/RAG_LUND/rag-anything/server/templates/index.html)

  ⎿  Read(D:/HUAWEI/RAG_LUND/rag-anything/raganything/services/local_rag.py)

  ⎿  Read(D:/HUAWEI/RAG_LUND/rag-anything/raganything/query.py)

  ⎿  Search(pattern: "aquery|aquery_llm|aquery_with_multimodal", path: "D:/HUAWEI/RAG_LUND/rag-anything/raganything")

  ⎿  Search(pattern: "rag-anything/raganything/*.py", path: "D:/HUAWEI/RAG_LUND")

  ⎿  Search(pattern: "aquery|only_need_prompt|only_need_context", path: "D:/HUAWEI/RAG_LUND/lightrag")

  ⎿  Search(pattern: "graphml|graph_chunk_entity_relation", path: "D:/HUAWEI/RAG_LUND/rag-anything")

  ⎿  Read(D:/HUAWEI/RAG_LUND/rag-anything/raganything/constants.py)

  ⎿  Search(pattern: "async def aquery", path: "D:/HUAWEI/RAG_LUND/lightrag/lightrag/lightrag.py")

  ⎿  Search(pattern: "class QueryParam", path: "D:/HUAWEI/RAG_LUND/lightrag/lightrag/base.py")

  ⎿  Read(D:/HUAWEI/RAG_LUND/lightrag/lightrag/lightrag.py · lines 2422-2541)

  ⎿  Read(D:/HUAWEI/RAG_LUND/lightrag/lightrag/lightrag.py · lines 2477-2726)

  ⎿  Read(D:/HUAWEI/RAG_LUND/lightrag/lightrag/lightrag.py · lines 2726-2845)

  ⎿  Search(pattern: "graphs|graph_label|graphml", path: "D:/HUAWEI/RAG_LUND/lightrag/lightrag/api")

  ⎿  Read(D:/HUAWEI/RAG_LUND/lightrag/lightrag/api/routers/graph_routes.py)

  ⎿  Search(pattern: "async def get_knowledge_graph|def get_knowledge_graph", path:
           "D:/HUAWEI/RAG_LUND/lightrag/lightrag/lightrag.py")
  ⎿  Search(pattern: "async def get_graph_labels|def get_graph_labels", path:
           "D:/HUAWEI/RAG_LUND/lightrag/lightrag/lightrag.py")
  ⎿  Search(pattern: "StaticFiles|static|mount", path: "D:/HUAWEI/RAG_LUND/rag-anything/server/app.py")

  ⎿  Search(pattern: "FileResponse|StreamingResponse|send_file", path: "D:/HUAWEI/RAG_LUND/rag-anything/server/app.py")

  ⎿  Search(pattern: "_ensure_lightrag_initialized|self\.lightrag", path:
           "D:/HUAWEI/RAG_LUND/rag-anything/raganything/raganything.py")
  ⎿  Response:
       I now have a comprehensive understanding of the entire codebase. Let me produce the detailed implementation plan.

       ---
       RAG WebUI System -- Detailed Implementation Plan

       Part 1: Backend API Design (Changes to server/app.py)

       All changes are confined to D:\HUAWEI\RAG_LUND\rag-anything\server\app.py. No modifications to raganything/ core or
       lightrag/ core.

       1.1 New Imports Needed

       import shutil
       import time
       import networkx as nx
       from fastapi import BackgroundTasks
       from fastapi.responses import FileResponse, StreamingResponse
       from fastapi.staticfiles import StaticFiles
       from starlette.responses import Response
       from lightrag import QueryParam

       1.2 Three-Layer Storage Setup

       Add constants and directory creation at module level:

       UPLOADS_DIR = Path(os.getenv("RAGANYTHING_UPLOADS_DIR", "./rag-anything/uploads"))
       # OUTPUT_DIR and WORKING_DIR_ROOT already exist via LocalRagSettings

       In the ingest endpoint, save the original file to UPLOADS_DIR/{doc_id}/ before processing, and remove the
       tmp_path.unlink() cleanup of the original (keep it in uploads/).

       1.3 API Endpoints -- Complete List

       Existing Endpoints (Modified)

       ┌───────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────
       ──────┐
       │   Endpoint    │                                                  Changes
             │
       ├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ POST /ingest  │ Save original file to uploads/{doc_id}/{original_filename} before processing. Return {"doc_id": str,
             │
       │               │ "filename": str}.
             │
       ├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ POST /query   │ Major rewrite: use lightrag.aquery_llm() instead of service.query(). Return structured response with
             │
       │               │ answer, entities, relationships, chunks, references, metadata.
             │
       ├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ GET           │ Add stats per workspace: file count, entity count, chunk count. Add uploaded_files list.
             │
       │ /workspaces   │
             │
       └───────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────
       ──────┘

       New Endpoints

       File Management:

       ┌─────────────────────────────────────┬────────┬──────────────────────────┬─────────┬──────────────────────────────────
       ──────┐
       │              Endpoint               │ Method │       Description        │ Request │                Response
             │
       ├─────────────────────────────────────┼────────┼──────────────────────────┼─────────┼──────────────────────────────────
       ──────┤
       │ /uploads/{doc_id}                   │ GET    │ List uploaded original   │ -       │ {"files": [{"name": str, "size":
       int,  │
       │                                     │        │ files                    │         │ "uploaded_at": str}]}
             │
       ├─────────────────────────────────────┼────────┼──────────────────────────┼─────────┼──────────────────────────────────
       ──────┤
       │ /uploads/{doc_id}/{filename}        │ GET    │ Serve original file      │ -       │ FileResponse (binary)
             │
       │                                     │        │ (PDF) for PDF.js         │         │
             │
       ├─────────────────────────────────────┼────────┼──────────────────────────┼─────────┼──────────────────────────────────
       ──────┤
       │ /workspace/{doc_id}                 │ DELETE │ Delete workspace +       │ -       │ {"status": "ok"}
             │
       │                                     │        │ output + uploads         │         │
             │
       ├─────────────────────────────────────┼────────┼──────────────────────────┼─────────┼──────────────────────────────────
       ──────┤
       │ /output/{doc_id}/images/{path:path} │ GET    │ Serve parsed images from │ -       │ FileResponse (binary)
             │
       │                                     │        │  output dir              │         │
             │
       └─────────────────────────────────────┴────────┴──────────────────────────┴─────────┴──────────────────────────────────
       ──────┘

       Enhanced Query:

       ┌───────────┬────────┬───────────────────────────────────────────┬────────────────────────┬────────────────────────────
       ──────┐
       │ Endpoint  │ Method │                Description                │        Request         │             Response
             │
       ├───────────┼────────┼───────────────────────────────────────────┼────────────────────────┼────────────────────────────
       ──────┤
       │ /query/v2 │ POST   │ Rich query returning full structured data │ QueryRequestV2 (below) │ Full structured response
       (below) │
       └───────────┴────────┴───────────────────────────────────────────┴────────────────────────┴────────────────────────────
       ──────┘

       QueryRequestV2 model:
       class QueryRequestV2(BaseModel):
           doc_id: str
           query: str
           mode: str = "hybrid"
           top_k: int = DEFAULT_TOP_K
           chunk_top_k: int = DEFAULT_CHUNK_TOP_K
           enable_rerank: bool = True
           vlm_enhanced: bool = True
           # New fields:
           return_graph: bool = False        # Whether to include subgraph data
           graph_max_depth: int = 2
           graph_max_nodes: int = 50

       Response format:
       {
         "answer": "...(markdown string with [1][2] references)...",
         "data": {
           "entities": [{"entity_name": str, "entity_type": str, "description": str, "reference_id": str, ...}],
           "relationships": [{"src_id": str, "tgt_id": str, "description": str, "weight": float, ...}],
           "chunks": [{"content": str, "file_path": str, "chunk_id": str, "reference_id": str}],
           "references": [{"reference_id": str, "file_path": str}]
         },
         "metadata": {
           "query_mode": str,
           "keywords": {"high_level": [], "low_level": []},
           "processing_info": {...}
         },
         "graph": {  // Only if return_graph=true
           "nodes": [{"id": str, "label": str, "type": str, "description": str}],
           "edges": [{"source": str, "target": str, "label": str, "weight": float}]
         }
       }

       Implementation approach for /query/v2: Obtain the RAGAnything instance via service.get_rag(doc_id), then call
       rag.lightrag.aquery_llm() directly to get both the answer and structured data in a single call. For VLM-enhanced
       queries, we need a two-step approach: first call aquery_data() to get structured data, then call service.query() (which
        handles VLM) for the answer.

       Graph API:

       ┌───────────────────────────────┬────────┬────────────────┬───────────────────────────────────┬────────────────────────
       ──────┐
       │           Endpoint            │ Method │  Description   │              Request              │           Response
             │
       ├───────────────────────────────┼────────┼────────────────┼───────────────────────────────────┼────────────────────────
       ──────┤
       │ /graph/{doc_id}/labels        │ GET    │ Get all entity │ -                                 │ {"labels": [str]}
             │
       │                               │        │  labels        │                                   │
             │
       ├───────────────────────────────┼────────┼────────────────┼───────────────────────────────────┼────────────────────────
       ──────┤
       │                               │        │ Get subgraph   │                                   │ {"nodes": [...],
       "edges":    │
       │ /graph/{doc_id}/subgraph      │ GET    │ by entity      │ ?label=X&max_depth=2&max_nodes=50 │ [...]}
             │
       │                               │        │ label          │                                   │
             │
       ├───────────────────────────────┼────────┼────────────────┼───────────────────────────────────┼────────────────────────
       ──────┤
       │                               │        │ Graph          │                                   │ {"entity_count": int,
             │
       │ /graph/{doc_id}/stats         │ GET    │ statistics     │ -                                 │ "relation_count": int,
             │
       │                               │        │                │                                   │ "chunk_count": int}
             │
       ├───────────────────────────────┼────────┼────────────────┼───────────────────────────────────┼────────────────────────
       ──────┤
       │ /graph/{doc_id}/entity/{name} │ GET    │ Get single     │ -                                 │ Entity data dict
             │
       │                               │        │ entity details │                                   │
             │
       ├───────────────────────────────┼────────┼────────────────┼───────────────────────────────────┼────────────────────────
       ──────┤
       │                               │        │ Search         │                                   │
             │
       │ /graph/{doc_id}/search        │ GET    │ entities by    │ ?q=keyword&limit=20               │ {"results":
       [entity...]}     │
       │                               │        │ name           │                                   │
             │
       └───────────────────────────────┴────────┴────────────────┴───────────────────────────────────┴────────────────────────
       ──────┘

       Implementation: These all call through to the RAGAnything instance's lightrag property. For example,
       rag.lightrag.get_knowledge_graph(label, max_depth, max_nodes) returns a KnowledgeGraph object with nodes and edges. For
        stats, we read the graphml file via networkx.read_graphml() from
       rag_workspace/{doc_id}/graph_chunk_entity_relation.graphml.

       Graph Edit API (Phase 2 -- lower priority):

       ┌─────────────────────────────────┬────────┬──────────────────────────────────────────────────────┐
       │            Endpoint             │ Method │                     Description                      │
       ├─────────────────────────────────┼────────┼──────────────────────────────────────────────────────┤
       │ /graph/{doc_id}/entity/edit     │ POST   │ Edit entity (proxies to rag.lightrag.aedit_entity()) │
       ├─────────────────────────────────┼────────┼──────────────────────────────────────────────────────┤
       │ /graph/{doc_id}/relation/edit   │ POST   │ Edit relation                                        │
       ├─────────────────────────────────┼────────┼──────────────────────────────────────────────────────┤
       │ /graph/{doc_id}/entity/create   │ POST   │ Create entity                                        │
       ├─────────────────────────────────┼────────┼──────────────────────────────────────────────────────┤
       │ /graph/{doc_id}/relation/create │ POST   │ Create relation                                      │
       └─────────────────────────────────┴────────┴──────────────────────────────────────────────────────┘

       Workspace Stats:

       ┌───────────────────────────┬────────┬──────────────────┬──────────────────────────────────────────────────────────────
       ──────┐
       │         Endpoint          │ Method │   Description    │                              Response
             │
       ├───────────────────────────┼────────┼──────────────────┼──────────────────────────────────────────────────────────────
       ──────┤
       │ /workspace/{doc_id}/stats │ GET    │ Health dashboard │ {"files": int, "entities": int, "relations": int, "chunks":
       int,   │
       │                           │        │  data            │ "graphml_size": int, "upload_size_total": int}
             │
       └───────────────────────────┴────────┴──────────────────┴──────────────────────────────────────────────────────────────
       ──────┘

       1.4 Implementation Details for Key Endpoints

       POST /query/v2 -- Core Logic:

       @app.post("/query/v2")
       async def query_v2(payload: QueryRequestV2, ...):
           rag = await service.get_rag(payload.doc_id)
           await rag._ensure_lightrag_initialized()

           # Step 1: Get structured retrieval data (entities, relations, chunks)
           data_param = QueryParam(
               mode=payload.mode,
               top_k=payload.top_k,
               chunk_top_k=payload.chunk_top_k,
               enable_rerank=payload.enable_rerank,
           )
           retrieval_data = await rag.lightrag.aquery_data(payload.query, param=data_param)

           # Step 2: Get the actual answer (uses VLM if enabled)
           answer = await service.query(
               payload.doc_id, payload.query,
               mode=payload.mode,
               top_k=payload.top_k,
               chunk_top_k=payload.chunk_top_k,
               enable_rerank=payload.enable_rerank,
               vlm_enhanced=payload.vlm_enhanced,
           )

           # Step 3: Optionally get subgraph for visualization
           graph_data = None
           if payload.return_graph:
               keywords = retrieval_data.get("metadata", {}).get("keywords", {})
               ll_kws = keywords.get("low_level", [])
               # Use first keyword as label for graph query
               if ll_kws:
                   kg = await rag.lightrag.get_knowledge_graph(
                       node_label=ll_kws[0],
                       max_depth=payload.graph_max_depth,
                       max_nodes=payload.graph_max_nodes,
                   )
                   graph_data = {"nodes": [...], "edges": [...]}  # serialize KnowledgeGraph

           return {
               "answer": answer,
               "data": retrieval_data.get("data", {}),
               "metadata": retrieval_data.get("metadata", {}),
               "graph": graph_data,
           }

       GET /uploads/{doc_id}/{filename} -- PDF Serving:

       @app.get("/uploads/{doc_id}/{filename}")
       async def serve_upload(doc_id: str, filename: str, ...):
           file_path = UPLOADS_DIR / doc_id / filename
           if not file_path.exists():
               raise HTTPException(404)
           return FileResponse(str(file_path), media_type="application/pdf")

       GET /graph/{doc_id}/subgraph -- Graph Visualization Data:

       @app.get("/graph/{doc_id}/subgraph")
       async def get_subgraph(doc_id: str, label: str, max_depth: int = 2, max_nodes: int = 50, ...):
           rag = await service.get_rag(doc_id)
           await rag._ensure_lightrag_initialized()
           kg = await rag.lightrag.get_knowledge_graph(
               node_label=label, max_depth=max_depth, max_nodes=max_nodes
           )
           # Serialize KnowledgeGraph to {nodes, edges} format for Sigma.js
           nodes = [{"id": n.id, "label": n.id, "type": getattr(n, 'entity_type', ''), ...} for n in kg.nodes]
           edges = [{"source": e.source, "target": e.target, "label": e.description, ...} for e in kg.edges]
           return {"nodes": nodes, "edges": edges}

       Part 2: Frontend Page Layout and Interaction Flow

       2.1 Overall Layout

       The page uses a responsive three-panel layout:

       +--[Header Bar]-------------------------------------------------------+
       | Brand | API Key | Workspace Select | Upload | Mode | Graph | Theme  |
       +----------------------------------------------------------------------+
       |                    |                    |                             |
       |  PDF Panel (Left)  |   Divider (drag)   |   Chat Panel (Right)       |
       |  +--------------+  |                    |  +------------------------+|
       |  | PDF Toolbar   |  |                    |  | Reasoning Trace Panel  ||
       |  | page/zoom/ref |  |                    |  | (collapsible)          ||
       |  +--------------+  |                    |  +------------------------+|
       |  |              |  |                    |  |                        ||
       |  |  PDF.js      |  |                    |  |  Chat Messages         ||
       |  |  Viewer      |  |                    |  |  (markdown + KaTeX +   ||
       |  |              |  |                    |  |   tables + images)     ||
       |  |              |  |                    |  |                        ||
       |  +--------------+  |                    |  +------------------------+|
       |                    |                    |  | Input Area              ||
       |                    |                    |  +------------------------+|
       +----------------------------------------------------------------------+
       |  [Optional Bottom Panel: Knowledge Graph Viewer (Sigma.js)]          |
       +----------------------------------------------------------------------+

       Key layout decisions:
       - Left panel: PDF.js viewer (replacing the markdown reader). Still shows file list as a dropdown/selector in a toolbar.
       - Right panel: Chat with reasoning trace, rich rendering.
       - Bottom panel (toggle): Full-width graph visualization panel using Sigma.js. Opens when user clicks "Graph" button or
       clicks a graph-related action in the answer.

       2.2 Interaction Flows

       Flow 1: Initial Load
       1. Restore API key from localStorage
       2. Fetch /workspaces to populate workspace selector
       3. User selects workspace -> fetch /uploads/{doc_id} for PDF list + /files/{doc_id} for parsed file list
       4. Auto-load first PDF into PDF.js viewer

       Flow 2: Query with Tracing
       1. User types question, clicks Send
       2. Show reasoning trace panel with animated steps:
         - "Extracting keywords..." (shown immediately)
         - "Searching knowledge graph..." (after keywords extraction)
         - "Reranking 40 chunks..." (after retrieval)
         - "Generating answer..." (during LLM call)
       3. POST /query/v2 with return_graph=true
       4. On response:
         - Render answer with markdown + KaTeX + highlight.js
         - Parse [1][2] citations into clickable links
         - Show entity/chunk reference cards at bottom of answer
         - If graph data returned, update graph panel
         - Update reasoning trace with final stats

       Flow 3: Citation Click (Traceability)
       1. User clicks [1] in answer text
       2. Look up reference_id "1" in the response's references array
       3. Extract file_path from reference -> determine which PDF and page
       4. If PDF is loaded, jump to the corresponding page in PDF.js viewer
       5. Show hover card with chunk content excerpt

       Flow 4: Graph Interaction
       1. User clicks "Graph" toggle -> bottom panel expands
       2. Load subgraph data from response or fetch /graph/{doc_id}/subgraph?label=...
       3. Render with Sigma.js: entities as nodes, relationships as edges
       4. Click node -> show entity details sidebar
       5. Double-click node -> expand graph (fetch more neighbors)

       Flow 5: File Upload
       1. User clicks Upload button -> file input dialog
       2. POST /ingest with file (original saved to uploads/)
       3. Show progress indicator
       4. On completion, refresh workspace file list
       5. Auto-select new document

       2.3 PDF Viewer Integration

       PDF.js viewer setup:
       - Load PDF via URL: /uploads/{doc_id}/{filename}
       - Use pdfjsLib.getDocument(url) to load
       - Custom toolbar: page navigation, zoom, search
       - scrollToPage(n) function for citation jumping
       - Page mapping: derive page numbers from chunk file_path metadata (format: filename_page_N) or from content markers

       2.4 Rich Content Rendering

       Answer rendering pipeline:
       raw_answer (string)
         -> marked.parse()           // Markdown -> HTML
         -> KaTeX.renderToString()   // LaTeX formulas: $...$ and $$...$$
         -> hljs.highlightAll()      // Code blocks
         -> citationReplace()        // [1][2] -> clickable spans
         -> tableEnhance()           // Add sorting/scroll to tables
         -> imageEnhance()           // Add click-to-zoom lightbox

       Part 3: Frontend JS Module Organization

       Since this is a single index.html, organize the JS into clearly delimited IIFE sections:

       // ============================================================
       // MODULE 1: Config & State
       // ============================================================
       const AppState = { ... };
       const AppConfig = { ... };

       // ============================================================
       // MODULE 2: API Client
       // ============================================================
       const API = {
           async getWorkspaces() { ... },
           async queryV2(docId, query, opts) { ... },
           async getSubgraph(docId, label) { ... },
           async getUploads(docId) { ... },
           async ingestFile(file, docId) { ... },
           async deleteWorkspace(docId) { ... },
           async getStats(docId) { ... },
           async searchEntities(docId, q) { ... },
           // ... all API calls centralized
       };

       // ============================================================
       // MODULE 3: PDF Viewer
       // ============================================================
       const PDFViewer = {
           init(containerId) { ... },
           loadDocument(url) { ... },
           scrollToPage(pageNum) { ... },
           highlight(text) { ... },
           destroy() { ... },
       };

       // ============================================================
       // MODULE 4: Chat Engine
       // ============================================================
       const Chat = {
           sendMessage(text) { ... },
           renderAnswer(data) { ... },        // Rich rendering
           addUserMessage(text) { ... },
           addBotMessage(html) { ... },
           addThinking() { ... },
           clearWelcome() { ... },
       };

       // ============================================================
       // MODULE 5: Reasoning Trace
       // ============================================================
       const ReasoningTrace = {
           start() { ... },
           updateStep(step, status) { ... },  // "searching" -> "done"
           showKeywords(keywords) { ... },
           showStats(info) { ... },
           collapse() { ... },
       };

       // ============================================================
       // MODULE 6: Citation Manager
       // ============================================================
       const Citations = {
           register(references, chunks) { ... },
           onClick(refId) { ... },           // Triggers PDFViewer.scrollToPage
           onHover(refId, element) { ... },  // Shows floating card
           getPageFromRef(ref) { ... },
       };

       // ============================================================
       // MODULE 7: Graph Viewer (Sigma.js)
       // ============================================================
       const GraphViewer = {
           init(containerId) { ... },
           loadData(nodes, edges) { ... },
           focusNode(nodeId) { ... },
           expandNode(nodeId) { ... },
           clear() { ... },
           toggle() { ... },                 // Show/hide bottom panel
       };

       // ============================================================
       // MODULE 8: Rich Renderer
       // ============================================================
       const Renderer = {
           renderMarkdown(text) { ... },     // marked + KaTeX + hljs
           renderCitations(html) { ... },    // Replace [n] with clickable spans
           renderTables(container) { ... },  // Interactive table features
           renderImages(container) { ... },  // Click-to-zoom
           renderFormulas(container) { ... },// KaTeX pass
       };

       // ============================================================
       // MODULE 9: File Manager
       // ============================================================
       const FileManager = {
           refreshList() { ... },
           upload(file) { ... },
           selectPDF(filename) { ... },
           deleteWorkspace() { ... },
       };

       // ============================================================
       // MODULE 10: UI Controls
       // ============================================================
       const UI = {
           initTheme() { ... },
           initDivider() { ... },
           initTextarea() { ... },
           toggleGraphPanel() { ... },
           showModal(content) { ... },       // Lightbox for images
           showToast(msg) { ... },
       };

       // ============================================================
       // MODULE 11: App Init
       // ============================================================
       (function main() {
           UI.initTheme();
           UI.initDivider();
           UI.initTextarea();
           PDFViewer.init('pdfContainer');
           GraphViewer.init('graphContainer');
           // Restore state from localStorage
           // ...
       })();

       Part 4: File Changes List and Implementation Steps

       Files to Modify

       1. D:\HUAWEI\RAG_LUND\rag-anything\server\app.py -- Major changes:
         - Add UPLOADS_DIR constant and directory creation
         - Modify POST /ingest to save original file to uploads
         - Add POST /query/v2 endpoint with full structured response
         - Add GET /uploads/{doc_id} and GET /uploads/{doc_id}/{filename} for PDF serving
         - Add GET /output/{doc_id}/images/{path:path} for parsed image serving
         - Add GET /graph/{doc_id}/labels, /graph/{doc_id}/subgraph, /graph/{doc_id}/stats
         - Add GET /graph/{doc_id}/search
         - Add DELETE /workspace/{doc_id}
         - Add GET /workspace/{doc_id}/stats
         - Modify GET /workspaces to include upload info and stats
       2. D:\HUAWEI\RAG_LUND\rag-anything\server\templates\index.html -- Complete rewrite:
         - Replace entire file with new WebUI
         - CDN imports for marked.js, KaTeX, PDF.js, Sigma.js/graphology, highlight.js
         - Three-panel layout (PDF left, chat right, graph bottom)
         - All JS modules inline
         - All CSS inline (build on existing design system/theme variables)

       Files NOT Modified (important constraint)

       - Nothing under raganything/ (core library)
       - Nothing under lightrag/ (upstream library)

       Implementation Steps (Ordered)

       Phase 1: Backend Foundation (Steps 1-4)

       Step 1: Add uploads directory and modify /ingest
       - Create UPLOADS_DIR constant
       - In ingest(), after reading file content, save to UPLOADS_DIR/{doc_id}/{original_filename}
       - Keep existing tempfile processing logic for MinerU
       - Return extended response with filename

       Step 2: Add PDF and image file serving endpoints
       - GET /uploads/{doc_id} -- list files in uploads dir
       - GET /uploads/{doc_id}/{filename} -- serve PDF via FileResponse
       - GET /output/{doc_id}/images/{path:path} -- serve images from output dir with path traversal protection

       Step 3: Add /query/v2 endpoint
       - Define QueryRequestV2 model
       - Implement two-phase query: aquery_data() for structured data + service.query() for answer
       - Serialize response into the defined JSON format
       - Handle graph data optionally

       Step 4: Add graph and stats endpoints
       - /graph/{doc_id}/labels -- calls rag.lightrag.get_graph_labels()
       - /graph/{doc_id}/subgraph -- calls rag.lightrag.get_knowledge_graph()
       - /graph/{doc_id}/stats -- reads graphml file for counts
       - /graph/{doc_id}/search -- calls rag.lightrag.chunk_entity_relation_graph.search_labels()
       - DELETE /workspace/{doc_id} -- remove dirs + evict from _rag_instances
       - GET /workspace/{doc_id}/stats -- compute file counts, sizes

       Phase 2: Frontend Core (Steps 5-8)

       Step 5: Build HTML skeleton and CSS
       - Header bar with controls
       - Three-panel layout with resizable dividers
       - PDF viewer container (left)
       - Chat panel (right) with reasoning trace area
       - Graph panel (bottom, collapsible)
       - Preserve existing theme system (dark/light, amber accent)

       Step 6: Implement API Client + PDF Viewer
       - Central API object with all fetch calls
       - PDF.js integration: load document, render pages, page navigation
       - Citation-to-page mapping

       Step 7: Implement Chat Engine + Rich Renderer
       - Send query via /query/v2
       - Render answer with full pipeline: marked -> KaTeX -> hljs -> citations -> tables -> images
       - Reasoning trace with step indicators
       - Citation click -> PDF jump + hover card

       Step 8: Implement Graph Viewer
       - Sigma.js + graphology setup
       - Load nodes/edges from query response or dedicated endpoint
       - Node click for details
       - Force-directed layout
       - Toggle panel visibility

       Phase 3: Polish and Secondary Features (Steps 9-11)

       Step 9: File upload UI + workspace management
       - Upload button with drag-and-drop
       - Progress indicator during ingest
       - Workspace deletion with confirmation
       - Stats dashboard

       Step 10: Interactive features
       - Image lightbox (click to zoom)
       - Table sorting/scrolling
       - Formula rendering edge cases
       - Mode switch (strict/creative) mapped to query mode parameter

       Step 11: Graph editing (future)
       - Entity/relation edit forms
       - Proxy to LightRAG edit APIs
       - Optimistic UI updates

       Part 5: Verification Plan

       5.1 Backend Verification

       ┌───────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────
       ──────┐
       │             Test              │                                       How to Verify
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ File upload preserves PDF     │ Upload a PDF, check uploads/{doc_id}/ contains the original
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ PDF serving works             │ curl /uploads/{doc_id}/test.pdf returns valid PDF binary
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ Image serving works           │ curl /output/{doc_id}/images/xxx.png returns image
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ /query/v2 returns structured  │ POST query, verify response has answer, data.entities, data.chunks, data.references,
             │
       │ data                          │ metadata.keywords
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ Graph labels endpoint         │ GET /graph/{doc_id}/labels returns non-empty list for ingested workspace
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ Graph subgraph endpoint       │ GET /graph/{doc_id}/subgraph?label=X returns {nodes, edges}
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ Workspace delete              │ DELETE /workspace/{doc_id}, verify dirs removed
             │
       ├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────
       ──────┤
       │ Stats endpoint                │ GET /workspace/{doc_id}/stats returns counts
             │
       └───────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────
       ──────┘

       5.2 Frontend Verification

       ┌──────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
       │             Test             │                              How to Verify                              │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ PDF loads correctly          │ Select workspace with uploaded PDF, left panel shows rendered PDF pages │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Citation click jumps to page │ Click [1] in answer, PDF scrolls to referenced page                     │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Citation hover shows card    │ Hover [1], floating card shows chunk excerpt                            │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ KaTeX renders                │ Query that returns LaTeX in answer, verify formulas render              │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Graph visualization          │ Click "Graph" button, verify Sigma.js canvas shows nodes/edges          │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Graph node click             │ Click entity node, details panel shows entity info                      │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Reasoning trace              │ During query, verify step indicators animate in sequence                │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Theme toggle                 │ Click theme button, verify both panels switch correctly                 │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ File upload                  │ Upload new PDF, verify it appears in workspace and PDF viewer           │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Image zoom                   │ Click image in answer, lightbox opens full-size                         │
       ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
       │ Responsive layout            │ Resize window, verify panels adapt                                      │
       └──────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

       5.3 Integration Verification

       1. End-to-end flow: Upload PDF -> Wait for ingest -> Query about content -> Verify answer references map to correct PDF
        pages -> Click citation -> PDF jumps to page -> View graph of entities mentioned in answer
       2. Multi-workspace: Switch between workspaces, verify PDF viewer and graph update correctly
       3. API key: Test with wrong key, verify 401 responses and UI feedback

       Critical Files for Implementation

       - D:\HUAWEI\RAG_LUND\rag-anything\server\app.py -- All backend API additions and modifications
       - D:\HUAWEI\RAG_LUND\rag-anything\server\templates\index.html -- Complete frontend rewrite
       - D:\HUAWEI\RAG_LUND\rag-anything\raganything\services\local_rag.py -- Reference for LocalRagService API surface (read
       only, used to understand get_rag(), query(), ingest())
       - D:\HUAWEI\RAG_LUND\lightrag\lightrag\lightrag.py -- Reference for aquery_data(), aquery_llm(), get_knowledge_graph()
       APIs (read only)
       - D:\HUAWEI\RAG_LUND\lightrag\lightrag\api\routers\graph_routes.py -- Pattern reference for graph API design (read
       only)
  ⎿  Done (28 tool uses · 92.4k tokens · 5m 42s)




   User approved Claude's plan
  ⎿  Plan saved to: C:\Users\14647\.claude\plans\ethereal-humming-lynx.md · /plan to edit
     计划：全新 RAG WebUI 系统

     Context

     当前 WebUI 是内嵌在单个 index.html 中的简单对话+文档预览界面，/query 仅返回 {"answer":
     str}，无法展示检索链路、知识图谱、引用溯源等关键 RAG 能力。用户要求从"黑盒对话"转向"玻璃盒协同"，重新设计前端并扩展后端
     API。

     技术选型：Vanilla JS（单 HTML）+ Sigma.js（图谱）+ PDF.js（文档）+ KaTeX（公式）
     改动范围：仅 server/app.py + server/templates/index.html，不改 raganything/ 或 lightrag/ 核心

     ---
     一、后端 API 设计（server/app.py）

     三层存储

     ┌──────────┬─────────────────────────┬───────────────────────────────────┐
     │    层    │          目录           │               内容                │
     ├──────────┼─────────────────────────┼───────────────────────────────────┤
     │ 资料库   │ uploads/{doc_id}/       │ 原始上传文件（PDF 等）新增        │
     ├──────────┼─────────────────────────┼───────────────────────────────────┤
     │ 解析产物 │ output/{doc_id}/        │ MinerU hybrid_auto/*.md + images/ │
     ├──────────┼─────────────────────────┼───────────────────────────────────┤
     │ 工作空间 │ rag_workspace/{doc_id}/ │ GraphML + 向量数据库              │
     └──────────┴─────────────────────────┴───────────────────────────────────┘

     修改的端点

     POST /ingest：上传文件时同时保存原始文件到 uploads/{doc_id}/{filename}，不再 ingest 完删除。返回增加 filename 字段。

     POST /query：重写为调用 rag.lightrag.aquery_llm() 获取完整结构化数据 + 调用 service.query() 获取答案。响应格式：
     {
       "answer": "...[1][2]...",
       "data": {
         "entities": [{"entity_name", "entity_type", "description"}],
         "relationships": [{"src_id", "tgt_id", "description", "weight"}],
         "chunks": [{"content", "file_path", "chunk_id", "reference_id", "is_multimodal"}],
         "references": [{"reference_id", "file_path"}]
       },
       "metadata": {
         "query_mode": "hybrid",
         "keywords": {"high_level": [], "low_level": []},
         "processing_info": {"total_entities_found", "final_chunks_count", ...}
       }
     }

     实现方式：
     # Step 1: 获取结构化检索数据（不调用 LLM）
     retrieval = await rag.lightrag.aquery_data(query, param=data_param)
     # Step 2: 获取 LLM 答案（走现有 VLM 增强链路）
     answer = await service.query(doc_id, query, **kwargs)
     # Step 3: 合并返回

     GET /workspaces：增加每个 workspace 的统计信息和上传文件列表。

     新增端点

     ┌─────────────────────────────────────┬────────┬───────────────────────────────────────────────┐
     │                端点                 │  方法  │                     用途                      │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /uploads/{doc_id}                   │ GET    │ 列出原始上传文件                              │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /uploads/{doc_id}/{filename}        │ GET    │ 提供 PDF 二进制流（供 PDF.js 加载）           │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /output/{doc_id}/images/{path:path} │ GET    │ 提供解析产物中的图片                          │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /graph/{doc_id}/labels              │ GET    │ 获取所有实体标签                              │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /graph/{doc_id}/subgraph            │ GET    │ 获取子图（?label=X&max_depth=2&max_nodes=50） │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /graph/{doc_id}/stats               │ GET    │ 图谱统计（实体/关系/chunk 数量）              │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /graph/{doc_id}/search              │ GET    │ 搜索实体（?q=keyword&limit=20）               │
     ├─────────────────────────────────────┼────────┼───────────────────────────────────────────────┤
     │ /workspace/{doc_id}                 │ DELETE │ 删除工作空间（三层目录均删除）                │
     └─────────────────────────────────────┴────────┴───────────────────────────────────────────────┘

     图谱端点直接调用 LightRAG 已有 API：
     - rag.lightrag.get_graph_labels() → 标签列表
     - rag.lightrag.get_knowledge_graph(node_label, max_depth, max_nodes) → KnowledgeGraph 对象（含 nodes/edges）

     ---
     二、前端布局（server/templates/index.html）

     整体结构

     +--[Header]------------------------------------------------------------+
     | Logo | API Key | Workspace ▼ | Upload | [严格/创意] | Graph | Theme  |
     +----------------------------------------------------------------------+
     |                    |       |                                         |
     |   PDF Panel        | drag  |   Chat Panel                           |
     |  ┌──────────┐      |       |  ┌─────────────────────────────────────┐|
     |  │ Toolbar   │      |       |  │ Reasoning Trace (collapsible)      │|
     |  │ page/zoom │      |       |  │ "提取关键词..." → "重排序 40 片段…" │|
     |  ├──────────┤      |       |  ├─────────────────────────────────────┤|
     |  │          │      |       |  │                                     │|
     |  │ PDF.js   │      |       |  │  Chat Messages                     │|
     |  │ Viewer   │      |       |  │  (Markdown + KaTeX + 引用卡片)     │|
     |  │          │      |       |  │                                     │|
     |  └──────────┘      |       |  ├─────────────────────────────────────┤|
     |                    |       |  │ Input: [________] [Send]            │|
     |                    |       |  └─────────────────────────────────────┘|
     +----------------------------------------------------------------------+
     | [Graph Panel (底部, 可折叠)] Sigma.js 知识图谱 + 实体详情侧栏        |
     +----------------------------------------------------------------------+

     CDN 库

     ┌───────────────────────────────┬──────────────────┐
     │              库               │       用途       │
     ├───────────────────────────────┼──────────────────┤
     │ marked.js 4.x                 │ Markdown → HTML  │
     ├───────────────────────────────┼──────────────────┤
     │ KaTeX 0.16.x                  │ LaTeX 公式渲染   │
     ├───────────────────────────────┼──────────────────┤
     │ PDF.js 4.x                    │ PDF 渲染         │
     ├───────────────────────────────┼──────────────────┤
     │ sigma 3.x + graphology 0.26.x │ WebGL 图谱可视化 │
     ├───────────────────────────────┼──────────────────┤
     │ highlight.js 11.x             │ 代码高亮         │
     └───────────────────────────────┴──────────────────┘

     JS 模块划分（IIFE 区段）

     ┌──────────────────────┬────────────────────────────────────────────────────────┐
     │         模块         │                          职责                          │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ AppState + AppConfig │ 全局状态和配置                                         │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ API                  │ 所有 fetch 调用集中管理                                │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ PDFViewer            │ PDF.js 加载/渲染/页面跳转/高亮                         │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ Chat                 │ 消息发送/渲染/历史管理                                 │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ ReasoningTrace       │ 检索步骤进度指示器                                     │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ Citations            │ 引用注册/点击跳转/悬浮卡片                             │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ GraphViewer          │ Sigma.js 初始化/数据加载/节点交互                      │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ Renderer             │ marked + KaTeX + hljs + 引用替换 + 表格增强 + 图片灯箱 │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ FileManager          │ 上传/文件列表/workspace 管理                           │
     ├──────────────────────┼────────────────────────────────────────────────────────┤
     │ UI                   │ 主题切换/分割线拖拽/textarea 自适应/模态框             │
     └──────────────────────┴────────────────────────────────────────────────────────┘

     ---
     三、核心交互流程

     查询 + 溯源

     1. 用户输入问题 → POST /query
     2. ReasoningTrace 显示动画步骤："提取关键词..." → "检索知识图谱..." → "重排序 N 个片段..." → "生成回答..."
     3. 响应到达 → Renderer 处理 answer（Markdown → KaTeX → 代码高亮 → 引用链接化）
     4. Citations 注册 references/chunks 映射
     5. 用户点击 [1] → Citations 查找 file_path → PDFViewer.scrollToPage()
     6. 用户 hover [1] → 弹出浮动卡片显示 chunk 原文摘要

     图谱可视化

     1. 查询响应中的 metadata.keywords.low_level 作为种子实体
     2. 调用 /graph/{doc_id}/subgraph?label={keyword} 获取子图
     3. GraphViewer 用 Sigma.js 渲染（force-directed 布局）
     4. 节点颜色按 entity_type 映射（复用 LightRAG WebUI 色表）
     5. 点击节点 → 侧栏显示实体描述；双击 → 展开邻居

     文件上传

     1. 点击 Upload → 文件选择器
     2. POST /ingest（原始文件保存到 uploads/）
     3. 进度指示（ingest 是异步的，可显示 spinner）
     4. 完成后刷新 workspace 列表 + 自动加载 PDF

     ---
     四、实现步骤

     Phase 1：后端 API（app.py）

     ┌──────┬───────────────────────────────────────────────────────────────────────┐
     │ Step │                                 内容                                  │
     ├──────┼───────────────────────────────────────────────────────────────────────┤
     │ 1    │ 添加 UPLOADS_DIR 常量 + 目录创建；修改 /ingest 保存原始文件           │
     ├──────┼───────────────────────────────────────────────────────────────────────┤
     │ 2    │ 添加 /uploads/{doc_id} 和 /uploads/{doc_id}/{filename} 文件服务端点   │
     ├──────┼───────────────────────────────────────────────────────────────────────┤
     │ 3    │ 添加 /output/{doc_id}/images/{path:path} 图片服务端点                 │
     ├──────┼───────────────────────────────────────────────────────────────────────┤
     │ 4    │ 重写 /query：aquery_data() + service.query() 组合，返回完整结构化响应 │
     ├──────┼───────────────────────────────────────────────────────────────────────┤
     │ 5    │ 添加图谱端点：/graph/{doc_id}/labels、/subgraph、/stats、/search      │
     ├──────┼───────────────────────────────────────────────────────────────────────┤
     │ 6    │ 添加 DELETE /workspace/{doc_id} + GET /workspace/{doc_id}/stats       │
     ├──────┼───────────────────────────────────────────────────────────────────────┤
     │ 7    │ 修改 GET /workspaces 增加统计 + 上传文件信息                          │
     └──────┴───────────────────────────────────────────────────────────────────────┘

     Phase 2：前端核心（index.html）

     ┌──────┬─────────────────────────────────────────────────────────────────────────┐
     │ Step │                                  内容                                   │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 8    │ HTML 骨架 + CSS（三栏布局 + 底部图谱面板 + 主题系统）                   │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 9    │ API 模块 + UI 基础（主题/分割线/textarea）                              │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 10   │ PDFViewer 模块（PDF.js 集成、工具栏、页码跳转）                         │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 11   │ Chat + Renderer 模块（消息流 + markdown/KaTeX/hljs/引用/表格/图片渲染） │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 12   │ ReasoningTrace 模块（步骤动画指示器）                                   │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 13   │ Citations 模块（引用注册 + 点击跳转 + hover 卡片）                      │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 14   │ GraphViewer 模块（Sigma.js + graphology + 节点交互）                    │
     ├──────┼─────────────────────────────────────────────────────────────────────────┤
     │ 15   │ FileManager 模块（上传 + workspace 管理 + 删除确认）                    │
     └──────┴─────────────────────────────────────────────────────────────────────────┘

     Phase 3：集成和打磨

     ┌──────┬─────────────────────────────────────────────┐
     │ Step │                    内容                     │
     ├──────┼─────────────────────────────────────────────┤
     │ 16   │ 严格/创意模式切换（映射到不同 query 参数）  │
     ├──────┼─────────────────────────────────────────────┤
     │ 17   │ 图片灯箱（点击放大）+ 表格增强（排序/滚动） │
     ├──────┼─────────────────────────────────────────────┤
     │ 18   │ 健康看板（workspace stats 展示）            │
     ├──────┼─────────────────────────────────────────────┤
     │ 19   │ 端到端验证                                  │
     └──────┴─────────────────────────────────────────────┘

     ---
     五、验证方案

     1. 后端冒烟测试：RAGANYTHING_API_KEY=test uvicorn server.app:app，用 curl 逐个测试新端点
     2. PDF 加载：上传 PDF → 刷新 → 左栏 PDF.js 能否渲染
     3. 查询溯源：发送查询 → 答案含 [1] → 点击 → PDF 跳转到对应页
     4. 图谱可视化：查询后点击 Graph 面板 → Sigma.js 渲染节点/边
     5. KaTeX：查询含公式的文档 → 验证 $...$ 渲染
     6. 主题切换：深色/浅色切换，所有面板一致
     7. workspace 管理：删除 workspace → 确认三层目录均清除

     ---
     六、关键依赖文件（只读参考）

     ┌───────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
     │                     文件                      │                       参考内容                        │
     ├───────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
     │ raganything/services/local_rag.py             │ LocalRagService.get_rag()、query()、ingest() 方法签名 │
     ├───────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
     │ lightrag/lightrag/lightrag.py:2477            │ aquery_data() — 仅返回检索数据不调用 LLM              │
     ├───────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
     │ lightrag/lightrag/lightrag.py:2684            │ aquery_llm() — 完整结构化返回                         │
     ├───────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
     │ lightrag/lightrag/lightrag.py:1059            │ get_graph_labels() — 图谱标签列表                     │
     ├───────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
     │ lightrag/lightrag/lightrag.py:1063            │ get_knowledge_graph() — 子图查询                      │
     ├───────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
     │ lightrag/lightrag/api/routers/graph_routes.py │ 图谱 API 设计参考                                     │
     └───────────────────────────────────────────────┴───────────────────────────────────────────────────────┘

                                                                                                   02:55 PM claude-opus-4-6
● 计划已批准。开始实施。先创建任务列表，然后按 Phase 1 → 2 → 3 顺序执行。

**进度： 已经在\RAG_LUND\rag-anything\server\app.py 的第 365 行附近注入了 LocalRagService = Depends(get_service)。请读取该文件，并继续完成后续的业务逻辑**