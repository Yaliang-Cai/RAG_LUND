# Server 模块说明

本目录包含 RAGAnything 的 HTTP 服务层，基于 FastAPI 实现。

## 文件结构

```
server/
├── app.py              # FastAPI 应用主体，所有路由与业务逻辑
├── download_static.py  # 离线静态资源下载脚本（一次性运行）
├── static/             # 离线静态资源（运行 download_static.py 后生成）
│   ├── marked.min.js
│   ├── katex/
│   └── hljs/
└── templates/
    └── index.html      # WebUI 单页应用
```

---

## app.py 架构

### 实例生命周期

服务层只有 **两层单例**，不存在重复初始化：

```
FastAPI 进程
  └── _service: LocalRagService          # 模块级全局变量，整个进程唯一
        └── _rag_instances: Dict[str, RAGAnything]   # 按 workspace_id 缓存
              └── rag.lightrag: LightRAG             # 每个 RAGAnything 内部唯一
```

- `get_service()`（第 133 行）：FastAPI 依赖注入函数，首次调用时从环境变量读取 `LocalRagSettings` 并创建 `LocalRagService`，之后全部复用同一实例。
- `service.get_rag(workspace_id)`（`local_rag.py`）：在 `asyncio.Lock` 保护下查缓存，同一 `workspace_id` 只创建一次 `RAGAnything`。
- `RAGAnything._ensure_lightrag_initialized()`：内部 LightRAG 实例懒初始化，有 `if self.lightrag is not None` 守卫，幂等可重复调用。

### 在线 / 离线模式自动检测

启动时检查三个哨兵文件是否存在（第 46-51 行）：

```python
_USE_LOCAL_STATIC = all([
    (static_dir / "marked.min.js").exists(),
    (static_dir / "katex" / "katex.min.js").exists(),
    (static_dir / "hljs" / "highlight.min.js").exists(),
])
```

存在则 WebUI 使用本地文件，否则从 CDN 加载。切换方式：运行 `download_static.py`，重启服务自动生效。

### API 鉴权

所有受保护路由依赖 `verify_api_key`（`X-Api-Key` 请求头）。若环境变量 `RAGANYTHING_API_KEY` 未设置，则鉴权直接放行（开发模式）。

图谱 HTML 端点额外支持 `verify_api_key_or_query`，即也接受 `?key=` 查询参数，用于 `<iframe>` 嵌入场景。

---

## API 端点一览

### 文档内容

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | WebUI 主页 |
| `GET` | `/files/{workspace_id}` | 列出该工作空间下解析产物 `.md` 文件 |
| `GET` | `/content/{workspace_id}` | 读取 Markdown 内容（可选 `?filename=` 参数） |

### 文件上传与入库

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/ingest` | 上传文件并触发 RAG 入库（multipart/form-data） |
| `POST` | `/retry/{workspace_id}` | 用已上传文件重新触发入库（后台任务） |
| `GET` | `/uploads/{workspace_id}` | 列出已上传的原始文件 |
| `GET` | `/uploads/{workspace_id}/{filename}` | 下载已上传原始文件 |
| `GET` | `/output/{workspace_id}/images/{path}` | 获取解析产物图片 |

`/ingest` 的文件存储路径：
- `uploads/{workspace_id}/`：原始上传文件（永久保留，供 `/retry` 使用）
- `output/{workspace_id}/`：MinerU/Docling 解析产物（Markdown、图片）
- `working_dir_root/{workspace_id}/`：LightRAG 索引（知识图谱、向量库）

### 查询

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/query` | 一次性查询，返回 `{answer, data, metadata, graph}` |
| `POST` | `/query/stream` | SSE 流式查询，先发 `meta` 事件，再逐 token 发 `chunk`，最后发 `done` |

两个端点均支持相同的 `QueryRequest` 参数：

```json
{
  "workspace_id": "MyGraph",
  "query": "什么是...",
  "mode": "hybrid",
  "top_k": 60,
  "chunk_top_k": 10,
  "enable_rerank": true,
  "vlm_enhanced": false,
  "return_graph": false,
  "graph_max_depth": 2,
  "graph_max_nodes": 50
}
```

查询执行顺序（`/query`）：
1. `rag.lightrag.aquery_data()` — 纯检索，不调用 LLM，获取结构化数据
2. `service.query()` — 走完整 RAG 链路（含 VLM 增强可选分支）
3. `_get_query_subgraph()` — 可选，从关键词中提取子图（`return_graph=true` 时执行）

### 知识图谱

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/graph/{workspace_id}/labels` | 获取所有实体标签（优先 LightRAG API，退回 NetworkX GraphML）|
| `GET` | `/graph/{workspace_id}/subgraph` | 以指定节点为中心展开子图（`?label=&max_depth=&max_nodes=`）|
| `GET` | `/graph/{workspace_id}/stats` | 实体数、关系数、文件大小 |
| `GET` | `/graph/{workspace_id}/search` | 实体名称模糊搜索（`?q=&limit=`）|
| `GET` | `/graph/{workspace_id}/overview` | 按度数取 Top-N 节点的概览子图 |
| `GET` | `/graph/{workspace_id}/html` | 生成 pyvis 自包含 HTML 可视化（`?q=&theme=&max_nodes=`）|

图谱端点有两层退路策略：优先调用 LightRAG 的异步 API（需要初始化存储），失败则退回 NetworkX 直接读 `graph_chunk_entity_relation.graphml`，确保图谱可视化不依赖完整的 RAG 初始化。

### 工作空间管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `DELETE` | `/workspace/{workspace_id}` | 删除三层目录（uploads/output/workspace）并清除内存缓存 |
| `GET` | `/workspace/{workspace_id}/stats` | 文件数、实体数、关系数、chunk 数、磁盘用量 |
| `GET` | `/workspaces` | 列出所有工作空间及其状态 |

### 配置

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/config` | 返回 `constants.py` 中的查询默认参数，供 WebUI 初始化使用 |

---

## download_static.py

一次性脚本，将 WebUI 所需的前端资源下载到 `server/static/`：

- `marked.js` — Markdown 渲染
- `KaTeX`（含字体文件）— 数学公式渲染
- `highlight.js` — 代码块高亮

用法：

```bash
# 联网机器上运行一次
python server/download_static.py

# 强制重新下载所有文件
python server/download_static.py --force
```

下载完成后重启服务，`app.py` 会自动检测并切换到离线模式。

---

## 目录布局（运行时）

```
./
├── uploads/
│   └── {workspace_id}/           ← 原始上传文件
├── output/           (RAGANYTHING_OUTPUT_DIR)
│   └── {workspace_id}/
│       └── hybrid_auto/    ← 解析产物 Markdown 与图片
└── rag_storage/      (RAGANYTHING_WORKDIR_ROOT)
    └── {workspace_id}/           ← LightRAG 索引（向量库、知识图谱、KV 缓存）
        └── graph_chunk_entity_relation.graphml
```

三层目录均以 `workspace_id` 为子目录名隔离，`DELETE /workspace/{workspace_id}` 会同时清除全部三层。
