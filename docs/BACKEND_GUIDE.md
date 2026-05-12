# Backend 参考手册

## 快速启动

### 前置条件

```bash
# PostgreSQL（governance 层）
# env.example → .env，关键变量：
GOVERNANCE_DATABASE_URL=postgresql://user:pass@localhost:5432/raganything

# 其他必填
LLM_BINDING=openai          # 或 ollama / lmstudio
LLM_MODEL=gpt-4o
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
WORKING_DIR=./rag_storage   # LightRAG 索引根目录
OUTPUT_DIR=./output         # 文档解析产物
```

### 启动服务

```bash
cd rag-anything

# 安装依赖（首次）
pip install -e ".[all]"

# 启动 FastAPI（自动执行 PG migration、注册已有 workspace）
uvicorn server.app:app --host 0.0.0.0 --port 9621 --reload

# 启动 vLLM（本地部署时）
bash start_vllm.sh
```

### 测试

```bash
cd rag-anything
pytest                         # 全量测试
pytest tests/test_governance.py -v   # 仅 governance 集成测试（需 PG）
```

---

## 架构

```
rag-anything/
├── server/app.py              # FastAPI 入口：lifespan + 所有路由
└── raganything/
    ├── governance/            # 数据治理层（PostgreSQL）
    │   ├── db.py              # asyncpg pool + run_migrations()
    │   ├── service.py         # GovernanceService — 唯一触碰 PG 的地方
    │   ├── jobs.py            # JobRunner（asyncio 任务池）
    │   ├── callbacks.py       # IngestProvenanceCallback / JobProgressCallback
    │   ├── models.py          # Pydantic schemas（API 面）
    │   ├── settings.py        # GovernanceSettings（from env）
    │   └── migrations/
    │       └── 001_init.sql   # 5 张表，幂等可重跑
    ├── services/
    │   └── local_rag.py       # LocalRagService（RAGAnything 封装）
    ├── raganything.py         # RAGAnything 主类（QueryMixin + ProcessorMixin + BatchMixin）
    ├── parser.py              # MineruParser / DoclingParser
    ├── modalprocessors.py     # Image / Table / Equation / Generic 处理器
    ├── query.py               # QueryMixin（aquery / aquery_vlm_enhanced）
    ├── processor.py           # ProcessorMixin（文档入库流水线）
    └── constants.py           # 所有默认值（唯一来源）
```

---

## 运行时结构

```
FastAPI 进程（单 uvicorn worker）
│
├── lifespan 启动
│   ├── asyncpg pool  ──→  PostgreSQL（migration → backfill legacy workspaces）
│   ├── LocalRagService（SentenceTransformer / CrossEncoder / vLLM 连接）
│   ├── GovernanceService（封装所有 PG 操作）
│   └── JobRunner（asyncio 任务池，管理并发 ingest）
│
├── app.state（依赖注入）
│   ├── .pg_pool  .rag  .gov  .jobs  .gov_settings
│
└── 路由（薄层，只调 gov.x() / rag.x()）
```

---

## PostgreSQL 表（5 张，001_init.sql）

| 表 | 职责 |
|----|------|
| `workspaces` | workspace 元数据，frozen 标志 |
| `documents` | 文件注册表，file_hash 去重（UNIQUE workspace+hash） |
| `provenance` | chunk / entity / relation → doc_id 映射（per-doc delete 用） |
| `ingest_jobs` | job 状态、进度、错误记录 |
| `ingest_audit` | 追加写 audit log（ingest / delete / freeze / unfreeze） |

---

## 存储分工

| 存储 | 数据 |
|------|------|
| **PostgreSQL** | workspace 元数据、文档注册、provenance、job、audit |
| **Neo4j** | 知识图谱（实体、关系、多跳遍历） |
| **Qdrant** | 稠密 + 稀疏向量（chunk / entity / relation） |
| **文件系统** | `uploads/`、`output/`（解析 MD + 图片）、LightRAG KV JSON |

---

## API 端点速览

```
# 文档管理
GET  /files/{ws}                 已处理文件列表（output/ 中的 .md）
GET  /content/{ws}?filename=     返回解析后 Markdown 内容
GET  /uploads/{ws}/{filename}    原始上传文件（PDF 阅读器用）

# Ingest（job-based，非阻塞）
POST /ingest                     单文件入库 → 返回 job_id
POST /ingest/batch               多文件批量入库
POST /retry/{ws}                 重试失败文件

# Job 监控
GET  /jobs                       列出 job（可按 workspace_id 过滤）
GET  /jobs/{job_id}              单个 job 状态
DELETE /jobs/{job_id}            取消 job

# 查询
POST /query                      同步查询（blocking）
POST /query/stream               SSE 流式查询（meta → chunk… → done）

# 知识图谱（Neo4j）
GET  /graph/{ws}/overview        全图概览（限 max_nodes）
GET  /graph/{ws}/subgraph        种子节点展开子图
GET  /graph/{ws}/search          实体名称搜索
GET  /graph/{ws}/labels          所有实体标签
GET  /graph/{ws}/stats           节点/边统计

# Workspace 治理
GET  /workspaces                 列出所有 workspace
DELETE /workspace/{ws}           删除整个 workspace
PATCH /workspace/{ws}/freeze     冻结（禁止 ingest/delete）
PATCH /workspace/{ws}/unfreeze   解冻
DELETE /workspace/{ws}/document/{doc_id}  per-doc 删除（共享实体保护）
GET  /workspace/{ws}/stats       文档数、chunk 数等
GET  /workspace/{ws}/audit       workspace audit log
GET  /admin/audit                全局 audit log

# 其他
GET  /config                     当前服务配置
```

---

## /query/stream SSE 事件格式

```jsonc
// 1. 检索元数据（第一帧）
{"type": "meta", "data": {...}, "metadata": {"keywords": {...}}}

// 2. LLM token（多帧）
{"type": "chunk", "text": "..."}

// 3. 结束（含引用溯源）
{"type": "done", "graph": null, "source_nodes": [
  {"doc_id": "...", "filename": "paper.pdf", "page_num": 4, "excerpt": "..."}
]}

// 错误
{"type": "error", "text": "..."}
```

---

## ingest 流程（job-based）

```
POST /ingest
  └── GovernanceService.register_document()   # file_hash 去重
  └── JobRunner.submit()                       # 异步 asyncio 任务
        └── GovernanceService.run_ingest()
              ├── LocalRagService.process_file()   # 解析 → 向量化 → 图谱
              ├── IngestProvenanceCallback         # 捕获 chunk/entity/relation → doc_id
              └── GovernanceService.backfill_provenance()   # 写入 provenance 表
```

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| FastAPI lifespan | 连接在 startup 建立、shutdown 释放，无懒初始化泄漏 |
| app.state 依赖注入 | `Depends(get_service)` 从 `request.app.state` 取，多 worker 友好 |
| Job-based ingest | `POST /ingest` 立即返回 job_id，前端轮询 `/jobs/{id}` |
| file_hash 去重 | UNIQUE (workspace_id, file_hash)，重传同文件幂等 |
| per-doc delete | 查 provenance 表，只删该文件独占的 chunk/entity/relation |
| 共享实体保护 | 删除前检查 entity/relation 是否被其他文档引用，引用中则跳过 |
| frozen flag | patch freeze → PG 写标志，ingest/delete 路由入口检查，拒绝写操作 |
| orphan job 恢复 | lifespan 启动时将 running 状态的残留 job 标记为 crashed |
