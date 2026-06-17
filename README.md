# RAG_LUND 部署与同步指南

本仓库是本地化 RAG 实验与服务代码集合，主要由两个子项目组成：

- `lightrag/`：LightRAG 核心库、API Server、WebUI、存储后端适配与示例。
- `rag-anything/`：基于 LightRAG 的本地多模态 RAG 服务，包含 FastAPI 后端、React 前端、文档解析、入库、查询、图谱浏览和评测脚本。

根目录下的 `DocBench/` 和 `docs/` 是评测/内部文档目录。如果需要把代码同步到 GitLab，但不希望包含这两个根级目录，请使用本文后面的 GitLab 导出步骤，不要直接把当前仓库历史原样 push 到 GitLab。

## 目录结构

```text
.
├── .env.example                         # 根级 RAG-Anything 本地运行配置模板
├── lightrag/                            # LightRAG 子项目
├── rag-anything/                        # RAG-Anything 子项目
│   ├── raganything/                     # Python 包
│   ├── server/                          # FastAPI 后端与前端资源
│   │   ├── app.py
│   │   └── frontend/                    # React/Vite 前端源码
│   ├── scripts/                         # 工作区、评测、导出辅助脚本
│   └── examples/
├── DocBench/                            # 根级评测数据/脚本，不导出到 GitLab
├── docs/                                # 根级内部文档，不导出到 GitLab
└── RAGAnything_LocalRAG_VLM_Refactor_2026-02-20.md
```

只排除根级 `DocBench/` 和 `docs/`。子项目内部的 `lightrag/docs/`、`rag-anything/docs/` 属于项目说明文档，应保留。

## Linux 环境准备

推荐使用已有 conda 环境：

```bash
conda activate lightRAG
```

如果是新机器，建议使用 Python 3.10 或更高版本：

```bash
conda create -n lightRAG python=3.10 -y
conda activate lightRAG
python -m pip install -U pip wheel setuptools
```

安装两个子项目的本地可编辑依赖：

```bash
cd /data/h50056787/workspaces/RAG_LUND
pip install -e "./lightrag[api,offline-storage,offline-llm]"
pip install -e "./rag-anything[all,agentic]"
```

如果只跑 `rag-anything` Web 服务，也可以先安装最小依赖：

```bash
pip install -e "./lightrag[api,offline-storage]"
pip install -e "./rag-anything[all]"
```

Office 文档解析需要 LibreOffice：

```bash
sudo apt-get update
sudo apt-get install -y libreoffice
```

离线机器不要随意升级关键运行环境。当前本地 PostgreSQL 数据目录由 PostgreSQL 16 初始化，如果使用本文中的本地 PostgreSQL 数据目录，必须用 PostgreSQL 16 启动。

## 配置文件

根级配置模板在 `.env.example`：

```bash
cd /data/h50056787/workspaces/RAG_LUND
cp .env.example .env
```

启动 `rag-anything` 前加载根级 `.env`：

```bash
set -a
source .env
set +a
```

从 `rag-anything/` 子目录启动时使用：

```bash
set -a
source ../.env
set +a
```

关键配置项：

| 配置项 | 作用 | 示例 |
| --- | --- | --- |
| `NEO4J_URI` | Neo4j Bolt 地址 | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j 用户名 | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j 密码 | `your_neo4j_password` |
| `NEO4J_DATABASE` | Neo4j database | `neo4j` |
| `QDRANT_URL` | Qdrant REST 地址 | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key，本地无认证可留空 | 空 |
| `RAGANYTHING_API_KEY` | WebUI/API 访问密钥 | `your_api_key` |
| `VLLM_API_BASE` | OpenAI-compatible vLLM 地址 | `http://localhost:8001/v1` |
| `VLLM_API_KEY` | vLLM API key | `EMPTY` |
| `LLM_MODEL_NAME` | 文本生成模型名，需和 vLLM served name 一致 | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` |
| `VISION_VLLM_API_BASE` | 视觉模型单独 endpoint，可不设 | `http://localhost:8002/v1` |
| `VISION_MODEL_NAME` | 视觉模型名，可不设 | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` |
| `RAGANYTHING_EMBEDDING_MODEL_PATH` | 本地 embedding 模型路径 | `/path/to/bge-m3` |
| `RAGANYTHING_RERANK_MODEL_PATH` | 本地 reranker 模型路径 | `/path/to/bge-reranker-v2-m3` |
| `VISION_MODEL_PATH` | 本地视觉模型/tokenizer 路径 | `/path/to/vision_model` |
| `TIKTOKEN_CACHE_DIR` | 离线 tiktoken cache | `/path/to/tiktoken_cache` |
| `RAGANYTHING_WORKDIR_ROOT` | LightRAG KV/图谱工作区根目录 | `./rag_workspace` |
| `RAGANYTHING_OUTPUT_DIR` | MinerU/Markdown 输出目录 | `./output` |
| `RAGANYTHING_UPLOADS_DIR` | 上传文件目录 | `./uploads` |
| `RAGANYTHING_LOG_DIR` | 日志目录 | `./logs` |
| `RAGANYTHING_DEVICE` | 推理设备 | `cuda:0` |
| `CHUNKING_STRATEGY` | 切块策略 | `token` |
| `CHUNK_SIZE` | chunk token 长度 | `1200` |
| `CHUNK_OVERLAP_SIZE` | chunk overlap token 长度 | `100` |

可选检索/归一化配置：

```bash
RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION=true
RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION=true
RAGANYTHING_ENTITY_UPPERCASE_ALLOWLIST=LLM,RAG,API,BERT,6G
RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH=true
RAGANYTHING_RECOGNITION_TOP_K=20
RAGANYTHING_RECOGNITION_PROMPT_MAX_TOKENS=65536
RAGANYTHING_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS=8192
RAGANYTHING_RECOGNITION_PROMPT_RESERVED_TOKENS=200
```

## 启动顺序

推荐启动顺序：

```text
PostgreSQL -> Qdrant -> Neo4j -> Phoenix -> vLLM -> RAG-Anything Web
```

PostgreSQL 只在本地部署确实依赖 PG 存储或评测工作区时需要。Qdrant、Neo4j、vLLM 是当前本地 RAG 服务的核心依赖。Phoenix 是可选链路追踪服务。

### 1. PostgreSQL

端口：`5433`

当前数据目录由 PostgreSQL 16 初始化，必须使用 PostgreSQL 16 启动：

```bash
conda activate lightRAG
pg_ctl -D /data/h50056787/workspaces/RAG_LUND/database/pg_data/ start
```

查看状态：

```bash
pg_ctl -D /data/h50056787/workspaces/RAG_LUND/database/pg_data/ status
```

停止：

```bash
pg_ctl -D /data/h50056787/workspaces/RAG_LUND/database/pg_data/ stop
```

如果报 `incompatible with server` 或当前版本是 17，降回 16：

```bash
conda install -n lightRAG -c conda-forge postgresql=16
```

严禁删除 `database/pg_data/` 内容，数据库数据保存在这里。

### 2. Qdrant

端口：`6333`

```bash
cd /data/h50056787
./qdrant
```

Web UI：

```text
http://localhost:6333/dashboard
```

如果使用远程或 Docker Qdrant，只需要保证 `.env` 中的 `QDRANT_URL` 指向正确地址。

### 3. Neo4j

端口：`7474` Web UI，`7687` Bolt

系统默认 Java 11 不支持当前 Neo4j，需要指定 Java 21：

```bash
cd /data/h50056787/neo4j-community-2026.02.3/bin
JAVA_HOME=/data/h50056787/jdk-21.0.10+7 ./neo4j start
```

查看状态：

```bash
JAVA_HOME=/data/h50056787/jdk-21.0.10+7 ./neo4j status
```

停止：

```bash
JAVA_HOME=/data/h50056787/jdk-21.0.10+7 ./neo4j stop
```

Web UI：

```text
http://localhost:7474
```

### 4. Phoenix

Phoenix 用于链路追踪，可选。注意子命令是 `serve`：

```bash
conda activate lightRAG
python -m phoenix.server.main serve
```

如果不做 tracing，可以不启动 Phoenix。

### 5. vLLM

`rag-anything` 通过 OpenAI-compatible API 调用本地 vLLM。默认配置：

```bash
VLLM_API_BASE=http://localhost:8001/v1
VLLM_API_KEY=EMPTY
```

仓库里已有模型启动脚本，可按当前机器模型路径调整后运行：

```bash
cd /data/h50056787/workspaces/RAG_LUND/rag-anything
bash start_server_qwen3_vl.sh
```

也可以直接启动：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --served-model-name "your_model_name" \
  --trust-remote-code \
  --port 8001 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 65536
```

检查 vLLM 是否可用：

```bash
curl http://localhost:8001/v1/models
```

`LLM_MODEL_NAME` 和 `VISION_MODEL_NAME` 必须与 vLLM 的 `--served-model-name` 对齐，否则会出现模型不存在或请求失败。

### 6. RAG-Anything Web 服务

端口：`9621`

首次部署或前端代码更新后，先构建前端：

```bash
conda activate lightRAG
cd /data/h50056787/workspaces/RAG_LUND/rag-anything/server/frontend
npm ci
npm run build
```

启动后端：

```bash
conda activate lightRAG
cd /data/h50056787/workspaces/RAG_LUND/rag-anything
set -a
source ../.env
set +a
uvicorn server.app:app --host 0.0.0.0 --port 9621
```

访问：

```text
http://localhost:9621
```

开发调试时可以加 `--reload`：

```bash
uvicorn server.app:app --host 0.0.0.0 --port 9621 --reload
```

常规运行不要加 `--reload`。

## CLI 入库

`rag-anything` 支持不经过 WebUI 的命令行入库：

```bash
conda activate lightRAG
cd /data/h50056787/workspaces/RAG_LUND/rag-anything
set -a
source ../.env
set +a
python raganything/services/local_rag.py \
  --path /path/to/file-or-folder \
  --id workspace_name \
  --max-async-ingest 2
```

简写：

```bash
python raganything/services/local_rag.py -p /path/to/file-or-folder -i workspace_name
```

输出位置由 `.env` 控制：

- `RAGANYTHING_WORKDIR_ROOT`：图谱、KV、向量索引工作区。
- `RAGANYTHING_OUTPUT_DIR`：解析后的 Markdown、图片等输出。
- `RAGANYTHING_UPLOADS_DIR`：WebUI 上传文件目录。

如果入库文件夹，脚本会按支持的文件类型筛选顶层文件，并按 `--max-async-ingest` 分批并发入库。

## Web API

WebUI 使用同一个 FastAPI 服务。常用端点：

| Method | Path | 说明 |
| --- | --- | --- |
| `GET` | `/workspaces` | 列出工作区 |
| `POST` | `/ingest` | 上传并处理单个文档 |
| `POST` | `/ingest/batch` | 批量上传并处理 |
| `POST` | `/query` | 文本查询 |
| `POST` | `/query/multimodal` | 多模态查询 |
| `GET` | `/files/{workspace_id}` | 工作区文件列表 |
| `GET` | `/uploads/{workspace_id}` | 上传文件列表 |
| `GET` | `/graph/{workspace_id}/labels` | 图谱标签 |
| `GET` | `/graph/{workspace_id}/subgraph` | 子图 |
| `GET` | `/graph/{workspace_id}/stats` | 图谱统计 |
| `GET` | `/graph/{workspace_id}/search` | 图谱实体搜索 |
| `GET` | `/workspace/{workspace_id}/documents` | 文档状态 |
| `GET` | `/workspace/{workspace_id}/stats` | 工作区统计 |
| `DELETE` | `/workspace/{workspace_id}` | 删除工作区 |
| `GET` | `/config` | 服务端默认查询配置 |

如果设置了 `RAGANYTHING_API_KEY`，请求需要带 `X-API-Key`：

```bash
curl -H "X-API-Key: your_api_key" http://localhost:9621/workspaces
```

## LightRAG standalone

如果只启动 LightRAG 自带 API/WebUI：

```bash
cd /data/h50056787/workspaces/RAG_LUND/lightrag
cp env.example .env
```

编辑 `lightrag/.env`，重点配置：

- `HOST`、`PORT`
- `LIGHTRAG_API_KEY`
- `LLM_BINDING`、`LLM_MODEL`、`LLM_BINDING_HOST`、`LLM_BINDING_API_KEY`
- `EMBEDDING_BINDING`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_BINDING_HOST`
- `LIGHTRAG_KV_STORAGE`
- `LIGHTRAG_DOC_STATUS_STORAGE`
- `LIGHTRAG_GRAPH_STORAGE`
- `LIGHTRAG_VECTOR_STORAGE`
- Neo4j、Qdrant、Redis、PostgreSQL 等后端连接参数

启动：

```bash
conda activate lightRAG
cd /data/h50056787/workspaces/RAG_LUND/lightrag
lightrag-server
```

Docker Compose 方式：

```bash
cd /data/h50056787/workspaces/RAG_LUND/lightrag
cp env.example .env
docker compose up
```

更多 LightRAG 原生用法见 `lightrag/README.md`。

## 预构建工作区与外部解析结果

如果需要挂载已有 DocBench 工作区，可以复制工作区数据到当前运行目录：

```bash
cd /data/h50056787/workspaces/RAG_LUND/rag-anything
cp -r /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421/_workspace_cache/docbench_shared/v0_v1_v2/rag_workspaces/docbench_shared_graphbm25_20260421_v0_v1_v2 ./rag_data/rag_workspace/
chmod -R 755 ./rag_data/rag_workspace/docbench_shared_graphbm25_20260421_v0_v1_v2
```

把外部 PDF 和 MinerU 解析结果链接到工作区：

```bash
python scripts/link_external_workspace.py \
  --workspace-id docbench_shared_graphbm25_20260421_v0_v1_v2 \
  --docbench-root /data/y50056788/Yaliang/datasets_for_eval/data_for_DocBench \
  --mineru-root /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/DocBench/docbench_shared_results/mineru_outputs \
  --dry-run
```

确认 dry run 输出正确后去掉 `--dry-run`。

Multi-hop 工作区示例：

```text
/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0/2wiki/2wiki_hr2_v0
/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0/musique/musique_hr2_v0
```

## 常见问题

### PostgreSQL 报版本不兼容

确认 `pg_ctl --version` 是 PostgreSQL 16。不是 16 时执行：

```bash
conda install -n lightRAG -c conda-forge postgresql=16
```

### Neo4j 启动失败

确认使用 Java 21：

```bash
JAVA_HOME=/data/h50056787/jdk-21.0.10+7 ./neo4j status
```

### WebUI 空白或前端资源找不到

重新构建前端：

```bash
cd /data/h50056787/workspaces/RAG_LUND/rag-anything/server/frontend
npm ci
npm run build
```

然后重启 `uvicorn`。

### vLLM 请求失败

检查三件事：

```bash
curl http://localhost:8001/v1/models
echo "$VLLM_API_BASE"
echo "$LLM_MODEL_NAME"
```

`LLM_MODEL_NAME` 必须等于 `/v1/models` 返回的模型名，或等于启动 vLLM 时的 `--served-model-name`。

### Qdrant 或 Neo4j 数据没有命中

确认 `.env` 中的连接配置与实际服务一致：

```bash
echo "$QDRANT_URL"
echo "$NEO4J_URI"
echo "$NEO4J_USERNAME"
echo "$NEO4J_DATABASE"
```

如果切换 embedding 模型或 embedding 维度，旧向量索引通常不能复用，需要新建工作区或清理对应工作区数据。

## Git 工作流

日常更新当前仓库：

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
```

如果本地有未提交改动，不要直接 `git pull`。先提交、stash，或使用单独 worktree：

```bash
git fetch origin
git worktree add -b docs/readme-linux-gitlab ../projects-readme origin/main
cd ../projects-readme
```

提交文档修改：

```bash
git add README.md .gitignore
git commit -m "docs: add Linux deployment and GitLab export guide"
git push origin HEAD:main
```

## 在 Linux 上导出到 GitLab

目标：只把 `lightrag/`、`rag-anything/` 和其他必要根级文件同步到 GitLab，不包含根级 `DocBench/` 和 `docs/`。

重要：根级 `DocBench/` 和 `docs/` 目前在源仓库历史里已经被跟踪。直接执行 `git remote add gitlab ... && git push gitlab main` 会把历史和这两个目录一起推过去。推荐用 `git archive` 导出干净快照，再初始化 GitLab 仓库。

### 新建 GitLab 仓库的推荐做法

```bash
git clone https://github.com/Yaliang-Cai/RAG_LUND.git rag-lund
cd rag-lund
git checkout main
git pull --ff-only origin main
```

导出不含根级 `DocBench/`、`docs/`、`LightRAG-Qwen3VL-Local/` 的快照：

```bash
mkdir -p ../rag-lund-gitlab
git archive --format=tar HEAD -- \
  . \
  ':(exclude)DocBench' ':(exclude)DocBench/**' \
  ':(exclude)docs' ':(exclude)docs/**' \
  ':(exclude)LightRAG-Qwen3VL-Local' ':(exclude)LightRAG-Qwen3VL-Local/**' \
  | tar -x -C ../rag-lund-gitlab
```

初始化并推送到 GitLab：

```bash
cd ../rag-lund-gitlab
git init -b main
git add .
git status --short
git commit -m "Initial GitLab export"
git remote add origin <GITLAB_REPO_URL>
git push -u origin main
```

验证排除成功：

```bash
test ! -e DocBench
test ! -e docs
test ! -e LightRAG-Qwen3VL-Local
git ls-files | grep -E '^(DocBench|docs|LightRAG-Qwen3VL-Local)(/|$)' && exit 1 || echo "GitLab export scope OK"
```

### 同步到已存在的 GitLab 仓库

如果 GitLab 仓库已经存在，建议先 clone GitLab 仓库，再用导出的快照覆盖工作区：

```bash
git clone <GITLAB_REPO_URL> rag-lund-gitlab
cd rag-lund-gitlab
git checkout main
```

从 GitHub 源仓库导出快照到临时目录：

```bash
cd ..
git clone https://github.com/Yaliang-Cai/RAG_LUND.git rag-lund-source
cd rag-lund-source
git checkout main
git pull --ff-only origin main
mkdir -p ../rag-lund-export
git archive --format=tar HEAD -- \
  . \
  ':(exclude)DocBench' ':(exclude)DocBench/**' \
  ':(exclude)docs' ':(exclude)docs/**' \
  ':(exclude)LightRAG-Qwen3VL-Local' ':(exclude)LightRAG-Qwen3VL-Local/**' \
  | tar -x -C ../rag-lund-export
```

覆盖 GitLab 工作区并推送：

```bash
cd ../rag-lund-gitlab
rsync -a --delete --exclude='.git' ../rag-lund-export/ ./
git add -A
git status --short
git commit -m "Sync from RAG_LUND main"
git push origin main
```

如果 `git status --short` 没有输出，说明这次没有需要同步的变化。

## 不要提交的内容

根级 `.gitignore` 已覆盖常见本地文件、构建产物和运行数据。尤其不要提交：

- `.env`
- `.env.local`
- `.claude/`
- `.superpowers/`
- `.venv/`、`venv/`
- `database/`
- `query_cache/`
- `uploads/`
- `logs/`
- `rag-anything/server/frontend/node_modules/`
- `rag-anything/server/static/dist/`
- 根级 `DocBench/`
- 根级 `docs/`
- `LightRAG-Qwen3VL-Local/`
