# Qdrant 向量数据库配置指南

本项目使用 **Qdrant** 作为向量数据库后端（替代 Milvus Lite）。Qdrant 以单文件二进制方式运行，无需 Docker，数据持久化到本地磁盘。

---

## 一、下载并启动 Qdrant

### Windows

```powershell
# 下载最新版本（根据实际版本号调整 URL）
# 前往 https://github.com/qdrant/qdrant/releases 获取最新版本
Invoke-WebRequest -Uri "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip" `
  -OutFile qdrant.zip
Expand-Archive qdrant.zip -DestinationPath qdrant-bin

# 启动（默认监听 6333，数据写入 ./storage/）
.\qdrant-bin\qdrant.exe
```

### Linux / macOS

```bash
# Linux x86_64
curl -LO https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz
tar xzf qdrant-x86_64-unknown-linux-musl.tar.gz

# macOS (Apple Silicon)
curl -LO https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-apple-darwin.tar.gz
tar xzf qdrant-aarch64-apple-darwin.tar.gz

# 启动
./qdrant
```

启动后终端会输出：

```
...
INFO  qdrant::actix  > Actix Web server started at 0.0.0.0:6333
```

Web 控制台：http://localhost:6333/dashboard

---

## 二、配置环境变量

在项目根目录的 `.env` 文件中（或直接 export 到 shell）添加以下内容：

```env
# ── Qdrant 向量数据库 ──────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=          # 本地无认证时留空或删除此行

# ── Neo4j 图数据库（保持不变）──────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

> **无需**设置 `LIGHTRAG_VECTOR_STORAGE`，`local_rag.py` 已在代码层面硬编码为 `QdrantVectorDBStorage`。

---

## 三、安装 Python 依赖

```bash
pip install qdrant-client
```

> `qdrant-client` 也会在首次初始化时由 LightRAG 通过 `pipmaster` 自动安装，手动安装可跳过这一延迟。

---

## 四、数据存储位置

| 数据类型 | 位置 |
|---|---|
| Qdrant 向量数据 | `./storage/`（Qdrant 二进制所在目录下，可通过 `--storage-snapshots-path` 等参数自定义） |
| Neo4j 图数据 | Neo4j 数据目录（由 Neo4j 自身管理） |
| 解析产物 Markdown | `rag-anything/output/{workspace_id}/` |
| 原始上传文件 | `rag-anything/uploads/{workspace_id}/` |
| LightRAG KV 缓存 | `rag-anything/rag_workspace/{workspace_id}/` |

Qdrant 使用**集合（collection）**隔离不同 namespace，用 `workspace_id` 做 payload 过滤实现多工作空间共享同一 Qdrant 实例。集合命名格式为：

```
lightrag_vdb_chunks_{model_suffix}
lightrag_vdb_entities_{model_suffix}
lightrag_vdb_relationships_{model_suffix}
```

BM25 sparse indexing is controlled by `DEFAULT_QDRANT_ENABLE_SPARSE_BM25`
in `raganything/constants.py`. When it is `True`, Qdrant collection names use
the `_bm25` suffix, for example:

```
lightrag_vdb_chunks_{model_suffix}_bm25
lightrag_vdb_entities_{model_suffix}_bm25
lightrag_vdb_relationships_{model_suffix}_bm25
```

Set `DEFAULT_QDRANT_ENABLE_SPARSE_BM25 = False` before starting the process
when querying or extending older dense-only Qdrant collections. RAG-Anything
entry points write the corresponding LightRAG/Qdrant environment setting from
this constant at startup, so changing the constant and restarting the process is
the normal switch path.

---

## 五、迁移已有 Milvus 工作空间

切换向量库后，原 Milvus 数据**不会自动迁移**。已有工作空间需要重新入库：

### 方式 A：通过 Web UI

1. 启动服务 `uvicorn server.app:app --host 0.0.0.0 --port 9621`
2. 在工作空间列表选择需要迁移的工作空间
3. 点击 **Retry** 按钮，服务器会从 `uploads/{workspace_id}/` 重新处理所有已上传文件

### 方式 B：通过 API

```bash
curl -X POST http://localhost:9621/retry/{workspace_id}
```

### 方式 C：通过 CLI

```bash
python raganything/services/local_rag.py \
  -p ./path/to/original/files/ \
  -i {workspace_id}
```

> 如果原始文件已在 `rag-anything/uploads/{workspace_id}/`，用方式 A 或 B 最方便。

---

## 六、验证 Qdrant 工作正常

入库完成后，访问 Qdrant 控制台确认集合已创建：

```
http://localhost:6333/dashboard
```

或通过 API：

```bash
curl http://localhost:6333/collections
```

应返回类似：

```json
{
  "result": {
    "collections": [
      {"name": "lightrag_vdb_chunks_bge-m3"},
      {"name": "lightrag_vdb_entities_bge-m3"},
      {"name": "lightrag_vdb_relationships_bge-m3"}
    ]
  }
}
```

---

## 七、自定义 Qdrant 数据目录（可选）

默认情况下 Qdrant 数据写在二进制所在目录的 `./storage/`。可通过配置文件指定路径：

创建 `config.yaml`（与 Qdrant 二进制同目录）：

```yaml
storage:
  storage_path: /data/qdrant/storage
```

然后启动时指定：

```bash
./qdrant --config-path config.yaml
```

---

## 八、常见问题

**Q: 启动时报 `Connection refused` 或 `QDRANT_URL not set`**

确认 Qdrant 进程正在运行，且 `QDRANT_URL` 已正确设置。

```bash
curl http://localhost:6333/healthz   # 应返回 {"title": "qdrant - vector search engine", ...}
```

**Q: 切换后查询返回空结果**

向量数据在新 Qdrant 实例中为空，需按第五节重新入库。

**Q: 多个工作空间会互相干扰吗**

不会。Qdrant 通过 `workspace_id` payload 字段过滤，不同工作空间的向量严格隔离。

**Q: 能否同时运行多个 Qdrant 实例**

单实例即可支持所有工作空间。不需要也不建议多实例。
