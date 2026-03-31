# Neo4j 图数据库配置指南

本项目使用 **Neo4j** 作为知识图谱存储后端（`Neo4JStorage`）。Neo4j 负责存储实体节点、实体关系，以及支持图遍历查询（local/global/hybrid 模式）。

---

## 一、安装并启动 Neo4j

### 推荐：Neo4j Community Edition（本地免费）

**Windows / macOS / Linux — 通用方式**

1. 前往 https://neo4j.com/download-center/#community 下载 Neo4j Community Server
2. 解压后进入目录，启动：

```bash
# Linux / macOS
bin/neo4j start

# Windows (PowerShell)
bin\neo4j.bat start
```

3. 访问 http://localhost:7474，使用默认账号 `neo4j / neo4j` 登录，**首次登录必须修改密码**。

### 或者用 Docker（更方便）

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme \
  neo4j:5.26-community
```

> **版本要求**：Neo4j 5.x（推荐）或 4.4.x。Community Edition 完全够用。

---

## 二、配置环境变量

在 `D:\HUAWEI\RAG_LUND\.env` 中（已预填，按实际修改密码）：

```env
# ── Neo4j ──────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme          # ← 改成你的实际密码
NEO4J_DATABASE=neo4j             # Community Edition 固定为 neo4j，不要改
```

**Community Edition 重要限制**：不支持 `CREATE DATABASE`，因此必须显式设置 `NEO4J_DATABASE=neo4j`，否则 Neo4JStorage 会尝试创建数据库并失败后回退，产生警告日志。

### 可选调优参数（已有默认值，一般无需修改）

```env
NEO4J_MAX_CONNECTION_POOL_SIZE=100
NEO4J_CONNECTION_TIMEOUT=30
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=30
NEO4J_MAX_TRANSACTION_RETRY_TIME=30
NEO4J_MAX_CONNECTION_LIFETIME=300
NEO4J_LIVENESS_CHECK_TIMEOUT=30
NEO4J_KEEP_ALIVE=true
```

---

## 三、多工作空间数据隔离

本项目通过向 LightRAG 传入 `workspace=workspace_id` 实现隔离。每个工作空间在 Neo4j 中使用独立的**节点标签**（Node Label），例如：

| 工作空间 | Neo4j 节点标签 |
|---|---|
| `My_Paper` | `` `My_Paper` `` |
| `Project_A` | `` `Project_A` `` |

查询时自动按标签过滤，不同工作空间的数据完全隔离，无需多个数据库。

> **注意**：`NEO4J_WORKSPACE` 环境变量可强制所有工作空间使用同一标签（兼容旧版设计），正常使用时**不要设置**此变量，否则多工作空间数据会混合。

---

## 四、Neo4j 自动完成的初始化

首次向工作空间入库时，`Neo4JStorage.initialize()` 自动执行：

1. **B-Tree 索引**：在 `entity_id` 属性上创建索引，加速节点查找
   ```cypher
   CREATE INDEX IF NOT EXISTS FOR (n:`{workspace_id}`) ON (n.entity_id)
   ```

2. **全文检索索引**：用于实体名称模糊搜索
   - 支持中文（CJK 分析器），自动检测并回退到标准分析器
   - 索引名：`entity_id_fulltext_idx_{workspace_id}`

无需手动建库或建表。

---

## 五、验证 Neo4j 工作正常

入库完成后，在 Neo4j Browser（http://localhost:7474）中执行：

```cypher
// 查看所有工作空间标签
CALL db.labels()

// 查看某工作空间的节点数量（替换 My_Paper 为实际 workspace_id）
MATCH (n:`My_Paper`) RETURN count(n)

// 查看实体样本
MATCH (n:`My_Paper`) RETURN n LIMIT 10

// 查看关系样本
MATCH (a:`My_Paper`)-[r]->(b:`My_Paper`) RETURN a, r, b LIMIT 20
```

---

## 六、删除工作空间数据

通过 Web UI 删除工作空间时，Neo4j 中对应标签的所有节点和关系会被自动清理。

手动清理（替换 `My_Paper` 为实际 workspace_id）：

```cypher
MATCH (n:`My_Paper`)
DETACH DELETE n
```

---

## 七、常见问题

**Q: 启动时报 `ServiceUnavailable` 或连接超时**

确认 Neo4j 已启动且 `bolt://localhost:7687` 可达：
```bash
curl http://localhost:7474
```

**Q: 日志出现 `does not support creating databases` 警告**

你没有设置 `NEO4J_DATABASE=neo4j`，或者用的是 Community Edition 但尝试了自定义数据库名。在 `.env` 中明确加上 `NEO4J_DATABASE=neo4j`。

**Q: 切换 Bolt 端口**

```env
NEO4J_URI=bolt://localhost:7688   # 自定义端口
```

**Q: 使用 Neo4j AuraDB（云端）**

```env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-aura-password
NEO4J_DATABASE=neo4j
```

**Q: 中文实体搜索效果差**

检查 Neo4j 日志，确认全文索引使用了 `cjk` 分析器。如果 Neo4j 版本较旧（< 4.4），CJK 分析器可能不支持，会自动回退到标准分析器，中文分词精度略降，但功能正常。

**Q: `NEO4J_WORKSPACE` 应该设置吗**

**不要设置**。该变量仅用于兼容旧版单工作空间部署。设置后所有工作空间将共享同一节点标签，导致数据混合。
