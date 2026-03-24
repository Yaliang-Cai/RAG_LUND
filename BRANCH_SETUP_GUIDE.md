# neo4j-milvus Branch 配置指南

本 branch 相比 main 的主要区别：

## 核心差异

| 特性 | main | neo4j-milvus |
|------|------|--------------|
| 图数据库后端 | NetworkX（内存） | Neo4j 或 NetworkX |
| 向量数据库 | Faiss（本地文件） | Milvus 或 Faiss |
| 同义词链接（V2） | ❌ | ✅ |
| PPR 多跳推理（V3） | ❌ | ✅ |

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

# 如果使用 Neo4j，需要额外安装
pip install neo4j

# 如果使用 Milvus，需要额外安装
pip install pymilvus
```

---

## 配置 1：知识图谱后端（Neo4j vs NetworkX）

### 方案 A：NetworkX（推荐默认）

无需额外配置，LightRAG 自动使用内存 NetworkX。

```python
from lightrag import LightRAG

rag = LightRAG(working_dir="./rag_storage")  # 默认使用 NetworkX
```

### 方案 B：Neo4j

**前提：** Neo4j 服务已启动

```bash
# 启动 Neo4j（本地或远程）
# 例如：docker run -d -p 7687:7687 -p 7474:7474 neo4j

# 离线机上，修改 constants.py 或通过环境变量配置
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

**代码配置：**

```python
from lightrag import LightRAG
from lightrag.kg import Neo4JStorage

rag = LightRAG(
    working_dir="./rag_storage",
    graph_storage_cls=Neo4JStorage,
)
```

---

## 配置 2：向量数据库（Milvus vs Faiss）

### 方案 A：Milvus Lite（推荐离线）

单文件数据库，无需启动服务。

```bash
# 只需安装 pymilvus
pip install pymilvus
```

**代码配置：**

```python
from lightrag import LightRAG
from lightrag.kg import MilvusVectorDBStorage

rag = LightRAG(
    working_dir="./rag_storage",
    vector_storage_cls=MilvusVectorDBStorage,
    vector_storage_kwargs={
        "milvus_db_uri": "sqlite:///./milvus.db"  # 本地单文件
    }
)
```

### 方案 B：Milvus Server

连接到已启动的 Milvus 服务。

```bash
# 启动 Milvus（Docker）
docker run -d -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

**代码配置：**

```python
from lightrag import LightRAG
from lightrag.kg import MilvusVectorDBStorage

rag = LightRAG(
    working_dir="./rag_storage",
    vector_storage_cls=MilvusVectorDBStorage,
    vector_storage_kwargs={
        "milvus_host": "localhost",
        "milvus_port": 19530,
    }
)
```

### 方案 C：Faiss（默认，如 main branch）

本地向量索引文件。

```python
from lightrag import LightRAG

rag = LightRAG(working_dir="./rag_storage")  # 默认使用 Faiss
```

---

## 配置 3：V2 同义词链接

### 启用 V2

编辑 `rag-anything/raganything/constants.py`：

```python
# 行 ~139-158
DEFAULT_ENABLE_SYNONYM_LINKING = True  # 改为 True
DEFAULT_SYNONYMY_THRESHOLD = 0.8       # embedding 相似度阈值
DEFAULT_SYNONYMY_TOPK = 100            # KNN top-K
DEFAULT_SYNONYMY_MIN_ENTITY_LEN = 2    # 过滤短实体
```

或通过环境变量覆盖：

```bash
export ENABLE_SYNONYM_LINKING=true
export SYNONYMY_THRESHOLD=0.8
```

### V2 效果

- **索引阶段**：自动检测同义词实体对，建立 SYNONYM 关系边
- **查询阶段**：图遍历时会利用 SYNONYM 边扩展实体集合，增加召回

---

## 配置 4：V3 PPR 多跳推理

### 启用 V3

V3 **不在** constants.py，而是在查询时通过 `QueryParam` 传入：

```python
from lightrag.base import QueryParam

# 构造查询参数
param = QueryParam(
    mode="mix",                  # 查询模式
    enable_multi_hop=True,       # ⭐ V3 启用
    multi_hop_depth=2,           # PPR 深度
    ppr_top_k=50,                # PPR 返回的 chunk 数
)

# 查询
result = await rag.aquery(
    query="你的问题",
    param=param
)
```

### V3 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `enable_multi_hop` | False | 启用 PPR 多跳 |
| `multi_hop_depth` | 2 | 图的搜索深度 |
| `ppr_top_k` | 50 | PPR 返回的 chunk 数量 |
| `ppr_damping` | 0.5 | PPR damping 因子（自定义需修改源码） |

### V3 效果

- 对实体进行 Personalized PageRank
- 考虑图的多跳结构，不只是直接邻居
- chunk 按 PPR 分数排序，优先返回重要片段

---

## 完整配置示例

### 场景 1：离线机器（推荐）

```python
from lightrag import LightRAG
from lightrag.kg import MilvusVectorDBStorage
from lightrag.base import QueryParam

# 创建 RAG 实例
rag = LightRAG(
    working_dir="./rag_storage",
    vector_storage_cls=MilvusVectorDBStorage,
    vector_storage_kwargs={"milvus_db_uri": "sqlite:///./milvus.db"}
)

# 索引文档（V2 同义词链接自动启用，需要先编辑 constants.py）
await rag.ainsert_file(file_path="./document.pdf", doc_id="doc1")

# 查询（V3 多跳推理）
result = await rag.aquery(
    query="什么是知识图谱？",
    param=QueryParam(enable_multi_hop=True, multi_hop_depth=2)
)

print(result["response"])
```

### 场景 2：使用 Neo4j

```python
from lightrag import LightRAG
from lightrag.kg import Neo4JStorage, MilvusVectorDBStorage

rag = LightRAG(
    working_dir="./rag_storage",
    graph_storage_cls=Neo4JStorage,
    vector_storage_cls=MilvusVectorDBStorage,
)
```

---

## 环境变量优先级

某些参数可通过环境变量覆盖 constants.py：

```bash
# V2 参数
export ENABLE_SYNONYM_LINKING=true
export SYNONYMY_THRESHOLD=0.8
export SYNONYMY_TOPK=100

# Neo4j 配置
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# 其他（来自 constants.py）
export WORKING_DIR=./rag_storage
export DEVICE=cuda:0
```

---

## 关键文件位置

```
neo4j-milvus branch 新增/修改：
├── lightrag/lightrag/
│   ├── synonym_linking.py        ← V2 实现
│   ├── ppr.py                    ← V3 实现
│   ├── operate.py                ← 集成 V2/V3 的关键改动
│   ├── base.py                   ← QueryParam 新增字段
│   └── kg/
│       ├── neo4j_impl.py         ← Neo4j 后端
│       └── milvus_impl.py        ← Milvus 后端
├── rag-anything/raganything/
│   └── constants.py              ← V2 默认参数在这里
└── DEVLOG_neo4j_milvus.md        ← 详细技术日志
```

---

## 常见问题

### Q: V2 和 V3 可以都启用吗？

**A:** 可以。V2 在索引时建立同义词边，V3 在查询时利用这些边做多跳推理。两者协同工作。

### Q: 如果只想用 V3，不用 V2？

**A:** 可以。修改 `constants.py` 保持 `DEFAULT_ENABLE_SYNONYM_LINKING = False`，查询时仍可传 `enable_multi_hop=True`。

### Q: 如何调整 PPR 的 damping 因子？

**A:** 需要修改源码 `lightrag/ppr.py` 中的 `personalized_pagerank()` 函数。

### Q: 离线机上没有网络，如何安装 Neo4j 驱动？

**A:** 在有网络的机器上 `pip download neo4j`，传输 `.whl` 文件到离线机，离线安装：
```bash
pip install neo4j-*.whl
```

---

## 与 main branch 的区别总结

```
main branch:
  - 图: NetworkX（内存，重启丢失）
  - 向量: Faiss（本地文件）
  - V2: ❌ 无
  - V3: ❌ 无

neo4j-milvus branch:
  - 图: Neo4j（持久化）或 NetworkX
  - 向量: Milvus（持久化）或 Faiss
  - V2: ✅ 同义词链接（编辑 constants.py 启用）
  - V3: ✅ PPR 多跳推理（QueryParam 传入）
```

---

就这么简单。你的离线机可以直接 `pip install` 依赖，然后根据需要修改 constants.py 或通过环境变量配置。
