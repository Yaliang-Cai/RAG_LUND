# 统一配置指南 — 仅编辑 constants.py

所有 neo4j-milvus branch 的配置都在一个文件里：

```
rag-anything/raganything/constants.py
```

## 配置项速查表

### V2：同义词链接

```python
# 约第 197 行
DEFAULT_ENABLE_SYNONYM_LINKING = True    # 启用/禁用
DEFAULT_SYNONYMY_THRESHOLD = 0.8         # 相似度阈值
DEFAULT_SYNONYMY_TOPK = 100              # KNN top-K
DEFAULT_SYNONYMY_MIN_ENTITY_LEN = 2      # 最小实体长度
```

### V3：PPR 多跳推理

```python
# 约第 206 行
DEFAULT_ENABLE_MULTI_HOP = True          # 启用/禁用
DEFAULT_MULTI_HOP_DEPTH = 2              # 图搜索深度
DEFAULT_PPR_DAMPING = 0.5                # PPR damping 因子
DEFAULT_PPR_TOP_K = 50                   # 返回的 chunk 数量
DEFAULT_PASSAGE_NODE_WEIGHT = 0.05       # chunk 权重系数
```

### 存储后端选择

#### 图数据库（Knowledge Graph）

```python
# 约第 216 行
# 选项 1：NetworkX（内存，不持久化，推荐简单测试）
DEFAULT_GRAPH_STORAGE_TYPE = "networkx"

# 选项 2：Neo4j（持久化，推荐生产）
DEFAULT_GRAPH_STORAGE_TYPE = "neo4j"
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"
```

#### 向量数据库

```python
# 约第 224 行
# 选项 1：NanoVectorDB（本地文件，默认，与 main 一致）
DEFAULT_VECTOR_STORAGE_TYPE = "nanovectordb"

# 选项 2：Milvus Lite（单文件，推荐持久化离线）
DEFAULT_VECTOR_STORAGE_TYPE = "milvus"
DEFAULT_MILVUS_DB_URI = "sqlite:///./milvus.db"

# 选项 3：Milvus Server（连接到已启动的服务）
DEFAULT_VECTOR_STORAGE_TYPE = "milvus"
DEFAULT_MILVUS_DB_URI = "http://localhost:19530"
```

---

## 常见配置场景

### 场景 1：离线简单测试（默认）

```python
DEFAULT_GRAPH_STORAGE_TYPE = "networkx"      # 内存图，不持久
DEFAULT_VECTOR_STORAGE_TYPE = "nanovectordb" # 本地向量文件（与 main 一致）
DEFAULT_ENABLE_SYNONYM_LINKING = False       # V2 关闭
DEFAULT_ENABLE_MULTI_HOP = False             # V3 关闭
```

### 场景 2：离线生产环境（推荐）

```python
DEFAULT_GRAPH_STORAGE_TYPE = "neo4j"         # 持久化图
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"

DEFAULT_VECTOR_STORAGE_TYPE = "milvus"       # 持久化向量
DEFAULT_MILVUS_DB_URI = "sqlite:///./milvus.db"

DEFAULT_ENABLE_SYNONYM_LINKING = True        # V2 启用
DEFAULT_ENABLE_MULTI_HOP = True              # V3 启用
```

### 场景 3：只用 V2，不用 V3

```python
DEFAULT_ENABLE_SYNONYM_LINKING = True        # V2
DEFAULT_ENABLE_MULTI_HOP = False             # V3 关闭
```

---

## 如何使用

### 第 1 步：编辑 constants.py

```bash
nano rag-anything/raganything/constants.py

# 或用编辑器打开，找到上面的配置项，修改为你的需求
```

### 第 2 步：代码中直接使用（自动读取 constants.py）

使用 `LocalRagService`（推荐）：

```python
from raganything.services.local_rag import LocalRagService

async def main():
    # 自动从 constants.py 读取所有配置
    service = LocalRagService.from_env()

    # 索引（V2 会自动启用，如果 DEFAULT_ENABLE_SYNONYM_LINKING=True）
    await service.ainsert_from_folder("./documents", "my_graph")

    # 查询（V3 会自动启用，如果 DEFAULT_ENABLE_MULTI_HOP=True）
    result = await service.aquery("问题", "my_graph")
    print(result["response"])
```

或直接使用 `LightRAG`（需要手动传递存储后端）：

```python
from lightrag import LightRAG
from lightrag.kg import Neo4JStorage, MilvusVectorDBStorage
from raganything.constants import (
    DEFAULT_GRAPH_STORAGE_TYPE,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USERNAME,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_VECTOR_STORAGE_TYPE,
    DEFAULT_MILVUS_DB_URI,
)

# 根据 constants.py 的设置选择存储后端
if DEFAULT_GRAPH_STORAGE_TYPE == "neo4j":
    graph_storage_cls = Neo4JStorage
    graph_storage_kwargs = {
        "uri": DEFAULT_NEO4J_URI,
        "username": DEFAULT_NEO4J_USERNAME,
        "password": DEFAULT_NEO4J_PASSWORD,
    }
else:
    graph_storage_cls = None  # 默认 NetworkX
    graph_storage_kwargs = {}

if DEFAULT_VECTOR_STORAGE_TYPE == "milvus":
    vector_storage_cls = MilvusVectorDBStorage
    vector_storage_kwargs = {"milvus_db_uri": DEFAULT_MILVUS_DB_URI}
else:
    vector_storage_cls = None  # 默认 Faiss
    vector_storage_kwargs = {}

# 创建 RAG 实例
rag = LightRAG(
    working_dir="./rag_storage",
    graph_storage_cls=graph_storage_cls,
    graph_storage_kwargs=graph_storage_kwargs,
    vector_storage_cls=vector_storage_cls,
    vector_storage_kwargs=vector_storage_kwargs,
)
```

---

## 查询时控制 V3

即使在 constants.py 中 `DEFAULT_ENABLE_MULTI_HOP=False`，也可以在查询时临时启用：

```python
from lightrag.base import QueryParam

result = await rag.aquery(
    query="问题",
    param=QueryParam(
        enable_multi_hop=True,      # 覆盖常量设置
        multi_hop_depth=3,          # 覆盖常量设置
        ppr_top_k=100,              # 覆盖常量设置
    )
)
```

---

## 环境变量覆盖

如果需要，可以通过环境变量临时覆盖 constants.py：

```bash
export ENABLE_SYNONYM_LINKING=true
export ENABLE_MULTI_HOP=true
export GRAPH_STORAGE_TYPE=neo4j
export NEO4J_URI=bolt://localhost:7687
export VECTOR_STORAGE_TYPE=milvus
export MILVUS_DB_URI=sqlite:///./milvus.db
```

代码中读取：

```python
import os
from raganything.constants import DEFAULT_ENABLE_SYNONYM_LINKING

enable_v2 = os.getenv("ENABLE_SYNONYM_LINKING", str(DEFAULT_ENABLE_SYNONYM_LINKING)).lower() == "true"
```

---

## 完整工作流

```python
import asyncio
from raganything.services.local_rag import LocalRagService

async def main():
    # 1. 初始化（自动读取 constants.py 所有配置）
    service = LocalRagService.from_env()

    # 2. 索引文档（V2 如果启用，会建立 SYNONYM 边）
    print("Indexing...")
    await service.ainsert_from_folder("./documents", "my_graph")

    # 3. 查询（V3 如果启用，会做 PPR 多跳推理）
    print("Querying...")
    result = await service.aquery(
        query="这个文档讲了什么？",
        graph_name="my_graph"
    )

    print("回答:", result["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

**就这么简单：**
1. 编辑 `rag-anything/raganything/constants.py`
2. 运行代码，自动生效

无需在代码里改参数！
