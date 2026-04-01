# 开发日志：`neo4j-milvus` 分支

> 基线提交：`257f887 keep only path-based extract filtering`（main 分支）
> 最后更新：2026-04-01

---

## 零、快速使用说明

### 0.1 安装依赖

```bash
pip install neo4j qdrant-client
```

### 0.2 配置：`.env` 文件

将 `.env.example` 复制为 `.env`，填入实际值：

```bash
# 图数据库
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# 向量数据库（Qdrant，本地单二进制，无需 Docker）
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=              # 本地无认证时留空

# WebUI
RAGANYTHING_API_KEY=your_api_key

# LLM
VLLM_API_BASE=http://localhost:8001/v1
LLM_MODEL_NAME=your_model

# 模型路径
RAGANYTHING_EMBEDDING_MODEL_PATH=/path/to/bge-m3
RAGANYTHING_RERANK_MODEL_PATH=/path/to/bge-reranker-v2-m3
```

`.env` 在 `raganything.py` 启动时通过 `find_dotenv(usecwd=True)` 自动加载，无论从哪个子目录运行都能找到项目根目录的 `.env`。

### 0.3 后端激活方式

> **重要**：`LIGHTRAG_GRAPH_STORAGE` / `LIGHTRAG_VECTOR_STORAGE` 两个环境变量**只对 `lightrag_server.py`（API Server）生效**，`LocalRagService` 不读取它们。

| 使用方式 | 后端选择方式 |
|---|---|
| `lightrag_server.py` | `api/config.py` 读取 env var |
| `LocalRagService` / `RAGAnything` | `local_rag.py: _build_rag()` 的 `lightrag_kwargs` 显式传参 |
| 直接 `LightRAG(...)` | 构造函数参数 |

`local_rag.py: _build_rag()` 当前配置：
```python
lightrag_kwargs={
    "graph_storage": "Neo4JStorage",
    "vector_storage": "QdrantVectorDBStorage",
    "workspace": workspace_id,   # Neo4j 隔离，每个工作空间独立 label
}
```

连接凭证（`NEO4J_URI`、`QDRANT_URL` 等）由 LightRAG 内部的 `check_storage_env_vars()` 从环境变量读取。

### 0.4 功能开关速查表

**A. LightRAG 初始化参数（影响 Indexing）**

| 开关 | 默认值 | 说明 |
|------|--------|------|
| `enable_entity_disambiguation` | `True` | V1：实体消歧（`name|type` 复合 ID） |
| `enable_synonym_linking` | `False` | V2：同义词 SYNONYM 边构建 |
| `synonymy_threshold` | `0.8` | V2：cosine 阈值 |
| `synonymy_topk` | `100` | V2：KNN 候选数量 |
| `synonymy_min_entity_len` | `2` | V2：最短实体名（字符数） |

**B. QueryParam 参数（影响单次查询）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_multi_hop` | `False` | V3：启用 PPR 多跳推理 |
| `multi_hop_depth` | `2` | V3：BFS 子图提取深度 |
| `ppr_damping` | `0.5` | V3：PPR damping 因子 α |
| `ppr_top_k` | `50` | V3：PPR 返回的最高分 chunk 数 |
| `passage_node_weight` | `0.05` | V3：chunk VDB 分数在 PPR seed 中的缩放系数 |

### 0.5 完整初始化示例

```python
from lightrag import LightRAG
from lightrag.base import QueryParam

rag = LightRAG(
    working_dir="./rag_storage",
    enable_entity_disambiguation=True,   # V1（默认已开）
    enable_synonym_linking=True,         # V2（需手动开启）
    synonymy_threshold=0.8,
)

await rag.ainsert_file("document.pdf", doc_id="doc1")

result = await rag.aquery(
    query="问题",
    param=QueryParam(
        mode="hybrid",
        enable_multi_hop=True,   # V3
        ppr_top_k=50,
    )
)
```

### 0.6 消融实验 Baseline（完全回退到 main）

```python
# 所有增强关闭 → 与 main 分支物理路径 100% 一致
rag = LightRAG(
    working_dir="./baseline",
    enable_entity_disambiguation=False,
    enable_synonym_linking=False,
)
result = await rag.aquery(query="问题", param=QueryParam(mode="hybrid"))
```

在 `local_rag.py: _build_rag()` 的 `lightrag_kwargs` 中注释掉 `graph_storage`/`vector_storage` 即可回退到 NetworkX + NanoVectorDB。

---

## 一、分支目标

在 LightRAG + rag-anything 代码库上实现四层阶梯式增强，通过 Feature Toggles 保证可独立开关、可消融对比：

| 版本 | 功能 | 开关 | 默认值 |
|------|------|------|--------|
| V0 | Neo4j + Qdrant 存储后端 | `local_rag.py: _build_rag()` lightrag_kwargs | 默认 NetworkX + NanoVectorDB |
| V1 | Entity Disambiguation（实体消歧） | `enable_entity_disambiguation` | `True` |
| V2 | Synonym Linking（同义词边） | `enable_synonym_linking` | `False` |
| V3 | PPR Multi-hop Reasoning（多跳推理） | `enable_multi_hop`（QueryParam） | `False` |

**关键原则**：全部开关设为 `False` 时，代码物理执行路径与 main 分支 100% 一致。

---

## 二、算法讲解

### 2.1 V1：Entity Disambiguation（实体消歧）

#### 问题背景

LightRAG 原版用 `compute_mdhash_id(entity_name, prefix="ent-")` 生成实体 ID。"苹果"这个实体，无论指水果、公司还是手机型号，都会被合并为同一个图节点。

#### 算法原理

在实体 ID 计算中加入 `entity_type` 作为区分维度：

```
entity_id      = f"{entity_name}|{entity_type}"   # 图节点 key
vdb_id         = md5(entity_id + "ent-")           # VDB hash
```

举例：
- `"苹果"（ORGANIZATION）` → 图节点 `"苹果|ORGANIZATION"`，VDB ID `md5("苹果|ORGANIZATIONent-")`
- `"苹果"（FOOD）` → 图节点 `"苹果|FOOD"`，VDB ID `md5("苹果|FOODent-")`

#### 回退保证

```python
def compute_entity_id(entity_name, entity_type="", enable_disambiguation=True):
    if enable_disambiguation and entity_type:
        return f"{entity_name}|{entity_type}"
    return entity_name   # 关闭时 == 原版
```

---

### 2.2 V2：Synonym Linking（同义词边）

#### 问题背景

"AI" 和 "人工智能"、"Beijing" 和 "北京" 在图中无连接，跨写法检索召回率低。

#### 算法原理（HippoRAG2 对齐版）

用已计算好的 entity embedding 向量直接做 KNN，将余弦相似度超过阈值的实体对连接为 SYNONYM 边。

**执行步骤：**

1. 批量获取所有实体的 embedding 向量（`entities_vdb.get_all_entity_ids()` + `get_vectors_by_ids()`）
2. 过滤长度 ≤ `min_entity_len` 的短实体（避免标点/单字成为枢纽）
3. 逐实体 KNN 查询，传入预计算向量而非文本：
   ```python
   neighbors = await entities_vdb.query(
       query="",
       query_embedding=entity_vec,
       top_k=synonymy_topk,       # 默认 100
   )
   ```
4. 对相似度 ≥ `synonymy_threshold`（默认 0.8）的邻居建 SYNONYM 边

**集成点**：`lightrag.py: ainsert()` 在 `merge_nodes_and_edges()` 完成后执行，`enable_synonym_linking=False` 时零开销。

#### 关键参数对比

| 参数 | 我们 | HippoRAG2 | 差距 |
|------|------|-----------|------|
| 阈值 | 0.8 | 0.8 | 对齐 ✅ |
| topk | 100 | ~2047 | 20x（VDB 接口限制） |
| 短实体过滤 | `min_entity_len=2` | `len > 2` | 对齐 ✅ |
| 向量来源 | 预计算 embedding | 精确矩阵乘法 | 方向一致，实现不同 |

---

### 2.3 V3：PPR Multi-hop Reasoning（多跳推理）

#### 问题背景

"A 公司 CEO 的母校在哪个城市？"需要沿 `A公司 → CEO张三 → 北京大学 → 北京` 多跳传播，单跳检索无法覆盖。

#### 算法原理（HippoRAG2 对齐版：异构图 + 虚拟 chunk 节点 + 双信号 seed）

**步骤 1：构建双信号 Seed**
```python
# 信号 1：entity VDB 分数 + relation VDB 分数（取最大）
entity_seeds[entity.id] = entity.score
entity_seeds[relation.src/tgt] = max(..., relation.score)

# 信号 2：chunk VDB 分数 × passage_node_weight（0.05）
chunk_seeds[chunk.id] = chunk.score * 0.05
```

**步骤 2：构建异构图**
- entity-chunk 边：通过 `entity.source_id` 字段反向映射
- entity-entity 边：来自知识图谱

**步骤 3：运行 PPR（networkx）**
```python
ppr_scores = nx.pagerank(G, alpha=damping, personalization=seed_weights)
# 从 PPR 分数中提取 chunk 节点，按分数降序取 top_k
```

**步骤 4：PPR chunks 以最高优先级合并**（优先于 VDB chunks）

**PPR 参数直觉：**
- `damping=0.5`：50% 继续游走，50% 回 seed，与 HippoRAG2 一致
- `passage_node_weight=0.05`：值过大退化为 VDB 排序，值过小 chunk 节点初始权重太低

#### 集成点

| 位置 | 说明 |
|------|------|
| `operate.py: _perform_kg_search()` | V3 入口 |
| `operate.py: _ppr_rank_chunks()` | 核心：构建图 + seed + 运行 PPR |
| `operate.py: _merge_all_chunks()` | PPR chunks 最高优先级合并 |

---

### 2.4 V0：Neo4j + Qdrant 存储后端

#### Neo4j（图存储）

- 移除 `neo4j_impl.py` 中的 `pipmaster` 自动安装，改为 `try/except ImportError`（离线兼容）
- `get_subgraph_for_ppr()` 用单条 Cypher 替代 BFS 循环（V3 优化）：
  ```cypher
  MATCH path = (seed)-[*1..{max_depth}]-(neighbor)
  WHERE seed.entity_id IN $seed_ids
  RETURN DISTINCT neighbor.entity_id, properties(neighbor), ...
  ```
- 每个工作空间通过 `workspace_id` 参数获得独立的 Neo4j node label，实现数据隔离

#### Qdrant（向量存储）

- 以本地单二进制 `qdrant` 进程方式运行，默认端口 6333，无需 Docker
- 通过 `QDRANT_URL` 环境变量配置连接，支持丰富的调优选项（HNSW、量化等）
- 详见 `rag-anything/docs/qdrant_setup.md`

---

## 三、各版本变更明细

### V0：Neo4j + Qdrant 离线兼容

**`lightrag/lightrag/kg/neo4j_impl.py`**
- 移除 `pipmaster`，改为 `try/except ImportError`
- 新增 `get_subgraph_for_ppr()` Cypher 实现

**`lightrag/lightrag/kg/qdrant_impl.py`**
- 接受 `entity_type` 参数，改用 `compute_entity_vdb_id`

**`rag-anything/raganything/services/local_rag.py: _build_rag()`**
- `lightrag_kwargs` 中指定 `graph_storage: "Neo4JStorage"` + `vector_storage: "QdrantVectorDBStorage"` + `workspace: workspace_id`

**`.env.example`**（新建）
- Neo4j 连接参数 + Qdrant URL（`LIGHTRAG_GRAPH_STORAGE`/`LIGHTRAG_VECTOR_STORAGE` 仅对 API Server 有效）

---

### V1：Entity Disambiguation — 实体消歧

**开关**：`lightrag/lightrag/lightrag.py: LightRAG.enable_entity_disambiguation`（默认 `True`）

**新增工厂函数**（`lightrag/lightrag/utils.py`）：
```python
def compute_entity_id(entity_name, entity_type="", enable_disambiguation=True):
    if enable_disambiguation and entity_type:
        return f"{entity_name}|{entity_type}"
    return entity_name

def compute_entity_vdb_id(entity_name, entity_type="", enable_disambiguation=True):
    composite = compute_entity_id(entity_name, entity_type, enable_disambiguation)
    return compute_mdhash_id(composite, prefix="ent-")
```

**替换调用点（20+ 处）**：`operate.py`、`utils_graph.py`、`lightrag.py`、`modalprocessors.py`、`processor.py`

**分组守卫**（消融实验关键保证，`operate.py: merge_nodes_and_edges()`）：
```python
if _disambig:
    group_key = compute_entity_id(entity_name, entity.get("entity_type", ""), True)
    all_nodes[group_key].append(entity)
else:
    all_nodes[entity_name].extend(entities)   # 与 main 100% 一致
```

---

### V2：Synonym Linking — 同义词链接

**开关**：`lightrag/lightrag/lightrag.py: LightRAG.enable_synonym_linking`（默认 `False`）

**核心文件**：`lightrag/lightrag/synonym_linking.py`

**集成点**：`lightrag.py: ainsert()` 末段，`merge_nodes_and_edges()` 完成后

---

### V3：PPR Multi-hop Reasoning — 多跳推理

**开关**：`lightrag/lightrag/base.py: QueryParam.enable_multi_hop`（默认 `False`）

**核心文件**：`lightrag/lightrag/ppr.py`

**集成点**：`operate.py: _perform_kg_search()` → `_ppr_rank_chunks()` → `_merge_all_chunks()`

---

## 四、文件变更总览

### 修改的文件

| 文件 | 涉及版本 | 说明 |
|------|----------|------|
| `lightrag/lightrag/utils.py` | V1 | 工厂函数 `compute_entity_id`, `compute_entity_vdb_id` |
| `lightrag/lightrag/base.py` | V1/V3 | `delete_entity` 加 `entity_type`；`QueryParam` 扩展（V3 字段） |
| `lightrag/lightrag/lightrag.py` | V0/V1/V2 | Feature Toggles + synonym linking 集成；`entity_id` 加入 `entities_vdb` meta_fields |
| `lightrag/lightrag/operate.py` | V1/V3 | 实体 ID 替换 + 分组守卫 + edge remap + `_ppr_rank_chunks()` + `_merge_all_chunks()` |
| `lightrag/lightrag/utils_graph.py` | V1 | 图操作工具函数 entity ID 更新；`adelete_by_entity` 加 `entity_type` |
| `lightrag/lightrag/kg/neo4j_impl.py` | V0/V3 | pipmaster 移除 + PPR 子图 Cypher |
| `lightrag/lightrag/kg/qdrant_impl.py` | V1 | `delete_entity` 加 `entity_type`，改用 `compute_entity_vdb_id` |
| `lightrag/lightrag/kg/nano_vector_db_impl.py` | V1 | 同上 |
| `lightrag/lightrag/kg/faiss_impl.py` | V1 | 同上 |
| `lightrag/lightrag/kg/postgres_impl.py` | V1 | 消歧模式下 `WHERE entity_name=$2 AND entity_type=$3` |
| `lightrag/lightrag/kg/mongo_impl.py` | V1 | 同上（qdrant 系列） |
| `lightrag/lightrag/api/routers/document_routes.py` | V1 | `DeleteEntityRequest` 加可选 `entity_type` 字段 |
| `rag-anything/raganything/modalprocessors.py` | V1 | 多模态处理器 entity ID 更新 |
| `rag-anything/raganything/processor.py` | V1 | 处理器 entity ID 更新 |
| `rag-anything/raganything/raganything.py` | V0 | `load_dotenv` 改用 `find_dotenv(usecwd=True)` |
| `rag-anything/raganything/services/local_rag.py` | V0 | `_build_rag()` 指定 Neo4J + Qdrant + workspace 隔离 |
| `rag-anything/server/app.py` | patch | `DELETE /workspace` 端点：清除 Neo4j + Qdrant + KV 存储 |

### 新增文件

| 文件 | 版本 | 说明 |
|------|------|------|
| `lightrag/lightrag/synonym_linking.py` | V2 | 同义词边构建 |
| `lightrag/lightrag/ppr.py` | V3 | PPR 计算 |
| `.env.example` | V0 | 环境变量模板（含 Neo4j + Qdrant） |
| `rag-anything/docs/qdrant_setup.md` | patch | Qdrant 安装、配置、迁移、排错指南 |
| `rag-anything/docs/neo4j_setup.md` | patch | Neo4j 安装、工作空间隔离、索引初始化 |

---

## 五、消融实验开关矩阵

| 配置 | V1 disambig | V2 synonym | V3 multi_hop | 预期行为 |
|------|:-----------:|:----------:|:------------:|---------|
| **基准组** | `False` | `False` | `False` | 与 main 100% 一致 |
| V1 only | `True` | `False` | `False` | composite key，无 SYNONYM 边，无 PPR |
| V1+V2 | `True` | `True` | `False` | composite key + SYNONYM 边 |
| V1+V3 | `True` | `False` | `True` | composite key + PPR 多跳 |
| V1+V2+V3 | `True` | `True` | `True` | 全功能：消歧 + 同义词 + PPR |

**V2+V3 协同**：SYNONYM 边增加图连通性，PPR 可跨同义词边界传播召回相关 chunk。

### 基准组路径一致性保证

| 代码路径 | 全关时行为 | 与 main 一致 |
|----------|-----------|:---:|
| 分组逻辑 | `all_nodes[entity_name].extend(entities)` | ✅ |
| Entity VDB ID | `compute_mdhash_id(entity_name, prefix="ent-")` | ✅ |
| Graph node key | `entity_name`（无 `|type` 后缀） | ✅ |
| Synonym linking | 跳过 | ✅ |
| PPR expansion | 跳过 | ✅ |
| 存储后端 | `NanoVectorDBStorage` / `NetworkXStorage` | ✅ |

---

## 六、与 HippoRAG2 的差距分析

### V2 对比

| 维度 | HippoRAG2 | 我们 | 差距 |
|------|-----------|------|------|
| 向量来源 | 精确矩阵乘法 | 预计算 embedding + VDB 查询 | 方向一致 ✅ |
| KNN 规模 | top-2047 | top-100 | 20x（VDB 接口限制） |
| 阈值 | 0.8 | 0.8 | 对齐 ✅ |
| 短实体过滤 | `len > 2` | `min_entity_len=2` | 对齐 ✅ |

### V3 对比

| 维度 | HippoRAG2 | 我们 | 差距 |
|------|-----------|------|------|
| 图节点类型 | Entity + Passage + Fact | Entity + virtual Chunk | Fact 节点缺失 |
| Seed 信号 | 双信号（entity + passage DPR × 0.05） | 双信号（entity VDB + chunk VDB × 0.05） | 方向一致 ✅ |
| PPR 输出 | Passage 分数直接排序 | Chunk 分数直接排序 | 对齐 ✅ |
| Recognition Memory | LLM fact reranking | 无 | **最大差距** |
| Fact 三元组节点 | 独立节点 | 无 | 架构限制 |

---

## 七、已知局限 & 待改进事项

1. **V2 增量模式未启用**：`build_synonym_edges(new_entity_ids=None)` 每次处理全部实体，大图效率低。后续可传入本批次新增 entity_id 列表实现增量模式。

2. **Recognition Memory 缺失**：HippoRAG2 最核心的创新（LLM 对候选三元组重排序），当前无法在不引入独立 fact 存储的情况下实现。

3. **V2 topk 受 VDB 限制**：无法达到 HippoRAG2 的 top-2047 批量矩阵乘法，性能差距 20x。

4. **Neo4j PPR Cypher 待验证**：`get_subgraph_for_ppr` 的 Cypher 在真实 Neo4j 实例上的语法正确性需要实测。基类 BFS 默认实现可作为 fallback。

---

## 八、补丁记录

### 2026-03-30：远端同步 + 实体消歧 Bug 修复

从 `origin/neo4j-milvus` 拉取 21 个新 commit，主要内容：SurGE 评估流水线增强、rerank 可观察性改进、服务级弹性（resilience / callbacks / single-flight init）。

**Bug 1 — 实体图节点 ID 重复拼接 `entity_type`**

现象：消歧模式开启时，图中节点 ID 为 `entity_name|entity_type|entity_type`（`|type` 重复）。

根因：`merge_nodes_and_edges` 以 composite key (`name|type`) 作为 key 传入 `_merge_nodes_then_upsert`，函数内部又调用了一次 `compute_entity_id`，导致二次拼接。

修复（`operate.py: _merge_nodes_then_upsert`）：
```python
# 从 nodes_data[0] 取回真正的 plain name
canonical_entity_name = entity_name
if nodes_data:
    first = nodes_data[0].get("entity_name")
    if isinstance(first, str) and first:
        canonical_entity_name = first

composite_id = entity_name   # entity_name 已经是正确的图节点 key，不再重复调用
```

**Bug 2 — `delete_entity()` 无法定位消歧模式下的向量**

现象：消歧模式开启时，`delete_entity(entity_name)` 计算的 VDB hash 为 `hash(name)`，而插入时存储的是 `hash(name|type)`，删除静默失败。

修复范围：

| 层 | 文件 | 变更 |
|----|------|------|
| 抽象接口 | `base.py` | `delete_entity(entity_name, entity_type="")` |
| 编排层 | `utils_graph.py: adelete_by_entity` | 加 `entity_type` 参数；用 `compute_entity_id` 得到 `node_key` |
| LightRAG 入口 | `lightrag.py` | `adelete_by_entity` / `delete_by_entity` 透传 `entity_type` |
| API | `document_routes.py: DeleteEntityRequest` | 加可选 `entity_type: str = ""` 字段 |
| VDB 实现（6 个） | `qdrant_impl`, `nano_vector_db_impl`, `faiss_impl`, `mongo_impl` 等 | 接受 `entity_type`，改用 `compute_entity_vdb_id` |
| Postgres 特殊处理 | `postgres_impl` | 消歧模式下 `WHERE entity_name=$2 AND entity_type=$3` |

---

### 2026-03-31：Milvus → Qdrant 迁移 + Neo4j workspace 隔离

**向量后端切换（commit `006287d`）**

将 `local_rag.py: _build_rag()` 中的 `MilvusVectorDBStorage` 替换为 `QdrantVectorDBStorage`。LightRAG 已内置完整的 Qdrant 实现（`lightrag/kg/qdrant_impl.py`），只需配置 `QDRANT_URL`。

`.env` 变化：
- 移除：`MILVUS_URI`、`MILVUS_DB_NAME`
- 新增：`QDRANT_URL`

**Neo4j workspace 隔离 Bug（commit `bd5c861`）**

Bug：`_build_rag()` 从未将 `workspace` 传给 LightRAG，导致 `Neo4JStorage` 所有工作空间使用同一个 `"base"` node label，数据混用。

修复：
```python
# local_rag.py: _build_rag()
lightrag_kwargs={
    ...
    "workspace": workspace_id,   # 每个工作空间独立 Neo4j label
}
```

---

### 2026-04-01：entity_id VDB payload 缺失 + WebUI 工作空间删除

**entity_id 缺失 Qdrant payload（commit `3f05fd8`）**

Bug：Qdrant payload 中没有 `entity_id` 字段（composite key `name|type`），导致 Neo4j/NetworkX 节点查找时 ID 不匹配。

修复（`lightrag/lightrag/lightrag.py`）：将 `entity_id` 加入 `entities_vdb` 的 `meta_fields`，使 Qdrant payload 携带完整 composite key。

**WebUI 工作空间删除**

新增 `DELETE /workspace` API 端点（`rag-anything/server/app.py`）：
- 并行清除 Neo4j 节点/边、Qdrant 集合、KV 存储
- 返回 `drop_errors` 供排错
- WebUI 新增「删除工作空间」按钮，需输入工作空间名确认，删除后自动刷新列表
