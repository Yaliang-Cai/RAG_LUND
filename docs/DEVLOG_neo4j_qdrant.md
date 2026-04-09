# 开发日志：`neo4j-qdrant` 分支

> 基线提交：`257f887 keep only path-based extract filtering`（main 分支）
> 最后更新：2026-04-09

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

| 使用方式                          | 后端选择方式                                               |
| --------------------------------- | ---------------------------------------------------------- |
| `lightrag_server.py`              | `api/config.py` 读取 env var                               |
| `LocalRagService` / `RAGAnything` | `local_rag.py: _build_rag()` 的 `lightrag_kwargs` 显式传参 |
| 直接 `LightRAG(...)`              | 构造函数参数                                               |

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

| 开关                           | 默认值  | 说明                      |
| ------------------------------ | ------- | ------------------------- |
| `enable_entity_disambiguation` | `True`  | V1：实体消歧（`name       | type` 复合 ID） |
| `enable_synonym_linking`       | `False` | V2：同义词 SYNONYM 边构建 |
| `synonymy_threshold`           | `0.8`   | V2：cosine 阈值           |
| `synonymy_topk`                | `2048`  | V2：API 兼容参数（保留接口，内部已改为全量精确 matmul，不再做 KNN） |
| `synonymy_min_entity_len`      | `2`     | V2：最短实体名（字符数）  |

**B. QueryParam 参数（影响单次查询）**

| 参数                  | 默认值  | 说明                                       |
| --------------------- | ------- | ------------------------------------------ |
| `mode`                | `"mix"` | 检索模式；新增 `"rrf"` 选项                |
| `rrf_k`               | `60`    | RRF 平滑常数 k（仅 `mode="rrf"` 时生效）   |
| `enable_multi_hop`    | `False` | V3：启用 PPR 多跳推理                      |
| `multi_hop_depth`     | `2`     | V3：BFS 子图提取深度                       |
| `ppr_damping`         | `0.5`   | V3：PPR damping 因子 α                     |
| `ppr_top_k`           | `50`    | V3：PPR 返回的最高分 chunk 数              |
| `passage_node_weight` | `0.05`  | V3：chunk VDB 分数在 PPR seed 中的缩放系数 |

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

| 版本 | 功能                                | 开关                                         | 默认值                       |
| ---- | ----------------------------------- | -------------------------------------------- | ---------------------------- |
| V0   | Neo4j + Qdrant 存储后端             | `local_rag.py: _build_rag()` lightrag_kwargs | 默认 NetworkX + NanoVectorDB |
| V1   | Entity Disambiguation（实体消歧）   | `enable_entity_disambiguation`               | `True`                       |
| V2   | Synonym Linking（同义词边）         | `enable_synonym_linking`                     | `False`                      |
| V3   | PPR Multi-hop Reasoning（多跳推理） | `enable_multi_hop`（QueryParam）             | `False`                      |

**关键原则**：全部开关设为 `False` 时，代码物理执行路径与 main 分支 100% 一致。

---

## 二、算法讲解

### 2.1 V0：Neo4j + Qdrant 存储后端

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

### 2.2 V1：Entity Disambiguation（实体消歧）

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

### 2.3 V2：Synonym Linking（同义词边）

#### 问题背景

"AI" 和 "人工智能"、"Beijing" 和 "北京" 在图中无连接，跨写法检索召回率低。

#### 算法原理（HippoRAG2 对齐版）

一次性取回所有实体 embedding，在本地用 numpy 精确矩阵乘法计算全量余弦相似度，将超过阈值的实体对连接为 SYNONYM 边。零 VDB 往返，结果精确。

**执行步骤：**

1. 获取图中所有实体标签（`knowledge_graph_inst.get_all_labels()`）
2. 批量取回所有实体的 embedding 向量（`entities_vdb.get_vectors_by_ids()`）
3. 过滤长度 ≤ `min_entity_len` 的短实体（查询侧；避免标点/单字成为枢纽）
4. L2 归一化后做分批矩阵乘法（默认每批 1000 行，避免一次性展开 N×N 矩阵）：
   ```python
   Q /= np.linalg.norm(Q, axis=1, keepdims=True)   # 归一化
   sim_batch = Q[i:i+1000] @ R.T                   # 点积 == cosine similarity
   rows, cols = np.where(sim_batch >= synonymy_threshold)
   ```
5. 对超过 `synonymy_threshold`（默认 0.8）的实体对建 SYNONYM 边（全量模式取上三角，避免重复）

**增量模式**：传入 `new_entity_ids` 时，仅以新实体为查询侧，参考侧仍为全量实体，确保新实体能与已有实体建边。

**集成点**：`lightrag.py: ainsert()` 在 `merge_nodes_and_edges()` 完成后执行，`enable_synonym_linking=False` 时零开销。

#### 关键参数对比

| 参数       | 我们               | HippoRAG2    | 差距               |
| ---------- | ------------------ | ------------ | ------------------ |
| 阈值       | 0.8                | 0.8          | 对齐 ✅             |
| topk       | 2048               | ~2047        | 对齐 ✅（Qdrant 无上限，已取消 20x 限制）|
| 短实体过滤 | `min_entity_len=2` | `len > 2`    | 对齐 ✅             |
| 向量来源   | 预计算 embedding + 本地 numpy matmul | 精确矩阵乘法 | 对齐 ✅（零 VDB 往返，精确余弦）|

---

### 2.4 V3：PPR Multi-hop Reasoning（多跳推理）

#### 问题背景

"A 公司 CEO 的母校在哪个城市？"需要沿 `A公司 → CEO张三 → 北京大学 → 北京` 多跳传播，单跳检索无法覆盖。

#### 算法原理（HippoRAG2 对齐版：异构图 + 虚拟 chunk 节点 + 双信号 seed）

**步骤 1：构建双信号 Entity Seed（`operate.py:5383-5406`）**

```python
# 信号 1a：entity VDB 分数（_get_node_data 已检索）
for nd in node_datas:
    eid = nd.get("entity_id", nd.get("entity_name", ""))
    entity_seed_weights[eid] = max(entity_seed_weights.get(eid, 0), nd["vdb_score"])

# 信号 1b：relation VDB 分数 → 关系两端实体也获得 seed 权重（取 max）
rel_results = await relationships_vdb.query(query, top_k=query_param.top_k)
for rel in rel_results:
    score = rel["distance"]
    for field in ("src_id", "tgt_id"):
        entity_seed_weights[rel[field]] = max(entity_seed_weights.get(rel[field], 0), score)
```

两个信号合并到同一个 `entity_seed_weights` 字典，对同一实体取最大值。

**步骤 2：提取子图（`base.py:733` / `neo4j_impl.py:1103`）**

```python
seed_ids = list(entity_seed_weights.keys())
subgraph_nodes, subgraph_edges = await knowledge_graph_inst.get_subgraph_for_ppr(
    seed_ids, max_depth=query_param.multi_hop_depth  # 默认 2
)
```

- **基础实现**（`base.py:745-780`）：BFS 循环 `max_depth` 轮，每轮对当前边界节点调用 `get_node()` + `get_node_edges()` + `get_edge()`。边 dict 仅含 `{src, tgt, weight}`，**不含 `source_id`**（已知差距，见 TODO 方案 B）。
- **Neo4j 优化实现**（`neo4j_impl.py:1103`）：单条 Cypher `MATCH path = (seed)-[*1..{max_depth}]-(neighbor)` 替代 BFS N×M 次串行 IO。

**步骤 3：构建虚拟 chunk 节点 + chunk-entity 边（`operate.py:5417-5439`）**

```python
# 从节点 source_id 字段反向映射（逗号分隔）
chunk_to_entities: dict[str, list[str]] = {}
for node in subgraph_nodes:
    for chunk_id in split_string_by_multi_markers(node["source_id"], [GRAPH_FIELD_SEP]):
        chunk_to_entities.setdefault(chunk_id.strip(), []).append(node["entity_id"])

chunk_nodes = [{"chunk_id": cid} for cid in chunk_to_entities]
chunk_entity_edges = [
    {"chunk_id": cid, "entity_id": eid}
    for cid, eids in chunk_to_entities.items()
    for eid in eids
]
```

注意：仅使用**节点**的 `source_id`，边的 `source_id` 当前未用（见 TODO 方案 B）。

**步骤 4：构建 Chunk Seed 权重（DPR 分 × passage_node_weight，`operate.py:5441-5460`）**

```python
chunk_results = await chunks_vdb.query(query, top_k=ppr_top_k * 2)
# min-max 归一化后缩放
normalized = (score - min_s) / (max_s - min_s)
chunk_seed_weights[cid] = normalized * passage_node_weight   # 默认 0.05

# 仅对已出现在 chunk_to_entities 中的 chunk 赋 seed 权重
# （子图外的 VDB chunk 不参与 PPR）
```

**步骤 5：运行 PPR（`ppr.py:personalized_pagerank`，`operate.py:5462-5472`）**

```python
# ppr.py 内部图构建
G = nx.Graph()
G.add_nodes_from(entity_ids, node_type="entity")
G.add_edges_from(entity_edges, weight=edge["weight"])   # 原图权重
G.add_nodes_from(chunk_ids, node_type="chunk")
G.add_edges_from(chunk_entity_edges, weight=1.0)        # 固定权重（已知差距，见 TODO 方案 A/B）

# 合并双信号 seed，归一化到 sum=1
personalization = {**entity_seed_weights, **chunk_seed_weights}
personalization = {k: v / sum(personalization.values()) for k, v in personalization.items()}

pr = nx.pagerank(G, alpha=damping, personalization=personalization, weight="weight")

# 仅提取 chunk 节点的 PPR 分数，降序取 top_k
chunk_scores = [(nid, pr[nid]) for nid in chunk_node_ids]
chunk_scores.sort(key=lambda x: x[1], reverse=True)
return chunk_scores[:top_k]
```

- `damping`（即 nx.pagerank 的 `alpha`）越高，分布越集中在 seed 附近，多跳效果越弱；越低传播越广。
- `nx.PowerIterationFailedConvergence` 时 fallback 到 seed chunk 直接排序。

**步骤 6：取回 chunk 内容（`operate.py:5477-5495`）**

```python
chunk_data_list = await text_chunks_db.get_by_ids(ranked_chunk_ids)
# 附加 source_type="ppr" 和 ppr_score 字段
```

**步骤 7：PPR chunks 最高优先级合并（`operate.py:4720-4760`，`_merge_all_chunks`）**

```python
if ppr_chunks:
    # 1. PPR chunks（图结构排序）
    merged = deduplicate(ppr_chunks)
    # 2. vector_chunks 补充（去重）
    merged += [c for c in vector_chunks if c["chunk_id"] not in seen]
    return merged
    # 注意：enable_multi_hop 时，entity/relation chunk 推导被跳过
```

**参数直觉：**

| 参数                  | 默认值 | 直觉                                              |
| --------------------- | ------ | ------------------------------------------------- |
| `ppr_damping`         | 0.5    | 50% 继续游走，50% 回 seed；与 HippoRAG2 一致      |
| `passage_node_weight` | 0.05   | 过大→退化为 VDB 排序；过小→chunk 节点初始权重太低 |
| `multi_hop_depth`     | 2      | BFS 深度；Neo4j 用 Cypher 变长路径                |
| `ppr_top_k`           | 50     | PPR 输出的最大 chunk 数；VDB 查询用 top_k × 2     |

#### 集成点

| 位置                                    | 行号（参考） | 说明                                    |
| --------------------------------------- | ------------ | --------------------------------------- |
| `operate.py: _perform_kg_search()`      | 4467-4491    | V3 入口：`enable_multi_hop` 守卫        |
| `operate.py: _ppr_rank_chunks()`        | 5358-5495    | 主编排：seed → 子图 → PPR → 取回内容    |
| `ppr.py: personalized_pagerank()`       | 19-110       | NetworkX 图构建 + PPR 计算 + chunk 抽取 |
| `base.py: get_subgraph_for_ppr()`       | 733-780      | 基础 BFS 实现（通用后端）               |
| `neo4j_impl.py: get_subgraph_for_ppr()` | 1103-1152    | Cypher 优化实现（Neo4j）                |
| `operate.py: _merge_all_chunks()`       | 4700-4760    | PPR chunks 最高优先级合并，vector 补充  |

---

### 2.5 RRF：Reciprocal Rank Fusion 查询模式

#### 问题背景

`mix` 模式将三路 chunk 列表（vector / entity / relation）以 round-robin 轮流取，保证多样性但完全忽略排名信号——排名第 1 和排名第 50 的 chunk 获得相同地位。当某个 chunk 同时被多路检索高度认可时，round-robin 无法放大这一共识信号。

#### 算法原理

RRF 公式（`operate.py:4665`）：

```
score(chunk) = Σ_{source i}  1 / (k + rank_i)
```

- `rank_i`：chunk 在第 i 个来源列表中的排名（从 1 开始）
- `k`：平滑常数（默认 60，来自原论文），防止头部排名过度主导
- 同一 chunk 在多个来源中出现时，分数**累加**；仅出现一次的 chunk 分数较低

实现位于 `_rrf_merge()`，接收三个排名列表，按 chunk_id 去重合并后按 RRF 分降序输出。

#### 与其他 mode 的对比

| mode  | 召回来源                         | chunk 合并方式                |
| ----- | -------------------------------- | ----------------------------- |
| `mix` | vector + entity + relation       | round-robin 轮流取            |
| `rrf` | vector + entity + relation（同） | RRF 公式，共识 chunk 得分叠加 |

召回阶段完全相同，差异只在 `_merge_all_chunks()` 的合并路径（`operate.py:4791`）。

#### 参数

| 参数    | 默认值 | 说明                                             |
| ------- | ------ | ------------------------------------------------ |
| `rrf_k` | `60`   | 平滑常数；越小头部排名越主导，越大各位次趋于均衡 |

---

## 三、各版本变更明细

### V0：Neo4j + Qdrant 离线兼容

**开关**：
**`rag-anything/raganything/services/local_rag.py: _build_rag()`**
- `lightrag_kwargs` 中指定 `graph_storage: "Neo4JStorage"` + `vector_storage: "QdrantVectorDBStorage"` + `workspace: workspace_id`

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

### RRF：Reciprocal Rank Fusion 查询模式

**开关**：

```python
from lightrag.base import QueryParam

result = await rag.aquery(
    query="问题",
    param=QueryParam(mode="rrf", rrf_k=60)
)
```

也可与 PPR 叠加（PPR 优先级更高，RRF 仅在 `enable_multi_hop=False` 时生效）：

```python
# PPR 开启时，entity/relation chunk 推导被跳过，_merge_all_chunks 不走 RRF 分支
# 因此 rrf 与 enable_multi_hop=True 同时设置无意义
```

#### 集成点

| 位置                                | 说明                                         |
| ----------------------------------- | -------------------------------------------- |
| `lightrag/lightrag/base.py:89,229`  | `QueryParam.mode` 新增 `"rrf"`；`rrf_k` 字段 |
| `lightrag/lightrag/operate.py:4665` | `_rrf_merge()` 实现                          |
| `lightrag/lightrag/operate.py:4791` | `_merge_all_chunks()` RRF 分支入口           |


---

## 四、文件变更总览

### 修改的文件

| 文件                                               | 涉及版本  | 说明                                                                                                 |
| -------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `lightrag/lightrag/utils.py`                       | V1        | 工厂函数 `compute_entity_id`, `compute_entity_vdb_id`                                                |
| `lightrag/lightrag/base.py`                        | V1/V3/RRF | `delete_entity` 加 `entity_type`；`QueryParam` 扩展（V3 字段 + `rrf_k`）                             |
| `lightrag/lightrag/lightrag.py`                    | V0/V1/V2  | Feature Toggles + synonym linking 集成；`entity_id` 加入 `entities_vdb` meta_fields                  |
| `lightrag/lightrag/operate.py`                     | V1/V3/RRF | 实体 ID 替换 + 分组守卫 + edge remap + `_ppr_rank_chunks()` + `_merge_all_chunks()` + `_rrf_merge()` |
| `lightrag/lightrag/utils_graph.py`                 | V1        | 图操作工具函数 entity ID 更新；`adelete_by_entity` 加 `entity_type`                                  |
| `lightrag/lightrag/kg/neo4j_impl.py`               | V0/V3     | pipmaster 移除 + PPR 子图 Cypher                                                                     |
| `lightrag/lightrag/kg/qdrant_impl.py`              | V1        | `delete_entity` 加 `entity_type`，改用 `compute_entity_vdb_id`                                       |
| `lightrag/lightrag/kg/nano_vector_db_impl.py`      | V1        | 同上                                                                                                 |
| `lightrag/lightrag/kg/faiss_impl.py`               | V1        | 同上                                                                                                 |
| `lightrag/lightrag/kg/postgres_impl.py`            | V1        | 消歧模式下 `WHERE entity_name=$2 AND entity_type=$3`                                                 |
| `lightrag/lightrag/kg/mongo_impl.py`               | V1        | 同上（qdrant 系列）                                                                                  |
| `lightrag/lightrag/api/routers/document_routes.py` | V1        | `DeleteEntityRequest` 加可选 `entity_type` 字段                                                      |
| `rag-anything/raganything/modalprocessors.py`      | V1        | 多模态处理器 entity ID 更新                                                                          |
| `rag-anything/raganything/processor.py`            | V1        | 处理器 entity ID 更新                                                                                |
| `rag-anything/raganything/raganything.py`          | V0        | `load_dotenv` 改用 `find_dotenv(usecwd=True)`                                                        |
| `rag-anything/raganything/services/local_rag.py`   | V0        | `_build_rag()` 指定 Neo4J + Qdrant + workspace 隔离                                                  |
| `rag-anything/server/app.py`                       | patch     | `DELETE /workspace` 端点：清除 Neo4j + Qdrant + KV 存储                                              |

### 新增文件

| 文件                                   | 版本  | 说明                                 |
| -------------------------------------- | ----- | ------------------------------------ |
| `lightrag/lightrag/synonym_linking.py` | V2    | 同义词边构建                         |
| `lightrag/lightrag/ppr.py`             | V3    | PPR 计算                             |
| `.env.example`                         | V0    | 环境变量模板（含 Neo4j + Qdrant）    |
| `rag-anything/docs/qdrant_setup.md`    | patch | Qdrant 安装、配置、迁移、排错指南    |
| `rag-anything/docs/neo4j_setup.md`     | patch | Neo4j 安装、工作空间隔离、索引初始化 |

---

## 五、消融实验开关矩阵

| 配置       | V1 disambig | V2 synonym | V3 multi_hop | 预期行为                             |
| ---------- | :---------: | :--------: | :----------: | ------------------------------------ |
| **基准组** |   `False`   |  `False`   |   `False`    | 与 main 100% 一致                    |
| V1 only    |   `True`    |  `False`   |   `False`    | composite key，无 SYNONYM 边，无 PPR |
| V1+V2      |   `True`    |   `True`   |   `False`    | composite key + SYNONYM 边           |
| V1+V3      |   `True`    |  `False`   |    `True`    | composite key + PPR 多跳             |
| V1+V2+V3   |   `True`    |   `True`   |    `True`    | 全功能：消歧 + 同义词 + PPR          |

**V2+V3 协同**：SYNONYM 边增加图连通性，PPR 可跨同义词边界传播召回相关 chunk。

### 基准组路径一致性保证

| 代码路径        | 全关时行为                                      | 与 main 一致 |
| --------------- | ----------------------------------------------- | :----------: |
| 分组逻辑        | `all_nodes[entity_name].extend(entities)`       |      ✅       |
| Entity VDB ID   | `compute_mdhash_id(entity_name, prefix="ent-")` |      ✅       |
| Graph node key  | `entity_name`（无 `                             | type` 后缀） | ✅ |
| Synonym linking | 跳过                                            |      ✅       |
| PPR expansion   | 跳过                                            |      ✅       |
| 存储后端        | `NanoVectorDBStorage` / `NetworkXStorage`       |      ✅       |

---

## 六、与 HippoRAG2 的差距分析

### V2 对比

| 维度       | HippoRAG2    | 我们                                         | 差距    |
| ---------- | ------------ | -------------------------------------------- | ------- |
| 向量来源   | 精确矩阵乘法 | 预计算 embedding + 本地 numpy matmul          | 对齐 ✅  |
| KNN 规模   | top-2047     | 全量精确（无 top-k 上限）                     | 对齐 ✅  |
| 阈值       | 0.8          | 0.8                                          | 对齐 ✅  |
| 短实体过滤 | `len > 2`    | `min_entity_len=2`                           | 对齐 ✅  |

### V3 对比

| 维度               | HippoRAG2                             | 我们                                    | 差距          |
| ------------------ | ------------------------------------- | --------------------------------------- | ------------- |
| 图节点类型         | Entity + Passage + Fact               | Entity + virtual Chunk                  | Fact 节点缺失 |
| Seed 信号          | 双信号（entity + passage DPR × 0.05） | 双信号（entity VDB + chunk VDB × 0.05） | 方向一致 ✅    |
| PPR 输出           | Passage 分数直接排序                  | Chunk 分数直接排序                      | 对齐 ✅        |
| Recognition Memory | LLM fact reranking                    | 无                                      | **最大差距**  |
| Fact 三元组节点    | 独立节点                              | 无                                      | 架构限制      |

---

## 七、已知局限 & 待改进事项

1. **V2 增量模式函数已支持，调用侧未启用**：`build_synonym_edges()` 已实现 `new_entity_ids` 增量路径，但 `lightrag.py` 调用时仍传 `new_entity_ids=None`，每次 insert 都重算全图。后续可在 `ainsert()` 中收集本批次新增 entity_id 列表并传入以减少重复计算。

2. **Recognition Memory 缺失**：HippoRAG2 最核心的创新（LLM 对候选三元组重排序），当前无法在不引入独立 fact 存储的情况下实现。

3. ~~**V2 topk 受 VDB 限制**~~：已改用本地 numpy matmul，全量精确余弦，零 VDB 往返，差距已消除。

4. **Neo4j PPR Cypher 待验证**：`get_subgraph_for_ppr` 的 Cypher 在真实 Neo4j 实例上的语法正确性需要实测。基类 BFS 默认实现可作为 fallback。
