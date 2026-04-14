# 开发日志：`neo4j-qdrant` 分支

> 基线提交：`257f887 keep only path-based extract filtering`（main 分支）
> 最后更新：2026-04-14

---

## 零、快速使用说明

### 0.1 依赖

```bash
pip install neo4j qdrant-client fast-pagerank
```

### 0.2 配置：`.env` 文件

```bash
# 图数据库
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# 向量数据库（Qdrant，本地单二进制，无需 Docker）
QDRANT_URL=http://localhost:6333

# LLM
VLLM_API_BASE=http://localhost:8001/v1
LLM_MODEL_NAME=your_model

# 模型路径
RAGANYTHING_EMBEDDING_MODEL_PATH=/path/to/bge-m3
RAGANYTHING_RERANK_MODEL_PATH=/path/to/bge-reranker-v2-m3
```

### 0.3 后端激活方式

> `LIGHTRAG_GRAPH_STORAGE` / `LIGHTRAG_VECTOR_STORAGE` 环境变量**只对 `lightrag_server.py` 生效**，`LocalRagService` 不读取它们。

| 使用方式                          | 后端选择方式                                               |
| --------------------------------- | ---------------------------------------------------------- |
| `lightrag_server.py`              | `api/config.py` 读取 env var                               |
| `LocalRagService` / `RAGAnything` | `local_rag.py: _build_rag()` 的 `lightrag_kwargs` 显式传参 |

`local_rag.py: _build_rag()` 当前配置：
```python
lightrag_kwargs={
    "graph_storage": "Neo4JStorage",
    "vector_storage": "QdrantVectorDBStorage",
    "workspace": workspace_id,
}
```

### 0.4 功能开关速查

**LightRAG 初始化参数（影响 Indexing）**

| 开关                           | 默认值  | 说明                                                |
| ------------------------------ | ------- | --------------------------------------------------- |
| `enable_entity_disambiguation` | `True`  | V1：实体消歧（`name\|type` 复合 ID）                |
| `enable_synonym_linking`       | `False` | V2：同义词 SYNONYM 边构建                           |
| `synonymy_threshold`           | `0.8`   | V2：cosine 阈值                                     |
| `synonymy_topk`                | `2048`  | V2：保留接口，内部已改为全量精确 matmul，不再做 KNN |
| `synonymy_min_entity_len`      | `2`     | V2：最短实体名（字符数）                            |

**QueryParam 参数（影响单次查询）**

| 参数                    | 默认值  | 说明                                                         |
| ----------------------- | ------- | ------------------------------------------------------------ |
| `mode`                  | `"mix"` | `hybrid` / `mix` / `rrf` / `ppr_local` / `ppr`               |
| `rrf_k`                 | `60`    | RRF 平滑常数（仅 `mode="rrf"` 时生效）                       |
| `enable_multi_hop`      | `False` | **已废弃**，请改用 `mode="ppr_local"`，保留仅为向后兼容      |
| `multi_hop_depth`       | `2`     | 仅 `ppr_local`：BFS 子图提取深度                             |
| `ppr_damping`           | `0.5`   | PPR damping 因子 α（`ppr` 和 `ppr_local` 均生效）            |
| `ppr_top_k`             | `50`    | PPR 返回的最高分 chunk 数                                    |
| `passage_node_weight`   | `0.05`  | chunk seed 总权重相对于 entity seed 总权重的比例             |
| `hub_penalty_threshold` | `50`    | 度数超过此值的实体 seed 权重除以 log(1+degree)；0 = 禁用惩罚 |

### 0.5 版本总览

| 版本 | 功能                              | 开关                                         | 默认值                       |
| ---- | --------------------------------- | -------------------------------------------- | ---------------------------- |
| V0   | Neo4j + Qdrant 存储后端           | `local_rag.py: _build_rag()` lightrag_kwargs | 默认 NetworkX + NanoVectorDB |
| V1   | Entity Disambiguation（实体消歧） | `enable_entity_disambiguation`               | `True`                       |
| V2   | Synonym Linking（同义词边）       | `enable_synonym_linking`                     | `False`                      |
| V3   | PPR Multi-hop Reasoning（局部）   | `mode="ppr_local"`                           | —                            |
| V3b  | PPR Global（全图传播）            | `mode="ppr"`                                 | —                            |

**关键原则**：全部开关设为 `False` 时，代码物理执行路径与 main 分支 100% 一致。

### 0.6 完整初始化示例

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

# V3b 全图 PPR（推荐）
result = await rag.aquery("问题", param=QueryParam(mode="ppr", ppr_top_k=50))

# 消融 baseline（与 main 100% 一致）
result = await rag.aquery("问题", param=QueryParam(mode="hybrid"))
```

---

## 一、算法说明

### 1.1 V1：Entity Disambiguation（实体消歧）

LightRAG 原版以 `entity_name` 为唯一 key，"苹果（公司）"和"苹果（食物）"被合并为同一节点。

**解法**：加入 `entity_type` 作为区分维度：

```
entity_id = f"{entity_name}|{entity_type}"    # 图节点 key
vdb_id    = md5(entity_id + "ent-")            # VDB hash
```

回退保证：`enable_entity_disambiguation=False` 时退回 `entity_name`，与 main 完全一致。

---

### 1.2 V2：Synonym Linking（同义词边）

"AI"与"人工智能"在图中无连接，跨写法检索召回率低。

**解法**（HippoRAG2 对齐）：一次性取回所有实体 embedding，本地 numpy 精确矩阵乘法计算全量 cosine，超过阈值的实体对建 SYNONYM 边。零 VDB 往返，结果精确。

**增量模式**：传入 `new_entity_ids` 时，仅以新实体为查询侧，参考侧仍为全量实体。  
**集成点**：`lightrag.py: ainsert()` 在 `merge_nodes_and_edges()` 完成后执行。

| 参数       | 我们                            | HippoRAG2      |
| ---------- | ------------------------------- | -------------- |
| 阈值       | 0.8                             | 0.8 ✅          |
| 向量来源   | 预计算 embedding + numpy matmul | 精确矩阵乘法 ✅ |
| 短实体过滤 | `min_entity_len=2`              | `len > 2` ✅    |

---

### 1.3 V3：PPR Multi-hop（ppr_local）

**问题**："A 公司 CEO 的母校在哪个城市？"需要沿 `A公司→CEO→北京大学→北京` 多跳传播。

**算法步骤**：

1. **双信号 Entity Seed**：entity VDB 分数 + relation VDB 分数（取两端实体的 max）
2. **Hub 节点惩罚**：度数 > `hub_penalty_threshold` 的实体 seed 权重 ÷ log(1+degree)
3. **子图提取**：BFS depth=`multi_hop_depth`；Neo4j 用单条 Cypher 替代 N×M 串行 IO
4. **虚拟 chunk 节点**：从节点和边的 `source_id` 反向映射，构建 entity↔chunk 边
5. **Chunk Seed**：VDB 分数 min-max 归一化后 × `passage_node_weight`（0.05）
6. **分离归一化**：entity seeds → sum=1.0，chunk seeds → sum=passage_node_weight，合并后再归一化
7. **NetworkX PPR**：`nx.pagerank(alpha=damping, personalization=combined_seeds)`
8. **输出**：取 chunk 节点 PPR 分数降序 top_k，PPR chunks 最高优先级合并，vector 补充

**集成点**：

| 位置                                    | 说明                                    |
| --------------------------------------- | --------------------------------------- |
| `operate.py: _ppr_rank_chunks()`        | 主编排：seed → 子图 → PPR → 取回内容    |
| `ppr.py: personalized_pagerank()`       | NetworkX 图构建 + PPR 计算 + chunk 抽取 |
| `base.py: get_subgraph_for_ppr()`       | 基础 BFS 实现（通用后端）               |
| `neo4j_impl.py: get_subgraph_for_ppr()` | Cypher 优化实现（Neo4j）                |
| `operate.py: _merge_all_chunks()`       | PPR chunks 最高优先级合并，vector 补充  |

---

### 1.4 V3b：全图 PPR（ppr）

**问题**：`ppr_local` BFS depth=2 截断子图，答案路径超 2 跳时相关 chunk 不可见。

**解法**：引入 `GlobalPPREngine`（`ppr_engine.py`）：
- 第一次调用时从 Neo4j 拉取全部节点和边，构建 scipy csr_matrix 缓存到内存
- 后续 query 直接复用缓存（140K 节点规模下 ~20-50ms/query）
- `invalidate()` 在 insert 后调用，触发下次重新加载
- chunk→entity 边权重使用 embedding cosine similarity（fallback 1.0）

**集成点**：

| 位置                                       | 说明                               |
| ------------------------------------------ | ---------------------------------- |
| `ppr_engine.py`（新增）                    | GlobalPPREngine：缓存 + sparse PPR |
| `neo4j_impl.py: get_all_nodes_and_edges()` | 全图拉取，含边的 source_id         |
| `operate.py: _ppr_rank_chunks_global()`    | 全图 PPR 编排路径                  |

---

### 1.5 RRF 查询模式

`mix` 以 round-robin 合并三路 chunk，忽略排名信号。RRF 公式：

```
score(chunk) = Σ_{source i}  1 / (k + rank_i)
```

同一 chunk 被多路高排名时分数叠加，共识信号被放大。`k=60` 防止头部排名过度主导。

| mode  | 合并方式                      |
| ----- | ----------------------------- |
| `mix` | round-robin 轮流取            |
| `rrf` | RRF 公式，共识 chunk 得分叠加 |

召回阶段完全相同，差异只在 `_merge_all_chunks()`（`operate.py:4791`）。

---

## 二、文件变更总览

### 修改的文件

| 文件                                             | 涉及版本  | 说明                                                                                          |
| ------------------------------------------------ | --------- | --------------------------------------------------------------------------------------------- |
| `lightrag/lightrag/utils.py`                     | V1        | 工厂函数 `compute_entity_id`, `compute_entity_vdb_id`                                         |
| `lightrag/lightrag/base.py`                      | V1/V3/RRF | `QueryParam` 扩展（V3 字段 + `rrf_k` + mode Literal 新增 `ppr`/`ppr_local`）                  |
| `lightrag/lightrag/lightrag.py`                  | V0/V1/V2  | Feature Toggles + synonym linking 集成                                                        |
| `lightrag/lightrag/operate.py`                   | V1/V3/RRF | 实体 ID 替换 + 分组守卫 + `_ppr_rank_chunks()` + `_ppr_rank_chunks_global()` + `_rrf_merge()` |
| `lightrag/lightrag/kg/neo4j_impl.py`             | V0/V3/V3b | pipmaster 移除 + PPR 子图 Cypher + `get_all_nodes_and_edges()`                                |
| `lightrag/lightrag/kg/qdrant_impl.py`            | V1        | `delete_entity` 改用 `compute_entity_vdb_id`                                                  |
| `lightrag/lightrag/kg/postgres_impl.py`          | V1        | 消歧模式下 `WHERE entity_name=$2 AND entity_type=$3`                                          |
| `rag-anything/raganything/services/local_rag.py` | V0        | `_build_rag()` 指定 Neo4J + Qdrant + workspace 隔离                                           |
| `rag-anything/server/app.py`                     | patch     | `DELETE /workspace` 端点：清除 Neo4j + Qdrant + KV 存储                                       |

### 新增文件

| 文件                                   | 版本 | 说明                               |
| -------------------------------------- | ---- | ---------------------------------- |
| `lightrag/lightrag/synonym_linking.py` | V2   | 同义词边构建                       |
| `lightrag/lightrag/ppr.py`             | V3   | PPR 计算（NetworkX）               |
| `lightrag/lightrag/ppr_engine.py`      | V3b  | GlobalPPREngine：缓存 + sparse PPR |
| `.env.example`                         | V0   | 环境变量模板                       |

---

## 三、消融实验矩阵

| 配置       | V1 disambig | V2 synonym | V3 multi_hop | 预期行为                             |
| ---------- | :---------: | :--------: | :----------: | ------------------------------------ |
| **基准组** |   `False`   |  `False`   |   `False`    | 与 main 100% 一致                    |
| V1 only    |   `True`    |  `False`   |   `False`    | composite key，无 SYNONYM 边，无 PPR |
| V1+V2      |   `True`    |   `True`   |   `False`    | composite key + SYNONYM 边           |
| V1+V3      |   `True`    |  `False`   |    `True`    | composite key + PPR 多跳             |
| V1+V2+V3   |   `True`    |   `True`   |    `True`    | 全功能：消歧 + 同义词 + PPR          |

**V2+V3 协同**：SYNONYM 边增加图连通性，PPR 可跨同义词边界传播。

---

## 四、与 HippoRAG2 的差距分析

| 维度                | HippoRAG2                             | 我们                                    | 状态          |
| ------------------- | ------------------------------------- | --------------------------------------- | ------------- |
| 图节点类型          | Entity + Passage + Fact               | Entity + virtual Chunk                  | Fact 节点缺失 |
| Seed 信号           | 双信号（entity + passage DPR × 0.05） | 双信号（entity VDB + chunk VDB × 0.05） | 方向一致 ✅    |
| Hub 节点抑制        | Recognition Memory 过滤               | log(1+degree) 惩罚                      | 对齐 ✅        |
| Seed 归一化         | 分离归一化                            | 分离归一化                              | 对齐 ✅        |
| PPR 图范围          | 全图传播                              | 全图传播（mode=ppr）                    | 对齐 ✅        |
| chunk→entity 边权重 | embedding cosine                      | embedding cosine（fallback 1.0）        | 基本对齐 ✅    |
| Recognition Memory  | LLM fact reranking                    | 无                                      | **最大差距**  |

---

## 五、已知局限

1. **Recognition Memory 缺失**：HippoRAG2 最核心创新（LLM 对候选三元组重排序），当前无法在不引入独立 fact 存储的情况下实现。
