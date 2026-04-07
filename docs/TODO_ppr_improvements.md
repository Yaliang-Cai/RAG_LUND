# V3 PPR 多跳推理优化方案

## 背景

当前 V3 基于 HippoRAG2 风格的 Personalized PageRank (PPR) 实现了图上多跳推理，核心逻辑分布在以下文件：

| 文件 | 职责 |
|------|------|
| `lightrag/lightrag/ppr.py` | PPR 算法核心（NetworkX） |
| `lightrag/lightrag/operate.py:5274-5411` | `_ppr_rank_chunks()` 调用编排 |
| `lightrag/lightrag/operate.py:4633-4693` | `_merge_all_chunks()` 结果合并 |
| `lightrag/lightrag/base.py:726-773` | `get_subgraph_for_ppr()` 基础实现（BFS） |
| `lightrag/lightrag/kg/neo4j_impl.py:1103-1152` | `get_subgraph_for_ppr()` Neo4j 优化实现 |

### 当前流程

```
用户查询
  ↓
[1] VDB 检索：entities_vdb + relationships_vdb → 候选实体 + vdb_score
  ↓
[2] 构建 entity_seed_weights（实体 VDB 分 + 关系 VDB 分取 max）
  ↓
[3] get_subgraph_for_ppr(seed_ids, depth=2) → 子图节点 + 边
  ↓
[4] 从节点 source_id 构建虚拟 chunk 节点 + chunk-entity 边（weight=1.0）
  ↓
[5] chunks_vdb DPR 分 × passage_node_weight(0.05) → chunk_seed_weights
  ↓
[6] networkx.pagerank(damping=0.5) → chunk 节点 PPR 分排序
  ↓
[7] 取回 chunk 内容，PPR chunks 优先合并入上下文
```

### 已识别的问题

| 问题 | 影响 |
|------|------|
| chunk-entity 边统一 weight=1.0 | 忽略共现频率信息 |
| 边的 `source_id` 未用于 chunk 映射 | 遗漏通过关系而非实体关联的文档 |
| 高度数 hub 节点（如"中国"）未被惩罚 | PPR 信号被通用实体稀释 |
| damping=0.5 固定 | 对简单和复杂查询一视同仁 |
| PPR 输出未经语义重排 | 图结构高分但内容无关的 chunk 排在前面 |
| SYNONYM 边与普通关系边权重相同 | 同义关系的传播强度被高估 |

---

## 优化方案

### 方案 A：Hub 节点惩罚（P0）

**问题**：VDB 检索可能把高连接度的通用实体（"技术"、"公司"、"中国"）选为种子，
导致子图膨胀，PPR 信号被大量无关邻居稀释。

**方案**：在构建 `entity_seed_weights` 后，用已有的 `node_degrees_batch()` 查询种子实体的度数，
对高度数节点施加对数惩罚：

```python
adjusted_weight = weight / log(1 + degree)
```

**实现位置**：`operate.py:5298-5325`（`_ppr_rank_chunks` Step 1 之后）

**复用的已有接口**：
- `knowledge_graph_inst.node_degrees_batch(seed_ids)` — 基类有逐条 fallback，Neo4j 有批量 Cypher

**兼容性**：✅ 所有图后端（Neo4j / 其他）均兼容

**参数建议**：
```python
# QueryParam 新增
hub_penalty_threshold: int = 50   # 度数超过此值才施加惩罚
```

---

### 方案 B：边 source_id 扩充 chunk-entity 映射（P1）

**问题**：当前虚拟 chunk 节点仅从**节点**的 `source_id` 建立。
边（关系）也有 `source_id`，记录了哪些 chunk 描述了该关系，
但 `get_subgraph_for_ppr` 把它丢弃了。

例：chunk_42 描述了"华为与5G"这条关系，但"华为"节点的 `source_id` 里可能没有 chunk_42。
这个 chunk 在当前实现中对 PPR 不可见。

**方案**：扩展 `get_subgraph_for_ppr` 接口，在边的返回 dict 中额外携带 `source_id`；
在 `_ppr_rank_chunks` Step 3 中同时从节点和边的 `source_id` 建立 chunk 映射：

```python
# 当前：仅从节点 source_id 建映射
for node in subgraph_nodes:
    for chunk_id in split(node["source_id"]):
        chunk_to_entities[chunk_id].append(node["entity_id"])

# 扩展后：节点 + 边的 source_id 都用上
for edge in subgraph_edges:
    for chunk_id in split(edge.get("source_id", "")):
        chunk_to_entities[chunk_id].append(edge["src"])
        chunk_to_entities[chunk_id].append(edge["tgt"])
```

**需要改动的文件**：

1. `base.py:762-766` — 基础 BFS 实现 `get_edge()` 返回已含 `source_id`，在构建 edge dict 时加上：
   ```python
   "source_id": edge_data.get("source_id", "")
   ```

2. `neo4j_impl.py:1143-1148` — Cypher 已 `RETURN properties(r) AS rprops`，在构建 edge dict 时加上：
   ```python
   "source_id": rprops.get("source_id", "")
   ```

**兼容性**：✅ 两处改动各约 1 行，不破坏现有接口

---

### 方案 C：SYNONYM 边折扣（P1，与方案 B 配套）

**问题**：V2 同义链接创建的 SYNONYM 边与普通事实关系边在 PPR 传播中权重相同，
但语义相似性不等于事实关系，不应有同等强度的传播。

**方案**：在方案 B 扩展边返回信息后，同时携带 `edge_type` 字段。
在 `ppr.py` 的图构建阶段对 SYNONYM 边施加折扣系数：

```python
# ppr.py 图构建
for edge in entity_edges:
    w = float(edge.get("weight", 1.0))
    if edge.get("edge_type") == "SYNONYM":
        w *= synonym_edge_discount   # 默认 0.5
    G.add_edge(src, tgt, weight=w)
```

**参数建议**：
```python
# QueryParam 新增
synonym_edge_discount: float = 0.5   # SYNONYM 边权重折扣
```

**兼容性**：✅ `edge_type` 字段对非 SYNONYM 边为 None，逻辑有默认分支

---

### 方案 D：自适应 damping（P0）

**问题**：固定 `damping=0.5` 对所有查询一视同仁。
PageRank 中 `damping`（即 `alpha`）越高，PageRank 分布越集中在种子附近，
传播越少；越低则传播越广，多跳效果越强。

| 查询类型 | 期望行为 | 建议 damping |
|---------|---------|-------------|
| "X 是什么" | 集中在直接相关实体 | 0.6–0.7 |
| "A 通过什么影响 C" | 允许远距离传播 | 0.3–0.4 |
| 默认 | 折中 | 0.5 |

**方案**：根据种子实体数量自动调整，无需 LLM 调用：

```python
n_seeds = len(entity_seed_weights)
if n_seeds <= 2:
    # 少数精确种子 → 允许更多传播
    effective_damping = max(query_param.ppr_damping - 0.15, 0.2)
elif n_seeds >= 8:
    # 大量种子 → 集中在附近
    effective_damping = min(query_param.ppr_damping + 0.1, 0.8)
else:
    effective_damping = query_param.ppr_damping
```

**实现位置**：`operate.py:5378`（PPR 调用前）

**兼容性**：✅ 纯 Python 逻辑，与所有后端无关

---

### 方案 E：PPR 输出 + Reranker 二次排序（P0）

**问题**：PPR 输出的 chunk 排序纯粹基于图结构传播分数，
可能选出"图上重要但内容无关"的 chunk。
系统已有 reranker（`apply_rerank_if_enabled()`），但未应用到 PPR 结果。

**方案**：PPR 返回 top_k chunks 后，调用已有的 reranker 对内容做语义重排，
再用加权融合得到最终分数：

```python
# operate.py _ppr_rank_chunks 末尾
if query_param.enable_rerank and result_chunks:
    reranked = await apply_rerank_if_enabled(
        query=query,
        retrieved_docs=result_chunks,
        global_config=global_config,
        top_n=len(result_chunks),
    )
    # 融合：α × PPR分(归一化) + (1-α) × rerank分
    alpha = query_param.ppr_rerank_alpha   # 默认 0.6
    ppr_scores = {c["chunk_id"]: c["ppr_score"] for c in result_chunks}
    max_ppr = max(ppr_scores.values()) or 1.0
    for chunk in reranked:
        cid = chunk.get("chunk_id")
        ppr_norm = ppr_scores.get(cid, 0.0) / max_ppr
        rerank_s = chunk.get("rerank_score", 0.0)
        chunk["ppr_score"] = alpha * ppr_norm + (1 - alpha) * rerank_s
    result_chunks = sorted(reranked, key=lambda x: x["ppr_score"], reverse=True)
```

**参数建议**：
```python
# QueryParam 新增
ppr_rerank_alpha: float = 0.6   # PPR 分权重，(1-alpha) 为 rerank 分权重
```

**兼容性**：✅ `apply_rerank_if_enabled` 是后端无关的；reranker 未配置时自动跳过

---

### 方案 F：文档级聚合加分（P2）

**问题**：PPR 独立评估每个 chunk，忽略了"同一文档多个 chunk 都被选中"
这一强信号——多个相关 chunk 来自同一文档，说明该文档整体与查询高度相关。

**方案**：在 `_merge_all_chunks()` 的 PPR 优先合并阶段，
统计每个 `full_doc_id` 出现的 chunk 数，对 ppr_score 施加文档聚合加分：

```python
# _merge_all_chunks PPR 分支
from collections import Counter
doc_chunk_counts = Counter(
    c.get("full_doc_id") for c in ppr_chunks if c.get("full_doc_id")
)
for chunk in ppr_chunks:
    doc_id = chunk.get("full_doc_id")
    if doc_id and doc_chunk_counts[doc_id] > 1:
        bonus = math.log(1 + doc_chunk_counts[doc_id]) * 0.05
        chunk["ppr_score"] = chunk.get("ppr_score", 0.0) + bonus
ppr_chunks.sort(key=lambda x: x.get("ppr_score", 0.0), reverse=True)
```

**数据来源**：`full_doc_id` 存储在 `text_chunks_db` KV 记录中（`TextChunkSchema`），
PPR 取回 chunk 内容时（`operate.py:5395-5404`）已可读到。

**兼容性**：✅ 依赖 `text_chunks_db` KV 存储，所有后端均有此字段

---

## 兼容性汇总

| 方案 | Neo4j | Base图后端 | Qdrant | NanoVectorDB | 是否需改 API |
|------|-------|-----------|--------|--------------|------------|
| A. Hub 惩罚 | ✅ | ✅ | — | — | 否 |
| B. 边 source_id 扩充 | ✅ 改1行 | ✅ 改1行 | — | — | 是（轻微） |
| C. SYNONYM 折扣 | ✅ 改1行 | ✅ 改1行 | — | — | 是（轻微，与B配套） |
| D. 自适应 damping | ✅ | ✅ | — | — | 否 |
| E. PPR + Reranker | ✅ | ✅ | ✅ | ✅ | 否 |
| F. 文档级聚合 | ✅ | ✅ | ✅ | ✅ | 否 |

---

## 新增 QueryParam 参数汇总

```python
# base.py QueryParam 中新增
hub_penalty_threshold: int = 50
"""度数超过此值的实体在 PPR 种子权重中受对数惩罚。0 表示禁用。"""

synonym_edge_discount: float = 0.5
"""SYNONYM 边在 PPR 图中的权重折扣系数（需方案 B/C 支持）。"""

ppr_rerank_alpha: float = 0.6
"""PPR 分与 rerank 分的融合权重。1.0 = 纯 PPR，0.0 = 纯 rerank。"""
```

现有参数无需修改：

- `enable_multi_hop: bool = False`
- `multi_hop_depth: int = 2`
- `ppr_damping: float = 0.5`（自适应 damping 在此基础上浮动）
- `ppr_top_k: int = 50`
- `passage_node_weight: float = 0.05`

---

## 实施顺序建议

```
阶段 1（无 API 改动，可独立验证）
  └─ A. Hub 节点惩罚
  └─ D. 自适应 damping
  └─ E. PPR + Reranker

阶段 2（统一扩展 get_subgraph_for_ppr 接口）
  └─ B. 边 source_id 扩充（改 base.py + neo4j_impl.py 各 1 行）
  └─ C. SYNONYM 折扣（依赖 B）

阶段 3（锦上添花）
  └─ F. 文档级聚合
```

---

## 验证方案

1. **单元测试**：构造包含 hub 节点、SYNONYM 边的测试图，验证各方案对 PPR 输出排序的影响
2. **集成测试**：对比开启/关闭各方案时，固定查询的 PPR chunk 排序变化
3. **端到端**：准备多跳查询集（如"A 通过什么机制影响 C"），对比回答质量
4. **延迟测试**：各方案的额外查询开销（主要是方案 A 的 `node_degrees_batch` 调用）
