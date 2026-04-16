# RRF v2 Upgrade Plan

## 目标

将 `mode="rrf"` 升级为 5 路检索 → RRF 融合 → rerank → threshold 过滤的完整 pipeline。

---

## 当前架构（v1）

```
dense vector (query_embedding)  → top-k chunks  ─┐
entity graph path               → chunks        ─┤ RRF → rerank → top-k
relation graph path             → chunks        ─┘
```

---

## 目标架构（v2）

```
dense  (top 20)        ─┐
sparse / SPLADE (top 20)─┤
HyDE vector (top 20)   ─┤ RRF → rerank → threshold → top-k
entity  → chunks       ─┤
relation → chunks      ─┘
```

每路独立检索，结果在 RRF 前去重，RRF 输出后再 rerank + threshold 过滤。

---

## 改动模块

### 1. Qdrant Collection 重建（ingestion 侧）

- Collection schema 新增 `sparse` 命名向量字段（SPLADE 格式）
- 入库时同时计算并存储 dense + sparse 两种向量
- 影响范围：`lightrag/lightrag/storage/` Qdrant 相关存储类

**待定参数**
- SPLADE 模型选型（本地推理 or API）

---

### 2. Sparse 检索路（query 侧）

- `_get_vector_context` 新增 sparse 检索分支，或拆出独立函数 `_get_sparse_context`
- 查询时向 Qdrant 发送 sparse 向量，取 top 20
- 若 sparse 字段不存在则跳过（向后兼容）

---

### 3. HyDE 检索路

- 新增 `_generate_hyde_embedding(query, llm_func, embedding_func, n)` 异步函数：
  1. 调用 LLM 生成 `n` 个假设性文档
  2. 对 query + n 个文档分别 embed
  3. 取均值 → HyDE vector
- 用 HyDE vector 查 Qdrant dense，取 top 20
- 与原 dense 路独立，结果分别进入 RRF

**待定参数**
- `hyde_n`：生成文档数（候选 1 / 3）
- `hyde_prompt`：指导 LLM 生成文档的 prompt 模板
- 是否复用 `llm_response_cache` 缓存 HyDE 文档

---

### 4. `_rrf_merge` 扩展

- 当前已支持任意数量 ranking_lists，无需修改核心逻辑
- 调用方传入 `[dense_chunks, sparse_chunks, hyde_chunks, entity_chunks, relation_chunks]`，空列表自动跳过

**待定参数**
- `rrf_k`：已有，默认 60

---

### 5. QueryParam 新增字段

```python
# RRF v2
hyde_n: int = 3                  # HyDE 假设文档数量
hyde_enabled: bool = True        # 是否启用 HyDE 路
sparse_enabled: bool = True      # 是否启用 sparse 路
rrf_chunk_top_k: int = 20        # 每路检索取 top-k
```

---

### 6. 去重时机

| 阶段 | 位置 |
|------|------|
| 各路内部 | Qdrant 已去重 |
| dense / sparse / HyDE 三路之间 | 进入 RRF 前，按 chunk_id 去重（保留首次出现） |
| entity / relation chunk 与前三路之间 | 进入 RRF 前统一去重 |
| RRF 输出后 | `_rrf_merge` 已按 chunk_id 合分，天然去重 |

---

### 7. 后处理（rerank + threshold）

- rerank 已有基础设施，RRF 模式当前是否走 rerank 待确认
- threshold 过滤：rerank score < threshold 的 chunk 丢弃
- 最终保留 `chunk_top_k` 个 chunks

**待定参数**
- `rrf_rerank_threshold`：rerank score 下限（候选 0.0 ~ 0.5）

---

## 实施顺序

1. Qdrant schema 重建 + ingestion sparse 写入
2. `_get_sparse_context` 实现 + 向后兼容检测
3. `_generate_hyde_embedding` + HyDE 检索路
4. `kg_query` 中组装 5 路，调用 `_rrf_merge`
5. `QueryParam` 新增字段
6. rerank threshold 过滤
7. 集成测试：各路可单独开关，验证降级行为

---

## 非目标

- 不改动 local / global / hybrid / mix / ppr 等其他 mode
- 不改 `_rrf_merge` 核心算法
- sparse 向量不建独立 collection（选用同 collection 命名向量方案）
