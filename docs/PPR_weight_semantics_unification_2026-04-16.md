# PPR 权重语义统一改造（2026-04-16）

## 目标
- 统一 `FACTUAL / SYNONYM / chunk-entity` 在 PPR 中的权重语义。
- 避免 factual 与 synonym 元数据混杂，保证写入与重建路径一致。
- 保持检索侧可切换 synonym 映射，且不要求重建历史库（你本轮会重建）。

## 已落地改动

### 1) FACTUAL：写入侧 `weight_raw + log1p`
- 文件：`lightrag/lightrag/operate.py`
- 规则：
  - `weight_raw = 已有 factual raw + 新 source_id 增量(max per source_id) + 无 source_id 增量`
  - `weight = log1p(weight_raw)`
- 图谱边与关系向量 payload 均写入：
  - `weight`
  - `weight_raw`
  - `edge_type=FACTUAL`
  - `provenance=relation_extraction`
- 兼容旧边：
  - 若旧边无 `weight_raw`，回退使用旧 `weight` 作为基线。

### 2) 重建路径与合并路径同语义
- 文件：`lightrag/lightrag/operate.py`（`_rebuild_single_relationship`）
- 关系重建时同样写入 `weight_raw` 与 `weight=log1p(weight_raw)`，与常规 merge 保持一致。

### 3) SYNONYM：存储不变，检索可映射
- 存储层保持：
  - `synonym_linking.py` 仍写入 `weight = cos`（原始余弦相似度）
- 检索层新增模式参数：
  - `ppr_synonym_weight_mode="raw"`（默认）=> `w_syn = cos`
  - `ppr_synonym_weight_mode="plus_one"` => `w_syn = 1 + cos`
- 生效位置：
  - `lightrag/lightrag/ppr.py`
  - `lightrag/lightrag/ppr_engine.py`

### 4) chunk-entity：固定 1.0
- `ppr.py` 与 `ppr_engine.py` 统一使用 `chunk-entity weight = 1.0`。
- `ppr_engine.py` 中旧的 `chunk_embeddings` 余弦分支已移除语义影响（参数仅兼容保留，不参与权重计算）。

### 5) Neo4j PPR 取边字段补齐
- 文件：`lightrag/lightrag/kg/neo4j_impl.py`
- `get_subgraph_for_ppr` 与 `get_all_nodes_and_edges` 现在返回/透传：
  - `weight`
  - `weight_raw`
  - `source_id`
  - `edge_type`
  - `provenance`

### 6) rag-anything 多模态 belongs_to 对齐
- 文件：
  - `rag-anything/raganything/processor.py`
  - `rag-anything/raganything/modalprocessors.py`
- `belongs_to` 权重从 `10.0` 改为 `1.0`。
- 直接写图路径补齐 factual 元数据模板：
  - `weight=1.0`
  - `weight_raw=1.0`
  - `edge_type=FACTUAL`
  - `provenance=relation_extraction`

### 7) 新参数与服务层透传
- `lightrag/lightrag/base.py`：
  - `QueryParam.ppr_synonym_weight_mode: Literal["raw","plus_one"] = "raw"`
- `lightrag` API：
  - `api/routers/query_routes.py` 新增请求字段 `ppr_synonym_weight_mode`
- `rag-anything`：
  - `raganything/constants.py` 新增 `DEFAULT_PPR_SYNONYM_WEIGHT_MODE = "raw"`
  - `raganything/services/local_rag.py` 读取环境变量
    - `RAGANYTHING_PPR_SYNONYM_WEIGHT_MODE`
  - `raganything/query.py` multimodal cache key 纳入 `ppr_synonym_weight_mode`
  - `rag-anything/server/app.py` QueryRequest 与调用链透传该字段

## 本次验证
- 离线测试 + 语义测试：通过
- 真实 Neo4j + Qdrant 集成测试：支持自动跳过无服务环境；本地执行结果为 skip（无服务）

## 删除链路补充修复（重建中断/无缓存场景）
- 文件：`lightrag/lightrag/operate.py`
- 问题：
  - 之前 `rebuild_knowledge_from_chunks` 在无 cached extraction 时直接返回，可能导致关系边保留被删除文档的旧 `source_id`。
- 修复：
  - 无缓存时不再直接退出，改为继续执行 rebuild 流程。
  - `_rebuild_single_relationship` 在无关系抽取数据时，启用 source-only fallback：
    - 仍更新 `source_id` 为 remaining chunks；
    - 用比例法收敛 `weight_raw`（按 remaining/source_count 比例缩放当前 raw）；
    - 重新写回图边与关系向量记录，保持 `FACTUAL` 模板字段一致。
- 效果：
  - 删除某文档后，即使缺少 LLM cache，也不会把被删 chunk 残留在关系边 `source_id` 中。

## 3-doc 验证结论（本轮）
- 使用 3 个文档验证：
  - 建库写入完整（KV / 图谱 / 向量）；
  - factual/synonym/chunk-entity 权重语义符合改造目标；
  - 模拟“重建中断后重试”可恢复，最终数据一致；
  - 删除 1 个 doc 后，其它 docs 的图/向量/KV 保持完整。

## 备注
- `custom_kg` 关系权重未做自动缩放，保持原语义。
- 你当前会重建图谱，因此本次不包含历史边回填工具。

## 2026-04-17 更新：Recognition-Memory 提示词 Token 保护
- 作用范围：global PPR 的 recognition-memory LLM 步骤（`_recognition_memory_filter`）。
- 新增硬性 token 上限参数：
  - `recognition_prompt_max_tokens`（默认 `65536`）
  - `recognition_prompt_reserved_tokens`（默认 `4096`）
- 实际可用预算：`max_tokens - reserved_tokens`。
- 当候选内容超预算时，构造器会按预算裁剪 entity 行 / fact 行，并保持顺序优先级（不裁剪 query）。
- 发生裁剪时会记录 `warning` 日志，便于在实验日志中直接定位是否触发了 token 保护。
- 目标：避免 recognition prompt 超出模型上下文窗口，保证 PPR seed filtering 稳定执行。
- 配置入口：
  - LightRAG：`RECOGNITION_PROMPT_MAX_TOKENS`、`RECOGNITION_PROMPT_RESERVED_TOKENS`
  - LocalRAG 环境变量：`RAGANYTHING_RECOGNITION_PROMPT_MAX_TOKENS`、`RAGANYTHING_RECOGNITION_PROMPT_RESERVED_TOKENS`
