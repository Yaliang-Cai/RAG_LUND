# 实体表面名规范化与严格关系端点开关说明（2026-04-17）

## 适用范围
- 运行链路：`LocalRagService`（`Neo4j + Qdrant`）。
- 覆盖阶段：实体/关系抽取、关系合并与重建、query 关键词后处理、global PPR recognition-memory。
- 设计目标：所有新增行为均由开关控制，且彼此正交；全部关闭时可回退到旧行为。

## 开关与参数

### 1) `enable_entity_surface_normalization`
- 位置：
  - `raganything.constants.DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION`
  - `lightrag.constants.DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION`
- 默认值：`True`
- 开启效果：
  - 仅在抽取解析阶段生效。
  - 执行顺序：先 sanitize/filter，再做 surface normalization。
  - 逐词规范化规则：
    - allowlist/缩写词：转大写；
    - 含内部大小写语义的词（如 `OpenAI`、`iPhone`）：保持原样；
    - 其余大小写不敏感词：转 Title Case。
  - 示例：`Machine learning -> Machine Learning`。

### 2) `enable_keyword_case_normalization`
- 位置：
  - `raganything.constants.DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION`
  - `lightrag.constants.DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION`
- 默认值：`True`
- 开启效果：
  - query `high_level_keywords`：默认小写，保留有语义的大写/混合大小写专名与缩写；
  - query `low_level_keywords`：复用实体规范化逻辑；
  - relation keyword 合并：大小写不敏感去重，并输出规范化结果；
  - `high_level_keywords` 与 relation `keywords` 使用同一套归一化函数，保证检索口径一致。

### 3) `entity_uppercase_allowlist`
- 位置：
  - `raganything.constants.DEFAULT_ENTITY_UPPERCASE_ALLOWLIST`
  - `lightrag.constants.DEFAULT_ENTITY_UPPERCASE_ALLOWLIST`
- 默认值：预置缩写列表（`LLM`、`RAG`、`API`、`BERT`、`6G` 等）。
- 作用：
  - 为实体规范化与关键词大小写归一提供统一缩写白名单。

### 4) `strict_relation_endpoint_entity_match`
- 位置：
  - `raganything.constants.DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
  - `lightrag.constants.DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
- 默认值：`True`
- 开启效果：
  - 关系写入与重建时，若任一端点实体不存在，则跳过该关系并清理对应边/向量；
  - 防止 fallback 自动写入 `UNKNOWN` 端点导致图谱污染。

### 5) `recognition_prompt_max_tokens`
### 6) `recognition_prompt_reserved_tokens`
- 位置：
  - `raganything.constants.DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS`
  - `raganything.constants.DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS`
  - `lightrag.constants.DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS`
  - `lightrag.constants.DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS`
- 默认值：
  - `recognition_prompt_max_tokens = 65536`
  - `recognition_prompt_reserved_tokens = 4096`
- 作用：
  - 仅影响 `mode="ppr"` 下的 global recognition-memory LLM prompt；
  - 生效预算为 `max_tokens - reserved_tokens`；
  - 超预算时自动裁剪候选实体/候选关系行（不裁剪 query），避免 prompt 触发模型上下文超限。

## 环境变量映射

### LocalRagService（`RAGANYTHING_*`）
- `RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION`
- `RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION`
- `RAGANYTHING_ENTITY_UPPERCASE_ALLOWLIST`
- `RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
- `RAGANYTHING_RECOGNITION_PROMPT_MAX_TOKENS`
- `RAGANYTHING_RECOGNITION_PROMPT_RESERVED_TOKENS`

### LightRAG（`ENABLE_*` / 直连变量）
- `ENABLE_ENTITY_SURFACE_NORMALIZATION`
- `ENABLE_KEYWORD_CASE_NORMALIZATION`
- `ENTITY_UPPERCASE_ALLOWLIST`
- `STRICT_RELATION_ENDPOINT_ENTITY_MATCH`
- `RECOGNITION_PROMPT_MAX_TOKENS`
- `RECOGNITION_PROMPT_RESERVED_TOKENS`

## 回退矩阵（正交）
- `enable_entity_surface_normalization=False`
  - 实体/关系端点不做 surface normalization。
- `enable_keyword_case_normalization=False`
  - query/relation 关键词保持原抽取大小写，不做归一化。
- `strict_relation_endpoint_entity_match=False`
  - 恢复旧兜底行为（允许缺端点关系继续写入并可能生成 `UNKNOWN`）。
- 将上述开关全部关闭：
  - 行为可回退到旧逻辑。
