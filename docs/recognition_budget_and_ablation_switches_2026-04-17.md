# Recognition Token 预算与 Ablation 开关基线（2026-04-17）
## 1. 本次改动目标
- 统一 `recognition_top_k` 的默认值来源，去掉脚本和代码中的分散硬编码。
- 让 global PPR 的 recognition LLM 步骤具备“输入预算 + 输出预算 + 预留预算”的闭环控制。
- 固化 ablation 运行时“除 V1/V2/V3 外，其余新增开关默认开启”的实验基线。

## 2. Recognition 预算语义（新增）
- `recognition_prompt_max_tokens`：模型上下文硬上限（通常等于模型 max context length，例如 65536）。
- `recognition_prompt_output_max_tokens`：recognition 这一步的 LLM 输出上限（默认 8192，可调）。
- `recognition_prompt_reserved_tokens`：系统包装和额外开销预留（默认 200）。

实际 prompt 可用预算：
`prompt_budget = recognition_prompt_max_tokens - recognition_prompt_output_max_tokens - recognition_prompt_reserved_tokens`

系统会保证最小 prompt 预算地板（256 token），并在超过预算时自动裁剪候选 entity/fact 行，同时打印 warning。

## 3. 当前默认值
- `recognition_top_k = 20`
- `recognition_prompt_max_tokens = 65536`
- `recognition_prompt_output_max_tokens = 8192`
- `recognition_prompt_reserved_tokens = 200`

## 4. Ablation 严谨性基线（当前）
- 由 ablation profile 控制的仅有：`enable_entity_disambiguation`、`enable_synonym_linking`、`enable_multi_hop`（以及对应 PPR 参数）。
- 其余新增开关在 DocBench/SurGE 评测入口统一固定为开启，避免受外部环境变量漂移影响：
  - `enable_entity_surface_normalization = true`
  - `enable_keyword_case_normalization = true`
  - `strict_relation_endpoint_entity_match = true`
  - `enable_type_based_context_window_override = true`
  - `context_zero_window_content_types = page_number,page_footnote,footer,header,ref_text`
  - recognition 预算参数固定为“当前默认值”中的基线。

## 5. 与 4/13 版本对比（recognition_top_k）
- 2026-04-13 版本中，ablation CLI 默认 `recognition_top_k = 10`。
- 当前统一为 `20`，并改为从常量读取，避免 `evaluate_shared` / `run_ablation_evals` / `QueryParam` 出现默认值分裂。
