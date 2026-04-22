# Rerank Batch Backoff（2026-04-22）

## 目标

- 给本地 CrossEncoder rerank 增加显存保护，避免一次性用过大 batch 触发 OOM。
- 保持当前质量优先策略：继续对同一批候选做完整 rerank，不把 `rerank_score_scope` 改成预裁剪。
- 统一覆盖 chunk rerank 和 KG entity/relation rerank，因为它们都走 `raganything/services/local_rag.py` 的同一个 `build_rerank_func()`。

## 新增常量

位于 `raganything/constants.py`：

- `DEFAULT_RERANK_BATCH_SIZE = 32`
- `DEFAULT_RERANK_ENABLE_OOM_BACKOFF = True`
- `DEFAULT_RERANK_MIN_BATCH_SIZE = 4`

对应 `LocalRagSettings` 字段：

- `rerank_batch_size`
- `rerank_enable_oom_backoff`
- `rerank_min_batch_size`

支持环境变量覆盖：

- `RAGANYTHING_RERANK_BATCH_SIZE`
- `RAGANYTHING_RERANK_ENABLE_OOM_BACKOFF`
- `RAGANYTHING_RERANK_MIN_BATCH_SIZE`

## 执行语义

- 默认先用 `batch_size=32` 调 `CrossEncoder.predict(...)`。
- 仅当异常满足 OOM-like 判定时才进入退避。
- 退避链固定为：`32 -> 16 -> 8 -> 4`。
- 每次退避都会：
  - 丢弃当前尝试的全部中间结果；
  - best-effort 执行 `gc.collect()`；
  - 若 `torch.cuda` 可用，则再执行 `torch.cuda.empty_cache()`；
  - 用更小 batch 从头重跑整次 rerank。
- 到 `batch_size=4` 仍 OOM：
  - 本次 rerank 直接返回空结果；
  - 上层沿用现有 fallback，回退到原始召回结果，不返回半成品 rerank 分数。

## 重要边界

- 不承诺不同 batch size 下 score 数值完全一致。GPU 推理通常不能保证 bit-identical。
- 本次能保证的是：同一次成功 rerank 的最终结果只来自某一个 batch size 的一次完整重跑，不会混用不同 batch size 的部分结果。
- 非 OOM 异常不会进入 backoff 链路，仍按普通 rerank 错误处理。

## 可观测性

启动日志新增：

- `Rerank batch guard configured: batch_size=..., oom_backoff=..., min_batch_size=...`

运行日志新增：

- `Rerank OOM backoff: ... Retrying full rerank from scratch.`
- `Rerank OOM fallback: ... Falling back to original retrieved items without rerank.`
